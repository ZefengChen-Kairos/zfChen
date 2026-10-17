"""3D Gaussian splatting of one object from a posed RGB-D dataset (gs_static.py / gs_dataset.py output), with gsplat.

Losses per training view (one view per step, crops of different sizes):
  colour   L1 + 0.2 (1 - SSIM) on object pixels (mask 255); a per-view affine colour correction (gain, bias per
           channel) absorbs exposure / white-balance differences between cameras and windows
  alpha    rendered opacity -> 1 on object pixels, -> 0 on background pixels (mask 0); ignore pixels (128) free
  depth    expected depth vs the measured depth on object pixels (weight decays, RealSense depth is noisy)
Densification / pruning: gsplat DefaultStrategy.  SH degree grows to 3.  Held-out views (every 7th) for PSNR.
Outputs in --out: point_cloud.ply (INRIA 3DGS format: SuperSplat / antimatter15 / any 3DGS viewer), turntable.mp4,
views.jpg (GT | render for held-out and training views), metrics.json.

    python -m latte_imex.rig.gs_train --data DIR --out DIR [--iters 15000]
"""
import argparse, json, math, os, time

import numpy as np
import cv2
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814


def load(data, device):
    T = json.load(open(os.path.join(data, "transforms.json")))
    views = []
    for f in T["frames"]:
        img = cv2.imread(os.path.join(data, "images", f["name"] + ".png"))[..., ::-1].copy()
        m = cv2.imread(os.path.join(data, "masks", f["name"] + ".png"), 0)
        d = cv2.imread(os.path.join(data, "depths", f["name"] + ".png"), -1).astype(np.float32) * 1e-3
        c2o = np.array(f["c2o"], np.float64)
        K = np.array([[f["fx"], 0, f["cx"]], [0, f["fy"], f["cy"]], [0, 0, 1]], np.float32)
        views.append(dict(name=f["name"], img=torch.tensor(img, device=device).float() / 255.0,
                          obj=torch.tensor(m == 255, device=device), bg=torch.tensor(m == 0, device=device),
                          depth=torch.tensor(d, device=device), K=torch.tensor(K, device=device),
                          viewmat=torch.tensor(np.linalg.inv(c2o), device=device).float(), W=img.shape[1], H=img.shape[0],
                          c2o=c2o))
    z = np.load(os.path.join(data, "init_points.npz"))
    return views, z["xyz"].astype(np.float32), z["rgb"].astype(np.float32) / 255.0


def ssim(a, b, w=11):
    """a, b [1, 3, H, W] in 0..1."""
    g = torch.exp(-((torch.arange(w, device=a.device) - w // 2) ** 2) / (2 * 1.5 ** 2)).float()
    g = (g / g.sum())
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x, k, padding=w // 2, groups=3)
    mu_a, mu_b = f(a), f(b)
    s_aa, s_bb, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))).mean()


def knn_scale(xyz, k=4):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz).query(xyz, k=k)
    return np.sqrt(np.mean(d[:, 1:] ** 2, 1)).clip(2e-4, 5e-3)


def save_ply(path, params):
    means = params["means"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(len(means), -1)                 # [N, 3]
    shN = params["shN"].detach().cpu().numpy().transpose(0, 2, 1).reshape(len(means), -1)   # INRIA: channel-major
    op = params["opacities"].detach().cpu().numpy()[:, None]
    sc = params["scales"].detach().cpu().numpy()
    q = F.normalize(params["quats"].detach(), dim=-1).cpu().numpy()
    cols = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(shN.shape[1])] + \
           ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.concatenate([means, np.zeros_like(means), sh0, shN, op, sc, q], 1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(arr)).encode())
        for c in cols:
            f.write(f"property float {c}\n".encode())
        f.write(b"end_header\n")
        f.write(arr.tobytes())


def look_at(eye, target, up=(0, 0, 1.0)):
    """world -> camera (OpenCV: x right, y down, z forward)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ eye
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--sh", type=int, default=3)
    ap.add_argument("--w-alpha", type=float, default=0.3)
    ap.add_argument("--w-depth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    torch.manual_seed(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    views, xyz, rgb = load(a.data, dev)
    for i, v in enumerate(views):
        v["test"] = len(views) >= 14 and i % 7 == 3
    test = [v for v in views if v["test"]]
    train = [v for v in views if not v["test"]]
    cams = np.array([v["c2o"][:3, 3] for v in views])
    centre = np.median(xyz, 0)
    scene_scale = float(np.linalg.norm(cams - centre, axis=1).max() * 1.1)
    print(f"{len(views)} views ({len(train)} train, {len(test)} test), {len(xyz)} init points, scene scale {scene_scale:.3f} m",
          flush=True)
    N = len(xyz)
    s = knn_scale(xyz)
    K_sh = (a.sh + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(xyz, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(s, device=dev)[:, None].repeat(1, 3))),
        "quats": torch.nn.Parameter(F.normalize(torch.rand(N, 4, device=dev) - 0.5, dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.3, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(rgb, device=dev) - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(N, K_sh - 1, 3, device=dev)),
    })
    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / a.iters))
    cc = torch.nn.Parameter(torch.zeros(len(views), 6, device=dev))          # per view: log-gain (3), bias (3)
    cc_opt = torch.optim.Adam([cc], lr=1e-3)
    vid = {v["name"]: i for i, v in enumerate(views)}
    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(a.iters * 0.7), reset_every=3000,
                               refine_every=100, prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01,
                               prune_scale3d=0.1, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(v, sh_deg):
        out, alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=torch.cat([params["sh0"], params["shN"]], 1),
            viewmats=v["viewmat"][None], Ks=v["K"][None], width=v["W"], height=v["H"], sh_degree=sh_deg,
            near_plane=0.01, far_plane=5.0, render_mode="RGB+ED", packed=False)
        return out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0], info

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    for step in range(a.iters):
        v = train[rng.integers(len(train))]
        sh_deg = min(a.sh, step // 1000)
        rgb_r, dep_r, alp, info = render(v, sh_deg)
        i = vid[v["name"]]
        rgb_c = rgb_r * torch.exp(cc[i, :3]) + cc[i, 3:]
        o, b = v["obj"], v["bg"]
        gt = v["img"]
        l1 = (rgb_c - gt).abs()[o].mean()
        fill = torch.where(o[..., None], gt, rgb_c.detach())
        ls = 1 - ssim(rgb_c.permute(2, 0, 1)[None], fill.permute(2, 0, 1)[None])
        la = (1 - alp[o]).mean() + (alp[b].mean() if b.any() else 0.0)
        dm = o & (v["depth"] > 0)
        wd = a.w_depth * max(0.1, 1 - step / (0.6 * a.iters))
        ld = ((dep_r[dm] - v["depth"][dm]).abs().clamp(max=0.02) / 0.005).mean() if dm.any() else 0.0
        loss = 0.8 * l1 + 0.2 * ls + a.w_alpha * la + wd * ld + 1e-3 * (cc[i] ** 2).sum()
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        cc_opt.step(); cc_opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 500 == 0 or step == a.iters - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}  l1 {l1.item():.4f}  alpha {float(la):.4f}  depth {float(ld):.3f}  "
                  f"gaussians {len(params['means'])}  {time.time() - t0:.0f} s", flush=True)

    # ---------------------------------------------------------------- outputs
    save_ply(os.path.join(a.out, "point_cloud.ply"), params)
    res = dict(n_gaussians=len(params["means"]), iters=a.iters, wall_s=time.time() - t0, views=len(views))
    rows, ps = [], []
    with torch.no_grad():
        for v in (test or train)[:8] + train[:4]:
            rgb_r, _, alp, _ = render(v, a.sh)
            i = vid[v["name"]]
            rgb_c = (rgb_r * torch.exp(cc[i, :3]) + cc[i, 3:]).clamp(0, 1)
            if v["test"]:
                mse = ((rgb_c - v["img"]) ** 2)[v["obj"]].mean().item()
                ps.append(10 * math.log10(1 / max(mse, 1e-10)))
            g = (v["img"].cpu().numpy() * 255).astype(np.uint8)
            r = (rgb_c.cpu().numpy() * 255).astype(np.uint8)
            tile = np.hstack([g, r])[..., ::-1].copy()
            cv2.putText(tile, ("test " if v["test"] else "train ") + v["name"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tile = cv2.resize(tile, (480, int(480 * tile.shape[0] / tile.shape[1])))
            rows.append(tile)
        Wm = max(t.shape[1] for t in rows)
        cv2.imwrite(os.path.join(a.out, "views.jpg"), np.vstack([cv2.copyMakeBorder(t, 0, 4, 0, Wm - t.shape[1], cv2.BORDER_CONSTANT) for t in rows]))
        res["test_psnr_object"] = float(np.mean(ps)) if ps else None
        # turntable around the object's vertical axis (z of the object frame; for the pour dataset: pitcher axis)
        import imageio
        Kt = torch.tensor([[700.0, 0, 320], [0, 700.0, 320], [0, 0, 1]], device=dev)
        lo, hi = np.percentile(xyz, 2, 0), np.percentile(xyz, 98, 0)
        tgt = (lo + hi) / 2
        rad = float(np.linalg.norm(hi - lo)) * 1.6
        frames = []
        for k in range(180):
            th = 2 * math.pi * k / 180
            el = math.radians(25 + 15 * math.sin(2 * th))
            eye = tgt + rad * np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
            vm = torch.tensor(look_at(eye, tgt), device=dev).float()
            v = dict(viewmat=vm, K=Kt, W=640, H=640)
            rgb_r, _, alp, _ = render(v, a.sh)
            rgb_w = rgb_r + (1 - alp[..., None])                 # white background
            frames.append((rgb_w.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        imageio.mimwrite(os.path.join(a.out, "turntable.mp4"), frames, fps=30, quality=8)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
