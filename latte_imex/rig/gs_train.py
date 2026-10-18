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


# CAFEDE KONA "Allround" 700 mL milk jug (the rig pitcher): outer radius [mm] vs height [mm], catalogue drawing
KONA700_Z = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 118, 122]
KONA700_R = [38.0, 39.3, 43.5, 44.0, 44.0, 44.0, 44.0, 43.7, 43.5, 43.3, 42.8, 42.6, 42.3, 42.2, 42.1, 42.1]


class Shape:
    """Known pitcher shape (catalogue profile, 1 mm wall, spout lip ~53.5 mm) as a geometric prior.  The catalogue shape is
    placed and scaled to the reconstruction: bottom z0 and height H from the depth points along the object z axis,
    radial scale from the mid-height radius; spout azimuth given (handle opposite, left free: thin wire + hand).
    dist(P): distance [mm] of points P [m] to the wall shell (outer and inner wall, inside bottom); 0 in the spout and
    handle regions."""

    def __init__(self, xyz, spout_az_deg, device):
        P = xyz * 1e3
        z, r = P[:, 2], np.hypot(P[:, 0], P[:, 1])
        self.z0 = float(np.percentile(z, 0.5)); self.H = float(np.percentile(z, 99.5)) - self.z0
        mid = (z > self.z0 + 0.25 * self.H) & (z < self.z0 + 0.6 * self.H)
        rc = np.interp(np.linspace(0.25, 0.6, 50) * 122.0, KONA700_Z, KONA700_R).mean()
        self.rs = float(np.median(r[mid]) / rc)
        self.spout = np.radians(spout_az_deg)
        self.lut = torch.tensor(np.interp(np.arange(0, 122.01, 0.5), KONA700_Z, KONA700_R) * self.rs, device=device).float()
        self.dev = device
        print(f"shape prior: bottom z0 {self.z0:.1f} mm, height {self.H:.1f} mm (catalogue 122), radial scale "
              f"{self.rs:.3f}, spout azimuth {spout_az_deg:.0f} deg", flush=True)

    def r_out(self, z_mm):
        zc = ((z_mm - self.z0) / self.H * 122.0).clamp(0, 122.0)
        return self.lut[(zc / 0.5).round().long()]

    def dist(self, P):
        mm = P * 1e3
        x, y, z = mm[:, 0], mm[:, 1], mm[:, 2]
        r = torch.hypot(x, y)
        ro = self.r_out(z)
        ve = F.relu(self.z0 - z) + F.relu(z - (self.z0 + self.H))
        d_wall = torch.minimum((r - ro).abs(), (r - (ro - 1.0)).abs())
        d_side = torch.sqrt(d_wall ** 2 + ve ** 2)
        d_bot = torch.sqrt(F.relu(r - ro) ** 2 + torch.minimum((z - self.z0).abs(), (z - self.z0 - 1.5).abs()) ** 2)
        d = torch.minimum(d_side, d_bot)
        az = torch.atan2(y, x)
        da = torch.remainder(az - self.spout + math.pi, 2 * math.pi) - math.pi
        top = self.z0 + self.H
        spout = (da.abs() < math.radians(35)) & (z > top - 30) & (z < top + 2) & (r > ro - 3) & (r < 53.5 * self.rs + 3)
        dh = torch.remainder(az - self.spout, 2 * math.pi) - math.pi
        handle = (dh.abs() < math.radians(30)) & (z > self.z0 + 3) & (z < top + 4) & (r < ro + 45)
        return torch.where(spout | handle, torch.zeros_like(d), d)

    def init_points(self, xyz, rgb):
        """Surface samples of the shell (outer wall, inner wall near the rim, inside bottom, spout lip) coloured from the
        nearest depth point, plus the depth points that lie on the shell or in the handle region."""
        from scipy.spatial import cKDTree
        ro = lambda zz: np.interp(np.clip((zz - self.z0) / self.H * 122.0, 0, 122), KONA700_Z, KONA700_R) * self.rs
        S = []
        for zz in np.arange(self.z0 + 0.5, self.z0 + self.H, 1.2):
            R = ro(zz); ph = np.arange(0, 2 * np.pi, 1.2 / R)
            S.append(np.stack([R * np.cos(ph), R * np.sin(ph), np.full_like(ph, zz)], 1))
            if zz > self.z0 + self.H - 35:
                S.append(np.stack([(R - 1) * np.cos(ph), (R - 1) * np.sin(ph), np.full_like(ph, zz)], 1))
        for rr in np.arange(0, ro(self.z0 + 2) - 1, 1.5):
            ph = np.arange(0, 2 * np.pi, 1.5 / max(rr, 1.5))
            S.append(np.stack([rr * np.cos(ph), rr * np.sin(ph), np.full_like(ph, self.z0 + 1.5)], 1))
        for zz in np.arange(self.z0 + self.H - 25, self.z0 + self.H, 1.0):
            for da in np.radians(np.arange(-30, 31, 2.0)):
                for rr in np.arange(ro(zz), 53.5 * self.rs, 1.0):
                    a_ = self.spout + da
                    S.append(np.array([[rr * np.cos(a_), rr * np.sin(a_), zz]]))
        S = np.concatenate(S) * 1e-3
        d, j = cKDTree(xyz).query(S, distance_upper_bound=0.006)
        col = np.where(np.isfinite(d)[:, None], rgb[np.minimum(j, len(rgb) - 1)], np.median(rgb, 0))
        keep = (self.dist(torch.tensor(xyz, device=self.dev)) < 3.0).cpu().numpy()
        print(f"shape prior init: {len(S)} surface samples + {keep.sum()} of {len(xyz)} depth points", flush=True)
        return np.concatenate([S, xyz[keep]]).astype(np.float32), np.concatenate([col, rgb[keep]]).astype(np.float32)


class CadShape:
    """Kona 700 mL CAD (pitcher_kona700_prior.npz from latte_imex/pitcher_geom/make_cad_prior.py: surface samples +
    unsigned distance grid, mm, spout +x, inside bottom at z = 0) as the geometric prior: thin wall, flat facets,
    spout lip and both handle straps.  Placed like Shape (bottom z0 and height H from the depth points, radial scale
    from the mid-height radius) and turned so that the spout is at --spout-az.  free_handle: no distance penalty
    behind the body (x_cad < -40 mm), where the barista's hand holds the handle in the pour recordings."""

    def __init__(self, xyz, spout_az_deg, device, prior, free_handle=False):
        Z = np.load(prior)
        self.dev = device
        self.free_handle = free_handle
        self.grid = torch.tensor(Z["dist"].astype(np.float32), device=device)[None, None]     # [1, 1, X, Y, Z]
        self.lo = torch.tensor(Z["lo"], device=device).float()
        self.n = torch.tensor(Z["dist"].shape, device=device).float()
        self.pts, self.nrm = Z["pts"].astype(float), Z["nrm"].astype(float)
        P = xyz * 1e3
        z, r = P[:, 2], np.hypot(P[:, 0], P[:, 1])
        mid = (z > np.percentile(z, 0.5) + 30) & (z < np.percentile(z, 0.5) + 70)
        print(f"CAD prior: depth/CAD mid radius ratio {np.median(r[mid]) / float(Z['r_mid']):.3f} (metric data: scale "
              f"kept at 1)", flush=True)
        self.rs = self.sz = 1.0                                  # RealSense depth and CAD are both metric
        # place: bottom height and spout azimuth by a grid search (mean clipped distance of the depth points)
        zb = float(np.percentile(z, 0.5)) - float(self.pts[:, 2].min())
        sub = torch.tensor(xyz[np.random.default_rng(0).choice(len(xyz), min(len(xyz), 20000), replace=False)],
                           device=device).float()
        best = (1e9, zb, spout_az_deg)
        for dz in np.arange(-6.0, 6.01, 1.0):
            for daz in np.arange(-30.0, 30.01, 2.0):
                self.z0, self.spout = zb + dz, math.radians(spout_az_deg + daz)
                sc = float(self.dist(sub).clamp(max=5.0).mean())
                if sc < best[0]:
                    best = (sc, zb + dz, spout_az_deg + daz)
        for dz in np.arange(-1.0, 1.01, 0.25):
            for daz in np.arange(-2.0, 2.01, 0.5):
                self.z0, self.spout = best[1] + dz, math.radians(best[2] + daz)
                sc = float(self.dist(sub).clamp(max=5.0).mean())
                if sc < best[0]:
                    best = (sc, best[1] + dz, best[2] + daz)
        self.z0, self.spout = best[1], math.radians(best[2])
        print(f"CAD prior placed: CAD z = 0 at {self.z0:.1f} mm, spout azimuth {best[2]:.1f} deg (given {spout_az_deg:.0f}), "
              f"mean clipped distance of the depth points {best[0]:.2f} mm, {len(self.pts)} surface samples"
              f"{', handle free' if free_handle else ''}", flush=True)

    def to_cad(self, mm):
        c, s = math.cos(self.spout), math.sin(self.spout)
        x = (c * mm[:, 0] + s * mm[:, 1]) / self.rs
        y = (-s * mm[:, 0] + c * mm[:, 1]) / self.rs
        return x, y, (mm[:, 2] - self.z0) / self.sz

    def to_obj(self, q):
        c, s = math.cos(self.spout), math.sin(self.spout)
        x, y = q[:, 0] * self.rs, q[:, 1] * self.rs
        return np.stack([c * x - s * y, s * x + c * y, q[:, 2] * self.sz + self.z0], 1)

    def dist(self, P):
        x, y, z = self.to_cad(P * 1e3)
        q = torch.stack([x, y, z], 1)
        u = (q - self.lo) / (self.n - 1) * 2 - 1                     # grid_sample: last dim (x, y, z) -> (W, H, D)
        g = torch.stack([u[:, 2], u[:, 1], u[:, 0]], 1).view(1, -1, 1, 1, 3)
        d = F.grid_sample(self.grid, g, mode="bilinear", padding_mode="border", align_corners=True).view(-1)
        hi = self.lo + self.n - 1
        d = d + (F.relu(self.lo - q) + F.relu(q - hi)).norm(dim=1)
        d = d * min(self.rs, self.sz)
        if self.free_handle:
            d = torch.where(x < -40.0, torch.zeros_like(d), d)
        return d

    def init_points(self, xyz, rgb):
        """CAD surface samples coloured from the nearest depth point, plus the depth points on the CAD surface (and,
        with free_handle, those around the handle: hand)."""
        from scipy.spatial import cKDTree
        S = self.to_obj(self.pts) * 1e-3
        d, j = cKDTree(xyz).query(S, distance_upper_bound=0.006)
        col = np.where(np.isfinite(d)[:, None], rgb[np.minimum(j, len(rgb) - 1)], np.median(rgb, 0))
        keep = (self.dist(torch.tensor(xyz, device=self.dev).float()) < 3.0).cpu().numpy()
        print(f"CAD prior init: {len(S)} surface samples + {keep.sum()} of {len(xyz)} depth points", flush=True)
        return np.concatenate([S, xyz[keep]]).astype(np.float32), np.concatenate([col, rgb[keep]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--sh", type=int, default=3)
    ap.add_argument("--w-alpha", type=float, default=0.3)
    ap.add_argument("--w-depth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--volume", default=None, help="rmax,zmin,zmax [m] of the object frame: Gaussians outside are removed "
                    "every 500 steps and before export (floaters on the table, background)")
    ap.add_argument("--grow-grad2d", type=float, default=0.0004)
    ap.add_argument("--shape", default=None, help="'kona700': catalogue profile, 'kona_cad': the CAD (pitcher_kona700_prior.npz) as a prior (init on the surface, "
                    "distance penalty, pruning); needs --spout-az")
    ap.add_argument("--spout-az", type=float, default=90.0, help="spout azimuth [deg] (atan2(y, x)) in the object frame")
    ap.add_argument("--w-shape", type=float, default=0.05)
    ap.add_argument("--prior", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitcher_kona700_prior.npz"),
                    help="CAD prior for --shape kona_cad")
    ap.add_argument("--free-handle", action="store_true", help="kona_cad: no shape penalty around the handle (hand)")
    a = ap.parse_args()
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    torch.manual_seed(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    views, xyz, rgb = load(a.data, dev)
    shape = (Shape(xyz, a.spout_az, dev) if a.shape == "kona700" else
             CadShape(xyz, a.spout_az, dev, a.prior, a.free_handle) if a.shape == "kona_cad" else None)
    if shape is not None:
        xyz, rgb = shape.init_points(xyz, rgb)
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
    s = knn_scale(xyz).astype(np.float32)
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
                               refine_every=100, prune_opa=0.005, grow_grad2d=a.grow_grad2d, grow_scale3d=0.01,
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

    vol = [float(x) for x in a.volume.split(",")] if a.volume else None

    def outside():
        m = params["means"].detach()
        out = torch.zeros(len(m), dtype=torch.bool, device=m.device)
        if vol is not None:
            out |= (torch.hypot(m[:, 0], m[:, 1]) > vol[0]) | (m[:, 2] < vol[1]) | (m[:, 2] > vol[2])
        if shape is not None:
            out |= shape.dist(m) > 5.0
        return out

    def prune_outside(st):
        try:
            from gsplat.strategy.ops import remove
            mk = outside()
            if mk.any():
                remove(params=params, optimizers=optimizers, state=st, mask=mk)
        except Exception as e:                            # older / newer gsplat: export-time filtering still applies
            print("  (in-training pruning skipped:", e, ")", flush=True)

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
        if shape is not None:                              # keep the Gaussians on the known shell (1.5 mm tolerance)
            loss = loss + a.w_shape * (F.relu(shape.dist(params["means"]) - 1.5) ** 2).mean()
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        cc_opt.step(); cc_opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if (vol is not None or shape is not None) and step % 500 == 499 and step < a.iters - 1:
            prune_outside(state)
        if step % 500 == 0 or step == a.iters - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}  l1 {l1.item():.4f}  alpha {float(la):.4f}  depth {float(ld):.3f}  "
                  f"gaussians {len(params['means'])}  {time.time() - t0:.0f} s", flush=True)

    # ---------------------------------------------------------------- outputs
    if vol is not None or shape is not None:
        with torch.no_grad():
            keep = ~outside()
            for k in list(params.keys()):
                params[k] = torch.nn.Parameter(params[k][keep])
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
