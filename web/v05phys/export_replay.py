"""export_replay.py -- inlet replays of V0.5 (Codex inlet) and of the all-physical set (strictB3 / strictfp90) for the web bench.

For each pattern the inlet the solver receives is sampled at t_k = k/120 s (k = 0 .. ceil(T*120)) from
  * V0.5 original: latte_imex.v05_program.Program (Codex controls, common_parameters, return law v05)
  * all-physical : ref2/drv.py PhysV05 with the strictB3 options (P4 closure c_p 1.36, beta 3, nu 1e-3; P2 per-stroke U and
                   B_dep/p_dep/kappa_r; pulsation <= 0.41; lateral jet oscillation = hand speed (latfix, wiggle patterns);
                   flow_phase 0 for tulip/leaf/swan; layered_heart = strictfp90 (flow_phase pi/2, post-hoc action re-guess))
and written, quantised to uint16 (base64, little-endian, value = lo + (hi - lo) * q / 65535), to web/v05phys/<name>.json.
Also renders the 256^2 final images (live-canvas orientation: row 0 = +y_M = T frame row 0) and the real-video images.

usage: python3 -I export_replay.py [--only heart,...]
"""
import argparse, base64, json, math, os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
REPO = "/home/user/zfChen"
SCR = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review"
REF2 = os.path.join(SCR, "v05allphys", "ref2")
sys.path.insert(0, REPO); sys.path.insert(0, REF2)
import numpy as np
from PIL import Image
from latte_imex.v05_program import Program, load_control
import drv as DRV  # ref2/drv.py (independent all-physical driver)

WEB = os.path.join(REPO, "web")
OUT = os.path.join(WEB, "v05phys")
PATTERNS = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
CN = dict(heart="大白心", push_heart="推推乐", layered_heart="千层心", tulip="压纹郁金香", leaf="树叶", swan="天鹅")
FPS = 120
D_M, G, ML, V0 = 0.08, 9.81, 512.0, 0.35
# strictB3 options exactly as ref2/round4.sh, round3b.sh
STRICT = dict(heart=dict(tag="strictB3", latfix=False, fp="auto"), push_heart=dict(tag="strictB3", latfix=False, fp="auto"),
              tulip=dict(tag="strictB3", latfix=True, fp=0.0), leaf=dict(tag="strictB3", latfix=True, fp=0.0),
              swan=dict(tag="strictB3", latfix=True, fp=0.0), layered_heart=dict(tag="strictfp90", latfix=True, fp=math.pi / 2))
LABEL_CN = {"reposition (no flow)": "移位（无流量）", "stop": "停", "wiggle": "摆动", "pour": "注入", "fine cut": "细切"}


def label_cn(s):
    head, _, kind = s.partition(": ")
    n = head.replace("phase ", "")
    return f"第 {n} 笔 · {LABEL_CN.get(kind, kind)}"


def v05_set(pattern):
    prog = Program(load_control(pattern))
    P = dict(prog.config["parameters"])
    P["return_law"] = "v05" if prog.config.get("return_law", {}).get("mode") == "return_speed" else "v1"
    return prog, P, None


def phys_set(pattern):
    E = json.load(open(os.path.join(SCR, "v05allphys", "allphys_experiments.json")))
    e, ue = E["P4"], E["P2"]
    params = dict(e["params"])
    for key in ("B_dep", "p_dep", "kappa_r"):
        params[key] = ue["params"][key]
    o = STRICT[pattern]
    fp = next((ch["value"] for ch in e["changes"] if ch.get("field") == "flow_phase"), None) if o["fp"] == "auto" else o["fp"]
    spec = ue["inlet"][pattern]
    prog = DRV.PhysV05(pattern, spec["U"], spec["kappa_Q"], params, 0.41, fp, latfix=o["latfix"])
    P = dict(prog.config["parameters"])
    P["return_law"] = "v05" if prog.config.get("return_law", {}).get("mode") == "return_speed" else "v1"
    # cross-check against the parameters the saved 256^2 run used
    m = json.load(open(os.path.join(REF2, "runs", f"{o['tag']}_{pattern}_N256", "metrics.json")))
    for k, v in m["params"].items():
        assert (P[k] == v) if isinstance(v, str) else abs(P[k] - v) < 1e-12, (pattern, k, P[k], v)
    return prog, P, dict(tag=o["tag"], flow_phase=fp, latfix=o["latfix"], kappa_Q=spec["kappa_Q"],
                         U_ms={int(k): v for k, v in spec["U"].items()})


def sample(prog, T, kind):
    K = int(math.ceil(T * FPS - 1e-9))
    rows, ks = [], []
    for k in range(K + 1):
        t = k / FPS
        if t >= T - 1e-12:   # V05Script.sample: inactive at t >= T (the run stops at T)
            ph = prog.phase_index(T - 1e-9)
            x, y = (rows[-1][1], rows[-1][2]) if rows else (0.0, 0.0)
            rows.append((0, x, y, 0, 0, 0, 0, 0, 0, 0, 0, 0)); ks.append(ph)
            continue
        inl = prog(t)
        ph = prog.phase_index(t)
        a = bool(inl.active and inl.S_eff > 0)
        rows.append((1 if a else 0, inl.x_hit[0], inl.x_hit[1], inl.S_eff if a else 0.0, inl.u_in[0] if a else 0.0,
                     inl.u_in[1] if a else 0.0, inl.U_perp if a else 0.0, inl.d_jet if a else 0.0, inl.r1 if a else 0.0,
                     inl.r2 if a else 0.0, inl.phi if a else 0.0, inl.scan_speed))
        ks.append(ph)
    return np.array(rows, float), np.array(ks, int)


def q16(a, lo=None, hi=None):
    a = np.asarray(a, float)
    lo = float(np.min(a)) if lo is None else lo
    hi = float(np.max(a)) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    q = np.round((a - lo) / (hi - lo) * 65535).clip(0, 65535).astype("<u2")
    return dict(lo=lo, hi=hi, b64=base64.b64encode(q.tobytes()).decode("ascii"))


def deq16(e):
    q = np.frombuffer(base64.b64decode(e["b64"]), "<u2").astype(float)
    return e["lo"] + (e["hi"] - e["lo"]) * q / 65535


def build(pattern):
    out = dict(name=pattern, cn=CN[pattern], fps=FPS, frame="M (solver): x right, y up; spout toward -y, barista/handle at +y",
               units="x, y, d, r1, r2 [D]; S [1/s]; ux, uy, U, scan [D/s]; phi [rad]; Q [mL/s]; z [cm]; D = 0.08 m",
               sets={})
    for key, maker in (("v05", v05_set), ("phys", phys_set)):
        prog, P, meta = maker(pattern)
        T = prog.duration
        R, ks = sample(prog, T, key)
        act = R[:, 0] > 0
        S, U, d = R[:, 3], R[:, 6], R[:, 7]
        kQ = P["kappa_Q"]
        Q = np.where(act, S / kQ * ML, 0.0)                                    # received flow [mL/s]
        Zc = np.where(act, U * U / np.maximum(d, 1e-12), 0.0)
        chi = np.where(act, 1.0 / (1.0 + (Zc / P["B_dep"]) ** P["p_dep"]), 0.0)
        Ums = U * D_M
        if key == "phys":
            z = np.where(act, (Ums ** 2 - V0 ** 2) / (2 * G) * 100, np.nan)  # spout height [cm]
            # between strokes: linear in time between the neighbouring strokes (display only)
            idx = np.arange(len(z)); good = ~np.isnan(z)
            z = np.interp(idx, idx[good], z[good])
        else:
            z = None
        cols = dict(active=R[:, 0], x=R[:, 1], y=R[:, 2], S=S, ux=R[:, 4], uy=R[:, 5], U=U, d=d, r1=R[:, 8], r2=R[:, 9],
                    phi=R[:, 10], scan=R[:, 11], Q=Q, chi=chi)
        if z is not None:
            cols["z"] = z
        enc = {}
        for k, a in cols.items():
            if k == "active":
                enc[k] = dict(lo=0.0, hi=1.0, b64=base64.b64encode(a.astype(np.uint8).tobytes()).decode("ascii"), u8=True)
            elif k == "phi":
                enc[k] = q16(a, -math.pi, math.pi)
            elif k == "chi":
                enc[k] = q16(a, 0.0, 1.0)
            else:
                enc[k] = q16(a)
        enc["phase"] = dict(u8=True, lo=0.0, hi=1.0, b64=base64.b64encode(ks.astype(np.uint8).tobytes()).decode("ascii"))
        labels = []
        for i, p in enumerate(prog.phases):
            tmid = 0.5 * (prog.events[i] + prog.events[i + 1])
            labels.append(label_cn(prog.phase_label(tmid)))
        # per-stroke physical summary (active samples)
        strokes = []
        for i in sorted(set(ks[act].tolist())):
            s = act & (ks == i)
            row = dict(k=i, label=labels[i], t0=float(prog.events[i]), t1=float(prog.events[i + 1]),
                       Q=[float(Q[s].min()), float(Q[s].max())], U_ms=[float(Ums[s].min()), float(Ums[s].max())],
                       uh_ms=[float(np.hypot(R[s, 4], R[s, 5]).min() * D_M), float(np.hypot(R[s, 4], R[s, 5]).max() * D_M)],
                       r_mm=[float(np.sqrt(R[s, 8] * R[s, 9]).min() * D_M * 1e3), float(np.sqrt(R[s, 8] * R[s, 9]).max() * D_M * 1e3)],
                       chi_Qw=float((chi[s] * Q[s]).sum() / max(Q[s].sum(), 1e-12)))
            if z is not None:
                row["z_cm"] = float(z[s].mean())
            strokes.append(row)
        out["sets"][key] = dict(T=T, n=len(R), params=P, labels=labels, cols=enc, strokes=strokes, meta=meta)
    return out


# ------------------------------------------------------------------------------------------------- images
BROWN = np.array([0x6b, 0x40, 0x1f], float); MILK = np.array([0xf7, 0xf2, 0xe8], float); OUTSIDE = np.array([0x5a, 0x3a, 0x1e], float)


def colour(c, mask, gamma=True):
    v = np.clip(c / 0.8, 0, 1) ** 0.8 if gamma else np.clip(c, 0, 1)
    rgb = BROWN * (1 - v[..., None]) + MILK * v[..., None]
    rgb[~mask] = OUTSIDE
    return rgb


def save_img(rgb, path, size=256):
    im = Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8))
    if im.size[0] != size:
        im = im.resize((size, size), Image.LANCZOS)
    im.save(path, "WEBP", quality=88, method=6)


def c_from_state(path):
    z = np.load(path)
    if "c" in z.files:
        return z["c"].astype(float), z["mask"].astype(bool)
    mask = z["mask"]
    return np.where(mask, z["m"] / np.where(mask, z["l"], 1.0), 0.0), mask


# leaf and swan targets are not in the canonical frame of the simulations (latte_imex/stage_scripts/leaf.json: B side at -y_M,
# 180 deg from the convention; swan.json: rot 0, B->D axis at +45 deg instead of -90 deg): the same fixed correction as
# review/v05real/compare.py (PHI, rotate_T) is applied to the real-video images shown next to the simulation
PHI = dict(heart=0.0, push_heart=0.0, layered_heart=0.0, tulip=0.0, leaf=180.0, swan=135.0)


def rotate_T(fT, phi_deg, order=1):
    """copy of review/v05real/compare.py rotate_T: T-frame field -> canonical T-frame field, f_can(q) = f(R(phi) q)."""
    from scipy import ndimage
    M = fT.shape[0]
    if phi_deg == 0.0:
        return fT.copy()
    g = (np.arange(M) + 0.5) / M - 0.5
    X, Y = np.meshgrid(g, g)
    t = math.radians(phi_deg)
    px = math.cos(t) * X - math.sin(t) * Y
    py = math.sin(t) * X + math.cos(t) * Y
    I = (-py + 0.5) * M - 0.5
    J = (px + 0.5) * M - 0.5
    if fT.ndim == 2:
        out = ndimage.map_coordinates(fT, [I, J], order=order, mode="nearest")
    else:
        out = np.stack([ndimage.map_coordinates(fT[..., k], [I, J], order=order, mode="nearest") for k in range(fT.shape[2])], -1)
    return out[::-1, :]


def images(pattern):
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    c, m = c_from_state(os.path.join(REPO, "runs", "v05_256", pattern, "final_state.npz"))
    save_img(colour(c[::-1], m[::-1]), os.path.join(OUT, "img", f"{pattern}_v05.webp"))       # live-canvas orientation
    tag = STRICT[pattern]["tag"]
    c, m = c_from_state(os.path.join(REF2, "runs", f"{tag}_{pattern}_N256", "c.npz"))
    save_img(colour(c[::-1], m[::-1]), os.path.join(OUT, "img", f"{pattern}_phys.webp"))
    # real video: whiteness (T frame, row 0 at the top = physical top view = live-canvas orientation)
    t = np.load(os.path.join(REPO, "latte_imex", "targets", f"{pattern}.npz"))
    w_can = np.clip(rotate_T(t["w"].astype(float), PHI[pattern]), 0.0, 1.0)
    save_img(colour(w_can, t["mask"].astype(bool), gamma=False), os.path.join(OUT, "img", f"{pattern}_realw.webp"))
    # real video: rectified photo of the target frame (same T grid)
    from latte_imex import video_extract as VE
    fr = np.load(os.path.join(REPO, "latte_imex", "video_targets", pattern, "frames.npz"))
    rgb = np.asarray(Image.open(os.path.join(REPO, "latte_imex", "targets", f"{pattern}_frame.png")).convert("RGB"), float) / 255
    M = int(t["M"])
    rect = VE.rectify(rgb, fr["ell"][-1], M, float(fr["rot"]), bool(fr["flip"]))
    disk = VE.disk_mask(M)
    rect = rotate_T(rect, PHI[pattern]) * 255.0
    rect[~disk] = OUTSIDE
    save_img(rect, os.path.join(OUT, "img", f"{pattern}_photo.webp"))
    # sanity: the rectified photo's luminance correlates with the whiteness target in the same orientation
    L = rect.mean(-1)
    w = w_can; mk = t["mask"].astype(bool) & disk
    return dict(corr_photo_whiteness=float(np.corrcoef(L[mk], w[mk])[0, 1]),
                corr_photo_whiteness_flipped=float(np.corrcoef(L[::-1][mk], w[mk])[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-img", action="store_true")
    ap.add_argument("--img-only", action="store_true")
    a = ap.parse_args()
    pats = a.only.split(",") if a.only else PATTERNS
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for p in pats:
        r = {}
        if not a.img_only:
            o = build(p)
            with open(os.path.join(OUT, f"{p}.json"), "w") as f:
                json.dump(o, f, ensure_ascii=False, separators=(",", ":"))
            r = dict(T={k: s["T"] for k, s in o["sets"].items()}, n={k: s["n"] for k, s in o["sets"].items()},
                     size_kB=os.path.getsize(os.path.join(OUT, f"{p}.json")) / 1024)
        if not a.no_img:
            r["img"] = images(p)
        report[p] = r
        print(p, json.dumps(r))
    json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_report_img.json" if a.img_only else "export_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
