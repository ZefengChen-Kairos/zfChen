"""Check every stage script against its extracted clip: impact point and pitcher side.

    python -m latte_imex.video_targets.check_stages            # all six -> <name>/check_overlay.png, check_stages.json

For each 5 fps frame inside a pour stage (low_pour / wiggle / push / retreat / lift / high_cut):
  * d_foot = distance between the scripted impact point (stage_point, M frame) and the hand-picked stream feet
             script['foot_measurements_M'] at their times [cup diameters]; stage_point leaves out the wiggle, so wiggle
             stages are reported separately (their d_foot includes the wiggle half-amplitude).  (frames.npz['tip'] is
             the extractor's pitcher-tip estimate near the rim, not the foot, and is not used.)
  * spout  = direction (M frame) of the white within 0.03-0.15 D of the scripted foot during low_pour / wiggle / retreat
             stages (mean unit vector from the foot; frames where it is shorter than 0.2 are skipped as undirected).  The
             jet's horizontal momentum carries new foam downstream, so this is the spout direction = the script's
             Barista yaw (default -pi/2: spout toward -y_M, pitcher body at +y_M; leaf +pi/2, swan +pi/4).  A script
             with the pitcher on the wrong side gives errors near 180 deg.  The occluded pixels are NOT used: with the
             oblique camera the pitcher's shadow on the rectified disc is displaced along the view direction.
"""
import json
import math
import os

import numpy as np

from latte_imex import video_extract as VE

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "stage_scripts")
POUR = ("low_pour", "wiggle", "push", "retreat", "lift", "high_cut")


script_yaw = VE.script_yaw


def check(name):
    d = VE.load_stages(os.path.join(SCRIPTS, f"{name}.json"))
    z = np.load(os.path.join(HERE, name, "frames.npz"))
    M = int(z["M"]); x = (np.arange(M) + 0.5) / M - 0.5
    X, Y = np.meshgrid(x, x)
    disk = z["disk"].astype(bool)
    rows = []
    for k, t in enumerate(z["t"]):
        p, s = VE.stage_point(d, float(t))
        if p is None or s["kind"] not in POUR:
            continue
        pT = VE.m_to_t(p)
        side_err = float("nan"); az = float("nan")
        if s["kind"] in ("low_pour", "wiggle", "retreat"):
            r = np.hypot(X - pT[0], Y - pT[1])
            wb = disk & z["valid"][k] & (z["white"][k] > 0.5) & (r > 0.03) & (r < 0.15)
            if wb.sum() > 0.003 * disk.sum():
                u = np.stack([X[wb] - pT[0], Y[wb] - pT[1]], 1)
                v = VE.t_to_m((u / np.linalg.norm(u, axis=1, keepdims=True)).mean(0))
                if np.hypot(*v) > 0.2:
                    az = math.atan2(v[1], v[0])
                    side_err = math.degrees((az - script_yaw(d) + math.pi) % (2 * math.pi) - math.pi)
        rows.append(dict(k=k, t=round(float(t), 2), kind=s["kind"], p_M=[round(float(a), 3) for a in p],
                         white_dir_deg=round(math.degrees(az), 1) if az == az else None,
                         spout_err_deg=round(side_err, 1) if side_err == side_err else None))
    feet = {float(k): np.array(v, float) for k, v in (d.get("foot_measurements_M") or {}).items()}
    dw, dn = [], []
    for tm, f in feet.items():
        p, s = VE.stage_point(d, tm)
        if p is not None:
            (dw if s["kind"] == "wiggle" else dn).append(float(np.hypot(*(p - f))))
    se = np.array([r["spout_err_deg"] for r in rows if r["spout_err_deg"] is not None], float)
    summ = dict(name=name, yaw_deg=round(math.degrees(script_yaw(d)), 1), n_frames=len(rows),
                n_feet=len(feet), d_foot_median=round(float(np.median(dn)), 3) if dn else None,
                d_foot_p90=round(float(np.percentile(dn, 90)), 3) if dn else None,
                d_foot_wiggle_median=round(float(np.median(dw)), 3) if dw else None,
                spout_ok_frac=round(float(np.mean(np.abs(se) < 60)), 2) if len(se) else None,
                spout_err_median_deg=round(float(np.median(se)), 1) if len(se) else None, frames=rows)
    plot(name, z, d, rows, summ, feet)
    return summ


def plot(name, z, d, rows, summ, feet, n=12):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not rows:
        return
    pick = [rows[i] for i in np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)]
    yaw = script_yaw(d)
    fig, axes = plt.subplots(2, (len(pick) + 1) // 2, figsize=(2.6 * ((len(pick) + 1) // 2), 5.6), dpi=90, squeeze=False)
    for ax, r in zip(axes.ravel(), pick):
        k = r["k"]
        w = z["white"][k].astype(float); v = z["valid"][k]
        img = np.stack([w, w, w], -1) * 0.8 + 0.1
        img[~v] = [0.75, 0.25, 0.25]
        img[~z["disk"].astype(bool)] = 1.0
        ax.imshow(img, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        pT = VE.m_to_t(np.array(r["p_M"]))
        ax.plot(*pT, "+", color="lime", ms=12, mew=2)
        near = [tm for tm in feet if abs(tm - r["t"]) <= 0.15]
        if near:
            ax.plot(*VE.m_to_t(feet[min(near, key=lambda tm: abs(tm - r["t"]))]), "x", color="cyan", ms=9, mew=2)
        bT = VE.m_to_t(np.array([math.cos(yaw), math.sin(yaw)]))
        ax.annotate("", xy=pT + 0.2 * bT, xytext=pT, arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
        ax.set_title(f"{r['t']:.1f}s {r['kind']}" + (f"\nspout err {r['spout_err_deg']} deg" if r["spout_err_deg"] is not None else ""), fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{name}: + script, x hand-picked foot, arrow = spout direction from yaw {summ['yaw_deg']} deg "
                 f"(T frame, red = occluded); script-foot median {summ['d_foot_median']} D, spout err median {summ['spout_err_median_deg']} deg", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name, "check_overlay.png"))
    plt.close(fig)


if __name__ == "__main__":
    out = {}
    for n in VE.PATTERNS:
        s = check(n)
        out[n] = s
        print(n, {k: v for k, v in s.items() if k != "frames"}, flush=True)
    json.dump(out, open(os.path.join(HERE, "check_stages.json"), "w"), indent=1)
