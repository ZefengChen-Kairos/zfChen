"""compare.py -- final simulated patterns vs the real video's final frame (six patterns).

Simulations (all 256^2, final state, M frame c_M[iy, ix]; physical top view = M frame, x right, y up):
  V05      V0.5 original (Codex inlet)              /home/user/zfChen/runs/v05_256/<p>/final_state.npz (c = m / l)
  P1       all-physical inlet except c_p = 0.3      v05allphys/runs/P1/<p>_N256.npz            (intermediate reference)
  PHYS     all-physical strictB3                     v05allphys/ref2/runs/strictB3_<p>_N256/c.npz
           (layered_heart: strictfp90 = strictB3 + flow-pulse peak on the inward swing, a post-hoc action re-guess)
Real: latte_imex/targets/<p>.npz (w, mask; T frame, M_t = 192, leaf 256), video white = w > 0.5.

Frame.  Solver M frame vs target T frame: (x, y)_M = (x, -y)_T, i.e. c_T = c_M[::-1, :] (frame_check.json).  The V0.5
controls are laid out in the canonical pattern frame (pour starts at B = +y_M, cut exits toward D = -y_M).  Four targets
(heart, push_heart, layered_heart, tulip) were rotated by hand into that frame (targets.FRAMES rot).  Two are documented
in the repo as NOT in it: leaf (stage_scripts/leaf.json: B at -y_M, "180 deg from the convention", compare as
c_M[:, ::-1]) and swan (stage_scripts/swan.json: rot 0 "not rotated into the canonical orientation", B->D axis at +45 deg
in M).  A fixed frame correction PHI (not fitted) brings these two targets into the canonical frame:
  PHI = 180 deg (leaf), 135 deg (swan: +45 - (-90)), 0 for the other four.
The target is resampled: w_can(q) = w(R(PHI) q) (exact for 0 and 180 deg; bilinear for the swan), mask = the same disk.
Every metric below is computed in this canonical frame on the target grid; the uncorrected numbers (c_M[::-1, :] vs the
target as stored) are kept for leaf and swan as 'uncorrected'.

Metrics (same operator for every simulation; cf_runner functions):
  sim to grid  bilinear, nearest-filled outside the sim cup (cf_runner.sample / nearest_fill)
  IoU          (s > 0.4) & mask vs (w > 0.5) & mask, and (s > 0.5) & mask vs the same video mask
  boundary     symmetric mean boundary distance [D]: boundary = white cell with a non-white 4-neighbour inside the cup;
               0.5 (mean_{A bdry} d(., B bdry) + mean_{B bdry} d(., A bdry)), Euclidean, cell size 1/M_t
  white frac   white cells / cup cells
  gaps         cf_runner.axis_gaps on lines x = 0, +-0.05 D (sim thr 0.4 and 0.5, video thr 0.5)
Alignments (never mixed):
  raw          as simulated (no transform; in the canonical frame)
  best         rigid transform of the simulated pattern about the cup centre: rotation theta in [-25, 25] deg (CCW in the
               physical top view), shift (dx, dy) in [-0.06, 0.06] D each (physical x right, y up), maximising IoU(c > 0.4);
               grid 1 deg / 0.01 D, then refined 0.25 deg / 0.0025 D; the same transform for every other metric.
Diagnostic (not an alignment): best IoU(c > 0.4) over theta in [-180, 180) step 5 deg and shifts {-0.06..0.06 step 0.03} D,
  also for the mirrored simulation -- only to show whether the optimum lies outside the +-25 deg window.
"""
import json, math, os, sys, time
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
REPO = "/home/user/zfChen"
sys.path.insert(0, os.path.join(REV, "v05audit"))
sys.path.insert(0, REPO)
import cf_runner as CF  # noqa: E402

PAT = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
PHI = dict(heart=0.0, push_heart=0.0, layered_heart=0.0, tulip=0.0, leaf=180.0, swan=135.0)
TH_MAX, SH_MAX = 25.0, 0.06


def sim_paths(p):
    phys_tag = "strictfp90" if p == "layered_heart" else "strictB3"
    return dict(V05=os.path.join(REPO, "runs", "v05_256", p, "final_state.npz"),
                P1=os.path.join(REV, "v05allphys", "runs", "P1", f"{p}_N256.npz"),
                PHYS=os.path.join(REV, "v05allphys", "ref2", "runs", f"{phys_tag}_{p}_N256", "c.npz")), phys_tag


def load_sim(path):
    z = np.load(path)
    m = z["mask"]
    if "c" in z.files:
        c = np.where(m, z["c"].astype(np.float64), np.nan)
    else:
        c = np.where(m, z["m"] / np.where(m, z["l"], 1.0), np.nan)
    return c, m


def rotate_T(fT, phi_deg, order=1):
    """T-frame field (cell-centred [iy, ix]) -> canonical T-frame field: f_can(q) = f(R(phi) q), q in the M frame."""
    M = fT.shape[0]
    if phi_deg == 0.0:
        return fT.copy()
    g = (np.arange(M) + 0.5) / M - 0.5
    X, Y = np.meshgrid(g, g)                      # canonical M-frame grid (rows <-> y_M)
    t = math.radians(phi_deg)
    px = math.cos(t) * X - math.sin(t) * Y
    py = math.sin(t) * X + math.cos(t) * Y        # source point in the target's M frame
    # target T frame: y_T = -y_M ; row index = (y_T + 0.5) M - 0.5
    I = (-py + 0.5) * M - 0.5
    J = (px + 0.5) * M - 0.5
    if fT.ndim == 2:
        out = ndimage.map_coordinates(fT, [I, J], order=order, mode="nearest")
    else:
        out = np.stack([ndimage.map_coordinates(fT[..., k], [I, J], order=order, mode="nearest") for k in range(fT.shape[2])], -1)
    return out[::-1, :]                           # canonical M -> canonical T


def target(p, canonical=True):
    w, wm = CF.load_target(p)
    if canonical and PHI[p] != 0.0:
        w = np.clip(rotate_T(w, PHI[p]), 0.0, 1.0)
    return w, wm


class Sampler:
    """sim c_M -> target grid under a rigid transform (theta CCW, shift in the physical/M frame); output T frame."""

    def __init__(self, cM, mM, Mt, mirror=False):
        self.N = cM.shape[0]
        ff = CF.nearest_fill(np.nan_to_num(cM), mM)
        self.ff = ff[:, ::-1] if mirror else ff
        g = (np.arange(Mt) + 0.5) / Mt - 0.5
        self.X, self.Y = np.meshgrid(g, g)
        self.Mt = Mt

    def field_T(self, th_deg=0.0, dx=0.0, dy=0.0):
        t = math.radians(th_deg)
        qx, qy = self.X - dx, self.Y - dy
        px = math.cos(t) * qx + math.sin(t) * qy     # R(-theta) q
        py = -math.sin(t) * qx + math.cos(t) * qy
        s = CF.sample(self.ff, self.N, px.ravel(), py.ravel()).reshape(self.Mt, self.Mt)
        return s[::-1, :]


def bdry(A, cup):
    nb = ndimage.binary_dilation(~A & cup, structure=ndimage.generate_binary_structure(2, 1))
    return A & nb


def boundary_distance(A, B, cup):
    Mt = A.shape[0]
    bA, bB = bdry(A, cup), bdry(B, cup)
    if not bA.any() or not bB.any():
        return dict(sym=float("nan"), a_to_b=float("nan"), b_to_a=float("nan"))
    dB = ndimage.distance_transform_edt(~bB) / Mt
    dA = ndimage.distance_transform_edt(~bA) / Mt
    ab, ba = float(dB[bA].mean()), float(dA[bB].mean())
    return dict(sym=0.5 * (ab + ba), a_to_b=ab, b_to_a=ba)


def gap_summary(g):
    return dict(n_axis=g["n_gaps_axis"], n_total=g["n_gaps_total"],
                per_line=[dict(x=L["x"], n=L["n_gaps"], widths_mm=L["widths_mm"], y_T=L["y_T"]) for L in g["lines"]],
                median_width_mm=g["median_width_mm"], n_dips_total=g["n_dips_total"])


def metrics(sT, w, wm):
    vw = (w > CF.THR_VID) & wm
    a4, a5 = (sT > 0.4) & wm, (sT > 0.5) & wm
    bd4, bd5 = boundary_distance(a4, vw, wm), boundary_distance(a5, vw, wm)
    g4 = CF.axis_gaps(np.where(wm, sT, np.nan), wm, 0.4)
    g5 = CF.axis_gaps(np.where(wm, sT, np.nan), wm, 0.5)
    return dict(iou_c04=CF.iou(a4, vw), iou_c05=CF.iou(a5, vw),
                bdist_c04_D=bd4["sym"], bdist_c04_dir_D=[bd4["a_to_b"], bd4["b_to_a"]], bdist_c05_D=bd5["sym"],
                white_frac_c04=float(a4.sum() / wm.sum()), white_frac_c05=float(a5.sum() / wm.sum()),
                gaps_c04=gap_summary(g4), gaps_c05=gap_summary(g5))


def best_align(smp, w, wm):
    vw = (w > CF.THR_VID) & wm

    def J(th, dx, dy):
        s = smp.field_T(th, dx, dy)
        return CF.iou((s > 0.4) & wm, vw)
    best = (-1.0, 0.0, 0.0, 0.0)
    ths = np.arange(-TH_MAX, TH_MAX + 1e-9, 1.0)
    shs = np.round(np.arange(-SH_MAX, SH_MAX + 1e-9, 0.01), 4)
    for th in ths:
        for dx in shs:
            for dy in shs:
                v = J(th, dx, dy)
                if v > best[0]:
                    best = (v, float(th), float(dx), float(dy))
    _, th0, dx0, dy0 = best
    for th in np.clip(th0 + np.arange(-1.0, 1.0 + 1e-9, 0.25), -TH_MAX, TH_MAX):
        for dx in np.clip(dx0 + np.arange(-0.01, 0.01 + 1e-9, 0.0025), -SH_MAX, SH_MAX):
            for dy in np.clip(dy0 + np.arange(-0.01, 0.01 + 1e-9, 0.0025), -SH_MAX, SH_MAX):
                v = J(th, dx, dy)
                if v > best[0]:
                    best = (v, float(th), float(dx), float(dy))
    v, th, dx, dy = best
    at = dict(theta=bool(abs(abs(th) - TH_MAX) < 1e-6), dx=bool(abs(abs(dx) - SH_MAX) < 1e-6), dy=bool(abs(abs(dy) - SH_MAX) < 1e-6))
    return dict(theta_deg=round(th, 3), dx_D=round(dx, 4), dy_D=round(dy, 4), iou_c04=v, at_bound=at)


def full_circle(cM, mM, w, wm):
    vw = (w > CF.THR_VID) & wm
    out = {}
    for mir in (False, True):
        smp = Sampler(cM, mM, w.shape[0], mirror=mir)
        best = (-1, 0, 0, 0)
        for th in np.arange(-180.0, 180.0, 5.0):
            for dx in (-0.06, -0.03, 0.0, 0.03, 0.06):
                for dy in (-0.06, -0.03, 0.0, 0.03, 0.06):
                    s = smp.field_T(th, dx, dy)
                    v = CF.iou((s > 0.4) & wm, vw)
                    if v > best[0]:
                        best = (v, float(th), dx, dy)
        out["mirrored" if mir else "as_simulated"] = dict(iou_c04=best[0], theta_deg=best[1], dx_D=best[2], dy_D=best[3])
    return out


def meta():
    return dict(
        units="D = 8 cm cup diameter; boundary distance in D (x 80 = mm); gap widths in mm",
        frames="sim c_M compared as c_M[::-1, :] in the target T frame; leaf and swan targets first rotated into the canonical "
               "pattern frame by PHI (documented frame correction, not fitted); figures show the physical top view (no mirror), "
               "canonical frame (B / pour start at top, cut exit / stem at bottom)",
        PHI_deg=PHI,
        thresholds=dict(sim=[0.4, 0.5], video=0.5),
        align_raw="as simulated, no transform (canonical frame)",
        align_best=f"rotation |theta| <= {TH_MAX} deg (CCW, physical top view) about the cup centre + shift |dx|,|dy| <= {SH_MAX} D "
                   "(physical x right, y up), maximising IoU(c>0.4) on a 1 deg / 0.01 D grid refined to 0.25 deg / 0.0025 D; "
                   "the same transform for all other metrics",
        diagnostic_full_circle="best IoU(c>0.4) over theta in [-180,180) step 5 deg x shifts {-0.06,-0.03,0,0.03,0.06} D, as simulated "
                               "and mirrored; NOT an alignment used in the tables",
        sims=dict(V05="V0.5 original, Codex inlet, runs/v05_256", P1="all-physical inlet with c_p = 0.3 (v05allphys P1)",
                  PHYS="all-physical strictB3 (layered_heart: strictfp90, flow-pulse peak at phase pi/2 = post-hoc action re-guess)"),
        gap_rule="cf_runner.axis_gaps: lines x = 0, -0.05, +0.05 D (T frame), 0.1 mm steps, white runs < 0.5 mm dropped, "
                 "gap = brown run >= 1 mm strictly between two white runs")


def main(pats):
    for p in pats:
        t0 = time.time()
        w, wm = target(p, canonical=True)
        Mt = w.shape[0]
        vg = CF.axis_gaps(np.where(wm, w, np.nan), wm, CF.THR_VID)
        rec = dict(M_target=Mt, PHI_deg=PHI[p],
                   video=dict(white_frac=float(((w > 0.5) & wm).sum() / wm.sum()), gaps=gap_summary(vg)), sims={})
        if PHI[p] != 0.0:
            w0, _ = target(p, canonical=False)
            rec["video"]["white_frac_uncorrected"] = float(((w0 > 0.5) & wm).sum() / wm.sum())
            rec["video"]["gaps_uncorrected"] = gap_summary(CF.axis_gaps(np.where(wm, w0, np.nan), wm, CF.THR_VID))
        paths, phys_tag = sim_paths(p)
        rec["phys_tag"] = phys_tag
        fields = {}
        for k, path in paths.items():
            cM, mM = load_sim(path)
            smp = Sampler(cM, mM, Mt)
            sraw = smp.field_T()
            raw = metrics(sraw, w, wm)
            gn = CF.axis_gaps(cM[::-1, :], mM[::-1, :], 0.4)
            raw["gaps_native_N256_c04"] = dict(n_axis=gn["n_gaps_axis"], n_total=gn["n_gaps_total"], median_width_mm=gn["median_width_mm"])
            raw["white_frac_native_c04"] = float(((cM > 0.4) & mM).sum() / mM.sum())
            ba = best_align(smp, w, wm)
            sb = smp.field_T(ba["theta_deg"], ba["dx_D"], ba["dy_D"])
            best = metrics(sb, w, wm)
            best["transform"] = ba
            r = dict(path=path, raw=raw, best=best, diagnostic_full_circle=full_circle(cM, mM, w, wm))
            if PHI[p] != 0.0:
                ru = metrics(sraw, w0, wm)
                bu = best_align(smp, w0, wm)
                r["uncorrected"] = dict(raw=dict(iou_c04=ru["iou_c04"], iou_c05=ru["iou_c05"], bdist_c04_D=ru["bdist_c04_D"]),
                                        best=bu)
            rec["sims"][k] = r
            fields[k] = dict(raw=sraw.astype(np.float32), best=sb.astype(np.float32))
            fc = r["diagnostic_full_circle"]
            print(p, k, f"raw IoU {raw['iou_c04']:.3f}/{raw['iou_c05']:.3f} bd {raw['bdist_c04_D']:.4f} wf {raw['white_frac_c04']:.3f} "
                  f"gaps {raw['gaps_c04']['n_axis']}/{raw['gaps_c04']['n_total']} | best IoU {best['iou_c04']:.3f}/{best['iou_c05']:.3f} "
                  f"bd {best['bdist_c04_D']:.4f} th {ba['theta_deg']} d ({ba['dx_D']},{ba['dy_D']}) gaps {best['gaps_c04']['n_axis']}/{best['gaps_c04']['n_total']}"
                  f" | full {fc['as_simulated']['iou_c04']:.3f}@{fc['as_simulated']['theta_deg']} mir {fc['mirrored']['iou_c04']:.3f}@{fc['mirrored']['theta_deg']}",
                  flush=True)
        print(p, "video", f"wf {rec['video']['white_frac']:.3f} gaps {vg['n_gaps_axis']}/{vg['n_gaps_total']} widths {[L['widths_mm'] for L in vg['lines']]}",
              f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(os.path.join(HERE, f"fields_{p}.npz"), w=w.astype(np.float32), wm=wm,
                            **{f"{k}_{a}": v for k, d in fields.items() for a, v in d.items()})
        json.dump(rec, open(os.path.join(HERE, f"metrics_{p}.json"), "w"), indent=1)


def merge():
    out = dict(meta=meta(), patterns={p: json.load(open(os.path.join(HERE, f"metrics_{p}.json"))) for p in PAT})
    json.dump(out, open(os.path.join(HERE, "metrics_all.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["merge"]:
        merge()
    else:
        main(sys.argv[1:] or PAT)
