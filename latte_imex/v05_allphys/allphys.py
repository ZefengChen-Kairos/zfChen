"""allphys.py -- "all inlet quantities physical" test of the V0.5 model (single-layer IMEX, closure 'chi', return law v05).

Nothing in latte_imex or v05audit is edited; this file reuses v05audit/apply_experiment.py (bridges, flow_ramps,
flow_modulation cap, jet_osc_hand) and v05audit/cf_runner.py / verify_compare.py (metrics).

Common physical inlet for every flowing stroke k of Codex's control (paths, timings, wiggles kept):
  F(t)     V0.5 flow incl. taper / pulsation (depth capped) / 0.1 s on-off ramps; bridges >= 0.3 s; lateral jet
           oscillation = hand lateral speed (jet_osc_hand)
  Q        = clip(F / kappa_Q, 1, 32 mL/s), one shared kappa_Q = 2.0 /D;  S_eff = kappa_Q Q  (consistent: a clipped
           flow also feeds the layer)
  U_perp   = sqrt(v0^2 + 2 g z_k), v0 = 0.35 m/s, z_k per stroke (free jet: low/wiggle/push 0.3-2.5 cm, cut 1.2-7 cm);
           'contact' strokes (P5 only): U_perp in [0.06, 0.27] m/s (= u0 sin beta), z ~ 0
  u_h      = Codex jet direction (incl. hand-speed lateral oscillation), |u_h| clipped to
           [0.10 m/s, min(0.40 m/s, 1.0 U_perp)]  (contact: [0.10, 0.27] m/s)
  footprint continuity ellipse: r = sqrt(Q/(pi U)), aspect sqrt(|v|/U), |v| = hypot(U, u_h), A = Q/|v|, d = 2 sqrt(A/pi)
  chi      = 1/(1 + (U^2/(B_dep d))^p_dep)  (solver, unchanged)

CLI
  python3 -I allphys.py fit                         -> allphys_experiments.json, fit_report.json (inverse design)
  python3 -I allphys.py one <exp> <pattern> <N>     -> runs/<exp>/<pattern>_N<N>.{npz,json}, img/<exp>_<pattern>_N<N>.png
  python3 -I allphys.py batch <exp,...> <pat,...|all> <N> [--workers 3]
  python3 -I allphys.py montage <exp> <N>
Units: D = 0.08 m (cup diameter), D^3 = 512 mL, s.  Solver speeds D/s.
"""
import argparse, copy, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(os.path.dirname(HERE), "v05audit")
REPO = "/home/user/zfChen"
sys.path.insert(0, A); sys.path.insert(0, REPO)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import apply_experiment as AX
from latte_imex.v05_program import Program, load_control
from latte_imex.solver import Inlet

PATTERNS = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")
D_M = 0.08; G = 9.81; ML = 512.0
V0 = 0.35                         # spout exit speed [m/s]
UH_LO, UH_HI, TAN_MAX = 0.10 / D_M, 0.40 / D_M, 1.0
UH_CONTACT_HI = 0.27 / D_M
Z_LOW = (0.003, 0.025); Z_CUT = (0.012, 0.07)          # free-jet spout heights [m]
U_CONTACT = (0.06, 0.27)                                # contact-pour normal speed [m/s]
KAPPA_Q = 2.0; Q_MIN = 1.0; Q_MAX = 32.0                # 1/D, mL/s
BRIDGE = 0.3; RAMP = 0.1
DT = 2e-3                                               # fit sampling [s]
CHI_CLIP = (0.02, 0.98)                                 # logit targets clipped here (>= 0.98 is "full", <= 0.02 "none")
EXPF = os.path.join(HERE, "allphys_experiments.json")
RUNS = os.path.join(HERE, "runs"); IMG = os.path.join(HERE, "img")


def U_of_z(z):
    return math.sqrt(V0 * V0 + 2 * G * z)


def z_of_U(U):
    return (U * U - V0 * V0) / (2 * G)


def common_changes(pulse):
    return [{"op": "bridges", "min_duration": BRIDGE}, {"op": "flow_ramps", "ramp_s": RAMP},
            {"op": "phase_field", "where": {"has": "flow_modulation"}, "field": "flow_modulation", "mode": "min",
             "value": pulse},
            {"op": "jet_osc_hand"}]


# ============================================================================== program
class PhysProgram(AX.XProgram):
    """XProgram (ramps, hand-speed jet oscillation, bridges, pulsation cap) + the physical inlet above.
    spec = {"kappa_Q": kQ, "U": {k: U [m/s]}, "mode": {k: "free"|"contact"}} (k = 0-based phase index)"""

    def __init__(self, config, changes, pattern, spec):
        super().__init__(config, changes, pattern)
        self.spec = spec
        self.last = None

    def __call__(self, t):
        k = self.phase_index(t)
        p = self.phases[k]
        tt = max(0.0, min(p["duration"], t - self.events[k]))
        swap = {}
        g = self._ramp(k, tt)
        if g != 1.0:
            swap["Q"] = p["Q"] * g
            swap["Q_end"] = p.get("Q_end", p["Q"]) * g
        if "jet_osc_hand" in self.dyn and AX.is_wiggle(p):
            f = tt / p["duration"]
            amp = p["amplitude"] * (1 - f) + p.get("amplitude_end", p["amplitude"]) * f
            swap.update(jet_amplitude_x=2 * math.pi * p["frequency"] * amp, jet_frequency=p["frequency"],
                        jet_phase=math.pi / 2)
        if swap:
            q = dict(p); q.update(swap)
            self.phases[k] = q
            try:
                inl = Program.__call__(self, t)
            finally:
                self.phases[k] = p
            pe = q
        else:
            inl = Program.__call__(self, t)
            pe = p
        self.last = None
        if not (inl.active and inl.S_eff > 0):
            return inl
        return self._phys(inl, pe, k, tt)

    def _phys(self, inl, pe, k, tt):
        sp = self.spec
        kQ = sp["kappa_Q"]
        F = inl.S_eff
        Qraw = F / kQ
        Q = min(max(Qraw, Q_MIN / ML), Q_MAX / ML)
        jx, jy = AX.raw_jet(pe, tt)
        vraw = math.hypot(jx, jy)
        U = sp["U"][str(k)] / D_M
        contact = sp["mode"][str(k)] == "contact"
        hi = UH_CONTACT_HI if contact else min(UH_HI, TAN_MAX * U)
        v = min(max(vraw, UH_LO), hi) if hi >= UH_LO else hi
        if vraw > 0:
            jx, jy = jx * v / vraw, jy * v / vraw
        else:
            jx, jy = 0.0, -v
        spd = math.hypot(U, v)
        aspect = spd / U
        r = math.sqrt(Q / (math.pi * U))
        d = 2 * math.sqrt(Q / spd / math.pi)
        self.last = (k, kQ * Q, Q, Qraw, vraw)
        return Inlet(active=True, x_hit=inl.x_hit, S_eff=kQ * Q, u_in=(jx, jy), U_perp=U, d_jet=d,
                     r1=r * math.sqrt(aspect), r2=r / math.sqrt(aspect), phi=math.atan2(jy, jx),
                     scan_speed=inl.scan_speed)


def build(exp, pattern):
    changes = exp["changes"]
    cfg = AX.apply_static(load_control(pattern), changes, pattern)
    cfg["parameters"].update(exp["params"])
    return PhysProgram(cfg, changes, pattern, exp["inlet"][pattern])


# ============================================================================== sampling / targets
def base_targets(pattern, dt=DT):
    """V0.5 per flowing phase: F-weighted chi (B 240, p 2), deposited amount, F-weighted 0.3 V_R, F_int"""
    prog = Program(load_control(pattern))
    P = prog.config["parameters"]
    out = {}
    for t in np.arange(0.0, prog.duration, dt) + 0.5 * dt:
        k = prog.phase_index(t)
        inl = prog(t)
        if not (inl.active and inl.S_eff > 0):
            continue
        d = max(inl.d_jet, 1e-9)
        Z = inl.U_perp ** 2 / d
        chi = 1 / (1 + (Z / P["B_dep"]) ** P["p_dep"])
        VR = min(inl.U_perp * chi ** (1 / (2 * P["p_dep"])), inl.U_perp, math.sqrt(P["B_dep"] * d))
        o = out.setdefault(k, dict(F=0.0, chiF=0.0, VRF=0.0, Q=0.0, U=0.0, uh=0.0, r=0.0))
        F = inl.S_eff
        o["F"] += F * dt; o["chiF"] += chi * F * dt; o["VRF"] += P["kappa_r"] * VR * F * dt
        o["Q"] += F / P["kappa_Q"] * F * dt; o["U"] += inl.U_perp * F * dt; o["uh"] += math.hypot(*inl.u_in) * F * dt
        o["r"] += math.sqrt(inl.r1 * inl.r2) * F * dt
    res = {}
    for k, o in out.items():
        W = o["F"]
        res[k] = dict(cls=AX.phase_class(prog.phases[k]), F_int=W, chi=o["chiF"] / W, dep=o["chiF"], radial=o["VRF"] / W,
                      Q_mLs=o["Q"] / W * ML, U_ms=o["U"] / W * D_M, uh_ms=o["uh"] / W * D_M, r_mm=o["r"] / W * D_M * 1e3)
    return res


def sample_phys(pattern, pulse, dt=DT):
    """samples of the physical program (U-independent part): per flowing phase arrays S, Q, vraw; clipping stats"""
    exp = dict(changes=common_changes(pulse), params={},
               inlet={pattern: dict(kappa_Q=KAPPA_Q, U={str(k): 0.5 for k in range(60)}, mode={str(k): "free" for k in range(60)})})
    prog = build(exp, pattern)
    rows = []
    for t in np.arange(0.0, prog.duration, dt) + 0.5 * dt:
        prog(t)
        if prog.last is not None:
            rows.append(prog.last)
    a = np.array(rows)
    out = {}
    for k in sorted(set(int(x) for x in a[:, 0])):
        s = a[a[:, 0] == k]
        out[k] = dict(cls=AX.phase_class(prog.phases[k]), S=s[:, 1], Q=s[:, 2], Qraw=s[:, 3], vraw=s[:, 4], dt=dt)
    clip = dict(t_flow=float(len(a) * dt), t_low=float((a[:, 3] < Q_MIN / ML).sum() * dt),
                t_high=float((a[:, 3] > Q_MAX / ML).sum() * dt),
                vol_raw_mL=float(a[:, 3].sum() * dt * ML), vol_mL=float(a[:, 2].sum() * dt * ML),
                Qraw_min_mLs=float(a[:, 3].min() * ML), Qraw_max_mLs=float(a[:, 3].max() * ML), duration=prog.duration)
    return out, clip


def stroke_fields(s, U, contact=False):
    """U [D/s] -> (Z, d, v) arrays for the stroke samples"""
    hi = UH_CONTACT_HI if contact else min(UH_HI, TAN_MAX * U)
    v = np.clip(s["vraw"], UH_LO, hi) if hi >= UH_LO else np.full_like(s["vraw"], hi)
    spd = np.hypot(U, v)
    d = 2 * np.sqrt(s["Q"] / spd / math.pi)
    return U * U / d, d, v


def chi_w(s, U, B, p, contact=False):
    Z, d, v = stroke_fields(s, U, contact)
    chi = 1 / (1 + (Z / B) ** p)
    return float((chi * s["S"]).sum() / s["S"].sum())


def logit(x):
    x = min(max(x, CHI_CLIP[0]), CHI_CLIP[1])
    return math.log(x / (1 - x))


def U_range(cls, contact=False):
    if contact:
        return U_CONTACT[0] / D_M, U_CONTACT[1] / D_M
    zr = Z_CUT if cls == "cut" else Z_LOW
    return U_of_z(zr[0]) / D_M, U_of_z(zr[1]) / D_M


def solve_U(s, target, B, p, contact=False, iters=40):
    """U in the stroke's range with chi_w(U) = clipped target (chi_w decreases with U); boundary if unreachable"""
    lo, hi = U_range(s["cls"], contact)
    tgt = min(max(target, CHI_CLIP[0]), CHI_CLIP[1])
    c_lo, c_hi = chi_w(s, lo, B, p, contact), chi_w(s, hi, B, p, contact)
    if c_lo <= tgt:
        return lo, c_lo
    if c_hi >= tgt:
        return hi, c_hi
    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        if chi_w(s, math.exp(m), B, p, contact) > tgt:
            a = m
        else:
            b = m
    U = math.exp(0.5 * (a + b))
    return U, chi_w(s, U, B, p, contact)


def design(smp, tg, B, p, contact_set=()):
    """per-stroke U for shared (B, p); returns objective (deposit-weighted mean squared logit error) and rows"""
    num = den = 0.0
    rows = {}
    for pat in smp:
        for k, s in smp[pat].items():
            c = (pat, k) in contact_set
            U, ch = solve_U(s, tg[pat][k]["chi"], B, p, c)
            e = logit(ch) - logit(tg[pat][k]["chi"])
            w = tg[pat][k]["dep"]
            num += w * e * e; den += w
            rows[(pat, k)] = (U, ch, e)
    return num / den, rows


def fit_inverse(smp, tg, contact_set=(), p_range=(1.0, 8.0)):
    """grid over (log B, p) then golden refinement; per-stroke U solved exactly inside each evaluation"""
    best = None
    grid = []
    for p in np.arange(p_range[0], p_range[1] + 1e-9, 0.5):
        for lB in np.linspace(math.log(50), math.log(2e5), 49):
            J, _ = design(smp, tg, math.exp(lB), p, contact_set)
            grid.append((J, lB, p))
            if best is None or J < best[0] - 1e-12:
                best = (J, lB, p)
    # local refinement (coordinate search)
    J, lB, p = best
    step_l, step_p = (math.log(2e5) - math.log(50)) / 48, 0.5
    for _ in range(30):
        improved = False
        for dl, dp in ((step_l, 0), (-step_l, 0), (0, step_p), (0, -step_p)):
            l2, p2 = lB + dl, min(max(p + dp, p_range[0]), p_range[1])
            J2, _ = design(smp, tg, math.exp(l2), p2, contact_set)
            if J2 < J - 1e-12:
                J, lB, p, improved = J2, l2, p2, True
        if not improved:
            step_l *= 0.5; step_p *= 0.5
            if step_l < 1e-3:
                break
    # the flat set: all grid points within 1e-3 of the optimum (non-unique optimum => report the range)
    flat = [(math.exp(l), pp) for (j, l, pp) in grid if j <= J + 1e-3]
    return dict(J=J, B=math.exp(lB), p=p, flat_B=[min(f[0] for f in flat), max(f[0] for f in flat)] if flat else None,
                flat_p=[min(f[1] for f in flat), max(f[1] for f in flat)] if flat else None, n_flat=len(flat))


def radial_w(s, U, B, p, contact=False):
    Z, d, v = stroke_fields(s, U, contact)
    chi = 1 / (1 + (Z / B) ** p)
    VR = np.minimum(np.minimum(U * chi ** (1 / (2 * p)), U), np.sqrt(B * d))
    return float((VR * s["S"]).sum() / s["S"].sum())


def fit_kr(smp, tg, Us, B, p, contact_set=()):
    """one kappa_r: F-weighted least squares in log of (kappa_r V_R,w) vs V0.5's 0.3 V_R,w per stroke"""
    num = den = 0.0
    for pat in smp:
        for k, s in smp[pat].items():
            a = radial_w(s, Us[(pat, k)], B, p, (pat, k) in contact_set)
            w = tg[pat][k]["F_int"]
            num += w * math.log(tg[pat][k]["radial"] / a); den += w
    return math.exp(num / den)


def fit_B_linear(smp, tg, Us, p):
    """X01 rule: one B minimising sum F_int (chi_w - chi_V05)^2 at fixed U per stroke"""
    def cost(B):
        return sum(tg[pat][k]["F_int"] * (chi_w(s, Us[(pat, k)], B, p) - tg[pat][k]["chi"]) ** 2
                   for pat in smp for k, s in smp[pat].items())
    grid = np.exp(np.linspace(math.log(5.0), math.log(2e6), 300))
    cs = [cost(B) for B in grid]
    i = int(np.argmin(cs))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(50):
        m1 = lo + (hi - lo) * 0.382; m2 = lo + (hi - lo) * 0.618
        if cost(m1) < cost(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def fit_kr_total(smp, tg, Us, B, p):
    """X01 rule (make_experiments.fit_kr): kappa_r = 0.3 sum(F V_R,V05) / sum(F V_R,new)"""
    num = den = 0.0
    for pat in smp:
        for k, s in smp[pat].items():
            num += tg[pat][k]["radial"] / 0.3 * tg[pat][k]["F_int"]
            den += radial_w(s, Us[(pat, k)], B, p) * (s["S"].sum() * s["dt"])
    return 0.3 * num / den


def stroke_table(smp, tg, Us, B, p, kr, contact_set=()):
    rows = []
    for pat in smp:
        for k, s in smp[pat].items():
            c = (pat, k) in contact_set
            U = Us[(pat, k)]
            Z, d, v = stroke_fields(s, U, c)
            chi = 1 / (1 + (Z / B) ** p)
            W = s["S"].sum()
            ch = float((chi * s["S"]).sum() / W)
            t = tg[pat][k]
            rows.append(dict(pattern=pat, phase=k + 1, cls=s["cls"], mode="contact" if c else "free",
                             z_cm=None if c else round(z_of_U(U * D_M) * 100, 3), U_ms=round(U * D_M, 4),
                             Q_mLs_w=round(float((s["Q"] * s["S"]).sum() / W * ML), 2),
                             Q_mLs_rng=[round(float(s["Q"].min() * ML), 2), round(float(s["Q"].max() * ML), 2)],
                             uh_ms_w=round(float((v * s["S"]).sum() / W * D_M), 4),
                             tan_w=round(float((v * s["S"]).sum() / W / U), 3),
                             r_mm_w=round(float((np.sqrt(s["Q"] / (math.pi * U)) * s["S"]).sum() / W * D_M * 1e3), 3),
                             d_mm_w=round(float((d * s["S"]).sum() / W * D_M * 1e3), 3),
                             chi_target=round(t["chi"], 4), chi=round(ch, 4), logit_err=round(logit(ch) - logit(t["chi"]), 3),
                             radial_target=round(t["radial"], 4), radial=round(kr * radial_w(s, U, B, p, c), 4),
                             dep_V05=round(t["dep"], 5), F_int=round(t["F_int"], 5),
                             V05=dict(Q_mLs=round(t["Q_mLs"], 3), U_ms=round(t["U_ms"], 4), uh_ms=round(t["uh_ms"], 4),
                                      r_mm=round(t["r_mm"], 3))))
    return rows


def fit_all():
    t0 = time.time()
    tg = {p: base_targets(p) for p in PATTERNS}
    experiments = {}
    report = dict(units="z cm, U/u_h m/s, Q mL/s, r/d mm, B_dep D/s^2, kappa_r -, radial target D/s", targets=None)
    report["targets"] = {p: {str(k + 1): v for k, v in tg[p].items()} for p in PATTERNS}
    base_params = dict(kappa_Q=KAPPA_Q, cp=0.3, beta=3.0, nu=1e-3)
    for pulse in (0.85, 0.41):
        smp, clip = {}, {}
        for p in PATTERNS:
            smp[p], clip[p] = sample_phys(p, pulse)
        tag = "85" if pulse > 0.5 else "41"
        report[f"clipping_pulse{tag}"] = clip
        if pulse > 0.5:
            # ---- P0: class heights 1.0 / 2.5 cm, p = 2, B and kappa_r by the X01 rules
            Us0 = {(pat, k): U_of_z(0.025 if s["cls"] == "cut" else 0.010) / D_M for pat in smp for k, s in smp[pat].items()}
            B0 = fit_B_linear(smp, tg, Us0, 2.0)
            kr0 = fit_kr_total(smp, tg, Us0, B0, 2.0)
            experiments["P0"] = mk_exp("P0", pulse, dict(base_params, B_dep=B0, p_dep=2.0, kappa_r=kr0), Us0, smp)
            report["P0"] = dict(B_dep=B0, p_dep=2.0, kappa_r=kr0, strokes=stroke_table(smp, tg, Us0, B0, 2.0, kr0))
        # ---- P1 / P2: inverse design
        fit = fit_inverse(smp, tg)
        J, rows = design(smp, tg, fit["B"], fit["p"])
        Us = {key: r[0] for key, r in rows.items()}
        kr = fit_kr(smp, tg, Us, fit["B"], fit["p"])
        name = "P1" if pulse > 0.5 else "P2"
        experiments[name] = mk_exp(name, pulse, dict(base_params, B_dep=fit["B"], p_dep=fit["p"], kappa_r=kr), Us, smp)
        report[name] = dict(fit, kappa_r=kr, strokes=stroke_table(smp, tg, Us, fit["B"], fit["p"], kr))
        # p fixed at 2 for comparison (how much of the match comes from p_dep)
        fit2 = fit_inverse(smp, tg, p_range=(2.0, 2.0))
        J2, rows2 = design(smp, tg, fit2["B"], 2.0)
        Us2 = {key: r[0] for key, r in rows2.items()}
        kr2 = fit_kr(smp, tg, Us2, fit2["B"], 2.0)
        report[name + "_p2"] = dict(fit2, kappa_r=kr2, strokes=stroke_table(smp, tg, Us2, fit2["B"], 2.0, kr2))
        if pulse > 0.5:
            experiments["P1_p2"] = mk_exp("P1_p2", pulse, dict(base_params, B_dep=fit2["B"], p_dep=2.0, kappa_r=kr2), Us2, smp)
            for nm, cp in (("P3", 1.0), ("P4", 1.36)):
                e = copy.deepcopy(experiments["P1"]); e["id"] = nm; e["params"]["cp"] = cp
                experiments[nm] = e
            # ---- P5: contact pours for strokes not matched in P1 (|logit err| > 0.4)
            bad = {key for key, r in rows.items() if abs(r[2]) > 0.4}
            report["P1_unmatched"] = sorted([f"{k[0]}#{k[1] + 1}" for k in bad])
            low_bad = {key for key in bad if smp[key[0]][key[1]]["cls"] != "cut" and rows[key][1] < tg[key[0]][key[1]]["chi"]}
            if low_bad:
                fit5 = fit_inverse(smp, tg, contact_set=low_bad)
                J5, rows5 = design(smp, tg, fit5["B"], fit5["p"], low_bad)
                Us5 = {key: r[0] for key, r in rows5.items()}
                kr5 = fit_kr(smp, tg, Us5, fit5["B"], fit5["p"], low_bad)
                experiments["P5"] = mk_exp("P5", pulse, dict(base_params, B_dep=fit5["B"], p_dep=fit5["p"], kappa_r=kr5),
                                           Us5, smp, low_bad)
                report["P5"] = dict(fit5, kappa_r=kr5, contact=sorted(f"{k[0]}#{k[1] + 1}" for k in low_bad),
                                    strokes=stroke_table(smp, tg, Us5, fit5["B"], fit5["p"], kr5, low_bad))
    report["fit_wall_s"] = round(time.time() - t0, 1)
    json.dump(experiments, open(EXPF, "w"), indent=1)
    json.dump(report, open(os.path.join(HERE, "fit_report.json"), "w"), indent=1, default=float)
    return experiments, report


def mk_exp(name, pulse, params, Us, smp, contact_set=()):
    inlet = {}
    for pat in smp:
        U = {str(k): Us[(pat, k)] * D_M for k in smp[pat]}
        mode = {str(k): ("contact" if (pat, k) in contact_set else "free") for k in smp[pat]}
        inlet[pat] = dict(kappa_Q=KAPPA_Q, U=U, mode=mode,
                          z_cm={str(k): (None if (pat, k) in contact_set else round(z_of_U(Us[(pat, k)] * D_M) * 100, 3))
                                for k in smp[pat]})
    return dict(id=name, pulse=pulse, changes=common_changes(pulse), params=params, inlet=inlet)


# ============================================================================== (beta, nu) screening
BETA_RNG = (0.12, 7.5); NU_RNG = (7e-5, 0.26)          # closure.md physical ranges [1/s], [D^2/s]


def make_screen(base, n=40):
    """n candidates: the V0.5 pair (3, 1e-3) + n-1 scrambled-Sobol points log-uniform in BETA_RNG x NU_RNG"""
    from scipy.stats import qmc
    E = json.load(open(EXPF))
    b = E[base]
    pts = qmc.Sobol(d=2, scramble=True, seed=7).random(64)[: n - 1]
    pairs = [(3.0, 1e-3)] + [(math.exp(math.log(BETA_RNG[0]) + u * math.log(BETA_RNG[1] / BETA_RNG[0])),
                               math.exp(math.log(NU_RNG[0]) + v * math.log(NU_RNG[1] / NU_RNG[0]))) for u, v in pts]
    extra_f = os.path.join(HERE, "allphys_extra.json")
    X = json.load(open(extra_f)) if os.path.exists(extra_f) else {}
    ids = []
    for i, (be, nu) in enumerate(pairs):
        e = copy.deepcopy(b); e["id"] = f"S{base}_{i:02d}"; e["params"]["beta"] = be; e["params"]["nu"] = nu
        e["screen"] = dict(base=base, beta=be, nu=nu)
        X[e["id"]] = e; ids.append(e["id"])
    json.dump(X, open(extra_f, "w"), indent=1)
    return ids


def rank(base, N=128):
    X = json.load(open(os.path.join(HERE, "allphys_extra.json")))
    out = []
    for k, e in X.items():
        if e.get("screen", {}).get("base") != base:
            continue
        rows = []
        for p in PATTERNS:
            fn = os.path.join(RUNS, "screen", k, f"{p}_N{N}.json")
            if os.path.exists(fn):
                rows.append(json.load(open(fn)))
        if len(rows) < len(PATTERNS):
            out.append(dict(id=k, beta=e["screen"]["beta"], nu=e["screen"]["nu"], n=len(rows), score=None))
            continue
        sc = [r["iou04"] + (r["r_hp"] if r["r_hp"] is not None else 0.0) for r in rows]
        out.append(dict(id=k, beta=e["screen"]["beta"], nu=e["screen"]["nu"], n=len(rows), score=float(np.mean(sc)),
                        iou04={r["pattern"]: round(r["iou04"], 3) for r in rows},
                        r_hp={r["pattern"]: (round(r["r_hp"], 3) if r["r_hp"] is not None else None) for r in rows},
                        wall=sum(r["solver"]["wall_s"] for r in rows)))
    out.sort(key=lambda r: -(r["score"] if r["score"] is not None else -9))
    return out


# ============================================================================== one run + metrics
def load_ref(p, N):
    if N == 256:
        z = np.load(os.path.join(REPO, "runs", "v05_256", p, "final_state.npz"))
        mask = z["mask"]
        c = np.where(mask, z["m"] / np.where(mask, z["l"], 1.0), np.nan)
        dep = json.load(open(os.path.join(REPO, "results", "v05_on_imex_256", p, "metrics.json")))["ledger"]["deposited"]
        return c, mask, dep
    z = np.load(os.path.join(A, "cf", "baseline", f"{p}_N{N}.npz"))
    dep = json.load(open(os.path.join(A, "cf", "baseline", f"{p}_N{N}.json")))["ledger"]["deposited"]
    return z["c"].astype(float), z["mask"], dep


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def get_exp(exp_id):
    E = json.load(open(EXPF))
    if exp_id in E:
        return E[exp_id]
    extra = os.path.join(HERE, "allphys_extra.json")          # screening candidates (beta, nu)
    if os.path.exists(extra):
        X = json.load(open(extra))
        if exp_id in X:
            return X[exp_id]
    raise KeyError(exp_id)


def metrics(c, mask, cr, mr, N):
    import cf_runner as CR
    import verify_compare as VC
    g = CR.axis_gaps(c[::-1], mask[::-1], 0.4); gr = CR.axis_gaps(cr[::-1], mr[::-1], 0.4)
    hc, rh = VC.hp_stats(c, mask, cr, mr, N)
    return dict(iou04=iou((c > 0.4) & mask, (cr > 0.4) & mr), iou05=iou((c > 0.5) & mask, (cr > 0.5) & mr),
                white=float(((c > 0.4) & mask).sum() / mask.sum()), white_ref=float(((cr > 0.4) & mr).sum() / mr.sum()),
                r_hp=rh, hp_contrast=hc,
                gaps=[g["n_gaps_axis"], g["n_gaps_total"]], gaps_ref=[gr["n_gaps_axis"], gr["n_gaps_total"]],
                dips=[g["n_dips_axis"], g["n_dips_total"]], dips_ref=[gr["n_dips_axis"], gr["n_dips_total"]],
                gap_widths_mm=[w for L in g["lines"] for w in L["widths_mm"]],
                gap_widths_ref_mm=[w for L in gr["lines"] for w in L["widths_mm"]])


def run_one(exp_id, pattern, N, force=False, outdir=None, fig=True):
    d = outdir or os.path.join(RUNS, exp_id)
    stem = os.path.join(d, f"{pattern}_N{N}")
    if os.path.exists(stem + ".json") and not force:
        return json.load(open(stem + ".json"))
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    exp = get_exp(exp_id)
    prog = build(exp, pattern)
    import latte_imex.run as R
    R.v05_program = lambda name: prog
    raw = os.path.join(d, "raw", f"{pattern}_N{N}")
    os.makedirs(raw, exist_ok=True)
    sol, m = R.run(pattern, N=N, out=raw, video=False, quiet=True, control="v05")
    wall = time.time() - t0
    c = sol.concentration(); mask = sol.g.mask.copy()
    traj = json.load(open(os.path.join(raw, "trajectory.json")))
    # realised chi per phase (solver frames, S-weighted)
    ph = {}
    for q in traj:
        if q["active"]:
            k = q["phase"].split(":")[0]
            o = ph.setdefault(k, [0.0, 0.0])
            o[0] += q["S"] * q["chi"]; o[1] += q["S"]
    chi_real = {k: round(v[0] / v[1], 4) for k, v in ph.items() if v[1] > 0}
    np.savez_compressed(stem + ".npz", c=c.astype(np.float32), mask=mask, N=N, T=m["T"])
    for f in ("snapshots_c.npz", "final_state.npz"):
        try:
            os.remove(os.path.join(raw, f))
        except OSError:
            pass
    cr, mr, dep_ref = load_ref(pattern, N)
    met = metrics(c, mask, cr, mr, N)
    st = m["stats"]; L = m["ledger"]
    res = dict(exp=exp_id, pattern=pattern, N=N, T=m["T"], **met, dep_ratio=L["deposited"] / dep_ref,
               deposited=L["deposited"], chi_realised=chi_real,
               solver=dict(wall_s=round(wall, 1), steps=st["steps"], max_substeps=st["max_substeps"], retries=st["retries"],
                           max_u=st["max_u"], clip_mass=st["clip_mass"], layer_error=L["layer_error"],
                           milk_error=L["milk_error"]),
               params=m["params"])
    json.dump(res, open(stem + ".json", "w"), indent=1, default=float)
    if fig:
        try:
            side_by_side(exp_id, pattern, N, c, mask, cr, mr, res, os.path.join(IMG, f"{exp_id}_{pattern}_N{N}.png"))
        except Exception as e:
            print(f"fig failed {e!r}", file=sys.stderr)
    return res


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("latte", [(0.36, 0.21, 0.10), (0.97, 0.94, 0.87)])
    cm.set_bad((0.86, 0.86, 0.86))
    return cm


def side_by_side(exp_id, pattern, N, c, mask, cr, mr, res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = _cmap()
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.5), dpi=120)
    for a, f, m, t in ((ax[0], cr, mr, f"V0.5 (N={N})  white {res['white_ref']:.3f}"),
                       (ax[1], c, mask, f"{exp_id}  white {res['white']:.3f}")):
        a.imshow(np.clip(np.where(m, f, np.nan)[::-1, :], 0, 1), cmap=cm, vmin=0, vmax=1, origin="lower",
                 extent=(-0.5, 0.5, -0.5, 0.5), interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=8)
    a_, b_ = ((c > 0.4) & mask)[::-1, :], ((cr > 0.4) & mr)[::-1, :]
    rgb = np.zeros(a_.shape + (3,)); rgb[...] = (0.36, 0.21, 0.10)
    rgb[a_ & b_] = (0.97, 0.94, 0.87); rgb[a_ & ~b_] = (0.90, 0.25, 0.20); rgb[~a_ & b_] = (0.25, 0.45, 0.90)
    rgb[~mask[::-1, :]] = (0.86, 0.86, 0.86)
    ax[2].imshow(rgb, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5), interpolation="nearest")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    rh = res["r_hp"]
    ax[2].set_title(f"red only {exp_id}, blue only V0.5\nIoU .4/.5 {res['iou04']:.3f}/{res['iou05']:.3f}  r_hp "
                    f"{(rh if rh is not None else float('nan')):.2f}", fontsize=8)
    fig.suptitle(f"{pattern}  {exp_id}  N={N}  gaps {res['gaps']} (V0.5 {res['gaps_ref']})  dips {res['dips']} "
                 f"(V0.5 {res['dips_ref']})  retries {res['solver']['retries']} max_sub {res['solver']['max_substeps']}",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def montage(exp_ids, N, path, patterns=PATTERNS):
    """rows = patterns, columns = V0.5 | exp1 | exp2 ..."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = _cmap()
    cols = ["V0.5"] + list(exp_ids)
    fig, ax = plt.subplots(len(patterns), len(cols), figsize=(2.2 * len(cols), 2.3 * len(patterns)), dpi=110, squeeze=False)
    for i, p in enumerate(patterns):
        cr, mr, _ = load_ref(p, N)
        for j, e in enumerate(cols):
            a = ax[i, j]; a.set_xticks([]); a.set_yticks([])
            if e == "V0.5":
                f, m, t = cr, mr, f"{p} V0.5"
            else:
                fn = os.path.join(RUNS, e, f"{p}_N{N}.npz")
                if not os.path.exists(fn):
                    a.axis("off"); continue
                z = np.load(fn); f, m = z["c"].astype(float), z["mask"]
                r = json.load(open(fn[:-4] + ".json"))
                t = f"{e} IoU {r['iou04']:.2f} r {r['r_hp'] if r['r_hp'] is None else round(r['r_hp'], 2)}"
            a.imshow(np.clip(np.where(m, f, np.nan)[::-1, :], 0, 1), cmap=cm, vmin=0, vmax=1, origin="lower",
                     extent=(-0.5, 0.5, -0.5, 0.5), interpolation="nearest")
            a.set_title(t, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def batch(exp_ids, patterns, N, workers=3, force=False, timeout=None):
    order = {"swan": 0, "leaf": 1, "tulip": 2, "layered_heart": 3, "push_heart": 4, "heart": 5}
    jobs = [(e, p) for e in exp_ids for p in sorted(patterns, key=lambda p: order.get(p, 9))]   # experiment-major
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = os.path.join(HERE, "batch.log")

    def one(j):
        e, p = j
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "one", e, p, str(N)] + (["--force"] if force else [])
        if e.startswith("S"):
            cmd.append("--nofig")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=HERE, timeout=timeout)
            out = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else f"FAIL: {r.stderr[-1200:]}"
        except subprocess.TimeoutExpired:
            out = f"TIMEOUT after {timeout} s (not evaluated)"
        msg = f"[{time.strftime('%H:%M:%S')}] {e} {p} N{N} ({time.time() - t0:.0f} s): {out}"
        with open(log, "a") as f:
            f.write(msg + "\n")
        return msg

    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "one", "batch", "montage", "screen", "rank", "mkconfirm"])
    ap.add_argument("exp", nargs="?")
    ap.add_argument("pattern", nargs="?", default="all")
    ap.add_argument("N", nargs="?", type=int, default=256)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--nofig", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "fit":
        E, Rp = fit_all()
        for k in ("P0", "P1", "P1_p2", "P2", "P2_p2", "P5"):
            if k in Rp:
                print(k, {q: Rp[k].get(q) for q in ("B_dep", "B", "p", "p_dep", "kappa_r", "J", "flat_B", "flat_p")})
        print("unmatched P1:", Rp.get("P1_unmatched"))
    elif a.cmd == "one":
        r = run_one(a.exp, a.pattern, a.N, force=a.force, fig=not a.nofig,
                    outdir=os.path.join(RUNS, "screen", a.exp) if a.exp.startswith("S") else None)
        print(json.dumps(dict(exp=r["exp"], pattern=r["pattern"], N=r["N"], iou04=round(r["iou04"], 3),
                              iou05=round(r["iou05"], 3), r_hp=r["r_hp"] and round(r["r_hp"], 3), white=round(r["white"], 3),
                              gaps=r["gaps"], dips=r["dips"], wall=r["solver"]["wall_s"], retries=r["solver"]["retries"],
                              max_sub=r["solver"]["max_substeps"])))
    elif a.cmd == "batch":
        pats = list(PATTERNS) if a.pattern == "all" else a.pattern.split(",")
        batch(a.exp.split(","), pats, a.N, a.workers, a.force)
    elif a.cmd == "screen":                      # allphys.py screen P4 0 128 --n 40   (exp, unused, N)
        ids = make_screen(a.exp, a.n)
        batch(ids, list(PATTERNS), a.N, a.workers, timeout=300)
        print(json.dumps(rank(a.exp, a.N)[:8], indent=1))
    elif a.cmd == "rank":
        print(json.dumps(rank(a.exp, a.N)[:12], indent=1))
    elif a.cmd == "mkconfirm":                   # allphys.py mkconfirm P4 P4b 128 : best screened (beta, nu) -> new experiment
        best = [r for r in rank(a.exp, a.N) if r["score"] is not None][0]
        E = json.load(open(EXPF))
        e = copy.deepcopy(E[a.exp]); e["id"] = a.pattern
        e["params"]["beta"] = best["beta"]; e["params"]["nu"] = best["nu"]
        e["screen_best"] = best
        E[a.pattern] = e
        json.dump(E, open(EXPF, "w"), indent=1)
        print(json.dumps(best))
    else:
        montage(a.exp.split(","), a.N, os.path.join(IMG, f"montage_{a.exp.replace(',', '_')}_N{a.N}.png"))
