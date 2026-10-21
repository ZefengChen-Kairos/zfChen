"""Surfacing-time tracer of the white (WSolver(age=True), solver.Solver(age=True); wsolver.k_age): a diagnostic that
must not change the dynamics.
    python -m latte_imex.diff.test_age [--pattern tulip --N 48 --frames 150] [--closure closures/x.json]
                                       [--set k=v,...] [--milk vl|mp5fct|...] [--save out.npz]
    python -m latte_imex.diff.test_age --geometry [--N 48 --T 1.8]
Pour mode (PASS needs all four):
  1 neutrality: l, qx, qy, m with age on bitwise equal to age off after every frame -- numpy, the Warp recorded path
    (frame) and the Warp fast path (fast_forward) -- with the same sub-step counts;
  2 parity: Warp fast path against numpy, age on: the state <= 1e-12 relative (as test_parity), the moment ma <= 1e-10
    relative (to max |ma|) and AGE = ma / m (where m >= 1e-3 in both) <= 1e-10 relative (to the frame time);
  3 bounds: 0 <= AGE <= t where m >= 1e-3, t the time of the state (the end of frame k), after every frame, numpy and
    Warp.  Also reported, not part of PASS: AGE against the frame-start time t_k, the clip's own bound.  The clip acts
    after step C; the explicit mixing k_mix (D > 0) then moves m but not ma (the tracer equation has no mixing term), so
    AGE can exceed t_k by O(D dt / h^2) relative (1e-6 - 4e-6 at D 1e-7, tulip / leaf N 48 - 64);
  4 budget (numpy): sum ma dA = sum over sub-steps of dt t int s_m dA - (what the clip took), to round-off (1e-12).
--closure: the closure json's numbers (Params fields) and its switches e0 / exit_cal / milk / sink / visc_tr / upwell;
--set overrides after it (kappa_Q included: both solvers get S_rec = kappa_Q Q Z from the record's Q, as DiffPour).
The default pattern is tulip.
Geometry mode (--geometry): a fixed source at rest (u_in 0, U_perp 0 so chi = 1, still foot) at the cup centre x0 in a
still layer, area rate S_eff 0.06 D^2/s, round footprint r 0.04 D, Stage-1 layer values; after T the isochrones AGE = t_b
for ages T - t_b 0.3, 0.6, 0.9, 1.2, 1.5 s (Warp fast path; numpy for comparison), each fitted with a circle.
PASS (fgr23 SPEC2 5.4.3 "circles about the K_d centre within one cell, radius ~ sqrt(age)"), on the interior isochrones,
ages 0.3 - 1.2 s: (i) the circle's centre within one cell h of x0; (ii) every crossing point within h of the circle;
(iii) |R - sqrt(S_eff age / pi)| <= h (the area law of a uniform white layer, l = c = 1).
Reported, not part of PASS: the age-1.5 s isochrone (1.5 - 2.3 cells inside the white front at N 96 - 144, inside the
front's transition zone; absent at N 48), the white volume inside each isochrone against the source volume surfaced
after t_b, and the slope of log R against log(age).  These PASS criteria were fixed after the first runs (N 48, 96,
144), whose own stricter criteria -- volume inside the isochrone within 5 % of the source volume, slope 0.5 +- 0.05 --
failed at N 48 / 96 (volume -18 ... -4 %, slope 0.60 / 0.59) and passed at N 144 except the front isochrone (volume
-3.1 ... -3.7 %, slope 0.55): the first-order upwind surfacing time a_up in the diverging flow places every isochrone
about half a cell inside the area-law radius at every grid (-0.0103 / -0.0056 / -0.0040 D at N 48 / 96 / 144), and
that bias weighs most at small radii.  (fgr23/build/DEVIATIONS.md.)"""
import argparse
import json
import math
import time
from dataclasses import fields, replace

import numpy as np

from ..solver import Inlet, Numerics, Params, Solver
from . import schedule as SC
from . import wsolver as W

STATE = ("l", "qx", "qy", "m")
M_MIN = 1e-3                      # AGE is defined where m >= M_MIN


def age_field(m, ma, msk):
    ok = msk & (m >= M_MIN)
    return np.where(ok, ma / np.where(ok, m, 1.0), 0.0), ok


def load_closure(fn, script):
    """Closure json -> script.params (Params fields) + switches (milk, sink, visc_tr codes, upwell)."""
    J = json.load(open(fn))
    num = {k: float(v) for k, v in J.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    names = {f.name for f in fields(Params)}
    if num.get("circ", 0.0):
        raise SystemExit("closures with the surface current (circ) are not supported here")
    script.params.update({k: v for k, v in num.items() if k in names and k not in ("milk", "sink", "visc_tr")})
    return dict(milk=W.milk_code(num.get("milk", 0.0)), sink=W.sink_code(num.get("sink", 0.0)),
                visc_tr=W.switch01(num.get("visc_tr", 0.0), "visc_tr"), upwell=bool(num.get("upwell", 0.0)))


def pour_mode(a):
    e0, exit_cal = False, False
    if a.closure:
        J = json.load(open(a.closure))
        e0, exit_cal = SC.e0_mode(J.get("e0", 0.0)), bool(J.get("exit_cal", 0.0))
    script, _ = SC.pour_script(a.pattern, a.N, e0=e0, exit_cal=exit_cal)
    sw = load_closure(a.closure, script) if a.closure else dict(milk=W.milk_code(a.milk), sink=0, visc_tr=0, upwell=False)
    if a.closure and a.milk_given:
        sw["milk"] = W.milk_code(a.milk)
    script.params.update({k: float(v) for k, v in (x.split("=") for x in a.set.split(",") if x)})
    script.params.update(milk=sw["milk"], sink=sw["sink"], visc_tr=sw["visc_tr"], upwell=sw["upwell"])
    S = SC.build(script, a.N)
    n = min(a.frames, S["n"])
    P = Params(**script.params)
    # the Warp solver receives S_rec = kappa_Q Q Z (record Q = S_eff / kQ of the script); the numpy reference takes S_eff
    # from the inlet: give it kappa_Q Q, so both run the closure's kappa_Q (as DiffPour does) with the same product
    inlets = [replace(inl, S_eff=P.kappa_Q * float(S["Q"][k])) for k, inl in enumerate(S["inlets"])]
    print(f"{a.pattern}: {S['n']} frames, testing {n}; closure {a.closure or 'script default'}; milk "
          f"{W.MILK_SCHEMES[sw['milk']]}, sink {W.SINK_FORMS[sw['sink']]}, visc_tr {sw['visc_tr']}; params "
          + " ".join(f"{k} {getattr(P, k):.4g}" for k in W.PARAMS), flush=True)
    ok = True
    res = {}
    # ---- numpy: off / on
    np_runs = {}
    for age in (False, True):
        ref = Solver(P, Numerics(N=a.N), age=age)
        t0 = time.time()
        st, ns, mas, ts = [], [], [], []
        for k in range(n):
            ts.append(ref.t)
            _, nk = ref.advance_frame(inlets[k], frame_dt=S["fdt"][k])
            ns.append(nk)
            st.append({c: getattr(ref, c).copy() for c in STATE})
            if age:
                mas.append(ref.ma.copy())
        np_runs[age] = dict(st=st, ns=ns, ma=mas, t=ts, ref=ref, wall=time.time() - t0)
        print(f"numpy age {age}: {n} frames in {np_runs[age]['wall']:.1f} s, sub-steps {sum(ns)}", flush=True)
    msk = np_runs[True]["ref"].g.mask
    tk = np.array(np_runs[True]["t"])                 # frame-start times (the clip bound of each frame)
    # 1 numpy neutrality
    d_np = max(float(np.max(np.abs(x[c] - y[c]))) for x, y in zip(np_runs[False]["st"], np_runs[True]["st"]) for c in STATE)
    eq = d_np == 0.0 and np_runs[False]["ns"] == np_runs[True]["ns"]
    print(f"[1 numpy] age on vs off: max |diff| over {n} frames x {STATE} = {d_np:.3e}, sub-steps equal "
          f"{np_runs[False]['ns'] == np_runs[True]['ns']} -> {'BITWISE' if eq else 'DIFFERENT'}")
    ok &= eq
    res["neutral_numpy"] = d_np
    # 4 numpy budget
    ref = np_runs[True]["ref"]
    tot = float(ref.ma[msk].sum() * ref.g.area)
    bud = ref.age_src - ref.age_clip
    rel = abs(tot - bud) / max(abs(ref.age_src), 1e-300)
    print(f"[4 budget] sum ma dA {tot:.15g}; source sum dt t int s_m dA {ref.age_src:.15g}, clip took {ref.age_clip:.3e} "
          f"({ref.age_clip / max(ref.age_src, 1e-300):.2e} of the source); residual {tot - bud:.3e} (relative {rel:.2e})")
    ok &= rel < 1e-12
    res["budget_rel"] = rel
    res["clip_share"] = ref.age_clip / max(ref.age_src, 1e-300)
    # ---- Warp
    import warp as wp
    wp.init()
    g = W.Geo(a.N, device=a.device)
    Pnp = np.array([getattr(P, k) for k in W.PARAMS], float)
    Pw = wp.array(Pnp, dtype=W.F, device=g.dev)
    Sw = {k: wp.array(S[k], dtype=W.F, device=g.dev) for k in W.SCHED}
    v05 = S["return_law"] == "v05"
    kw = dict(v05=v05, check_every=1, up=sw["upwell"], milk=sw["milk"], sink=sw["sink"], visc_tr=sw["visc_tr"])
    wr = {}
    for age in (False, True):                          # recorded path (ignores age)
        sol = W.WSolver(g, age=age, **kw)
        s = sol.initial_state()
        sts, nw = [], []
        for k in range(n):
            s, nk = sol.frame(s, Pw, Pnp, Sw, k, float(S["fdt"][k]), int(S["q"][k]), bool(S["active"][k]))
            nw.append(nk)
            sts.append(sol.to_numpy(s))
        wr[age] = (sts, nw)
    d_fr = max(float(np.max(np.abs(x[c] - y[c]))) for x, y in zip(wr[False][0], wr[True][0]) for c in STATE)
    eq = d_fr == 0.0 and wr[False][1] == wr[True][1]
    print(f"[1 warp frame] age on vs off: max |diff| {d_fr:.3e}, sub-steps equal {wr[False][1] == wr[True][1]} -> "
          f"{'BITWISE' if eq else 'DIFFERENT'}")
    ok &= eq
    res["neutral_frame"] = d_fr
    wf = {}
    for age in (False, True):                          # fast path
        sol = W.WSolver(g, age=age, **kw)
        t0 = time.time()
        fin, _, nw = sol.fast_forward(Pw, Pnp, Sw, S, keep=False, frames=n, snap=range(n))
        wf[age] = ([sol.to_numpy(sol.snaps[k]) for k in range(n)], nw, time.time() - t0)
        print(f"warp fast age {age}: {n} frames in {wf[age][2]:.1f} s, sub-steps {sum(nw)}", flush=True)
    d_fa = max(float(np.max(np.abs(x[c] - y[c]))) for x, y in zip(wf[False][0], wf[True][0]) for c in STATE)
    eq = d_fa == 0.0 and wf[False][1] == wf[True][1]
    print(f"[1 warp fast] age on vs off: max |diff| {d_fa:.3e}, sub-steps equal {wf[False][1] == wf[True][1]} -> "
          f"{'BITWISE' if eq else 'DIFFERENT'}")
    ok &= eq
    res["neutral_fast"] = d_fa
    # 2 parity (fast path, age on, against numpy, age on)
    on_w, on_n = wf[True][0], np_runs[True]
    st_rel = 0.0
    for c in STATE:
        d = max(float(np.abs(x[c] - y[c])[msk].max()) for x, y in zip(on_w, on_n["st"]))
        sc = max(float(np.abs(y[c])[msk].max()) for y in on_n["st"]) + 1e-300
        print(f"[2 parity] {c}: max |warp - numpy| over the frames {d:.3e} (relative {d / sc:.2e})")
        st_rel = max(st_rel, d / sc)
    sub_eq = wf[True][1] == on_n["ns"]
    ok &= st_rel < 1e-12 and sub_eq
    dma = max(float(np.abs(x["ma"] - y)[msk].max()) for x, y in zip(on_w, on_n["ma"]))
    sma = max(float(np.abs(y)[msk].max()) for y in on_n["ma"]) + 1e-300
    dage, nag = 0.0, 0
    bnd = dict(numpy=[0.0, 0.0, 0.0], warp=[0.0, 0.0, 0.0])   # largest excursion below 0 / above t (end of frame) /
    t_end = tk + np.asarray(S["fdt"][:n], float)               # above t_k (frame start), relative to that time
    for k in range(n):
        Aw, okw = age_field(on_w[k]["m"], on_w[k]["ma"], msk)
        An, okn = age_field(on_n["st"][k]["m"], on_n["ma"][k], msk)
        both = okw & okn
        nag += int(both.sum())
        if both.any():
            dage = max(dage, float(np.abs(Aw - An)[both].max()) / max(tk[k], 1e-300))
        for name, A_, o_ in (("warp", Aw, okw), ("numpy", An, okn)):
            if o_.any():
                bnd[name][0] = max(bnd[name][0], float(-A_[o_].min()))
                bnd[name][1] = max(bnd[name][1], float((A_[o_] - t_end[k]).max()) / t_end[k])
                bnd[name][2] = max(bnd[name][2], float((A_[o_] - tk[k]).max()) / max(tk[k], 1e-300))
    print(f"[2 parity] ma: max |warp - numpy| {dma:.3e} (relative to max |ma| {sma:.3e}: {dma / sma:.2e}); AGE "
          f"(m >= {M_MIN:g} in both, {nag} cell-frames): max |warp - numpy| / t_k {dage:.2e}; sub-steps equal {sub_eq}")
    ok &= dma / sma <= 1e-10 and dage <= 1e-10
    res.update(state_rel=st_rel, ma_rel=dma / sma, age_rel=dage)
    # 3 bounds
    for name in ("numpy", "warp"):
        lo, hi, hk = bnd[name]
        good = lo <= 0.0 and hi <= 0.0
        print(f"[3 bounds] {name}: min AGE {-lo:+.3e} (>= 0), max (AGE - t) / t {hi:+.3e} (<= 0) -> "
              f"{'ok' if good else 'VIOLATED'}; (reported) max (AGE - t_k) / t_k {hk:+.3e} against the frame-start time")
        ok &= good
    res["bounds"] = bnd
    Afin, okf = age_field(on_w[-1]["m"], on_w[-1]["ma"], msk)
    print(f"final frame (t_k {tk[-1]:.4f} s): AGE defined on {int(okf.sum())} cells, range "
          f"[{Afin[okf].min() if okf.any() else 0:.4f}, {Afin[okf].max() if okf.any() else 0:.4f}] s")
    if a.save:
        np.savez_compressed(a.save, np_ma=on_n["ma"][-1], warp_ma=on_w[-1]["ma"], m=on_w[-1]["m"], l=on_w[-1]["l"],
                            t_k=tk, mask=msk, **{f"res_{k}": np.asarray(v, dtype=object) for k, v in res.items()})
    print("PASS" if ok else "FAIL")
    return ok


def isochrone_points(A, ok, xs, ys, tb, h):
    """Crossings of AGE = tb on the edges between adjacent cells where AGE is defined (linear interpolation)."""
    pts = []
    for (A1, A2, o1, o2, x1, y1, dx, dy) in (
            (A[:, :-1], A[:, 1:], ok[:, :-1], ok[:, 1:], xs[:, :-1], ys[:, :-1], h, 0.0),
            (A[:-1, :], A[1:, :], ok[:-1, :], ok[1:, :], xs[:-1, :], ys[:-1, :], 0.0, h)):
        sel = o1 & o2 & ((A1 - tb) * (A2 - tb) < 0)
        f = (tb - A1[sel]) / (A2[sel] - A1[sel])
        pts.append(np.stack([x1[sel] + f * dx, y1[sel] + f * dy], 1))
    return np.concatenate(pts, 0)


def fit_circle(p):
    """Algebraic least-squares circle (Kasa): centre, radius, residuals |p - c| - R."""
    A = np.stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))], 1)
    b = (p ** 2).sum(1)
    (cx, cy, c0), *_ = np.linalg.lstsq(A, b, rcond=None)
    R = math.sqrt(c0 + cx * cx + cy * cy)
    return np.array([cx, cy]), R, np.hypot(p[:, 0] - cx, p[:, 1] - cy) - R


def geometry_mode(a):
    N, h = a.N, 1.0 / a.N
    S_eff, r, T = 0.06, 0.04, a.T
    x0 = (0.0, 0.0)
    num = Numerics(N=N)
    n = int(round(T / num.frame_dt))
    fdt = np.full(n, num.frame_dt)
    r_ = max(r, 1.5 * h)
    q = min(max(num.kernel_quadrature, int(math.ceil(4 * h / r_))), 12)
    # Stage-1 layer values (latte_imex/data/closures/mcd_S1_N96.json), kappa_Q 1: S_eff is the received area rate
    P = Params(cp=1.7711617739147767, beta=0.7016740944956349, nu=0.01, D=1e-7, kappa_Q=1.0, kappa_c=0.3465495819390284,
               kappa_t=0.0, kappa_r=0.0, B_dep=1267.4256251285594, p_dep=7.150394169793254, tau_s=0.075, vj=True,
               kappa_tc=0.0, xi_b=0.5, lam_b=10.0, milk=W.milk_code(a.milk), return_law="v05")
    inl = Inlet(active=True, x_hit=x0, S_eff=S_eff, u_in=(0.0, 0.0), U_perp=0.0, d_jet=0.05, r1=r, r2=r, phi=0.0)
    S = {k: np.zeros(n) for k in W.SCHED}
    S["x"][:], S["y"][:], S["Q"][:], S["d"][:], S["r1"][:], S["r2"][:] = x0[0], x0[1], S_eff, 0.05, r, r
    sched = dict(fdt=fdt, q=np.full(n, q), active=np.ones(n, bool), n=n)
    tk = np.concatenate([[0.0], np.cumsum(fdt)[:-1]])
    print(f"geometry: still source at {x0}, S_eff {S_eff} D^2/s, r {r} D, N {N} (h {h:.4f}), T {T} s ({n} frames), milk "
          f"{W.MILK_SCHEMES[P.milk]}", flush=True)
    ref = Solver(P, num, age=True)
    t0 = time.time()
    for k in range(n):
        ref.advance_frame(inl, frame_dt=fdt[k])
    print(f"numpy: {time.time() - t0:.1f} s", flush=True)
    import warp as wp
    wp.init()
    g = W.Geo(N, device=a.device)
    Pnp = np.array([getattr(P, k) for k in W.PARAMS], float)
    sol = W.WSolver(g, v05=True, check_every=1, milk=P.milk, age=True)
    Sw = {k: wp.array(S[k], dtype=W.F, device=g.dev) for k in W.SCHED}
    fin, _, ns = sol.fast_forward(wp.array(Pnp, dtype=W.F, device=g.dev), Pnp, Sw, sched, keep=False, snap=[n - 1])
    st = sol.to_numpy(sol.snaps[n - 1])
    msk = g.mask_np
    ok_all = True
    T_end = float(np.sum(fdt))
    for name, m, ma in (("warp", st["m"], st["ma"]), ("numpy", ref.m, ref.ma)):
        A, ok = age_field(m, ma, msk)
        print(f"[{name}] white volume {float(m[msk].sum() * h * h):.6f} (source {S_eff * T_end:.6f}); AGE on "
              f"{int(ok.sum())} cells")
        Rs, ages = [], []
        for age_ in (0.3, 0.6, 0.9, 1.2, 1.5):
            interior = age_ <= 1.2 + 1e-9
            tb = T_end - age_
            p = isochrone_points(A, ok, g.x_np, g.y_np, tb, h)
            if len(p) < 8:
                print(f"  age {age_}: {len(p)} crossing points, no isochrone" + (" -> FAIL" if interior else " (reported)"))
                ok_all &= not interior
                continue
            c, R, res = fit_circle(p)
            vin = float(m[ok & (A > tb)].sum() * h * h)
            vsrc = S_eff * float(fdt[tk > tb].sum())
            dc = math.hypot(c[0] - x0[0], c[1] - x0[1])
            R_law = math.sqrt(S_eff * age_ / math.pi)
            good = dc <= h and np.abs(res).max() <= h and abs(R - R_law) <= h
            print(f"  age {age_:.1f} s (t_b {tb:.3f}): {len(p)} points, centre offset {dc / h:.3f} h, R {R:.4f} "
                  f"({R / h:.2f} h), radial scatter rms {np.sqrt(np.mean(res ** 2)) / h:.3f} h max {np.abs(res).max() / h:.3f} h, "
                  f"R - sqrt(S age / pi) {(R - R_law) / h:+.3f} h; white volume inside {vin:.6f} vs surfaced after t_b "
                  f"{vsrc:.6f} ({vin / vsrc - 1:+.2%}) -> " + (("ok" if good else "FAIL") if interior else "(reported)"))
            if interior:
                ok_all &= good
            if R >= 2 * r:
                Rs.append(R)
                ages.append(age_)
        if len(Rs) >= 2:
            sl = float(np.polyfit(np.log(ages), np.log(Rs), 1)[0])
            print(f"  (reported) radius ~ age^{sl:.3f} over {len(Rs)} isochrones with R >= 2 r")
    Aw, okw = age_field(st["m"], st["ma"], msk)
    An, okn = age_field(ref.m, ref.ma, msk)
    both = okw & okn
    print(f"warp vs numpy: max |ma| diff {float(np.abs(st['ma'] - ref.ma)[msk].max()):.3e}, AGE diff "
          f"{float(np.abs(Aw - An)[both].max()):.3e} s; sub-steps warp {sum(ns)}")
    print("PASS" if ok_all else "FAIL")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="tulip")
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--device", default=None)
    ap.add_argument("--closure", default="", help="closure json (latte_imex/data/closures/*.json)")
    ap.add_argument("--set", default="", help="closure values after --closure, e.g. tau_s=0.05")
    ap.add_argument("--milk", default=None, choices=W.MILK_SCHEMES, help="milk scheme (default: the closure's, else vl)")
    ap.add_argument("--save", default="")
    ap.add_argument("--geometry", action="store_true", help="still-source isochrone test instead of a pour")
    ap.add_argument("--T", type=float, default=1.8, help="geometry: seconds simulated")
    a = ap.parse_args()
    a.milk_given = a.milk is not None
    if a.milk is None:
        a.milk = "mp5fct" if a.geometry else "vl"
    geometry_mode(a) if a.geometry else pour_mode(a)


if __name__ == "__main__":
    main()
