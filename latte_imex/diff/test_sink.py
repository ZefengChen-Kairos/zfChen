"""Area budget of the column entrainment sink (closure switch sink = 1, wsolver.SINK_FORMS; mcd22 SPEC 2.5, B3.4 /
B3.5) on a synthetic single pulse with the landing point fixed, through the Warp solver's recorded path
(WSolver.fields / substep / k_exchange: the kernels the fast path launches too).
    python -m latte_imex.diff.test_sink [--N 64]

Each sub-step (dt) is split at the exchange: transport (steps A-D) -> pre, exchange (k_exchange) -> post.
  B3.4  transport: sum(pre.l - st.l) h^2 = dt sum(s) h^2 (conservative step, to round-off); exchange: sum(pre.l -
        post.l) h^2 = E = sum b (1 - exp(-dt (kappa_e ej / l + W_o))) h^2 (the exact factor, to round-off), so
        sum(l) h^2 changes by dt sum(s) h^2 - E per sub-step and, with lam_b = 0, by the deposited minus the swallowed
        area over the pour; the milk is untouched by the exchange at eps_f = 0 (the column takes brown only).
  B3.5  sum(ej) h^2 = 4 Q Z / d (Z the in-cup share of the landing kernel), so the swallowing rate kappa_e sum(ej (1 -
        c)) h^2 = eps_c (4 Q / d) Z <1 - c>_K (<.>_K the landing-kernel mean).  The exchange applies it as the factor
        exp(-dt kappa_e ej / l) on the brown (exact for frozen l), so the area taken in one sub-step equals rate x dt
        only while kappa_e ej dt << 1: reported per frame (saturation E / (rate dt)) and as a time-step study (sub-steps
        x 1, 2, 4, 8), information for the grid / time-step convergence of runs with the sink.
Two cases: lam_b = 0 (the sink alone) and lam_b = 10 (with the open-surface exchange of the two-phase layer: the
per-sub-step identity holds with W_o included; the saturation is reported for the sink part alone)."""
import argparse
import math

import numpy as np
import warp as wp

from . import wsolver as W

D_MM = 85.5                                    # tulip cup diameter = solver length unit [mm]


def case(N, lam_b, n_on=60, n_off=30, mult=1, log=print, table=True):
    """Pulse of n_on frames at a fixed landing point, then n_off frames without stream.  Returns ok (table=True) or
    (swallowed, deposited [mm^2], frame-0 row) (table=False)."""
    wp.init()
    g = W.Geo(N, device="cpu")
    h = g.h
    # stream: 12 mL/s at U_perp 0.6 m/s (pulse values, mcd22 SPEC 2.5), d from Q = U_perp pi d^2 / 4 (5.0 mm)
    Q = 12e3 / D_MM ** 3                       # [D^3/s]
    U = 600.0 / D_MM                           # [D/s]
    d = math.sqrt(4.0 * Q / (math.pi * U))     # [D]
    r = max(0.5 * d, 1.5 * h)
    q = min(max(3, int(math.ceil(4 * h / r))), 12)
    n, fdt = n_on + n_off, 1.0 / 120.0
    act = np.r_[np.ones(n_on), np.zeros(n_off)]
    S = dict(x=np.full(n, -0.15), y=np.zeros(n), Q=Q * act, ux=0.8 * act, uy=np.zeros(n), U=U * act,
             d=np.where(act > 0, d, 1e-3), r1=np.full(n, r), r2=np.full(n, r), phi=np.zeros(n), vjx=np.zeros(n),
             vjy=np.zeros(n))
    Pd = {k: 0.0 for k in W.PARAMS}             # mcd22 Stage 2 (data/closures/mcd_S2_N96.json), lam_b as given
    Pd.update(cp=1.7711617739147767, beta=0.7016740944956349, nu=0.01, D=1e-7, kappa_Q=2.234172646174109,
              kappa_c=0.3465495819390284, B_dep=1267.4256251285594, p_dep=7.150394169793254, tau_s=0.075, vj=1.0,
              kappa_e=0.032, xi_b=0.5, lam_b=lam_b)
    Pnp = np.array([Pd[k] for k in W.PARAMS], float)
    P = wp.array(Pnp, dtype=W.F, device=g.dev)
    Sw = {k: wp.array(S[k], dtype=W.F, device=g.dev) for k in W.SCHED}
    sol = W.WSolver(g, v05=True, sink="column")
    msk = g.mask_np
    A = lambda a: float(a.numpy()[msk].sum() * h * h)
    st = sol.initial_state()
    nu_on = Pnp[W.NU] > 0
    worst = dict(transport=0.0, exchange=0.0, milk_ex=0.0, ej_sum=0.0)
    E_tot = dep_tot = 0.0
    A_start = A(st["l"])
    rows = []
    for k in range(n):
        sol.set_ex(Pnp)
        active = bool(act[k])
        fl = sol.fields(P, Sw, k, q, active, st)
        s_np, ej_np = fl[0].numpy(), fl[5].numpy()
        if active:                              # in-cup share of the landing kernel, as k_fields reads it
            Kraw, sums = sol.new(), sol.new1()
            sol.L(W.k_kernel_raw, [g.xc, g.yc, g.mask, Sw["x"], Sw["y"], Sw["ux"], Sw["uy"], P, W.F(0.0), Sw["r1"],
                                   Sw["r2"], Sw["phi"], sol.kq, W.F(h)], [Kraw, sums])
            sm = sums.numpy()
            Z = sm[0] / sm[1]
            want = 4.0 * Q * Z / d
            worst["ej_sum"] = max(worst["ej_sum"], abs(float(ej_np[msk].sum() * h * h) - want) / want)
        nsub = sol.n_substeps_host(st, fl, fdt, active) * mult
        dt = fdt / nsub
        sol.set_dt(dt)
        E_f = dep_f = rate_f = 0.0
        for _ in range(nsub):
            ex = sol.ex_on
            sol.ex_on = False
            pre = sol.substep(st, fl, P, nu_on, sol.mix_on(Pnp, dt))
            sol.ex_on = ex
            post = {c: sol.new() for c in ("l", "qx", "qy", "m")}
            sol.L(W.k_exchange, [pre["l"], pre["qx"], pre["qy"], pre["m"], sol.Kb, sol.Db, sol.Wf, fl[5], P, sol.dta,
                                 g.mask], [post["l"], post["qx"], post["qy"], post["m"]])
            dep = dt * float(s_np[msk].sum() * h * h)
            worst["transport"] = max(worst["transport"], abs(A(pre["l"]) - A(st["l"]) - dep))
            l_p, m_p = pre["l"].numpy()[msk], pre["m"].numpy()[msk]
            Wj = Pd["kappa_e"] * ej_np[msk] / l_p
            Wo = lam_b * (l_p - 1.0) / l_p
            E_exact = float(((l_p - m_p) * (1.0 - np.exp(-dt * (Wj + Wo)))).sum() * h * h)
            worst["exchange"] = max(worst["exchange"], abs(A(pre["l"]) - A(post["l"]) - E_exact))
            worst["milk_ex"] = max(worst["milk_ex"], abs(A(pre["m"]) - A(post["m"])))
            E_f += float(((l_p - m_p) * (1.0 - np.exp(-dt * Wj))).sum() * h * h)       # the sink part
            rate_f += dt * Pd["kappa_e"] * float((ej_np[msk] * (1.0 - np.clip(m_p / l_p, 0.0, 1.0))).sum() * h * h)
            dep_f += dep
            st = post
        E_tot += E_f
        dep_tot += dep_f
        if active and (k % 10 == 0 or k == n_on - 1):
            rows.append((k, nsub, dep_f, E_f, rate_f))
    A_end = A(st["l"])
    mm2 = D_MM ** 2
    if not table:
        return E_tot * mm2, dep_tot * mm2, rows[0]
    budget = abs((A_end - A_start) - (dep_tot - E_tot))
    log(f"N {N}, lam_b {lam_b:g}: Q {Q * D_MM ** 3 / 1e3:.1f} mL/s, d {d * D_MM:.2f} mm, U_perp {U * D_MM / 1e3:.2f} "
        f"m/s, eps_c {Pd['kappa_e']:g}: eps_c 4Q/d = {Pd['kappa_e'] * 4 * Q / d * mm2:.1f} mm^2/s; pulse {n_on} frames "
        f"+ {n_off} off")
    log("  frame nsub  deposited  swallowed  eps_c 4Q/d Z <1-c>_K dt   [mm^2 per frame]")
    for k, ns, dp, e, ra in rows:
        log(f"  {k:5d} {ns:4d}  {dp * mm2:9.4f}  {e * mm2:9.4f}  {ra * mm2:9.4f}   (saturation E / (rate dt) "
            f"{e / max(ra, 1e-300):.3f})")
    log(f"  totals: deposited {dep_tot * mm2:.3f} mm^2, swallowed by the sink {E_tot * mm2:.3f} mm^2 (ratio "
        f"{E_tot / max(dep_tot, 1e-300):.3f}); layer area change {(A_end - A_start) * mm2:.3f} mm^2"
        + (f", budget residual {budget * mm2:.2e} mm^2" if lam_b == 0 else " (the lam_b exchange changes it too)"))
    log(f"  worst per sub-step [D^2]: transport |dA - dt sum s| {worst['transport']:.2e}; exchange |dA - exact factor| "
        f"{worst['exchange']:.2e}; milk taken by the exchange {worst['milk_ex']:.2e}; |sum ej h^2 - 4QZ/d| / (4QZ/d) "
        f"{worst['ej_sum']:.2e}")
    return (worst["transport"] < 1e-12 and worst["exchange"] < 1e-13 and worst["milk_ex"] == 0.0
            and worst["ej_sum"] < 1e-12 and (lam_b != 0 or budget < 1e-11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    N = ap.parse_args().N
    ok = case(N, 0.0) & case(N, 10.0)
    print("time-step study (lam_b 0): sub-steps x mult -> swallowed / deposited over the pulse, frame-0 saturation")
    for mult in (1, 2, 4, 8):
        E, dep, r0 = case(N, 0.0, mult=mult, table=False)
        print(f"  x{mult}: swallowed {E:.3f} mm^2 of {dep:.3f} deposited ({E / dep:.4f}); frame 0: E / (rate dt) "
              f"{r0[3] / r0[4]:.4f}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
