"""Checks of the two switches of the 'chi' closure: Params.visc (symmetric viscous stress) and Params.offcup.

    NUMBA_NUM_THREADS=1 python -m latte_imex.test_visc                 # CPU checks
    NUMBA_ENABLE_CUDASIM=1 NUMBA_NUM_THREADS=1 python -m latte_imex.test_visc cuda   # + CPU vs CUDA (simulator, tiny N)

Symmetric viscosity: l u+ - dt div(2 nu l S(u+)) = l u-, S = sym grad u (V1 3.3-3.4), K = G^T W G with face-normal
strain rows (u_x on x faces, v_y on y faces), corner shear rows (u_y + v_x on 2x2 cells) and wall rows (u.n_b).
"""
import math
import os
import sys

import numpy as np

from .solver import Solver, Params, Numerics, Inlet, kernel_fraction


def _solver(N, nu=1e-3, **kw):
    return Solver(Params(nu=nu, closure="chi", return_law="v05", **kw), Numerics(N=N))


def _Ku(sol, ux, uy, lsn=None, wall=True):
    """K u (per unit area): (A u - L u)/dt with dt = 1."""
    g = sol.g; N = g.N
    lsn = np.ones((N, N)) if lsn is None else lsn
    apply, _ = sol.sym_visc_operator(lsn, 1.0)
    if not wall:
        cm, geo, nbx, nby = g.sym_geometry()
        g._symgeo = (cm, np.zeros_like(geo), nbx, nby)
        apply, _ = sol.sym_visc_operator(lsn, 1.0)
        g._symgeo = (cm, geo, nbx, nby)
    X = np.concatenate([np.where(g.mask, ux, 0.0), np.where(g.mask, uy, 0.0)])
    Y = apply(X) - np.concatenate([lsn, lsn]) * X
    return np.where(g.mask, Y[:N], 0.0), np.where(g.mask, Y[N:], 0.0)


def test_rigid(N=64, verbose=True):
    """Rigid translation: zero stress at every cell off the wall (the wall rows hold the no-penetration);
    rigid rotation about the cup centre: zero stress everywhere, wall included (u.n_b = 0)."""
    sol = _solver(N, nu=2e-3)
    g = sol.g
    l = np.where(g.mask, 1.0 + 0.3 * np.cos(5 * g.x) * np.sin(3 * g.y), 1.0)       # non-uniform layer
    cm, geo, _, _ = g.sym_geometry()
    interior = g.mask & (geo == 0)
    kx, ky = _Ku(sol, np.full((N, N), 0.7), np.full((N, N), -0.4), l)
    tr = float(max(np.abs(kx[interior]).max(), np.abs(ky[interior]).max()))
    wall_tr = float(max(np.abs(kx[~interior]).max(), np.abs(ky[~interior]).max()))
    kx, ky = _Ku(sol, -g.y, g.x, l)
    rot = float(max(np.abs(kx).max(), np.abs(ky).max()))
    if verbose:
        print(f"rigid: translation max|Ku| interior {tr:.1e} (wall cells {wall_tr:.1e}, no-penetration), "
              f"rotation max|Ku| everywhere {rot:.1e}", flush=True)
    assert tr < 1e-12 and rot < 1e-12 and wall_tr > 1e-3
    return dict(translation_interior=tr, rotation=rot)


def test_momentum(N=48, verbose=True):
    """Without the wall rows the viscous force sums to zero for any u (interior momentum conservation): every face /
    corner flux enters two cells with opposite signs."""
    sol = _solver(N)
    g = sol.g
    rng = np.random.default_rng(1)
    l = np.where(g.mask, 1 + 0.5 * rng.random((N, N)), 1.0)
    kx, ky = _Ku(sol, rng.standard_normal((N, N)), rng.standard_normal((N, N)), l, wall=False)
    sc = float(np.abs(kx).max())
    res = (abs(float(kx[g.mask].sum())) / sc, abs(float(ky[g.mask].sum())) / sc)
    if verbose:
        print(f"momentum: |sum K u| / max|K u| = {res[0]:.1e}, {res[1]:.1e}", flush=True)
    assert max(res) < 1e-12
    return dict(sum_x=res[0], sum_y=res[1])


def test_spd_nullspace(N=14, verbose=True):
    """Dense matrix at small N: A = L + dt K symmetric; K positive semi-definite with null space = {2 translations,
    1 rotation} without the wall rows (no checkerboard mode) and = {rotation} with them."""
    sol = _solver(N, nu=1.0)
    g = sol.g
    rng = np.random.default_rng(2)
    l = np.where(g.mask, 1 + 0.5 * rng.random((N, N)), 1.0)
    idx = np.flatnonzero(np.concatenate([g.mask, g.mask]).ravel())
    out = {}
    for wall in (True, False):
        cm, geo, nbx, nby = g.sym_geometry()
        if not wall:
            g._symgeo = (cm, np.zeros_like(geo), nbx, nby)
        apply, diag = sol.sym_visc_operator(l, 1.0)
        n2 = 2 * N * N
        A = np.empty((n2, n2))
        E = np.zeros(n2)
        for k in range(n2):
            E[k] = 1.0
            A[:, k] = apply(E.reshape(2 * N, N)).ravel()
            E[k] = 0.0
        g._symgeo = (cm, geo, nbx, nby)
        A = A[np.ix_(idx, idx)]
        Lm = np.concatenate([l, l]).ravel()[idx]
        Km = A - np.diag(Lm)
        asym = float(np.abs(A - A.T).max() / np.abs(A).max())
        dg = float(np.abs(np.diag(A) - diag.ravel()[idx]).max())
        ev = np.linalg.eigvalsh(0.5 * (Km + Km.T))
        tol = 1e-10 * ev.max()
        null = int((ev < tol).sum())
        out["wall" if wall else "nowall"] = dict(asym=asym, min_eig=float(ev.min()), null=null, diag_err=dg,
                                                 smallest_nonzero=float(ev[null] / ev.max()))
        if verbose:
            print(f"spd ({'with' if wall else 'without'} wall rows): asym {asym:.1e}, min eig {ev.min():.1e}, "
                  f"null dim {null}, smallest nonzero/largest {ev[null] / ev.max():.2e}, diag err {dg:.1e}", flush=True)
        assert asym < 1e-14 and ev.min() > -tol and dg < 1e-12
        assert null == (1 if wall else 3)
    # the checkerboard is not free: its Rayleigh quotient is of order the largest eigenvalue
    cb = np.where(g.mask, (-1.0) ** (np.arange(N)[None, :] + np.arange(N)[:, None]), 0.0)
    kx, ky = _Ku(sol, cb, 0 * cb, l, wall=False)
    rq = float((cb * kx).sum() / (cb * cb).sum())
    out["checkerboard_rayleigh"] = rq
    if verbose:
        print(f"checkerboard u: Rayleigh quotient {rq:.2e} (largest eigenvalue {ev.max():.2e})", flush=True)
    assert rq > 0.1 * ev.max()
    return out


def test_energy(N=48, verbose=True):
    """One implicit step never increases the discrete kinetic energy 1/2 sum l |u|^2 (any u, any l, any dt)."""
    sol = _solver(N, nu=5e-3)
    g = sol.g
    rng = np.random.default_rng(3)
    worst = -1.0
    for trial in range(4):
        l = np.where(g.mask, 0.5 + rng.random((N, N)), 1.0)
        ux, uy = rng.standard_normal((N, N)), rng.standard_normal((N, N))
        ux, uy = np.where(g.mask, ux, 0.0), np.where(g.mask, uy, 0.0)
        e0 = 0.5 * float((l * (ux ** 2 + uy ** 2))[g.mask].sum())
        dt = [1e-3, 1e-2, 1e-1, 1.0][trial]
        u1, v1, it = sol.sym_visc_step(ux, uy, l, dt)
        e1 = 0.5 * float((l * (u1 ** 2 + v1 ** 2))[g.mask].sum())
        worst = max(worst, e1 / e0)
        if verbose:
            print(f"energy: dt {dt:g}  E1/E0 = {e1 / e0:.6f}  (cg {it})", flush=True)
        assert e1 < e0
    return dict(max_ratio=worst)


def test_decay(N=128, verbose=True):
    """Sinusoidal modes, l = 1: shear u = (0, sin kx) decays at nu k^2, compression u = (sin kx, 0) at 2 nu k^2
    (the grad div u part of div(2 nu S)); measured in the cup interior (r < 0.3) over 20 implicit steps against
    exp(-rate t) and against the discrete backward-Euler factor 1/(1 + dt rate_h)^n."""
    nu, k, dt, n = 2e-3, 2 * math.pi * 3, 0.01, 20
    sol = _solver(N, nu=nu)
    g = sol.g
    l = np.ones((N, N))
    inner = g.mask & (g.r < 0.3)
    s = np.sin(k * g.x)
    sel = inner & (np.abs(s) > 0.5)
    kh = 4.0 * math.sin(k * g.h / 2) ** 2 / g.h ** 2        # discrete k^2 of the face difference
    out = {}
    for name, fac, (u, v) in (("shear", 1.0, (0 * s, s)), ("compression", 2.0, (s, 0 * s))):
        ux, uy = u.copy(), v.copy()
        for _ in range(n):
            ux, uy, _ = sol.sym_visc_step(ux, uy, l, dt)
        comp = uy if name == "shear" else ux
        other = ux if name == "shear" else uy
        amp = float(np.median(comp[sel] / s[sel]))
        spread = float(np.abs(comp[sel] / s[sel] - amp).max())
        rate = -math.log(amp) / (n * dt)
        exact = fac * nu * k * k
        disc = (1.0 + dt * fac * nu * kh) ** (-n)
        out[name] = dict(rate=rate, analytic=exact, rel_err=abs(rate - exact) / exact, amp=amp, amp_discrete=disc,
                         spread=spread, cross=float(np.abs(other[inner]).max()))
        if verbose:
            print(f"decay {name:11s}: rate {rate:.5f} /s, analytic {fac:.0f} nu k^2 = {exact:.5f} /s "
                  f"(rel err {abs(rate - exact) / exact:.2e}); amplitude {amp:.6f} vs discrete BE {disc:.6f}; "
                  f"interior spread {spread:.1e}, cross component {out[name]['cross']:.1e}", flush=True)
        assert abs(amp - disc) < 1e-4 and abs(rate - exact) / exact < 0.02 and spread < 1e-4
    r = out["compression"]["rate"] / out["shear"]["rate"]
    out["ratio"] = r
    if verbose:
        print(f"decay: compression / shear rate = {r:.4f} (2 with the grad div term, 1 without)", flush=True)
    assert abs(r - 2.0) < 0.01
    return out


def test_componentwise_unchanged(N=48, verbose=True):
    """visc='componentwise' + offcup='renormalise' is the V0.5 path: same as the defaults of that time. Here only the
    branch selection is checked (the bitwise check against the old code is done on the S1 runs)."""
    a = _solver(N, visc="componentwise", offcup="renormalise")
    b = _solver(N)
    inl = Inlet(active=True, x_hit=(0.1, -0.1), S_eff=0.05, U_perp=4.0, d_jet=0.03, r1=0.03, r2=0.02, u_in=(0.3, 0.0))
    for sol in (a, b):
        for _ in range(10):
            sol.advance_frame(inl)
    if verbose:
        print(f"branches: componentwise cg_visc {a.stats['cg_visc']}, symmetric cg_visc {b.stats['cg_visc']}; "
              f"deposited equal (foot inside): {a.deposited == b.deposited}", flush=True)
    assert a.stats["cg_visc"] == 0 and b.stats["cg_visc"] > 0 and a.deposited == b.deposited and b.spilled == 0.0
    return dict(ok=True)


def test_offcup(N=64, verbose=True):
    """Z = 1 inside; 0 < Z < 1 across the rim with deposited = chi Z S dt and spilled = (1 - Z) S dt; foot fully
    outside: Z = 0, nothing deposited, everything spilled, no exception; renormalise keeps all of it in the cup."""
    out = {}
    base = dict(S_eff=0.05, U_perp=4.0, d_jet=0.03, r1=0.04, r2=0.03, phi=0.3, u_in=(0.2, 0.1))
    for name, xh in (("inside", (0.1, -0.1)), ("rim", (0.0, -0.49)), ("outside", (0.0, -0.6)), ("off_grid", (0.0, -1.5))):
        inl = Inlet(active=True, x_hit=xh, **base)
        sol = _solver(N)
        Z = kernel_fraction(sol.g, sol.num, inl)
        T = 0.05
        chi = None
        while sol.t < T - 1e-12:
            chi, _ = sol.advance_frame(inl)
        L = sol.ledger()
        S = base["S_eff"]
        dep_ok = abs(L["deposited"] - chi * Z * S * sol.t) < 1e-12
        sp_ok = abs(L["spilled"] - (1 - Z) * S * sol.t) < 1e-12
        out[name] = dict(Z=Z, deposited=L["deposited"], spilled=L["spilled"], layer_error=L["layer_error"],
                         milk_error=L["milk_error"], offcup_frames=sol.stats["offcup_frames"])
        if verbose:
            print(f"offcup {name:8s}: Z {Z:.4f}  deposited {L['deposited']:.3e} (= chi Z S t: {dep_ok})  spilled "
                  f"{L['spilled']:.3e} (= (1-Z) S t: {sp_ok})  layer_err {L['layer_error']:.1e}", flush=True)
        assert dep_ok and sp_ok and abs(L["layer_error"]) < 1e-13 and abs(L["milk_error"]) < 1e-13
        if name == "inside":
            assert Z == 1.0 and L["spilled"] == 0.0
        elif name == "rim":
            assert 0.2 < Z < 0.8
        else:
            assert Z == 0.0 and L["deposited"] == 0.0
        if name == "rim":
            old = _solver(N, offcup="renormalise")
            while old.t < T - 1e-12:
                chi, _ = old.advance_frame(inl)
            assert abs(old.deposited - chi * S * old.t) < 1e-12 and old.spilled == 0.0
    # kernel clamp is counted
    sol = _solver(N)
    sol.advance_frame(Inlet(active=True, x_hit=(0.0, 0.0), S_eff=0.05, U_perp=4.0, d_jet=0.01, r1=0.5 / N, r2=0.5 / N))
    out["kernel_clamp"] = sol.stats["kernel_clamp"]
    assert sol.stats["kernel_clamp"] == 1
    return out


def test_cuda(N=20, T=0.06, verbose=True):
    """CPU Solver vs CudaSolver (simulator if NUMBA_ENABLE_CUDASIM=1) on a pour across the rim, both switch sets."""
    from .cuda_solver import CudaSolver
    out = {}
    for tag, sw in (("new", {}), ("old", dict(visc="componentwise", offcup="renormalise"))):
        # nu large enough that the viscous step matters at this tiny N (and the old CPU branch stays explicit)
        P = Params(cp=1.0, nu=2e-3, closure="chi", return_law="v05", kappa_r=0.1, **sw)
        num = Numerics(N=N)
        a, b = Solver(P, num), CudaSolver(P, num)
        k = 0
        while a.t < T - 1e-12:
            ang = 0.4 * k
            inl = Inlet(active=True, x_hit=(0.42 * math.cos(ang), 0.42 * math.sin(ang)), S_eff=0.08, U_perp=4.0,
                        d_jet=0.05, r1=0.09, r2=0.07, phi=ang, u_in=(0.3, -0.2), scan_speed=0.5)
            a.advance_frame(inl); b.advance_frame(inl)
            k += 1
        A, B = a.host_state(), b.host_state()
        rel = {kk: float(np.abs(A[kk] - B[kk]).max() / max(np.abs(A[kk]).max(), 1e-12)) for kk in A}
        La, Lb = a.ledger(), b.ledger()
        out[tag] = dict(rel=rel, spilled=(La["spilled"], Lb["spilled"]), cg_visc=(a.stats["cg_visc"], b.stats["cg_visc"]),
                        offcup=(a.stats["offcup_frames"], b.stats["offcup_frames"]))
        if verbose:
            print(f"cuda vs cpu [{tag}] N={N}: " + ", ".join(f"{kk} {v:.1e}" for kk, v in rel.items())
                  + f"; spilled {La['spilled']:.6e} / {Lb['spilled']:.6e}; cg_visc {a.stats['cg_visc']} / "
                  f"{b.stats['cg_visc']}; offcup frames {a.stats['offcup_frames']} / {b.stats['offcup_frames']}", flush=True)
        assert max(rel.values()) < 1e-8 and abs(La["spilled"] - Lb["spilled"]) < 1e-14
        assert a.stats["cg_visc"] == b.stats["cg_visc"]
    return out


def selftest(cuda=False):
    r = dict(rigid=test_rigid(), momentum=test_momentum(), spd=test_spd_nullspace(), energy=test_energy(),
             decay=test_decay(), branches=test_componentwise_unchanged(), offcup=test_offcup())
    if cuda:
        r["cuda"] = test_cuda()
    return r


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    selftest(cuda=len(sys.argv) > 1 and sys.argv[1] == "cuda")
    print("ALL TESTS PASSED")
