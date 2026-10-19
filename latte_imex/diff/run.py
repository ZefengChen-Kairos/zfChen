"""Checkpointed forward / backward of one pour with the Warp solver, and the differentiable loss.

Loss 'video_iou': per observed frame 1 - soft IoU (sum v p w / sum v (p + w - p w)) of the visible area, frames
averaged with the same time weights -- the smooth counterpart of the whole-video IoU used to evaluate fits.
Loss 'video' (default; whole-video supervision): every rectified top-view frame of the real pour (video_obs.py:
soft whiteness w and the observation mask v, fixed by the observation) is compared with the simulation at the frame
end nearest to its time, on the visible cup surface only; per frame the squared whiteness residual is normalised by
the number of visible pixels, frames are averaged with their share of the physical time,
    L = sum_j  dt_j / sum dt  *  sum_x v_j (p(c) - w_j)^2 / sum_x v_j,      p = sigmoid((c - 0.5) / tau),
or, with scales s_1..s_n (cup radii; 0 = the term above), the mean over the scales of the blurred residual
    sum_x (G_s*(v p) - G_s*(v w))^2 / G_s*v  /  sum_x G_s*v       (G_s: Gaussian of width s R)
i.e. the local mean whiteness compared over neighbourhoods of size s R, weighted by local visibility: white placed a
little off its observed position still lowers the coarse terms, so the gradient moves white towards the observation
instead of only removing it (the pixel term alone is lowest for a pattern that is absent rather than misplaced),
and the gradient of every frame term is injected into the backward pass at its frame (no state is ever reset to
the observation).  Optional extra term (w_final > 0, or loss 'final' alone): soft IoU of the final field against the
finished-pattern target, with the target gaps weighted,
    p = sigmoid((c - 0.5) / tau),  c = m / l,  t = target white (resampled onto the solver cells),
    L = 1 - sum w p t / sum w (p + t - p t),   w = valid * (1 + w_gap * target gaps)
where the target gaps are tulip_feas.gaps (brown inside the hull of the white pattern), so the brown wedges between
the petals weigh more than the rest of the cup.

    D = DiffPour("tulip", N=128)                       # schedule + target on the solver grid
    L, g = D.loss_grad(P)                              # P: dict of closure parameters; g: dL/dP (dict), + D.sched_grad
"""
from __future__ import annotations

import math
import time

import numpy as np
import warp as wp

from . import schedule as SC
from . import wsolver as W

F, A2, A1, I2 = W.F, W.A2, W.A1, W.I2


@wp.kernel
def k_softiou(l: A2, m: A2, tw: A2, ww: A2, mask: I2, tau: F, sums: A1):
    j, i = wp.tid()
    if mask[j, i] == 0:
        return
    w = ww[j, i]
    if w == F(0.0):
        return
    c = m[j, i] / l[j, i]
    p = F(1.0) / (F(1.0) + wp.exp(-(c - F(0.5)) / tau))
    t = tw[j, i]
    wp.atomic_add(sums, 0, w * p * t)
    wp.atomic_add(sums, 1, w * (p + t - p * t))


@wp.kernel
def k_loss(sums: A1, out: A1):
    out[0] = F(1.0) - sums[0] / (sums[1] + F(1e-30))


@wp.kernel
def k_scale(a: A2, s: F, out: A2):
    j, i = wp.tid()
    out[j, i] = s * a[j, i]


@wp.kernel
def k_obs(l: A2, m: A2, wo: A2, vo: A2, mask: I2, tau: F, sums: A1):
    j, i = wp.tid()
    if mask[j, i] == 0:
        return
    v = vo[j, i]
    if v == F(0.0):
        return
    c = m[j, i] / l[j, i]
    p = F(1.0) / (F(1.0) + wp.exp(-(c - F(0.5)) / tau))
    d = p - wo[j, i]
    wp.atomic_add(sums, 0, v * d * d)
    wp.atomic_add(sums, 1, v)


@wp.kernel
def k_iou_loss(sums: A1, wgt: F, out: A1):
    out[0] = wgt * (F(1.0) - sums[0] / (sums[1] + F(1e-30)))


@wp.kernel
def k_obs_loss(sums: A1, wgt: F, out: A1):
    out[0] = wgt * sums[0] / (sums[1] + F(1e-30))


@wp.kernel
def k_vp(l: A2, m: A2, vo: A2, mask: I2, tau: F, out: A2):
    """v * p(c) (visible soft whiteness of the simulation)."""
    j, i = wp.tid()
    out[j, i] = F(0.0)
    if mask[j, i] != 0:
        v = vo[j, i]
        if v != F(0.0):
            c = m[j, i] / l[j, i]
            out[j, i] = v / (F(1.0) + wp.exp(-(c - F(0.5)) / tau))


@wp.kernel
def k_blur(a: A2, wts: A1, r: int, N: int, axis: int, out: A2):
    """One pass of a separable truncated Gaussian, zero outside the grid (blur_np is the same on the host)."""
    j, i = wp.tid()
    acc = F(0.0)
    for d in range(2 * r + 1):
        if axis == 0:
            ii = i + d - r
            if ii >= 0 and ii < N:
                acc += wts[d] * a[j, ii]
        else:
            jj = j + d - r
            if jj >= 0 and jj < N:
                acc += wts[d] * a[jj, i]
    out[j, i] = acc


@wp.kernel
def k_ms(Bp: A2, Bw: A2, iBv: A2, coef: F, sums: A1):
    """sum (Bp - Bw)^2 / Bv  (= sum Bv (Bp/Bv - Bw/Bv)^2: local mean whiteness, weighted by local visibility)."""
    j, i = wp.tid()
    q = iBv[j, i]
    if q != F(0.0):
        d = Bp[j, i] - Bw[j, i]
        wp.atomic_add(sums, 0, coef * d * d * q)


@wp.kernel
def k_ms0(sum0: A1, frac: F, sums: A1):
    sums[0] = sums[0] + frac * sum0[0] / (sum0[1] + F(1e-30))


@wp.kernel
def k_ou(l: A2, qx: A2, qy: A2, o: A2, mask: I2, ox: A2, oy: A2):
    """o * u (film velocity on the trusted optical-flow pixels)."""
    j, i = wp.tid()
    ox[j, i] = F(0.0)
    oy[j, i] = F(0.0)
    if mask[j, i] != 0:
        w = o[j, i]
        if w != F(0.0):
            ox[j, i] = w * qx[j, i] / l[j, i]
            oy[j, i] = w * qy[j, i] / l[j, i]


@wp.kernel
def k_wsum(sums: A1, wgt: F, out: A1):
    out[0] = wgt * sums[0]


def gauss_w(sig):
    r = max(1, int(math.ceil(3 * sig)))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    return w / w.sum(), r


def blur_np(a, sig):
    """Host twin of k_blur (x pass, then y pass)."""
    w, r = gauss_w(sig)
    N = a.shape[0]
    for ax in (1, 0):
        pad = np.zeros_like(a)
        for d in range(2 * r + 1):
            sh = d - r
            src = np.roll(a, -sh, axis=ax)
            idx = np.arange(N) + sh
            ok = (idx >= 0) & (idx < N)
            pad += w[d] * (src * (ok[None, :] if ax == 1 else ok[:, None]))
        a = pad
    return a


FLOW_EXCL = 0.12        # [solver units] no velocity supervision this close to the landing point


def kernel_inside(S, geo, x, y):
    """Fraction of the footprint kernel inside the cup per frame (k_kernel_raw's sums[0] / sums[1], q x q quadrature)."""
    h, m = geo.h, geo.mask_np
    Z = np.ones(S["n"])
    for k in range(S["n"]):
        if not S["active"][k]:
            continue
        q = int(S["q"][k])
        r1, r2 = max(float(S["r1"][k]), 1.5 * h), max(float(S["r2"][k]), 1.5 * h)
        c, s = np.cos(float(S["phi"][k])), np.sin(float(S["phi"][k]))
        acc = np.zeros_like(geo.x_np)
        for ox in range(q):
            for oy in range(q):
                dx = geo.x_np + ((ox + 0.5) / q - 0.5) * h - x[k]
                dy = geo.y_np + ((oy + 0.5) / q - 0.5) * h - y[k]
                a, b = (c * dx + s * dy) / r1, (-s * dx + c * dy) / r2
                acc += np.maximum(0.0, 1.0 - a * a - b * b) ** 2
        tot = acc.sum()
        Z[k] = acc[m].sum() / tot if tot > 0 else 0.0
    return Z


def circ_fields(S, geo, x=None, y=None, dtype=np.float32):
    """Unit surface-current fields of every frame of the arrival record S (circulation.SurfaceCurrent: U, sigma+,
    sigma-, W_n), stored as dtype; driven by the received flow Q Z (Z: kernel_inside) at the landing points x, y (default S's)."""
    from ..circulation import SurfaceCurrent
    x = S["x"] if x is None else x
    y = S["y"] if y is None else y
    Z = kernel_inside(S, geo, x, y)
    C = SurfaceCurrent(geo.x_np, geo.y_np, geo.mask_np, geo.h)
    n, N = S["n"], geo.N
    U = np.zeros((n, 2, N, N), dtype)
    up, dn, Wn = (np.zeros((n, N, N), dtype) for _ in range(3))
    for k in range(n):
        U[k], up[k], dn[k], Wn[k] = C.step(float(S["fdt"][k]), bool(S["active"][k]), float(S["Q"][k] * Z[k]),
                                           (x[k], y[k]), (S["ux"][k], S["uy"][k]))
    return U, up, dn, Wn


EXPOSURE_TOL = 0.15     # frames whose median lightness departs from its +-1 s running median by more than this are not
                        # observed: the camera exposure drops when the pitcher is lifted and the whiteness classifier
                        # inflates the white fraction by 10-60 % for 0.3-0.4 s (tulip 5 %, push_heart 6 %, swan 2 % of
                        # the frames; none in heart, leaf).  None: keep every frame


def exposure_transients(t, L, valid, tol=EXPOSURE_TOL):
    """Boolean mask of the video frames with an exposure transient (see EXPOSURE_TOL)."""
    med = np.array([np.median(L[j][valid[j]]) if valid[j].any() else np.nan for j in range(len(t))])
    run = np.array([np.nanmedian(med[np.abs(t - tt) <= 1.0]) for tt in t])
    return np.abs(med - run) > tol * run


def video_on_grid(p, geo, t_end):
    """Whole-video observations (raw_data/rig/video_obs/<p>.npz) on the solver cells: list of (frame index, time
    weight, w, v) for the frames inside the simulated time (exposure transients dropped, EXPOSURE_TOL)."""
    import os
    from ..rig.real_program import CUP_R, REPO
    Z = np.load(os.path.join(REPO, "raw_data", "rig", "video_obs", f"{p}.npz"))
    t, Wo, Vo, ex = Z["t"], Z["w"].astype(float) / 255.0, Z["valid"], float(Z["extent"])
    bad = exposure_transients(t, Z["L"].astype(float), Vo) if EXPOSURE_TOL else np.zeros(len(t), bool)
    n = Wo.shape[1]
    U, V = geo.x_np / CUP_R, -geo.y_np / CUP_R
    col = np.clip(np.round((U + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
    row = np.clip(np.round((V + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
    inside = (U ** 2 + V ** 2 <= 1.0) & geo.mask_np
    keep = [j for j in range(len(t)) if t[j] <= t_end[-1] + 0.02 and (Vo[j][row, col] & inside).sum() > 50 and not bad[j]]
    tk = t[keep]
    mid = 0.5 * (tk[1:] + tk[:-1])
    dt = np.diff(np.r_[tk[0], mid, tk[-1]]) if len(tk) > 1 else np.ones(1)
    dt = np.maximum(dt, 1e-3)
    dt /= dt.sum()
    out = []
    for jj, j in enumerate(keep):
        k = int(np.argmin(np.abs(t_end - t[j])))
        out.append((k, float(dt[jj]), Wo[j][row, col], (Vo[j][row, col] & inside).astype(float), float(t[j])))
    return out


def target_on_grid(p, geo):
    """Target white / valid / gaps (tulip_feas.Target, disc coordinates) on the solver cells (nearest pixel)."""
    from ..rig.tulip_feas import Target
    from ..rig.real_program import CUP_R
    T = Target(p)
    U, V = geo.x_np / CUP_R, -geo.y_np / CUP_R
    n, ex = T.n, T.ex
    col = np.clip(np.round((U + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
    row = np.clip(np.round((V + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
    inside = (U ** 2 + V ** 2 <= 1.0) & geo.mask_np
    valid = T.valid[row, col] & inside
    return T.white[row, col].astype(float), valid, T.G[row, col] & valid, T


class DiffPour:
    def __init__(self, p, N=128, device=None, T_max=None, w_gap=2.0, tau=0.05, check_every=None, graphs=True,
                 loss="video", w_final=0.0, scales=(0.0,), e0=False, upwell=False, circ=False, w_flow=0.0,
                 flow_scale=0.15, flow_u0=0.05, exit_cal=False, log=print):
        self.p, self.N, self.log = p, N, log
        wp.init()
        script, pour = SC.pour_script(p, N, e0=e0, exit_cal=exit_cal)
        self.exit_cal = exit_cal
        self.script, self.pour, self.e0 = script, pour, e0
        self.S = SC.build(script, N, T_max=T_max)
        self.geo = W.Geo(N, device=device)
        if check_every is None:                     # PCG convergence read back every 4 iterations on the GPU
            check_every = 4 if self.geo.dev.is_cuda else 1
        self.sol = W.WSolver(self.geo, v05=self.S["return_law"] == "v05", check_every=check_every, graphs=graphs, up=upwell)
        dev = self.geo.dev
        self.Sw = {k: wp.array(self.S[k], dtype=F, device=dev, requires_grad=True) for k in W.SCHED}
        self.circ = circ
        if circ:                                         # momentum-driven surface current (circulation.py)
            self.sol.set_circ(*circ_fields(self.S, self.geo))
        self.foot_dir = SC.foot_dirs(self.S["ux"], self.S["uy"])
        self._foot = (0.0, 0.0)
        tw, valid, gaps, self.T = target_on_grid(p, self.geo)
        self.tw = wp.array(tw, dtype=F, device=dev)
        self.ww = wp.array(valid * (1.0 + w_gap * gaps), dtype=F, device=dev)
        self.tau = tau
        self.s1 = {k: float(script.params.get(k, 0.0)) for k in W.PARAMS}
        self.nsub = None
        self.mode, self.w_final = loss, (1.0 if loss == "final" else w_final)
        # backward replay: persistent buffers (fixed addresses for the per-frame graphs)
        z = lambda: wp.zeros((N, N), dtype=F, device=dev)
        self._Pg = wp.zeros(len(W.PARAMS), dtype=F, device=dev, requires_grad=True)
        self._ck_in = {c: z() for c in ("l", "qx", "qy", "m")}    # checkpoint of the frame being replayed
        self._seed = {c: z() for c in ("l", "qx", "qy", "m")}     # dL/d(state): after the frame in, before it out
        self._seen, self._nblk, self._bwd_graphs = set(), {}, graphs
        self.obs = []
        self.scales = tuple(float(x) for x in scales)
        from ..rig.real_program import CUP_R
        if loss in ("video", "video_iou"):
            t_end = np.cumsum(self.S["fdt"])
            for k, wt, wo, vo, tj in video_on_grid(p, self.geo, t_end):
                ob = dict(k=k, wt=wt, t=tj, wo=wp.array(wo, dtype=F, device=dev),
                          vo=wp.array(vo, dtype=F, device=dev), wo_np=wo, vo_np=vo > 0, ms=[])
                for sc in self.scales:                   # host side of the multi-scale terms (scales > 0)
                    if sc <= 0:
                        continue
                    sig = sc * CUP_R * N
                    Bv, Bw = blur_np(vo, sig), blur_np(vo * wo, sig)
                    keep = Bv > 0.05
                    iBv = np.where(keep, 1.0 / np.maximum(Bv, 1e-30), 0.0)
                    ob["ms"].append(dict(w=wp.array(gauss_w(sig)[0], dtype=F, device=dev), r=gauss_w(sig)[1],
                                         Bw=wp.array(Bw, dtype=F, device=dev), iBv=wp.array(iBv, dtype=F, device=dev),
                                         coef=1.0 / (len(self.scales) * max(Bv[keep].sum(), 1e-30))))
                self.obs.append(ob)
            if w_flow > 0:
                self._flow_terms(p, w_flow, flow_scale, flow_u0)
            # reference: the loss of a simulation with no white at all (p = 0) -- a fit must get well below it
            self.L_brown = 0.0
            for ob in self.obs:
                v, w = ob["vo_np"].astype(float), ob["wo_np"]
                t = (v * w * w).sum() / max(v.sum(), 1e-30) / len(self.scales) if 0.0 in self.scales else 0.0
                for m in ob["ms"]:
                    t += m["coef"] * float((m["Bw"].numpy() ** 2 * m["iBv"].numpy()).sum())
                self.L_brown += ob["wt"] * t
        log(f"{p}: {self.S['n']} frames, T {self.S['T']:.2f} s, N {N}, device {dev}, loss {loss}"
            + (", landing point: E0 measured stream foot" if e0 else "")
            + (f" ({len(self.obs)} observed frames, scales {self.scales} R; no-white reference loss "
               f"{self.L_brown:.4f})" if self.obs else ""))

    def _flow_terms(self, p, w_flow, scale, u0):
        """Surface-velocity supervision (rig/video_flow.py): per observed frame the film velocity u on the trusted
        optical-flow pixels o, compared after a Gaussian blur of width scale R,
            w_flow / u0^2 * sum (G*(o u) - G*(o u_obs))^2 / G*o / sum G*o   (both components),
        added to the frame's term (same time weight).  Caution: the frame-to-frame flow of the rectified view is noise
        dominated (trusted-pixel speeds 0.2-0.3 /s per frame against ~0.04 /s time-mean: registration jitter, exposure
        flicker, foam glints), so per-frame supervision is off by default (w_flow = 0)."""
        import os
        from ..rig.real_program import CUP_R, REPO
        Z = np.load(os.path.join(REPO, "raw_data", "rig", "video_obs", f"{p}_flow.npz"))
        tf, Uo, Vo, OK, ex = Z["t"], Z["u"].astype(float), Z["v"].astype(float), Z["ok"], float(Z["extent"])
        g, dev, N = self.geo, self.geo.dev, self.N
        n = Uo.shape[1]
        U, V = g.x_np / CUP_R, -g.y_np / CUP_R
        col = np.clip(np.round((U + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
        row = np.clip(np.round((V + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
        inside = (U ** 2 + V ** 2 <= 1.0) & g.mask_np
        sig = scale * CUP_R * N
        wts, r = gauss_w(sig)
        wd = wp.array(wts, dtype=F, device=dev)
        for ob in self.obs:
            j = int(np.argmin(np.abs(tf - ob["t"])))
            if abs(tf[j] - ob["t"]) > 0.06:
                continue
            o = OK[j][row, col] & inside
            k = ob["k"]
            if self.S["active"][k]:                      # the stream's impact zone: the flow there is not trusted
                o &= np.hypot(g.x_np - self.S["x"][k], g.y_np - self.S["y"][k]) > FLOW_EXCL
            o = o.astype(float)
            if o.sum() < 20:
                continue
            Bo = blur_np(o, sig)
            keep = Bo > 0.05
            ob["fl"] = dict(o=wp.array(o, dtype=F, device=dev), w=wd, r=r,
                            Bx=wp.array(blur_np(o * Uo[j][row, col], sig), dtype=F, device=dev),
                            By=wp.array(blur_np(o * Vo[j][row, col], sig), dtype=F, device=dev),
                            iBo=wp.array(np.where(keep, 1.0 / np.maximum(Bo, 1e-30), 0.0), dtype=F, device=dev),
                            coef=w_flow / (u0 * u0 * max(Bo[keep].sum(), 1e-30)))

    # ------------------------------------------------------------------ pieces
    def _apply_foot(self, P):
        """Landing-point correction (schedule.foot_shift) from P['foot_a'], P['foot_b'] (default 0).  With the surface
        current on, its host fields are recomputed for the shifted landing points; the adjoint does not see that part
        of the dependence (foot_a, foot_b gradients are partial; the foot fit is a diagnostic only, fit.py --foot)."""
        ab = (float(P.get("foot_a", 0.0)), float(P.get("foot_b", 0.0)))
        if ab != self._foot:
            x, y = SC.foot_shift(self.S["x"], self.S["y"], self.S["ux"], self.S["uy"], *ab)
            self.Sw["x"].assign(x)
            self.Sw["y"].assign(y)
            if self.circ:                                # the current follows the landing point
                self.sol.set_circ(*circ_fields(self.S, self.geo, x, y))
            self._foot = ab

    def _P(self, P, rec):
        Pnp = np.array([float(P.get(k, 0.0)) for k in W.PARAMS], float)
        return Pnp, wp.array(Pnp, dtype=F, device=self.geo.dev, requires_grad=rec)

    def forward(self, P, keep=True, nsub=None):
        """Final state + per-frame checkpoints (device arrays, fast path with CUDA graphs) + sub-step counts."""
        self._apply_foot(P)
        sol = self.sol
        sol.rec, sol.tape = False, None
        Pnp, Pw = self._P(P, False)
        return sol.fast_forward(Pw, Pnp, self.Sw, self.S, keep=keep, nsub=nsub)

    def _term(self, st, kind, ob=None, rec=False):
        """One loss term on a state (device arrays): 'final' soft IoU or one observed video frame.  Returns the value
        and, with rec, dL/dstate (device arrays)."""
        g = self.geo
        S = self.sol.clone_state(st, requires_grad=rec)
        sums = wp.zeros(2, dtype=F, device=g.dev, requires_grad=rec)
        out = wp.zeros(1, dtype=F, device=g.dev, requires_grad=rec)

        def launch():
            if kind == "final":
                wp.launch(k_softiou, dim=(g.N, g.N), inputs=[S["l"], S["m"], self.tw, self.ww, g.mask, F(self.tau), sums],
                          device=g.dev)
                wp.launch(k_loss, dim=1, inputs=[sums, out], device=g.dev)
            elif self.mode == "video_iou":             # 1 - soft IoU of the frame on its visible area
                wp.launch(k_softiou, dim=(g.N, g.N), inputs=[S["l"], S["m"], ob["wo"], ob["vo"], g.mask, F(self.tau), sums],
                          device=g.dev)
                wp.launch(k_iou_loss, dim=1, inputs=[sums, F(ob["wt"]), out], device=g.dev)
            elif self.scales == (0.0,):
                wp.launch(k_obs, dim=(g.N, g.N), inputs=[S["l"], S["m"], ob["wo"], ob["vo"], g.mask, F(self.tau), sums],
                          device=g.dev)
                wp.launch(k_obs_loss, dim=1, inputs=[sums, F(ob["wt"]), out], device=g.dev)
            else:                                        # multi-scale: mean over the scales of the blurred residual
                Lt = lambda kern, ins, d=(g.N, g.N): wp.launch(kern, dim=d, inputs=ins, device=g.dev)
                vp = wp.zeros((g.N, g.N), dtype=F, device=g.dev, requires_grad=rec)
                Lt(k_vp, [S["l"], S["m"], ob["vo"], g.mask, F(self.tau), vp])
                if 0.0 in self.scales:
                    sum0 = wp.zeros(2, dtype=F, device=g.dev, requires_grad=rec)
                    wp.launch(k_obs, dim=(g.N, g.N), inputs=[S["l"], S["m"], ob["wo"], ob["vo"], g.mask, F(self.tau),
                                                             sum0], device=g.dev)
                    Lt(k_ms0, [sum0, F(1.0 / len(self.scales)), sums], 1)
                for t in ob["ms"]:
                    bx = wp.zeros((g.N, g.N), dtype=F, device=g.dev, requires_grad=rec)
                    by = wp.zeros((g.N, g.N), dtype=F, device=g.dev, requires_grad=rec)
                    Lt(k_blur, [vp, t["w"], t["r"], g.N, 0, bx])
                    Lt(k_blur, [bx, t["w"], t["r"], g.N, 1, by])
                    Lt(k_ms, [by, t["Bw"], t["iBv"], F(t["coef"]), sums])
                if "fl" in ob:                           # surface-velocity supervision (_flow_terms)
                    fl = ob["fl"]
                    z = lambda: wp.zeros((g.N, g.N), dtype=F, device=g.dev, requires_grad=rec)
                    ox, oy = z(), z()
                    Lt(k_ou, [S["l"], S["qx"], S["qy"], fl["o"], g.mask, ox, oy])
                    for comp, Bo_ in ((ox, fl["Bx"]), (oy, fl["By"])):
                        bx, by = z(), z()
                        Lt(k_blur, [comp, fl["w"], fl["r"], g.N, 0, bx])
                        Lt(k_blur, [bx, fl["w"], fl["r"], g.N, 1, by])
                        Lt(k_ms, [by, Bo_, fl["iBo"], F(fl["coef"]), sums])
                Lt(k_wsum, [sums, F(ob["wt"]), out], 1)
        if rec:
            tape = wp.Tape()
            with tape:
                launch()
            tape.backward(loss=out)
            return float(out.numpy()[0]), {k: S[k].grad for k in ("l", "qx", "qy", "m")}
        launch()
        return float(out.numpy()[0]), None

    def loss_terms(self, final, after, rec=False):
        """Total loss from the final state and the states after the observed frames (after: k -> state).  With rec
        also the gradients: (dL/dfinal or None, {k: dL/dstate after frame k})."""
        L, gfin, gobs = 0.0, None, {}
        if self.w_final > 0:
            v, gfin = self._term(final, "final", rec=rec)
            L += self.w_final * v
            if rec:
                gfin = {c: self._scaled(a, self.w_final) for c, a in gfin.items()}
        for ob in self.obs:
            v, gr = self._term(after(ob["k"]), "video", ob, rec=rec)
            L += v
            if rec:
                gobs[ob["k"]] = gr if ob["k"] not in gobs else self._sum(gobs[ob["k"]], gr)
        return L, gfin, gobs

    def _scaled(self, a, s):
        out = wp.empty_like(a)
        wp.launch(k_scale, dim=a.shape, inputs=[a, F(s), out], device=self.geo.dev)
        return out

    def _sum(self, a, b):
        out = {}
        for c in a:
            o = wp.empty_like(a[c])
            self.sol.cp(a[c], o)
            wp.launch(W.k_add, dim=o.shape, inputs=[b[c], o], device=self.geo.dev)
            out[c] = o
        return out

    def _replay(self, k, fdt, q, act, nsub, Pnp, key):
        """Backward of frame k: replay it from the checkpoint (self._ck_in) under a tape, seed its output adjoint with
        self._seed and leave dL/d(state at the frame start) in self._seed; dL/dP and dL/d(schedule) accumulate in
        self._Pg.grad and self.Sw[...].grad.  All arrays come from the solver's pool and dt / frame index are read from
        device arrays, so on CUDA the whole replay (forward launches, PCG solves with their adjoint solves, backward
        launches) is captured once per (sub-steps, active, viscosity, mixing) into one graph and replayed -- the first
        frame of each kind runs eagerly (sizes the pool), the second captures."""
        sol = self.sol
        cs = ("l", "qx", "qy", "m")

        def body():
            sol.pool_begin(self._nblk.get(key))
            st = {c: sol.new() for c in cs}
            for c in cs:
                sol.cp(self._ck_in[c], st[c])
            tape = wp.Tape()
            sol.tape = tape
            with tape:
                out, _ = sol.frame(st, self._Pg, Pnp, self.Sw, k, fdt, q, act, nsub=nsub)
            for c in cs:
                sol.cp(self._seed[c], out[c].grad)
            tape.backward()
            for c in cs:
                sol.cp(st[c].grad, self._seed[c])
            self._nblk[key] = sol.pool_end()
            sol.tape = None                        # (Tape.reset() would also zero the accumulated dL/dP)

        R = sol.runner
        if R.use and self._bwd_graphs and key in self._seen:
            try:
                R(("bwd",) + key, body)
                return
            except Exception as e:
                sol.pool_on = False
                R.graphs.pop(("bwd",) + key, None)
                self._bwd_graphs = False
                self.log(f"[run] graph capture of the backward replay failed ({type(e).__name__}: {e}); eager replay")
        body()
        self._seen.add(key)

    def video_metrics(self, after):
        """Hard IoU (c > 0.5 vs w > 0.5) on the visible pixels of every observed frame: (time, IoU) list."""
        res = []
        msk = self.geo.mask_np
        for ob in self.obs:
            st = after(ob["k"])
            c = st["m"].numpy() / np.where(msk, st["l"].numpy(), 1.0)
            v = ob["vo_np"]
            a, b = (c > 0.5) & v, (ob["wo_np"] > 0.5) & v
            res.append((ob["t"], float((a & b).sum() / max((a | b).sum(), 1))))
        return res

    def loss(self, P, nsub=None):
        self._apply_foot(P)
        sol = self.sol
        sol.rec, sol.tape = False, None
        Pnp, Pw = self._P(P, False)
        final, _, ns = sol.fast_forward(Pw, Pnp, self.Sw, self.S, keep=False, nsub=nsub, snap=[o["k"] for o in self.obs])
        snaps = sol.snaps
        L, _, _ = self.loss_terms(final, lambda k: snaps[k])
        return L, self.sol.to_numpy(final), ns

    # ------------------------------------------------------------------ gradient
    def loss_grad(self, P, nsub=None):
        """Loss and dL/dP (dict over W.PARAMS); dL/d(schedule) in self.sched_grad (dict of numpy, per frame);
        per-frame video IoU in self.vmetrics."""
        t0 = time.time()
        final, ck, ns = self.forward(P, keep=True, nsub=nsub)
        self.nsub = ns
        t1 = time.time()
        sol, S = self.sol, self.S
        n = S["n"]
        after = lambda k: ck[k + 1] if k + 1 < n else final
        L, gfin, gobs = self.loss_terms(final, after, rec=True)
        self.vmetrics = self.video_metrics(after) if self.obs else []
        self.strip = []                            # 8 observed frames: (t, sim c, observed w, visible) for figures
        msk = self.geo.mask_np
        for ob in [self.obs[i] for i in np.linspace(0, len(self.obs) - 1, min(8, len(self.obs))).astype(int)]:
            st = after(ob["k"])
            c = np.where(msk, st["m"].numpy() / np.where(msk, st["l"].numpy(), 1.0), np.nan)
            self.strip.append((ob["t"], c.astype(np.float32), ob["wo_np"].astype(np.float32), ob["vo_np"]))
        for c in ("l", "qx", "qy", "m"):
            if gfin is not None:
                sol.cp(gfin[c], self._seed[c])
            else:
                self._seed[c].zero_()
        Pnp = np.array([float(P.get(k, 0.0)) for k in W.PARAMS], float)
        self._Pg.assign(Pnp)
        self._Pg.grad.zero_()
        for a in self.Sw.values():
            a.grad.zero_()
        sol.rec = True
        nu_on = bool(Pnp[W.NU] > 0)
        for k in range(n - 1, -1, -1):
            if k in gobs:
                for c, a in gobs.pop(k).items():
                    wp.launch(W.k_add, dim=a.shape, inputs=[a, self._seed[c]], device=self.geo.dev)
            for c in ("l", "qx", "qy", "m"):
                sol.cp(ck[k][c], self._ck_in[c])
            fdt, q, act = float(S["fdt"][k]), int(S["q"][k]), bool(S["active"][k])
            sol.set_dt(fdt / ns[k])
            sol.set_kq(k, q)
            self._replay(k, fdt, q, act, ns[k], Pnp, (ns[k], act, nu_on, sol.mix_on(Pnp, fdt / ns[k])))
            if k + 1 < n:
                ck[k + 1] = None                   # free the checkpoint (no longer needed as an observed state)
        sol.rec, sol.tape = False, None
        gP = self._Pg.grad.numpy()
        self.sched_grad = {k: v.grad.numpy().copy() for k, v in self.Sw.items()}
        self.timing = dict(forward_s=t1 - t0, backward_s=time.time() - t1)
        g = {k: float(gP[i]) for i, k in enumerate(W.PARAMS)}
        ex, ey = self.foot_dir                      # chain rule of the landing-point correction
        gx, gy = self.sched_grad["x"], self.sched_grad["y"]
        g["foot_a"] = float(np.sum(gx * ex + gy * ey))
        g["foot_b"] = float(np.sum(-gx * ey + gy * ex))
        return L, g, self.sol.to_numpy(final)
