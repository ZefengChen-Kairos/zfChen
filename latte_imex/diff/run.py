"""Checkpointed forward / backward of one pour with the Warp solver, and the differentiable loss.

Loss 'video' (default; whole-video supervision): every rectified top-view frame of the real pour (video_obs.py:
soft whiteness w and the observation mask v, fixed by the observation) is compared with the simulation at the frame
end nearest to its time, on the visible cup surface only; per frame the squared whiteness residual is normalised by
the number of visible pixels, frames are averaged with their share of the physical time,
    L = sum_j  dt_j / sum dt  *  sum_x v_j (p(c) - w_j)^2 / sum_x v_j,      p = sigmoid((c - 0.5) / tau),
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
def k_obs_loss(sums: A1, wgt: F, out: A1):
    out[0] = wgt * sums[0] / (sums[1] + F(1e-30))


def video_on_grid(p, geo, t_end):
    """Whole-video observations (raw_data/rig/video_obs/<p>.npz) on the solver cells: list of (frame index, time
    weight, w, v) for the frames inside the simulated time."""
    import os
    from ..rig.real_program import CUP_R, REPO
    Z = np.load(os.path.join(REPO, "raw_data", "rig", "video_obs", f"{p}.npz"))
    t, Wo, Vo, ex = Z["t"], Z["w"].astype(float) / 255.0, Z["valid"], float(Z["extent"])
    n = Wo.shape[1]
    U, V = geo.x_np / CUP_R, -geo.y_np / CUP_R
    col = np.clip(np.round((U + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
    row = np.clip(np.round((V + ex) / (2 * ex) * (n - 1)).astype(int), 0, n - 1)
    inside = (U ** 2 + V ** 2 <= 1.0) & geo.mask_np
    keep = [j for j in range(len(t)) if t[j] <= t_end[-1] + 0.02 and (Vo[j][row, col] & inside).sum() > 50]
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
                 loss="video", w_final=0.0, log=print):
        self.p, self.N, self.log = p, N, log
        wp.init()
        script, pour = SC.pour_script(p, N)
        self.script, self.pour = script, pour
        self.S = SC.build(script, N, T_max=T_max)
        self.geo = W.Geo(N, device=device)
        if check_every is None:                     # PCG convergence read back every 4 iterations on the GPU
            check_every = 4 if self.geo.dev.is_cuda else 1
        self.sol = W.WSolver(self.geo, v05=self.S["return_law"] == "v05", check_every=check_every, graphs=graphs)
        dev = self.geo.dev
        self.Sw = {k: wp.array(self.S[k], dtype=F, device=dev, requires_grad=True) for k in W.SCHED}
        tw, valid, gaps, self.T = target_on_grid(p, self.geo)
        self.tw = wp.array(tw, dtype=F, device=dev)
        self.ww = wp.array(valid * (1.0 + w_gap * gaps), dtype=F, device=dev)
        self.tau = tau
        self.s1 = {k: float(script.params[k]) for k in W.PARAMS}
        self.nsub = None
        self.mode, self.w_final = loss, (1.0 if loss == "final" else w_final)
        self.obs = []
        if loss == "video":
            t_end = np.cumsum(self.S["fdt"])
            for k, wt, wo, vo, tj in video_on_grid(p, self.geo, t_end):
                self.obs.append(dict(k=k, wt=wt, t=tj, wo=wp.array(wo, dtype=F, device=dev),
                                     vo=wp.array(vo, dtype=F, device=dev), wo_np=wo, vo_np=vo > 0))
        log(f"{p}: {self.S['n']} frames, T {self.S['T']:.2f} s, N {N}, device {dev}, loss {loss}"
            + (f" ({len(self.obs)} observed frames)" if self.obs else ""))

    # ------------------------------------------------------------------ pieces
    def _P(self, P, rec):
        Pnp = np.array([float(P[k]) for k in W.PARAMS], float)
        return Pnp, wp.array(Pnp, dtype=F, device=self.geo.dev, requires_grad=rec)

    def forward(self, P, keep=True, nsub=None):
        """Final state + per-frame checkpoints (device arrays, fast path with CUDA graphs) + sub-step counts."""
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
            else:
                wp.launch(k_obs, dim=(g.N, g.N), inputs=[S["l"], S["m"], ob["wo"], ob["vo"], g.mask, F(self.tau), sums],
                          device=g.dev)
                wp.launch(k_obs_loss, dim=1, inputs=[sums, F(ob["wt"]), out], device=g.dev)
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
        zero = lambda: {c: wp.zeros((self.N, self.N), dtype=F, device=self.geo.dev) for c in ("l", "qx", "qy", "m")}
        gs = gfin if gfin is not None else zero()
        Pnp, Pw = self._P(P, True)
        for a in self.Sw.values():
            a.grad.zero_()
        sol.rec = True
        for k in range(n - 1, -1, -1):
            if k in gobs:
                gs = self._sum(gs, gobs.pop(k))
            st = sol.clone_state(ck[k], requires_grad=True)
            tape = wp.Tape()
            sol.tape = tape
            with tape:
                out, _ = sol.frame(st, Pw, Pnp, self.Sw, k, float(S["fdt"][k]), int(S["q"][k]), bool(S["active"][k]),
                                   nsub=ns[k])
            tape.backward(grads={out[c]: gs[c] for c in ("l", "qx", "qy", "m")})
            gs = {c: st[c].grad for c in ("l", "qx", "qy", "m")}
            if k + 1 < n:
                ck[k + 1] = None                   # free the checkpoint (no longer needed as an observed state)
            del tape                               # (Tape.reset() would also zero the accumulated dL/dP)
        sol.rec, sol.tape = False, None
        gP = Pw.grad.numpy()
        self.sched_grad = {k: v.grad.numpy().copy() for k, v in self.Sw.items()}
        self.timing = dict(forward_s=t1 - t0, backward_s=time.time() - t1)
        return L, {k: float(gP[i]) for i, k in enumerate(W.PARAMS)}, self.sol.to_numpy(final)
