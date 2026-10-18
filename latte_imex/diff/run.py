"""Checkpointed forward / backward of one pour with the Warp solver, and the differentiable loss.

Loss (on the solver grid, final frame): soft IoU of the white pattern against the target,
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
    def __init__(self, p, N=128, device=None, T_max=None, w_gap=2.0, tau=0.05, log=print):
        self.p, self.N, self.log = p, N, log
        wp.init()
        script, pour = SC.pour_script(p, N)
        self.script, self.pour = script, pour
        self.S = SC.build(script, N, T_max=T_max)
        self.geo = W.Geo(N, device=device)
        self.sol = W.WSolver(self.geo, v05=self.S["return_law"] == "v05")
        dev = self.geo.dev
        self.Sw = {k: wp.array(self.S[k], dtype=F, device=dev, requires_grad=True) for k in W.SCHED}
        tw, valid, gaps, self.T = target_on_grid(p, self.geo)
        self.tw = wp.array(tw, dtype=F, device=dev)
        self.ww = wp.array(valid * (1.0 + w_gap * gaps), dtype=F, device=dev)
        self.tau = tau
        self.s1 = {k: float(script.params[k]) for k in W.PARAMS}
        self.nsub = None
        log(f"{p}: {self.S['n']} frames, T {self.S['T']:.2f} s, N {N}, device {dev}")

    # ------------------------------------------------------------------ pieces
    def _P(self, P, rec):
        Pnp = np.array([float(P[k]) for k in W.PARAMS], float)
        return Pnp, wp.array(Pnp, dtype=F, device=self.geo.dev, requires_grad=rec)

    def forward(self, P, keep=True, nsub=None):
        """Final state (numpy) + per-frame checkpoints (numpy) + sub-step counts."""
        sol, S = self.sol, self.S
        sol.rec, sol.tape = False, None
        Pnp, Pw = self._P(P, False)
        st = sol.initial_state()
        ck, ns = [], []
        for k in range(S["n"]):
            if keep:
                ck.append(sol.to_numpy(st))
            st, n_ = sol.frame(st, Pw, Pnp, self.Sw, k, float(S["fdt"][k]), int(S["q"][k]), bool(S["active"][k]),
                               nsub=None if nsub is None else nsub[k])
            ns.append(n_)
        return sol.to_numpy(st), ck, ns

    def loss_state(self, st, rec=False):
        """Loss of a final state (numpy dict); with rec also dL/dstate."""
        g = self.geo
        S = self.sol.state_from_numpy(**st, requires_grad=rec)
        sums = wp.zeros(2, dtype=F, device=g.dev, requires_grad=rec)
        out = wp.zeros(1, dtype=F, device=g.dev, requires_grad=rec)
        tape = wp.Tape() if rec else None
        if rec:
            with tape:
                wp.launch(k_softiou, dim=(g.N, g.N), inputs=[S["l"], S["m"], self.tw, self.ww, g.mask, F(self.tau), sums],
                          device=g.dev)
                wp.launch(k_loss, dim=1, inputs=[sums, out], device=g.dev)
            tape.backward(loss=out)
            return float(out.numpy()[0]), {k: S[k].grad.numpy().copy() for k in ("l", "qx", "qy", "m")}
        wp.launch(k_softiou, dim=(g.N, g.N), inputs=[S["l"], S["m"], self.tw, self.ww, g.mask, F(self.tau), sums],
                  device=g.dev)
        wp.launch(k_loss, dim=1, inputs=[sums, out], device=g.dev)
        return float(out.numpy()[0]), None

    def loss(self, P, nsub=None):
        st, _, ns = self.forward(P, keep=False, nsub=nsub)
        return self.loss_state(st)[0], st, ns

    # ------------------------------------------------------------------ gradient
    def loss_grad(self, P, nsub=None):
        """Loss and dL/dP (dict over W.PARAMS); dL/d(schedule) in self.sched_grad (dict of numpy, per frame)."""
        t0 = time.time()
        final, ck, ns = self.forward(P, keep=True, nsub=nsub)
        self.nsub = ns
        t1 = time.time()
        L, gs = self.loss_state(final, rec=True)
        sol, S = self.sol, self.S
        Pnp, Pw = self._P(P, True)
        for a in self.Sw.values():
            a.grad.zero_()
        sol.rec = True
        for k in range(S["n"] - 1, -1, -1):
            st = sol.state_from_numpy(**ck[k], requires_grad=True)
            tape = wp.Tape()
            sol.tape = tape
            with tape:
                out, _ = sol.frame(st, Pw, Pnp, self.Sw, k, float(S["fdt"][k]), int(S["q"][k]), bool(S["active"][k]),
                                   nsub=ns[k])
            tape.backward(grads={out[c]: wp.array(gs[c], dtype=F, device=self.geo.dev) for c in ("l", "qx", "qy", "m")})
            gs = {c: st[c].grad.numpy().copy() for c in ("l", "qx", "qy", "m")}
            del tape                               # (Tape.reset() would also zero the accumulated dL/dP)
        sol.rec, sol.tape = False, None
        gP = Pw.grad.numpy()
        self.sched_grad = {k: v.grad.numpy().copy() for k, v in self.Sw.items()}
        self.timing = dict(forward_s=t1 - t0, backward_s=time.time() - t1)
        return L, {k: float(gP[i]) for i, k in enumerate(W.PARAMS)}, final
