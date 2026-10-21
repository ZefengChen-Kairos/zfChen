"""Continuous-time pose fit of one or several pours (shared tag layout optional).

Objective (dimensionless):
  sum_tag-corner-coords  huber(r / sig_t, k_t)                                  cameras 0/1 colour, every detection
+ sum_depth-points        w_f * tukey(r, c_d) / sig_d^2                         cameras 0/1 depth (point-to-plane to CAD)
+ sum_prior-points        1/2 |alpha|^2 / sig_alpha^2 + 1/2 |a|^2 / sig_a^2     2 Gauss points per knot interval
Unknowns per pour: control rotations / origins (right perturbations), time offsets of the streams
  'c1' (camera-1 colour vs camera-0 colour), 'd0', 'd1' (depth of camera 0 / 1 vs camera-0 colour).
Shared (optional): layout parameters of the tags other than tag 4.
Gauss-Newton / Levenberg-Marquardt, dense per-pour normal equations, Schur complement on the shared layout.
"""
import numpy as np
import jax, jax.numpy as jnp
from scipy.spatial.transform import Rotation as Rot
from pv2lib import expm3, tag_corners, project, huber_w, huber_rho, tukey_w, tukey_rho, Layout, TAG
from spline import Traj

NUIS = ('c1', 'd0', 'd1')


def _tag_res(R, p, eps, tp, s, W, Kv, q, e, w1, cen, a, flat):
    """e: world-frame correction of camera 1's extrinsics (applied when w1 = 1): X -> exp(e_r)(X - cen) + cen + e_t."""
    R2 = R @ expm3(eps[:3]); p2 = p + eps[3:]
    X = tag_corners(tp, s, a, flat) @ R2.T + p2
    Xc = (X - cen) @ expm3(w1 * e[:3]).T + cen + w1 * e[3:]
    return (project(Xc, W, Kv) - q).ravel()


_TJ = {}


def tag_fun(a, flat):
    key = (round(float(a), 8), bool(flat))
    if key not in _TJ:
        f = lambda R, p, eps, tp, s, W, Kv, q, e, w1, cen: _tag_res(R, p, eps, tp, s, W, Kv, q, e, w1, cen, a, flat)
        fj = jax.jacfwd(f, argnums=(2, 3, 8))
        _TJ[key] = (jax.jit(jax.vmap(f, in_axes=(0, 0, None, 0, None, 0, 0, 0, None, 0, None))),
                    jax.jit(jax.vmap(lambda R, p, tp, s, W, Kv, q, e, w1, cen: fj(R, p, jnp.zeros(6), tp, s, W, Kv, q, e, w1, cen),
                                     in_axes=(0, 0, 0, None, 0, 0, 0, None, 0, None))))
    return _TJ[key]


class PourProblem:
    """Data of one pour.  tags: list of detection dicts (cam, step, tag, q); depth: dict with per-frame arrays."""

    def __init__(self, D, traj, tags, depth=None, nuis_free=('c1',), cad=None):
        self.D, self.traj = D, traj
        self.nuis = {k: 0.0 for k in NUIS}
        self.nuis_free = [k for k in NUIS if k in nuis_free]
        self.cad = cad
        self.tie_depth = True
        # tag observations
        self.tg = dict(cam=np.array([d['cam'] for d in tags]), step=np.array([d['step'] for d in tags]),
                       tag=np.array([d['tag'] for d in tags]), q=np.stack([d['q'] for d in tags]) if tags else np.zeros((0, 4, 2)))
        self.tg['t'] = np.array([D.ts[c][k] for c, k in zip(self.tg['cam'], self.tg['step'])])
        self.tg['W'] = np.stack([D.W2C[c] for c in self.tg['cam']]) if tags else np.zeros((0, 3, 4))
        self.tg['Kv'] = np.stack([D.Kv[c] for c in self.tg['cam']]) if tags else np.zeros((0, 4))
        self.tg['on'] = np.ones(len(tags), bool)
        # depth: frames f with t, cam; points X (world), frame index pf
        self.dp = depth
        if depth is not None:
            self.dp['on'] = np.ones(len(depth['X']), bool)

    # ------------------------------------------------------------------ helpers
    def tag_times(self):
        t = self.tg['t'].copy()
        t[self.tg['cam'] == 1] += self.nuis['c1']
        return t

    def dep_times(self):
        """depth frame times; with tie_depth (default) each camera's depth shares its colour frame's clock (one RealSense
        frameset): camera 0 depth at t, camera 1 depth at t + c1."""
        t = self.dp['ft'].copy()
        for c in (0, 1):
            off = (0.0 if c == 0 else self.nuis['c1']) if self.tie_depth else self.nuis[f'd{c}']
            t[self.dp['fcam'] == c] += off
        return t

    def n_ctrl(self):
        return 6 * self.traj.K

    def n_p(self):
        return self.n_ctrl() + len(self.nuis_free)

    def associate(self):
        """nearest CAD sample (pitcher frame) for every depth point at the current trajectory."""
        R, p = self.traj.eval(self.dep_times())
        pf = self.dp['pf']
        y = np.einsum('nji,nj->ni', R[pf], self.dp['X'] - p[pf])
        d, j = self.cad.kd.query(y, distance_upper_bound=0.03)
        ok = np.isfinite(d)
        j = np.where(ok, j, 0)
        self.dp['s'] = self.cad.Sp[j]; self.dp['n'] = self.cad.Np[j]; self.dp['assoc'] = ok
        return ok.mean()


class Fit:
    def __init__(self, probs, layout, sig_t=0.25, k_t=1.5, sig_d=0.0015, c_d=0.006, neff=None, sig_alpha=None,
                 sig_a=None, layout_free=(), fix_scale=True, nq=2, use_depth=True):
        self.probs, self.L = probs, layout
        self.sig_t, self.k_t, self.sig_d, self.c_d = sig_t, k_t, sig_d, c_d
        self.neff = neff
        self.sig_alpha, self.sig_a = sig_alpha, sig_a
        self.lfree = [(t, j) for (t, j) in layout_free]
        if not fix_scale:
            self.lfree.append(('s', 0))
        self.nq = nq
        self.use_depth = use_depth

    # ------------------------------------------------------------------ residual blocks
    def _frame_tags(self, P, L, lin):
        """tag residuals (n,8) [normalised by sig_t], and if lin: per-observation Jacobians wrt eps (n,8,6), layout (n,8,nL)."""
        f, fj = tag_fun(L.a, L.flat)
        tg = P.tg
        if not len(tg['cam']):
            return None
        t = P.tag_times()
        TP = np.stack([L.tp[x] for x in tg['tag']])
        w1 = (tg['cam'] == 1).astype(float)
        ex = (jnp.asarray(L.ext1), w1, jnp.asarray(L.ext_cen))
        if lin:
            i, u, R, p, G, gu = P.traj.eval_jac(t)
            r = np.asarray(f(jnp.asarray(R), jnp.asarray(p), jnp.zeros(6), jnp.asarray(TP), L.s, tg['W'], tg['Kv'], tg['q'], *ex))
            Je, Jt, Jx = [np.asarray(x) for x in fj(jnp.asarray(R), jnp.asarray(p), jnp.asarray(TP), L.s, tg['W'], tg['Kv'], tg['q'], *ex)]
            JL = np.zeros((len(t), 8, len(self.lfree)))
            for m, (tt, j) in enumerate(self.lfree):
                if tt == 's':
                    continue                                  # (scale: not used in the CT fit)
                if tt == 'e1':
                    JL[:, :, m] = Jx[:, :, j]
                    continue
                if tt == 'lean':                              # common tilt of every tag normal (conical wall)
                    JL[:, :, m] = Jt[:, :, 4]
                    continue
                sel = tg['tag'] == tt
                JL[sel, :, m] = Jt[sel, :, j]
            return r / self.sig_t, dict(i=i, G=G, gu=gu, Je=Je / self.sig_t, JL=JL / self.sig_t)
        R, p = P.traj.eval(t)
        r = np.asarray(f(jnp.asarray(R), jnp.asarray(p), jnp.zeros(6), jnp.asarray(TP), L.s, tg['W'], tg['Kv'], tg['q'], *ex))
        return r / self.sig_t, None

    def _depth(self, P, lin):
        dp = P.dp
        t = P.dep_times()
        if lin:
            i, u, R, p, G, gu = P.traj.eval_jac(t)
        else:
            R, p = P.traj.eval(t)
        pf = dp['pf']
        Rp = R[pf]
        y = np.einsum('nji,nj->ni', Rp, dp['X'] - p[pf])
        r = np.einsum('ni,ni->n', dp['n'], y - dp['s'])
        if not lin:
            return r, None
        Jr = np.cross(dp['n'], y)
        Jp = -np.einsum('nij,nj->ni', Rp, dp['n'])
        return r, dict(i=i, G=G, gu=gu, J=np.concatenate([Jr, Jp], 1))

    def depth_weights(self, P, r):
        w = tukey_w(r, self.c_d) * P.dp['assoc'] * P.dp['on']
        nf = np.bincount(P.dp['pf'], minlength=len(P.dp['ft'])).astype(float)
        fw = np.where(nf > 0, self.neff / np.maximum(nf, 1), 0.0) if self.neff else np.ones_like(nf)
        return w * fw[P.dp['pf']] / self.sig_d ** 2

    def cost(self, P, L):
        c = dict(tag=0.0, dep=0.0, prior=0.0)
        rt, _ = self._frame_tags(P, L, False)
        if rt is not None:
            c['tag'] = float((huber_rho(rt, self.k_t) * P.tg['on'][:, None]).sum())
        if self.use_depth and P.dp is not None:
            r, _ = self._depth(P, False)
            nf = np.bincount(P.dp['pf'], minlength=len(P.dp['ft'])).astype(float)
            fw = np.where(nf > 0, self.neff / np.maximum(nf, 1), 0.0) if self.neff else np.ones_like(nf)
            c['dep'] = float((tukey_rho(r, self.c_d) * P.dp['assoc'] * P.dp['on'] * fw[P.dp['pf']]).sum() / self.sig_d ** 2)
        if self.sig_a:
            i, u = P.traj.quad(self.nq)
            A = P.traj.acc(i, u)
            c['prior'] = float(0.5 * (self.wq * ((A[:, :3] / self.sig_alpha) ** 2).sum(1)).sum()
                               + 0.5 * (self.wq * ((A[:, 3:] / self.sig_a) ** 2).sum(1)).sum())
        return c

    @property
    def wq(self):
        return 1.0 / self.nq                  # per knot interval: mean over the interval of |a|^2 / sig^2

    # ------------------------------------------------------------------ normal equations of one pour
    def linearize(self, P, L):
        K = P.traj.K; nc = 6 * K; nn = len(P.nuis_free); nL = len(self.lfree)
        Hpp = np.zeros((nc + nn, nc + nn)); gp = np.zeros(nc + nn)
        Hpl = np.zeros((nc + nn, nL)); Hll = np.zeros((nL, nL)); gl = np.zeros(nL)
        nidx = {k: nc + m for m, k in enumerate(P.nuis_free)}

        M = K - 3

        def scatter(i, Hc, gc):
            SH = np.zeros((M, 24, 24)); sg = np.zeros((M, 24))
            np.add.at(SH, i, Hc); np.add.at(sg, i, gc)
            for m in np.unique(i):
                s0 = 6 * m
                Hpp[s0:s0 + 24, s0:s0 + 24] += SH[m]; gp[s0:s0 + 24] += sg[m]

        def add_frames(i, G, gu, He, ge, HeL, keys):
            # He (n,6,6), ge (n,6), HeL (n,6,nL); chain to controls i..i+3 and to the time offsets
            scatter(i, np.einsum('nai,nab,nbj->nij', G, He, G), np.einsum('nai,na->ni', G, ge))
            if nL:
                SL = np.zeros((M, 24, nL)); np.add.at(SL, i, np.einsum('nai,nab->nib', G, HeL))
                for m in np.unique(i):
                    Hpl[6 * m:6 * m + 24] += SL[m]
            keys = np.asarray(keys)
            for key in np.unique(keys):
                if key not in nidx:
                    continue
                msk = keys == key
                v = gu[msk] / P.traj.dt                                # d eps / d delta (n,6)
                k = nidx[key]
                Hev = np.einsum('nab,nb->na', He[msk], v)
                Hpp[k, k] += np.einsum('na,na->', v, Hev)
                gp[k] += np.einsum('na,na->', v, ge[msk])
                SC = np.zeros((M, 24)); np.add.at(SC, i[msk], np.einsum('nai,na->ni', G[msk], Hev))
                for m in np.unique(i[msk]):
                    Hpp[6 * m:6 * m + 24, k] += SC[m]; Hpp[k, 6 * m:6 * m + 24] += SC[m]
                if nL:
                    Hpl[k] += np.einsum('na,nab->b', v, HeL[msk])

        cost = dict(tag=0.0, dep=0.0, prior=0.0)
        # ---- tags (frame = one detection)
        rt, Jt = self._frame_tags(P, L, True)
        if rt is not None:
            w = huber_w(rt, self.k_t) * P.tg['on'][:, None]
            cost['tag'] = float((huber_rho(rt, self.k_t) * P.tg['on'][:, None]).sum())
            Je, JL = Jt['Je'], Jt['JL']
            He = np.einsum('nka,nk,nkb->nab', Je, w, Je); ge = np.einsum('nka,nk,nk->na', Je, w, rt)
            HeL = np.einsum('nka,nk,nkb->nab', Je, w, JL)
            keys = ['c1' if c == 1 else 'c0' for c in P.tg['cam']]
            add_frames(Jt['i'], Jt['G'], Jt['gu'], He, ge, HeL, keys)
            if nL:
                Hll += np.einsum('nka,nk,nkb->ab', JL, w, JL); gl += np.einsum('nka,nk,nk->a', JL, w, rt)
        # ---- depth (accumulate per frame first)
        if self.use_depth and P.dp is not None:
            r, Jd = self._depth(P, True)
            w = self.depth_weights(P, r)
            nf = np.bincount(P.dp['pf'], minlength=len(P.dp['ft'])).astype(float)
            fw = np.where(nf > 0, self.neff / np.maximum(nf, 1), 0.0) if self.neff else np.ones_like(nf)
            cost['dep'] = float((tukey_rho(r, self.c_d) * P.dp['assoc'] * P.dp['on'] * fw[P.dp['pf']]).sum() / self.sig_d ** 2)
            J = Jd['J']; nfr = len(P.dp['ft'])
            He = np.zeros((nfr, 6, 6)); ge = np.zeros((nfr, 6))
            np.add.at(He, P.dp['pf'], np.einsum('na,n,nb->nab', J, w, J))
            np.add.at(ge, P.dp['pf'], J * (w * r)[:, None])
            keys = [(('c0' if c == 0 else 'c1') if P.tie_depth else f"d{c}") for c in P.dp['fcam']]
            add_frames(Jd['i'], Jd['G'], Jd['gu'], He, ge, np.zeros((nfr, 6, nL)), keys)
        # ---- prior
        if self.sig_a:
            i, u = P.traj.quad(self.nq)
            A, JA = P.traj.acc_jac(i, u)
            sw = np.r_[[1 / self.sig_alpha ** 2] * 3, [1 / self.sig_a ** 2] * 3] * self.wq
            cost['prior'] = float(0.5 * (A * A * sw).sum())
            scatter(i, np.einsum('nai,a,naj->nij', JA, sw, JA), np.einsum('nai,a,na->ni', JA, sw, A))
        return Hpp, gp, Hpl, Hll, gl, cost

    # ------------------------------------------------------------------ apply an update
    def apply(self, P, dx):
        T2 = P.traj.copy(); T2.update(dx[:6 * T2.K])
        nu = dict(P.nuis)
        for m, k in enumerate(P.nuis_free):
            nu[k] = P.nuis[k] + dx[6 * T2.K + m]
        return T2, nu

    def total_cost(self, L=None):
        L = L or self.L
        out = []
        for P in self.probs:
            c = self.cost(P, L)
            out.append(c)
        return sum(sum(c.values()) for c in out), out

    def step(self, lam, verbose=False):
        """one LM step over all pours (Schur complement on the shared layout).  Returns accepted, new lam, costs."""
        lin = [self.linearize(P, self.L) for P in self.probs]
        c0 = sum(sum(x[5].values()) for x in lin)
        nL = len(self.lfree)
        while True:
            facs, sol = [], []
            S = np.zeros((nL, nL)); bS = np.zeros(nL)
            for (Hpp, gp, Hpl, Hll, gl, _) in lin:
                A = Hpp + lam * np.diag(np.diag(Hpp)) + 1e-9 * np.eye(len(gp))
                try:
                    cf = np.linalg.cholesky(A)
                except np.linalg.LinAlgError:
                    cf = None
                facs.append((A, cf))
            if nL:
                Hll_t = sum(x[3] for x in lin); gl_t = sum(x[4] for x in lin)
                S = Hll_t + lam * np.diag(np.diag(Hll_t)) + 1e-12 * np.eye(nL); bS = -gl_t.copy()
                for (A, cf), (Hpp, gp, Hpl, Hll, gl, _) in zip(facs, lin):
                    X = self._solve(A, cf, np.c_[Hpl, gp])
                    S -= Hpl.T @ X[:, :nL]; bS += Hpl.T @ X[:, nL]
                dl = np.linalg.solve(S, bS)
            else:
                dl = np.zeros(0)
            news = []
            for (A, cf), (Hpp, gp, Hpl, Hll, gl, _), P in zip(facs, lin, self.probs):
                dx = -self._solve(A, cf, gp + (Hpl @ dl if nL else 0))
                news.append(self.apply(P, dx))
            L2 = self.L.copy()
            for m, (t, j) in enumerate(self.lfree):
                if t == 's':
                    L2.s += dl[m]
                elif t == 'e1':
                    L2.ext1[j] += dl[m]
                elif t == 'lean':
                    for tt_ in L2.tp:
                        L2.tp[tt_][4] += dl[m]
                else:
                    L2.tp[t][j] += dl[m]
            # evaluate
            old = [(P.traj, dict(P.nuis)) for P in self.probs]
            for P, (T2, nu) in zip(self.probs, news):
                P.traj, P.nuis = T2, nu
            c1, parts = self.total_cost(L2)
            if np.isfinite(c1) and c1 <= c0:
                self.L = L2
                return True, max(lam / 5, 1e-10), c0, c1, parts, dl
            for P, (T, nu) in zip(self.probs, old):
                P.traj, P.nuis = T, nu
            lam *= 5
            if lam > 1e8:
                return False, lam, c0, c0, None, dl

    @staticmethod
    def _solve(A, cf, b):
        if cf is not None:
            from scipy.linalg import cho_solve
            return cho_solve((cf, True), b)
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def run(self, iters=60, tol=1e-9, verbose=True, outer=1, label='', c_sched=None):
        """outer rounds: (re-associate depth, set Tukey c) then LM to convergence."""
        hist = []
        for o in range(outer):
            if c_sched is not None:
                self.c_d = c_sched[min(o, len(c_sched) - 1)]
            if self.use_depth:
                for P in self.probs:
                    if P.dp is not None:
                        P.associate()
            lam = 1e-4
            for it in range(iters):
                ok, lam, c0, c1, parts, dl = self.step(lam)
                hist.append(c1)
                rel = (c0 - c1) / max(c0, 1e-12)
                if verbose and (it % 5 == 0 or not ok or rel < tol):
                    agg = {k: round(sum(p_[k] for p_ in parts), 3) for k in parts[0]} if parts else None
                    nu = [{k: round(v * 1e3, 2) for k, v in P.nuis.items() if k in P.nuis_free} for P in self.probs]
                    print(f'{label} outer {o} it {it} cost {c1:.4f} rel {rel:.2e} lam {lam:.0e} parts {agg} offsets_ms {nu}', flush=True)
                if not ok or (rel < tol and it > 2):
                    break
        return hist
