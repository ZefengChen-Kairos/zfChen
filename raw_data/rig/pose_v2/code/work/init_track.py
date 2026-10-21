"""Flip-free initial pose samples: every sample's distinct local minima (fixed layout, multi-start from all IPPE
alternatives) -> Viterbi over time: node cost = corner cost above the sample's best (in units of sig_t^2), transition
cost = (rotation change / (w_r dt))^2 + (origin change / (v_p dt))^2 with generous rates (w_r = 300 deg/s, v_p = 1 m/s,
plus 1 deg / 2 mm floors), so the chosen branch is the one consistent with its neighbours."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from pv2lib import frame_candidates, rot_deg


def viterbi_samples(D, L, samples, tt, by, sig_t=0.3, w_r=300.0, v_p=1.0, verbose=False):
    cands = []
    for g in samples:
        obs = [d for k in g for d in by[k] if d['tag'] in L.tp]
        sols = frame_candidates(D, L, obs, iters=30)
        c0 = sols[0][0]
        cands.append([(max(s[0] - c0, 0.0) / sig_t ** 2 * 2, s[1]) for s in sols[:4]])    # 2x: huber cost is 1/2 r^2
    n = len(cands)
    acc = [np.array([c for c, _ in cands[0]])]
    back = []
    for k in range(1, n):
        dt = max(tt[k] - tt[k - 1], 0.01)
        prev, cur = cands[k - 1], cands[k]
        T = np.zeros((len(prev), len(cur)))
        for a, (_, xa) in enumerate(prev):
            for b, (_, xb) in enumerate(cur):
                dr = rot_deg(xa, xb); dp = np.linalg.norm(xa[3:] - xb[3:])
                T[a, b] = (dr / (1.0 + w_r * dt)) ** 2 + (dp / (0.002 + v_p * dt)) ** 2
        tot = acc[-1][:, None] + T
        back.append(np.argmin(tot, 0))
        acc.append(tot.min(0) + np.array([c for c, _ in cur]))
    j = int(np.argmin(acc[-1])); path = [j]
    for k in range(n - 2, -1, -1):
        j = int(back[k][j]); path.append(j)
    path = path[::-1]
    X = np.array([cands[k][path[k]][1] for k in range(n)])
    n_alt = int(sum(p_ != 0 for p_ in path))
    if verbose:
        print('viterbi: samples', n, 'not the min-cost branch', n_alt, flush=True)
    return X, path, cands


def slerp_fn(t, X):
    o = np.argsort(t); t, X = t[o], X[o]
    keep = np.r_[True, np.diff(t) > 1e-4]; t, X = t[keep], X[keep]
    sl = Slerp(t, Rot.from_rotvec(X[:, :3]))

    def f(tc):
        tc = np.clip(tc, t[0], t[-1])
        return sl(tc).as_matrix(), np.stack([np.interp(tc, t, X[:, 3 + j]) for j in range(3)], 1)
    return f
