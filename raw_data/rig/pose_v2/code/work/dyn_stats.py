"""Motion bandwidth and acceleration statistics from the converged per-sample poses (step 1, flat layout, a = CAD).
(1) acceleration: local quadratic fits (window +-W s, >= 5 samples) to the TWO-CAMERA samples (cam-0 + cam-1 frames
    within 12 ms: no single-tag flip ambiguity) -> origin acceleration and body angular acceleration; noise-corrected
    variance  sig^2 = mean(est^2) - mean(noise variance of the estimate) (noise from the fit residuals).
(2) bandwidth: Lomb-Scargle periodogram of the axis tilt and of the spout-tip position (all samples, flip-cleaned) ->
    frequency where the spectrum meets the white-noise floor (median power above 6 Hz).
    python3 dyn_stats.py -> ../out/dyn_stats.json, ../img/bandwidth.jpg"""
import sys, json, pickle
import numpy as np
from scipy.signal import lombscargle
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *
from ctfit import tag_fun
import jax.numpy as jnp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

TIP = np.array([0.0, 0.050, 0.088])     # approximate spout tip in the tag frame (for the spectra only)
W = 0.15


def quad_acc(t, Y):
    """local quadratic fits; returns list of (acc (d,), noise var (d,))."""
    out = []
    for k in range(len(t)):
        m = np.abs(t - t[k]) <= W
        if m.sum() < 5:
            continue
        x = t[m] - t[k]
        X = np.stack([np.ones_like(x), x, x * x], 1)
        XtXi = np.linalg.inv(X.T @ X)
        c = XtXi @ X.T @ Y[m]
        res = Y[m] - X @ c
        s2 = (res ** 2).sum(0) / max(m.sum() - 3, 1)
        out.append((2 * c[2], 4 * XtXi[2, 2] * s2))
    return out


def rot_series(t, P, k0):
    R0 = Rot.from_rotvec(P[k0, :3])
    return (R0.inv() * Rot.from_rotvec(P[:, :3])).as_rotvec()


res = {}
fig, axs = plt.subplots(2, 5, figsize=(22, 8))
for ip, p in enumerate(POURS):
    o = pickle.load(open(f'{V2}/out/layout_{p}_flat_edge.pkl', 'rb'))
    t, P, S = o['t'], o['poses'], o['samples']
    two = np.array([len({c for c, _ in g}) == 2 for g in S])
    # (1) accelerations from two-camera samples
    L = Layout.from_dict(o['layout'])
    D = PourData(p, 'edge'); by = D.by_frame()
    f_, fj_ = tag_fun(L.a, L.flat)
    sig_px = o['rms']
    idx2 = np.nonzero(two)[0]
    cov = []
    for si in idx2:
        obs = [d for k in S[si] for d in by[k] if d['tag'] in L.tp]
        R0 = Rot.from_rotvec(P[si, :3]).as_matrix()
        Je, _ = fj_(jnp.asarray(np.repeat(R0[None], len(obs), 0)), jnp.asarray(np.repeat(P[si, 3:][None], len(obs), 0)),
                    jnp.asarray(np.stack([L.tp[d['tag']] for d in obs])), L.s, np.stack([D.W2C[d['cam']] for d in obs]),
                    np.stack([D.Kv[d['cam']] for d in obs]), np.stack([d['q'] for d in obs]))
        J = np.asarray(Je).reshape(-1, 6)
        cov.append(np.linalg.inv(J.T @ J) * sig_px ** 2)
    cov = np.array(cov)
    tt, PP = t[two], P[two]
    Rs = Rot.from_rotvec(PP[:, :3])
    at, vt, ar, vr = [], [], [], []
    for k in range(1, len(tt) - 1):
        h1, h2 = tt[k] - tt[k - 1], tt[k + 1] - tt[k]
        if max(h1, h2) > 0.12 or min(h1, h2) < 0.02:
            continue
        c = 2 / (h1 + h2)
        at.append(c * ((PP[k + 1, 3:] - PP[k, 3:]) / h2 - (PP[k, 3:] - PP[k - 1, 3:]) / h1))
        vt.append(c ** 2 * (np.diag(cov[k + 1])[3:] / h2 ** 2 + np.diag(cov[k])[3:] * (1 / h1 + 1 / h2) ** 2 + np.diag(cov[k - 1])[3:] / h1 ** 2))
        p12 = (Rs[k - 1].inv() * Rs[k]).as_rotvec(); p23 = (Rs[k].inv() * Rs[k + 1]).as_rotvec()
        ar.append(c * (p23 / h2 - p12 / h1))
        vr.append(c ** 2 * (np.diag(cov[k + 1])[:3] / h2 ** 2 + np.diag(cov[k])[:3] * (1 / h1 + 1 / h2) ** 2 + np.diag(cov[k - 1])[:3] / h1 ** 2))
    at, vt, ar, vr = map(np.array, (at, vt, ar, vr))
    sig_a = float(np.sqrt(max(np.mean(at ** 2) - np.mean(vt), 0)))
    sig_al = float(np.sqrt(max(np.mean(ar ** 2) - np.mean(vr), 0)))
    # cross-check: local quadratic windows +-0.25 s (>= 5 two-camera samples), noise from the fit residuals
    W2 = 0.25
    qa, qv = [], []
    for k in range(len(tt)):
        m = np.abs(tt - tt[k]) <= W2
        if m.sum() < 5:
            continue
        x = tt[m] - tt[k]; X = np.stack([np.ones_like(x), x, x * x], 1); XtXi = np.linalg.inv(X.T @ X)
        cc = XtXi @ X.T @ PP[m, 3:]; r_ = PP[m, 3:] - X @ cc
        qa.append(2 * cc[2]); qv.append(4 * XtXi[2, 2] * (r_ ** 2).sum(0) / max(m.sum() - 3, 1))
    qa, qv = np.array(qa), np.array(qv)
    sig_a_quad = float(np.sqrt(max(np.mean(qa ** 2) - np.mean(qv), 0)))
    # (2) spectra: all samples, tilt and tip position, Lomb-Scargle
    tilt, _ = tilt_az(Rot.from_rotvec(P[:, :3]).as_matrix())
    tip = Rot.from_rotvec(P[:, :3]).apply(TIP) + P[:, 3:]
    f = np.linspace(0.1, 14, 300)
    spec = {}
    for name, y in (('tilt_deg', tilt), ('tip_x_mm', tip[:, 0] * 1e3), ('tip_y_mm', tip[:, 1] * 1e3), ('tip_z_mm', tip[:, 2] * 1e3)):
        y = y - np.polyval(np.polyfit(t, y, 2), t)
        pw = lombscargle(t, y, 2 * np.pi * f, normalize=False) * 4 / len(t)     # ~ amplitude^2
        spec[name] = pw
    floor = {k: float(np.median(v[f > 6])) for k, v in spec.items()}
    fc = {}
    for k, v in spec.items():
        sm = np.convolve(v, np.ones(9) / 9, 'same')
        above = sm > 3 * floor[k]
        fc[k] = float(f[np.nonzero(above)[0].max()]) if above.any() else None
    gaps = np.diff(np.sort(t))
    res[p] = dict(n_samples=len(t), n_two_cam=int(two.sum()), frac_two_cam=float(two.mean()), n_triplets=len(at),
                  sig_a_ms2_per_comp=sig_a, rms_acc_est_ms2=float(np.sqrt(np.mean(at ** 2))), rms_noise_ms2=float(np.sqrt(np.mean(vt))),
                  sig_a_quad_window_ms2=sig_a_quad,
                  formal_sd_two_cam_median=dict(rot_deg=float(np.degrees(np.median(np.sqrt(cov[:, [0, 1, 2], [0, 1, 2]])))), pos_mm=float(np.median(np.sqrt(cov[:, [3, 4, 5], [3, 4, 5]])) * 1e3)),
                  sig_alpha_rad_s2_per_comp=sig_al, sig_alpha_deg_s2=float(np.degrees(sig_al)),
                  rms_alpha_est_deg_s2=float(np.degrees(np.sqrt(np.mean(ar ** 2)))), rms_alpha_noise_deg_s2=float(np.degrees(np.sqrt(np.mean(vr)))),
                  acc_p95_ms2=float(np.percentile(np.linalg.norm(at, axis=1), 95)),
                  band_edge_hz=fc, sample_gap_ms_p50_p95=(np.percentile(gaps, [50, 95]) * 1e3).round(1).tolist())
    print(p, json.dumps(res[p]), flush=True)
    ax = axs[0, ip]
    for k in ('tilt_deg',):
        ax.semilogy(f, spec[k], label=k); ax.axhline(floor[k], ls=':', c='k')
    ax.set_title(f'{p}: tilt spectrum (Lomb-Scargle)'); ax.set_xlabel('Hz'); ax.legend()
    ax = axs[1, ip]
    for k in ('tip_x_mm', 'tip_y_mm', 'tip_z_mm'):
        ax.semilogy(f, spec[k], label=k)
    ax.axhline(np.median([floor[k] for k in ('tip_x_mm', 'tip_y_mm', 'tip_z_mm')]), ls=':', c='k')
    ax.set_title(f'{p}: tip spectra'); ax.set_xlabel('Hz'); ax.legend()
fig.tight_layout(); fig.savefig(f'{V2}/img/bandwidth.jpg', dpi=70)
allp = {k: float(np.sqrt(np.mean([res[p][k] ** 2 for p in POURS]))) for k in ('sig_a_ms2_per_comp', 'sig_alpha_rad_s2_per_comp')}
res['pooled_rms'] = allp
print('pooled', allp)
json.dump(res, open(f'{V2}/out/dyn_stats.json', 'w'), indent=1)
