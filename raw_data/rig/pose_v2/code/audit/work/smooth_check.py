"""A6: over-smoothing / dynamics check against per-sample converged poses (v2 model: joint layout, lean, camera-1
extrinsic correction, camera-1 clock offset; fixed-layout Gauss-Newton on the tag corners of each sample, Huber):
  'j2'  = two-camera samples (cam-0 frame + cam-1 frame within 12 ms), time = mean corrected time;
  'c0'  = camera-0 frames alone (tags 4/5);   'c1' = camera-1 frames alone.
For each: axis tilt (output frame) per sample vs the v2 spline at the same time; high-pass parts (minus a 0.5 s
running mean of the v2 trace) -> regression slope sample_hp ~ v2_hp (1 = v2 carries the motion, >1 = v2 flattens);
residual vs v2 angular speed; same for the current track (tip.npz Slerp).
    python3 smooth_check.py -> ../out/smooth_check.json, ../out/samples_<p>.npz"""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *
from spline import Traj
from ctfit import tag_fun
import jax.numpy as jnp
A = f'{V2}/audit'
pours = sys.argv[1:] or list(POURS)
res = {}
for p in pours:
    F = pickle.load(open(f'{V2}/out/full_{p}_v2k.pkl', 'rb'))
    L = Layout.from_dict(F['layout']); c1 = F['nuis']['c1']
    tr = Traj(F['T0'], F['dt'], F['R'], F['P'])
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz'); B = Z['B_out_to_tag']
    D = PourData(p, 'edge', tags=set(L.tp))
    f_, fj_ = tag_fun(L.a, L.flat)
    by = D.by_frame()
    k0 = sorted([k for k in by if k[0] == 0], key=lambda k: k[1]); k1 = sorted([k for k in by if k[0] == 1], key=lambda k: k[1])
    t0s = np.array([D.ts[0][k[1]] for k in k0]); t1s = np.array([D.ts[1][k[1]] + c1 for k in k1])
    samples = [('c0', [k], t) for k, t in zip(k0, t0s)] + [('c1', [k], t) for k, t in zip(k1, t1s)]
    for k, t in zip(k0, t0s):
        j = int(np.argmin(np.abs(t1s - t)))
        if abs(t1s[j] - t) < 0.012:
            samples.append(('j2', [k, k1[j]], 0.5 * (t + t1s[j])))

    def solve(keys, t):
        obs = [d for k in keys for d in by[k]]
        TP = jnp.asarray(np.stack([L.tp[d['tag']] for d in obs])); n = len(obs)
        W = np.stack([D.W2C[d['cam']] for d in obs]); Kv = np.stack([D.Kv[d['cam']] for d in obs]); Q = np.stack([d['q'] for d in obs])
        w1 = np.array([float(d['cam'] == 1) for d in obs])
        if not (tr.T0 <= t <= tr.t_end): return None
        R0, p0 = tr.eval(np.array([t])); R0, p0 = R0[0], p0[0]
        ex = (jnp.asarray(L.ext1), w1, jnp.asarray(L.ext_cen))
        for it in range(15):
            Rr = np.repeat(R0[None], n, 0); pr = np.repeat(p0[None], n, 0)
            r = np.asarray(f_(jnp.asarray(Rr), jnp.asarray(pr), jnp.zeros(6), TP, L.s, W, Kv, Q, *ex)).ravel() / 0.18
            J = np.asarray(fj_(jnp.asarray(Rr), jnp.asarray(pr), TP, L.s, W, Kv, Q, *ex)[0]).reshape(-1, 6) / 0.18
            w = huber_w(r, 1.5)
            H = J.T @ (J * w[:, None]) + 1e-9 * np.eye(6); g = J.T @ (w * r)
            dx = -np.linalg.solve(H, g)
            R0 = R0 @ np.asarray(expm3(jnp.asarray(dx[:3]))); p0 = p0 + dx[3:]
            if np.abs(dx).max() < 1e-9: break
        C = np.linalg.inv(H)
        return R0, p0, float(np.sqrt(np.mean((r * 0.18) ** 2))), C
    rows = []
    for kind, keys, t in samples:
        s = solve(keys, t)
        if s is None: continue
        R0, p0, rms, C = s
        rows.append((kind, t, R0, p0, rms, np.sqrt(np.diag(C)[:3]).max()))
    kinds = np.array([r[0] for r in rows]); ts = np.array([r[1] for r in rows])
    Rs = np.stack([r[2] for r in rows]); ps = np.stack([r[3] for r in rows]); rms = np.array([r[4] for r in rows])
    Ro = Rs @ B[:3, :3]
    tilt_s, az_s = tilt_az(Ro)
    tg = np.arange(tr.T0 + 0.05, tr.t_end - 0.05, 0.005)
    Rg, pg = tr.eval(tg); tilt_g, _ = tilt_az(Rg @ B[:3, :3])
    wg = np.r_[np.degrees(np.linalg.norm((Rot.from_matrix(Rg[:-1]).inv() * Rot.from_matrix(Rg[1:])).as_rotvec(), axis=1)) / 0.005, 0]
    lowg = uniform_filter1d(tilt_g, 100, mode='nearest')            # 0.5 s running mean of the v2 tilt
    tilt_v = np.interp(ts, tg, tilt_g); low = np.interp(ts, tg, lowg); wv = np.interp(ts, tg, wg)
    # current track (tip.npz) tilt
    zc = np.load(f'{SCR}/pours/{p}/tip.npz'); tilt_c = np.interp(ts, zc['t'], zc['tilt'])
    np.savez_compressed(f'{A}/out/samples_{p}.npz', kind=kinds, t=ts, tilt=tilt_s, tilt_v2=tilt_v, tilt_cur=tilt_c, rms=rms, w_v2=wv, tg=tg, tilt_g=tilt_g, w_g=wg,
                        rotvec=Rot.from_matrix(Ro).as_rotvec(), pos=ps)
    out = {}
    for kd in ('j2', 'c0', 'c1'):
        m = (kinds == kd) & (rms < 1.5)
        if m.sum() < 10: continue
        d = tilt_s[m] - tilt_v[m]
        hp_s = tilt_s[m] - low[m]; hp_v = tilt_v[m] - low[m]; hp_c = tilt_c[m] - low[m]
        slope = float(np.polyfit(hp_v, hp_s, 1)[0]); slope_c = float(np.polyfit(hp_c, hp_s, 1)[0])
        fast = wv[m] > 30
        rec = dict(n=int(m.sum()), n_fast=int(fast.sum()), bias_deg=round(float(np.median(d)), 3), mad_deg=round(float(np.median(np.abs(d - np.median(d)))), 3),
                   p90abs_deg=round(float(np.percentile(np.abs(d - np.median(d)), 90)), 3),
                   slope_sample_on_v2_hp=round(slope, 3), slope_sample_on_cur_hp=round(slope_c, 3),
                   hp_rms_sample_v2_cur=[round(float(np.sqrt(np.mean(x ** 2))), 3) for x in (hp_s, hp_v, hp_c)],
                   resid_v2_rms_hp=round(float(np.sqrt(np.mean((hp_s - hp_v) ** 2))), 3), resid_cur_rms_hp=round(float(np.sqrt(np.mean((hp_s - hp_c) ** 2))), 3))
        if fast.sum() >= 5:
            df = d[fast] - np.median(d)
            rec['fast_w_gt30'] = dict(med_abs=round(float(np.median(np.abs(df))), 3), p90=round(float(np.percentile(np.abs(df), 90)), 3), signed_med=round(float(np.median(df)), 3),
                                     slope_hp=round(float(np.polyfit(hp_v[fast], hp_s[fast], 1)[0]), 3), cur_med_abs=round(float(np.median(np.abs(tilt_s[m][fast] - tilt_c[m][fast] - np.median(tilt_s[m] - tilt_c[m])))), 3))
        out[kd] = rec
    res[p] = out
    print(p, json.dumps(out), flush=True)
json.dump(res, open(f'{A}/out/smooth_check.json', 'w'), indent=1)
