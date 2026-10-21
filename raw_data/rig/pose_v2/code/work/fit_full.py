"""Step 2b: full continuous-time fit of one pour: tags (Huber) + CAD-to-depth point-to-plane (Tukey) of cameras 0/1
+ smoothness prior, time offsets c1 / d0 / d1 free, layout fixed (joint tags-only fit), CAD placement fixed (place_cad).
Depth weight: per point 1/sig_d^2 (sig_d 1.5 mm, the ICP rms) scaled per frame to N_EFF effective points, N_EFF set so
that a single frame's formal depth-only precision of the pouring tilt equals the audit's measured view-dependent floor
(0.5 deg, middle of 0.24-0.74 deg; never tuned on validation numbers).
    python3 fit_full.py <pour> <depth tag> <out tag> [cams=01] [init=tags|own]
cams '0' / '1' = leave-one-camera-out (the other camera's tags AND depth removed; initialised from the remaining
camera's own per-sample poses, Viterbi-cleaned, so the held-out camera never influences the fit)."""
import sys, time, pickle, json
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
from init_track import viterbi_samples, slerp_fn

FLOOR_DEG = 0.5
SIG_D = 0.0015


def neff_from_floor(P, cad, az_spout, sig_d=SIG_D, floor_deg=FLOOR_DEG):
    """N_eff such that the median frame's formal pouring-tilt sd with N_eff points equals floor_deg."""
    R, p = P.traj.eval(P.dep_times())
    pf = P.dp['pf']
    y = np.einsum('nji,nj->ni', R[pf], P.dp['X'] - p[pf])
    d, j = cad.kd.query(y, distance_upper_bound=0.01)
    ok = np.isfinite(d); j = np.where(ok, j, 0)
    n = cad.Np[j]
    J = np.concatenate([np.cross(n, y), -np.einsum('nij,nj->ni', R[pf], n)], 1) * ok[:, None]
    a = np.radians(az_spout - 90.0)
    upour = np.array([np.cos(a), np.sin(a), 0.0])          # pitcher-frame axis of the pouring tilt (perp. to spout)
    ratios = []
    for f in range(len(P.dp['ft'])):
        m = pf == f
        if m.sum() < 100:
            continue
        H = J[m].T @ J[m] / sig_d ** 2 + 1e-6 * np.eye(6)
        C = np.linalg.inv(H)
        sd = np.sqrt(upour @ C[:3, :3] @ upour)
        ratios.append(m.sum() * (sd / np.radians(floor_deg)) ** 2)
    return float(np.median(ratios)), np.array(ratios)


def load_depth(p, tag, cams):
    Z = np.load(f'{V2}/data/depth_{p}_{tag}.npz')
    keepf = np.isin(Z['fcam'], cams)
    newidx = -np.ones(len(keepf), int); newidx[keepf] = np.arange(keepf.sum())
    m = keepf[Z['pf']]
    return dict(X=Z['X'][m].astype(float), pf=newidx[Z['pf'][m]], fcam=Z['fcam'][keepf], fstep=Z['fstep'][keepf], ft=Z['ft'][keepf])


if __name__ == '__main__':
    p, dtag, otag = sys.argv[1], sys.argv[2], sys.argv[3]
    cams = tuple(int(c) for c in (sys.argv[4] if len(sys.argv) > 4 else '01'))
    TAGS_PKL = sys.argv[5] if len(sys.argv) > 5 else 'ct_tags_lean_e1'
    T = pickle.load(open(f'{V2}/out/{TAGS_PKL}.pkl', 'rb'))
    PL = json.load(open(f'{V2}/out/placement.json'))
    PL['az'] = float(PL['az'])
    L = Layout.from_dict(T['layout'])
    tr = T['pours'][p]
    D = PourData(p, 'edge', tags=set(L.tp))
    D.rows = [d for d in D.rows if d['cam'] in cams]
    cad = gcm.CAD(z0_mm=PL['z0'], az_deg=PL['az'])
    if PL.get('extra') is not None:
        cad.set_place(PL['z0'], PL['az'], np.array(PL['extra']))
    if cams == (0, 1):
        traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
        nuis0 = dict(c1=tr['nuis']['c1'], d0=0.0, d1=tr['nuis']['c1'])
        nfree = ('c1',)                                    # depth tied to its camera's colour clock (offsets d0/d1 unidentifiable)
    else:
        samples, tt, by = build_samples(D, cams=cams)
        X, path, cands = viterbi_samples(D, L, samples, tt, by, sig_t=T['sig_t'], verbose=True)
        f = slerp_fn(tt, X)
        traj = Traj(tr['T0'], tr['dt'], *f(Traj(tr['T0'], tr['dt'], tr['R'], tr['P']).ctrl_times()))
        nuis0 = dict(c1=tr['nuis']['c1'], d0=0.0, d1=tr['nuis']['c1'])
        nfree = ()                                           # c1 fixed at the two-camera value (camera 0 is the time origin)
    dp = load_depth(p, dtag, cams)
    Pp = PourProblem(D, traj, D.rows, dp, nuis_free=nfree, cad=cad)
    Pp.nuis.update(nuis0)
    Pp.associate()
    neff, ratios = neff_from_floor(Pp, cad, PL['az'])
    print(p, 'cams', cams, 'depth frames', len(dp['ft']), 'points', len(dp['X']), 'N_eff', round(neff, 1),
          'p10/p90', np.percentile(ratios, [10, 90]).round(1), flush=True)
    F = Fit([Pp], L, sig_t=T['sig_t'], k_t=T['k_t'], sig_d=SIG_D, c_d=0.008, neff=neff, sig_alpha=T['sig_alpha'],
            sig_a=T['sig_a'], layout_free=(), use_depth=True)
    t0 = time.time()
    F.run(iters=60, tol=1e-9, outer=4, label=f'{p}{cams}', c_sched=(0.010, 0.008, 0.006, 0.005))
    c = F.cost(Pp, L)
    rt = F._frame_tags(Pp, L, False)[0] * F.sig_t if len(D.rows) else np.zeros((0, 8))
    rd, _ = F._depth(Pp, False)
    out = dict(pour=p, cams=cams, T0=Pp.traj.T0, dt=Pp.traj.dt, R=Pp.traj.R, P=Pp.traj.P, nuis=dict(Pp.nuis), cost=c,
               neff=neff, sig_t=F.sig_t, sig_d=SIG_D, sig_a=F.sig_a, sig_alpha=F.sig_alpha, layout=L.as_dict(), place=PL,
               tag_res=rt, tag_cam=Pp.tg['cam'], tag_id=Pp.tg['tag'], tag_step=Pp.tg['step'],
               dep_res=rd.astype(np.float32), dep_pf=dp['pf'], dep_fcam=dp['fcam'], dep_assoc=Pp.dp['assoc'], time_s=time.time() - t0)
    for cc in cams:
        m = Pp.tg['cam'] == cc
        md = (dp['fcam'][dp['pf']] == cc) & Pp.dp['assoc'] & (np.abs(rd) < 0.005)
        print(p, 'cam', cc, 'tag rms px', round(float(np.sqrt(np.mean(rt[m] ** 2))), 4) if m.any() else None,
              'depth |r| med mm', round(float(np.median(np.abs(rd[md]))) * 1e3, 3), 'inlier frac', round(float(md.sum() / max(((dp['fcam'][dp['pf']] == cc)).sum(), 1)), 3), flush=True)
    print(p, 'offsets ms', {k: round(v * 1e3, 2) for k, v in Pp.nuis.items()}, 'cost', {k: round(v, 2) for k, v in c.items()}, f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(out, open(f'{V2}/out/full_{p}_{otag}.pkl', 'wb'))
