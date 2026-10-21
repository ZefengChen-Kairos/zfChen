"""Step 2a: tags-only continuous-time fit of all five pours jointly: one spline per pour, shared tag layout (flat,
a = CAD apothem, tags other than tag 4 free + psi4), camera-1 colour time offset per pour, smoothness prior.
    python3 fit_tags.py [dt_ms] [tag]   -> ../out/ct_tags[_tag].pkl"""
import sys, time, pickle, json
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
from init_track import viterbi_samples, slerp_fn

DT = float(sys.argv[1]) * 1e-3 if len(sys.argv) > 1 else 0.05
TAGNAME = sys.argv[2] if len(sys.argv) > 2 else ''
LEAN0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0       # initial common tag lean [deg]; free when given
EXT1 = len(sys.argv) > 4 and sys.argv[4] == 'e1'                 # free world-frame correction of camera 1's extrinsics
DYN = json.load(open(f'{V2}/out/dyn_stats.json'))
SIG_A = float(np.sqrt(np.mean([DYN[p]['rms_acc_est_ms2'] ** 2 for p in POURS])))                     # raw rms (upper bound)
SIG_AL = float(np.radians(np.sqrt(np.mean([DYN[p]['rms_alpha_est_deg_s2'] ** 2 for p in POURS]))))
POURS_RUN = [p for p in POURS]


def initial_layout():
    """tags 4/5: mean of the single-camera (cam-0) layouts; tags 1/0: mean of the joint 5-ms-sync layouts."""
    c0 = [pickle.load(open(f'{V2}/out/layout_{p}_flat_cam0_edge.pkl', 'rb'))['layout'] for p in POURS]
    s5 = [pickle.load(open(f'{V2}/out/layout_{p}_flat_sync5_edge.pkl', 'rb'))['layout'] for p in POURS]
    L = Layout.from_dict(s5[0])
    L.tags = [0, 1, 4, 5]
    tp = {}
    tp[4] = np.mean([d['tp'][4] for d in c0], 0)
    tp[5] = np.mean([d['tp'][5] for d in c0], 0)
    tp[1] = np.mean([d['tp'][1] for d in s5], 0)
    t0s = [d['tp'][0] for d in s5 if 0 in d['tp']]
    tp[0] = np.mean(t0s, 0)
    L.tp = {t: np.array(v, float) for t, v in tp.items()}
    L.tp[4][[0, 1, 3]] = 0
    L.tp = {t: np.r_[v[:4], 0.0, 0.0] for t, v in L.tp.items()}
    if LEAN0:
        for t in L.tp:
            L.tp[t][4] = np.radians(LEAN0)
    return L


if __name__ == '__main__':
    L = initial_layout()
    print('initial layout', L.pretty(), 'lean0', LEAN0, 'sig_a', SIG_A, 'sig_alpha deg/s2', np.degrees(SIG_AL), flush=True)
    probs = []
    for p in POURS_RUN:
        D = PourData(p, 'edge', tags=set(L.tp))
        samples, tt, by = build_samples(D)
        X, path, cands = viterbi_samples(D, L, samples, tt, by, verbose=True)
        f = slerp_fn(tt, X)
        t_all = np.concatenate([np.array([D.ts[c][k] for k in D.tk.steps[c]]) for c in CAMS])
        traj = Traj.make(t_all.min() - 0.02, t_all.max() + 0.06, DT, f)
        P = PourProblem(D, traj, D.rows, None, nuis_free=('c1',))
        P.init_samples = (tt, X)
        probs.append(P)
        print(p, 'knots', traj.K, 'tag obs', len(D.rows), flush=True)
    lfree = [(4, 2)] + [(t, j) for t in (0, 1, 5) for j in range(3)] + ([('lean', 0)] if LEAN0 else []) + ([('e1', j) for j in range(6)] if EXT1 else [])
    L.ext_cen = np.array([0.15, 0.22, -0.03])
    F = Fit(probs, L, sig_t=0.30, k_t=1.5, sig_alpha=SIG_AL, sig_a=SIG_A, layout_free=lfree, use_depth=False)
    t0 = time.time()
    for rnd in range(3):
        F.run(iters=80, tol=1e-10, label=f'round{rnd} sig_t {F.sig_t:.3f}')
        # variance component: robust rms of the normalised tag residuals
        rr = np.concatenate([F._frame_tags(P, F.L, False)[0].ravel() for P in probs]) * F.sig_t
        s_new = float(1.4826 * np.median(np.abs(rr)))
        print('round', rnd, 'tag residual rms px', round(float(np.sqrt(np.mean(rr ** 2))), 4), 'MAD sigma', round(s_new, 4),
              'layout', F.L.pretty(), f'{time.time() - t0:.0f}s', flush=True)
        if abs(s_new - F.sig_t) / F.sig_t < 0.05:
            break
        F.sig_t = s_new
    out = dict(layout=F.L.as_dict(), sig_t=F.sig_t, k_t=F.k_t, sig_a=SIG_A, sig_alpha=SIG_AL, dt=DT, pours={})
    for P in probs:
        r = F._frame_tags(P, F.L, False)[0] * F.sig_t
        out['pours'][P.D.p] = dict(T0=P.traj.T0, dt=P.traj.dt, R=P.traj.R, P=P.traj.P, nuis=dict(P.nuis), init_t=P.init_samples[0],
                                   init_X=P.init_samples[1], tag_res=r, tag_cam=P.tg['cam'], tag_id=P.tg['tag'], tag_step=P.tg['step'])
        for c in CAMS:
            m = P.tg['cam'] == c
            print(P.D.p, 'cam', c, 'rms px', round(float(np.sqrt(np.mean(r[m] ** 2))), 4), 'med |r|', round(float(np.median(np.abs(r[m]))), 4),
                  'offset c1 ms', round(P.nuis['c1'] * 1e3, 2), flush=True)
    pickle.dump(out, open(f'{V2}/out/ct_tags{TAGNAME}.pkl', 'wb'))
