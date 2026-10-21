"""Step 2 (final): joint continuous-time fit of all five pours from cameras 0/1 (tags + depth + prior) with the shared
unknowns estimated together: tag layout (tags 1/5/0 (phi, z, psi), psi4, common lean), camera-1 extrinsic correction
(6 dof, world frame) applied to camera-1 tags AND depth, CAD placement correction in the tag frame (axis tilt x/y,
translation; spin about the axis fixed: spout at 77.14 deg), per pour: spline + camera-1 time offset.
    python3 fit_joint.py [check] -> ../out/joint.pkl, ../out/full_<p>_v2j.pkl, ../out/placement_joint.json"""
import sys, time, pickle, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import *
from spline import Traj
from ctfit import PourProblem, Fit
from fit_full import neff_from_floor, SIG_D


def load_depth_raw(p, tag, Lold, cams=(0, 1)):
    """depth points (frame-sorted) + Xraw: camera-1 points back in camera 1's believed world (undo the correction the
    extraction applied)."""
    Z = np.load(f'{V2}/data/depth_{p}_{tag}.npz')
    keepf = np.isin(Z['fcam'], cams)
    newidx = -np.ones(len(keepf), int); newidx[keepf] = np.arange(keepf.sum())
    m = keepf[Z['pf']]
    X = Z['X'][m].astype(float); pf = newidx[Z['pf'][m]]
    o = np.argsort(pf, kind='stable'); X, pf = X[o], pf[o]
    fcam = Z['fcam'][keepf]
    Xraw = X.copy()
    c1 = fcam[pf] == 1
    Re = Rot.from_rotvec(Lold.ext1[:3]).as_matrix()
    Xraw[c1] = (X[c1] - Lold.ext_cen) @ Re.T + Lold.ext_cen + Lold.ext1[3:]
    return dict(X=X, Xraw=Xraw, pf=pf, fcam=fcam, fstep=Z['fstep'][keepf], ft=Z['ft'][keepf])


BASE_PL = 'placement.json'
OUT = 'v2j'


def build(pours=POURS, cams=(0, 1)):
    T = pickle.load(open(f'{V2}/out/ct_tags_lean_e1.pkl', 'rb'))
    PL = json.load(open(f'{V2}/out/{BASE_PL}'))
    L = Layout.from_dict(T['layout'])
    Lold = L.copy()
    cad = gcm.CAD(z0_mm=PL['z0'], az_deg=PL['az'])
    cad.set_place(PL['z0'], PL['az'], np.array(PL['extra']))
    L.pl_cen = cad.centre_p.copy()
    probs = []
    for p in pours:
        tr = T['pours'][p]
        D = PourData(p, 'edge', tags=set(L.tp))
        D.rows = [d for d in D.rows if d['cam'] in cams]
        dp = load_depth_raw(p, 't1', Lold, cams)
        P = PourProblem(D, Traj(tr['T0'], tr['dt'], tr['R'], tr['P']), D.rows, dp, nuis_free=('c1',), cad=cad)
        P.nuis['c1'] = tr['nuis']['c1']
        P.associate(L)
        P.neff, _ = neff_from_floor(P, cad, PL['az'])
        probs.append(P)
        print(p, 'N_eff', round(P.neff, 1), 'depth pts', len(dp['X']), flush=True)
    return T, PL, L, cad, probs


LFREE = [(4, 2)] + [(t, j) for t in (0, 1, 5) for j in range(3)] + [('lean', 0)] + [('e1', j) for j in range(6)] + \
        [('pl', 0), ('pl', 1), ('pl', 3), ('pl', 4), ('pl', 5)]


def placement_of(PL, L):
    """compose the base placement extra with the fitted correction L.pl (x -> R(x - c) + c + t, tag frame)."""
    Rp = Rot.from_rotvec(L.pl[:3]).as_matrix()
    Tp = np.eye(4); Tp[:3, :3] = Rp; Tp[:3, 3] = L.pl_cen + L.pl[3:] - Rp @ L.pl_cen
    return dict(z0=PL['z0'], az=PL['az'], extra=(Tp @ np.array(PL['extra'])).tolist(), source='fit_joint.py (joint tags + depth)')


if __name__ == '__main__':
    check = len(sys.argv) > 1 and sys.argv[1] == 'check'
    if len(sys.argv) > 2:
        BASE_PL, OUT = sys.argv[1], sys.argv[2]
    T, PL, L, cad, probs = build(POURS[2:3] if check else POURS)
    F = Fit(probs, L, sig_t=T['sig_t'], k_t=T['k_t'], sig_d=SIG_D, c_d=0.008, neff=None, sig_alpha=T['sig_alpha'],
            sig_a=T['sig_a'], layout_free=LFREE, use_depth=True)
    if check:
        L.ext1 = L.ext1 + np.array([0.003, -0.002, 0.004, 0.001, -0.002, 0.001]); L.pl = np.array([0.004, -0.003, 0, 0.001, 0.002, -0.001])
        P = probs[0]
        r0, Jd = F._depth(P, True, L)
        sel = np.random.default_rng(0).choice(len(r0), 2000, replace=False)
        for m, (t, j) in enumerate(F.lfree):
            if t not in ('e1', 'pl'):
                continue
            h = 1e-6
            L2 = L.copy()
            (L2.ext1 if t == 'e1' else L2.pl)[j] += h
            r1, _ = F._depth(P, False, L2)
            fd = (r1 - r0)[sel] / h; an = Jd['JL'][sel, m]
            print(t, j, 'max |fd - analytic| / max|fd|', float(np.abs(fd - an).max() / max(np.abs(fd).max(), 1e-12)))
        sys.exit()
    t0 = time.time()
    F.run(iters=40, tol=1e-9, outer=4, label='joint', c_sched=(0.010, 0.008, 0.006, 0.005))
    L = F.L
    PLj = placement_of(PL, L)
    json.dump(PLj, open(f'{V2}/out/placement_{OUT}.json', 'w'), indent=1)
    print('JOINT layout', L.pretty(), 'time', round(time.time() - t0), flush=True)
    out = dict(layout=L.as_dict(), place=PLj, sig_t=F.sig_t, sig_d=SIG_D, sig_a=F.sig_a, sig_alpha=F.sig_alpha, pours={})
    for P in probs:
        p = P.D.p
        c = F.cost(P, L)
        rt = F._frame_tags(P, L, False)[0] * F.sig_t
        rd, _ = F._depth(P, False, L)
        rec = dict(pour=p, cams=(0, 1), T0=P.traj.T0, dt=P.traj.dt, R=P.traj.R, P=P.traj.P, nuis=dict(P.nuis), cost=c,
                   neff=P.neff, sig_t=F.sig_t, sig_d=SIG_D, sig_a=F.sig_a, sig_alpha=F.sig_alpha, layout=L.as_dict(), place=PLj,
                   tag_res=rt, tag_cam=P.tg['cam'], tag_id=P.tg['tag'], tag_step=P.tg['step'],
                   dep_res=rd.astype(np.float32), dep_pf=P.dp['pf'], dep_fcam=P.dp['fcam'], dep_assoc=P.dp['assoc'])
        pickle.dump(rec, open(f'{V2}/out/full_{p}_{OUT}.pkl', 'wb'))
        out['pours'][p] = rec
        msg = []
        for cc in (0, 1):
            m = P.tg['cam'] == cc
            md = (P.dp['fcam'][P.dp['pf']] == cc) & P.dp['assoc']
            msg.append(f"cam{cc}: tag rms {np.sqrt(np.mean(rt[m] ** 2)):.3f} px, depth med|r| {np.median(np.abs(rd[md])) * 1e3:.3f} mm, "
                       f"signed med {np.median(rd[md & (np.abs(rd) < 0.005)]) * 1e3:+.3f} mm")
        print(p, 'c1 ms', round(P.nuis['c1'] * 1e3, 2), ' | '.join(msg), {k: round(v, 1) for k, v in c.items()}, flush=True)
    pickle.dump(out, open(f'{V2}/out/joint_{OUT}.pkl', 'wb'))
