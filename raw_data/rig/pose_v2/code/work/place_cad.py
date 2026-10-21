"""CAD placement in the v2 tag frame from the depth points of cameras 0/1 and a trajectory:
(a) profile over the spout azimuth (the 14-gon: tag 4 on a flat -> spout flat at k x 25.714 deg) and z0;
(b) free 6-dof constant correction (robust point-to-plane ICP, Tukey 5 mm) on top of the best nominal placement.
    python3 place_cad.py <traj.pkl> <depth tag> [pours...] -> ../out/place_<tag>.json"""
import sys, json, pickle
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2, POURS
from spline import Traj


def pitcher_points(p, T, tag, every=1):
    tr = T['pours'][p]
    traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
    Z = np.load(f'{V2}/data/depth_{p}_{tag}.npz')
    ft = Z['ft'].copy()
    for c in (0, 1):
        m = Z['fcam'] == c
        if T.get('depth_offsets'):
            ft[m] += tr['nuis'][f'd{c}']
        elif c == 1:
            ft[m] += tr['nuis'].get('c1', 0.0)
    R, pp = traj.eval(ft)
    pf = Z['pf']
    sel = (pf % every) == 0 if every > 1 else np.ones(len(pf), bool)
    X = Z['X'][sel].astype(float)
    return np.einsum('nji,nj->ni', R[pf[sel]], X - pp[pf[sel]])


def score(cad, Q, c=0.005):
    d, j = cad.kd.query(Q, distance_upper_bound=0.03)
    ok = np.isfinite(d)
    r = np.zeros(len(Q)); r[ok] = np.einsum('ij,ij->i', Q[ok] - cad.Sp[j[ok]], cad.Np[j[ok]])
    x = np.minimum(np.abs(np.where(ok, r, c)) / c, 1.0)
    return float(np.mean(c * c / 6 * (1 - (1 - x * x) ** 3))) / (c * c / 6), float(np.median(np.abs(r[ok]))) * 1e3


if __name__ == '__main__':
    trf, tag = sys.argv[1], sys.argv[2]
    pours = sys.argv[3:] or list(POURS)
    T = pickle.load(open(trf, 'rb'))
    cad = gcm.CAD(z0_mm=-28.5, az_deg=90.0)
    out = {}
    allQ = []
    for p in pours:
        Q = pitcher_points(p, T, tag)
        Q = Q[np.random.default_rng(0).choice(len(Q), min(len(Q), 120000), replace=False)]
        allQ.append(Q[::4])
        prof = {}
        for az in (51.43, 77.14, 90.0, 102.86, 128.57):
            row = []
            for z0 in np.arange(-33.0, -23.9, 1.0):
                cad.set_place(z0, az)
                row.append(score(cad, Q))
            prof[az] = row
        best = min(((az, z0, prof[az][k][0]) for az in prof for k, z0 in enumerate(np.arange(-33.0, -23.9, 1.0))), key=lambda x: x[2])
        cad.set_place(best[1], best[0])
        D, st = gcm.icp(cad, Q[::2], iters=40)
        de = gcm.describe(D, cad)
        out[p] = dict(n=len(Q), best_nominal=dict(az=best[0], z0=best[1], tukey_cost=best[2]),
                      profile={str(az): [[float(z0), v[0], v[1]] for z0, v in zip(np.arange(-33.0, -23.9, 1.0), row)] for az, row in prof.items()},
                      free6dof=dict(D=D.tolist(), **de, **{k: st[k] for k in ('rms_mm', 'med_abs_mm', 'frac_inl', 'cond')}))
        print(p, 'best nominal', best, 'free6dof on top: rot', round(de['rot_deg'], 2), 'xyz', np.round([de['rot_x_deg'], de['rot_y_deg'], de['rot_z_deg']], 2),
              'centre mm', np.round(de['tr_centre'], 2), 'tip mm', round(de['tr_tip_mm'], 2), 'rms', round(st['rms_mm'], 2), flush=True)
        for az in prof:
            k = int(np.argmin([v[0] for v in prof[az]]))
            print('   az', az, 'best z0', -33 + k, 'cost', round(prof[az][k][0], 4), 'med|r| mm', round(prof[az][k][1], 3))
    Q = np.concatenate(allQ)
    for az in (77.14, 102.86):
        cad.set_place(-28.5, az)
        D, st = gcm.icp(cad, Q, iters=40)
        de = gcm.describe(D, cad)
        out[f'all_az{az}'] = dict(free6dof=dict(D=D.tolist(), **de, **{k: st[k] for k in ('rms_mm', 'med_abs_mm', 'frac_inl', 'cond')}))
        print('ALL az', az, 'free6dof: rot xyz', np.round([de['rot_x_deg'], de['rot_y_deg'], de['rot_z_deg']], 2), 'centre', np.round(de['tr_centre'], 2),
              'rms', round(st['rms_mm'], 3), 'med', round(st['med_abs_mm'], 3))
    json.dump(out, open(f'{V2}/out/place_{tag}.json', 'w'), indent=1, default=float)
