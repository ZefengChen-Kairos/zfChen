"""Wall-profile test: depth points of each camera in the tag (pitcher) frame of a tags-only trajectory; signed
point-to-plane residual to the placed CAD (outward normal; + = observed surface outside the CAD) binned by height
(CAD z) and by azimuth sector; per camera.  Body-fixed shape (or a depth bias common to both sensors) -> same
residual(z) in both cameras; an extrinsic error of one camera -> that camera only.
    python3 profile_check.py <traj.pkl> <depth tag> <z0> <az> -> ../out/profile_<trajtag>.json, ../img/profile.jpg"""
import sys, json, pickle
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2, POURS
from spline import Traj
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

trf, dtag, z0, az = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
T = pickle.load(open(trf, 'rb'))
cad = gcm.CAD(z0_mm=z0, az_deg=az)
zb = np.arange(0, 125, 6.0)
res = {}
fig, axs = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
for ip, p in enumerate(POURS):
    tr = T['pours'][p]
    traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
    Z = np.load(f'{V2}/data/depth_{p}_{dtag}.npz')
    ft = Z['ft'] + np.where(Z['fcam'] == 1, tr['nuis']['c1'], 0.0)
    R, pp = traj.eval(ft)
    pf = Z['pf']
    y = np.einsum('nji,nj->ni', R[pf], Z['X'].astype(float) - pp[pf])
    d, j = cad.kd.query(y, distance_upper_bound=0.01)
    ok = np.isfinite(d); j = np.where(ok, j, 0)
    r = np.einsum('ij,ij->i', y - cad.Sp[j], cad.Np[j])
    zc = (y[:, 2] - z0 * 1e-3) * 1e3                                    # CAD height [mm]
    phi = np.degrees(np.arctan2(y[:, 1], y[:, 0]))
    rel = (phi - az + 180) % 360 - 180                                  # azimuth from the spout
    body = ok & (np.abs(r) < 0.006) & (np.abs(rel) > 50) & (np.abs(rel) < 130)    # side walls (no spout, no handle)
    out = {}
    for c in (0, 1):
        m = body & (Z['fcam'][pf] == c)
        prof = []
        for a, b in zip(zb[:-1], zb[1:]):
            mm = m & (zc >= a) & (zc < b)
            prof.append([float((a + b) / 2), float(np.median(r[mm]) * 1e3) if mm.sum() > 200 else None, int(mm.sum())])
        pr = np.array([[z, v] for z, v, n in prof if v is not None])
        slope = float(np.polyfit(pr[:, 0], pr[:, 1], 1)[0]) if len(pr) > 3 else None
        out[f'cam{c}'] = dict(profile=prof, slope_mm_per_100mm=slope * 100 if slope is not None else None,
                              lean_equiv_deg=float(np.degrees(np.arctan(-slope))) if slope is not None else None,
                              side_sector=dict(med_rel_az=float(np.median(rel[m]))))
        axs[ip].plot(pr[:, 1], pr[:, 0], 'o-', label=f'cam{c} (median sector {np.median(rel[m]):.0f} deg from spout)')
    res[p] = out
    axs[ip].axvline(0, c='k', lw=0.5); axs[ip].set_title(p); axs[ip].set_xlabel('signed residual to CAD [mm] (+ outside)')
    axs[ip].legend(fontsize=7)
    print(p, {c: (round(v['slope_mm_per_100mm'], 3), round(v['lean_equiv_deg'], 2), round(v['side_sector']['med_rel_az'], 0)) for c, v in out.items()}, flush=True)
axs[0].set_ylabel('CAD height z [mm]')
fig.tight_layout(); fig.savefig(f'{V2}/img/profile.jpg', dpi=70)
json.dump(res, open(f'{V2}/out/profile_{dtag}.json', 'w'), indent=1)
