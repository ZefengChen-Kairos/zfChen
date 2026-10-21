"""Hold-out camera check (camera 2 never enters the v2 fit): for every frame of camera c, render the placed CAD with
a pose source at the frame's own (nominal) timestamp and compare with the observed depth / colour (audit gcm.metrics,
silhouette_offsets, edge_offsets).  Pose sources: 'v2' (spline, v2 CAD placement) and 'cur' (tip.npz minus
fuse_occ.pose_inliers, Slerp = the downstream pose; CAD placement z0 -28.5 mm, spout az 90 deg as the 3DGS prior).
    python3 heldout_cam.py <pour> <v2.pkl> <cam> [every] -> ../out/heldout_<pour>_cam<c>.json"""
import sys, json, pickle, time
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2
from spline import Traj


def v2_source(p, V):
    tr = V['pours'][p]
    traj = Traj(tr['T0'], tr['dt'], tr['R'], tr['P'])
    cad = gcm.CAD(z0_mm=V['place']['z0'], az_deg=V['place']['az'])
    if V['place'].get('extra') is not None:
        cad.set_place(V['place']['z0'], V['place']['az'], np.array(V['place']['extra']))

    lay = tr.get('layout', {})
    e1 = np.array(lay.get('ext1', np.zeros(6)), float); cen = np.array(lay.get('ext_cen', np.zeros(3)), float)
    from scipy.spatial.transform import Rotation as _R
    Re = _R.from_rotvec(e1[:3]).as_matrix()

    def pose(t, cam=None):
        R, pp = traj.eval(np.array([t]))
        R, pp = R[0], pp[0]
        if cam == 1 and np.any(e1):                       # camera 1 sees the fitted (corrected) world through C
            R, pp = Re @ R, Re @ (pp - cen) + cen + e1[3:]
        return R, pp
    return pose, cad, (traj.T0, traj.t_end)


def run(p, V, cam, every=1, sources=('v2', 'cur')):
    P = gcm.Pour(p)
    srcs = {}
    if 'v2' in sources:
        pose, cad, rng = v2_source(p, V)
        srcs['v2'] = (pose, cad, cad.tip_pf.copy(), rng)
    if 'v2t' in sources:                                     # tags-only CT variant (same layout / placement, no depth)
        Vt = pickle.load(open(f'{V2}/out/v2t_all.pkl', 'rb'))
        pose, cad, rng = v2_source(p, Vt)
        srcs['v2t'] = (pose, cad, cad.tip_pf.copy(), rng)
    if 'v2tn' in sources:                                    # tags-only, common lean, NO camera-1 extrinsic correction
        Vn = pickle.load(open(f'{V2}/out/v2tn_all.pkl', 'rb'))
        pose, cad, rng = v2_source(p, Vn)
        srcs['v2tn'] = (pose, cad, cad.tip_pf.copy(), rng)
    if 'cur' in sources:
        cad0 = gcm.CAD(z0_mm=-28.5, az_deg=90.0)
        srcs['cur'] = (P.track, cad0, P.tip_p.copy(), (P.track.t[0], P.track.t[-1]))
    out = []
    t0 = time.time()
    for n, (k, ts) in enumerate(zip(P.tk.steps[cam], P.tk.ts[cam] - P.tk.t0)):
        if n % every:
            continue
        im = P.tk.color(cam, k); dep = P.tk.depth(cam, k).astype(np.float64) * 1e-3
        rec = dict(step=int(k), t=float(ts))
        for name, (pose, cad, tip_p, rng) in srcs.items():
            if ts < rng[0] - 0.02 or ts > rng[1] + 0.02:
                continue
            R, pp = pose(ts, cam) if name != 'cur' else pose(ts)
            P.tip_p = tip_p
            O = gcm.observe(P, cad, cam, k, ts, R, pp, im=im, dep=dep)
            m = gcm.metrics(P, cam, O)
            if m.get('n_body', 0) >= 300:
                so, nc = gcm.silhouette_offsets(O)
                if len(so) >= 20:
                    m['sil_med'] = float(np.median(so)); m['sil_absmed'] = float(np.median(np.abs(so)))
                e = gcm.edge_offsets(O)
                if len(e) >= 20:
                    m['edge_med'] = float(np.median(e)); m['edge_absmed'] = float(np.median(np.abs(e)))
            rec[name] = m
        out.append(rec)
        if len(out) % 40 == 0:
            print(p, cam, len(out), f'{time.time() - t0:.0f}s', {s: round(rec[s].get('iou', np.nan), 3) for s in srcs if s in rec}, flush=True)
    return out


if __name__ == '__main__':
    p, vf, cam = sys.argv[1], sys.argv[2], int(sys.argv[3])
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    tagname = sys.argv[5] if len(sys.argv) > 5 else ''
    sources = tuple(sys.argv[6].split(',')) if len(sys.argv) > 6 else ('v2', 'cur')
    V = pickle.load(open(vf, 'rb')) if 'v2' in sources else None
    out = run(p, V, cam, every, sources)
    json.dump(out, open(f'{V2}/out/heldout_{p}_cam{cam}{tagname}.json', 'w'), default=float)
    print('done', p, cam, len(out))
