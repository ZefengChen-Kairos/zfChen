"""A3: held-out camera check from the DELIVERED files (independent of the builder's heldout_cam.py / spline pickles).
Sources: 'v2out'  = out/<p>_pose_v2.npz (10 ms grid, Slerp/linear), CAD kona700 at z0 -29.0 mm, spout az 90 (the
                    documented output-frame placement), tip_p from the npz;
         'cur'    = tip.npz minus fuse_occ.pose_inliers (gcm.Pour.track), CAD z0 -28.5, az 90 (3DGS prior);
         'cur_pl' = the same current track with its own best constant CAD placement from cams 0/1 depth (cur_place.json).
Every frame of camera <cam> whose nominal time lies inside ALL sources' ranges.
    python3 cam_eval.py <pour> <cam> [every] -> ../out/cam<cam>_<pour>.json"""
import sys, json, time
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
p, cam = sys.argv[1], int(sys.argv[2]); every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
P = gcm.Pour(p)
Z = np.load(f'{V2}/out/{p}_pose_v2.npz')
trv = gcm.Track(Z['t'], Z['poses'])
cad_v = gcm.CAD(z0_mm=-29.0, az_deg=90.0)
cad_c = gcm.CAD(z0_mm=-28.5, az_deg=90.0)
CP = json.load(open(f'{V2}/audit/out/cur_place.json'))
cad_cp = gcm.CAD(z0_mm=-28.5, az_deg=90.0, extra=np.array(CP[p]['D']))
srcs = dict(v2out=(trv, cad_v, np.array(Z['tip_p'], float)), cur=(P.track, cad_c, P.tip_p.copy()), cur_pl=(P.track, cad_cp, P.tip_p.copy()))
lo = max(Z['t'][0], P.track.t[0]); hi = min(Z['t'][-1], P.track.t[-1])
out = []; t0 = time.time()
for n, (k, ts) in enumerate(zip(P.tk.steps[cam], P.tk.ts[cam] - P.tk.t0)):
    if n % every or ts < lo or ts > hi:
        continue
    im = P.tk.color(cam, k); dep = P.tk.depth(cam, k).astype(np.float64) * 1e-3
    rec = dict(step=int(k), t=float(ts))
    for name, (tr, cad, tip_p) in srcs.items():
        R, pp = tr(ts)
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
    if len(out) % 25 == 0:
        print(p, cam, len(out), f'{time.time() - t0:.0f}s', {s: round(rec[s].get('iou', np.nan), 3) for s in srcs}, flush=True)
json.dump(out, open(f'{V2}/audit/out/cam{cam}_{p}.json', 'w'), default=float)
print('done', p, cam, len(out), f'{time.time() - t0:.0f}s', flush=True)
