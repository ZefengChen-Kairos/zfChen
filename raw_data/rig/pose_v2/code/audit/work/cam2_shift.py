"""A10: held-out timing / dynamics test: camera-2 frames during fast motion (v2 |omega| > 20 deg/s); render each source at
t_frame + shift for shifts -60..+60 ms; IoU and depth-silhouette |offset| vs shift.  A track whose dynamics and timing
are right peaks near shift 0 (up to camera 2's own unknown clock offset, common to all sources)."""
import sys, json
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
CP = json.load(open(f'{V2}/audit/out/cur_place.json'))
SH = np.arange(-0.06, 0.0601, 0.02)
out = {}
for p in sys.argv[1:]:
    P = gcm.Pour(p); Z = np.load(f'{V2}/out/{p}_pose_v2.npz'); trv = gcm.Track(Z['t'], Z['poses'])
    S = np.load(f'{V2}/audit/out/samples_{p}.npz')
    cad_v = gcm.CAD(z0_mm=-29.0, az_deg=90.0); cad_cp = gcm.CAD(z0_mm=-28.5, az_deg=90.0, extra=np.array(CP[p]['D']))
    recs = []
    for k, ts in zip(P.tk.steps[2], P.tk.ts[2] - P.tk.t0):
        if ts < Z['t'][0] + 0.07 or ts > Z['t'][-1] - 0.07: continue
        if np.interp(ts, S['tg'], S['w_g']) < 20: continue
        im = P.tk.color(2, k); dep = P.tk.depth(2, k).astype(np.float64) * 1e-3
        rec = dict(t=float(ts), w=float(np.interp(ts, S['tg'], S['w_g'])))
        for name, tr, cad, tip in (('v2out', trv, cad_v, Z['tip_p']), ('cur_pl', P.track, cad_cp, P.tip_p.copy())):
            vals = []
            for sh in SH:
                R, pp = tr(ts + sh); P.tip_p = np.asarray(tip, float)
                O = gcm.observe(P, cad, 2, k, ts, R, pp, im=im, dep=dep)
                m = gcm.metrics(P, 2, O); so, _ = gcm.silhouette_offsets(O)
                vals.append([m.get('iou', np.nan), float(np.median(np.abs(so))) if len(so) >= 20 else np.nan])
            rec[name] = vals
        recs.append(rec)
    out[p] = recs
    print(p, 'fast cam-2 frames', len(recs), flush=True)
    for name in ('v2out', 'cur_pl'):
        A = np.array([r[name] for r in recs])           # (n, shifts, 2)
        rel = A[:, :, 0] - A[:, 3:4, 0]
        print('  ', name, 'median IoU gain vs shift 0:', {f'{s*1e3:+.0f}': round(float(np.nanmedian(rel[:, i])), 4) for i, s in enumerate(SH)},
              ' |sil| med:', {f'{s*1e3:+.0f}': round(float(np.nanmedian(A[:, i, 1])), 2) for i, s in enumerate(SH)}, flush=True)
json.dump(out, open(f'{V2}/audit/out/cam2_shift_{"_".join(sys.argv[1:])}.json', 'w'))
