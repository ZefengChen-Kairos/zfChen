"""A7: stream-origin check from the delivered npz (independent re-run): the audit's stream centre lines (frame rule as
in pose_v2/work/stream_check.py) vs the spout tip of: v2out (npz poses + npz tip_p), cur (tip.npz tip (0,53,93) mm),
cur_kona (current poses + production TIP_KONA), cur_pl (current poses + CAD tip under its own constant placement)."""
import sys, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
sys.path.insert(0, '/home/user/zfChen')
import gcm
from stream_check import compare, pct
from latte_imex.rig.full_pour import TIP_KONA
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
G = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/out'
CP = json.load(open(f'{V2}/audit/out/cur_place.json'))
res = {}
for p in ['tulip', 'leaf', 'heart', 'push_heart', 'swan']:
    P = gcm.Pour(p)
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz'); trv = gcm.Track(Z['t'], Z['poses'])
    cadp = gcm.CAD(z0_mm=-28.5, az_deg=90.0, extra=np.array(CP[p]['D']))
    Dp = np.array(CP[p]['D'])
    tips = dict(cur=P.tip_p, cur_kona=np.asarray(TIP_KONA, float), cur_pl=cadp.tip_pf)
    spouts = dict(cur=np.array([0, 1.0, 0]), cur_kona=np.array([0, 1.0, 0]), cur_pl=Dp[:3, :3] @ np.array([0, 1.0, 0]))
    S1 = {s['fi']: s for s in json.load(open(f'{G}/stream_{p}.json'))}
    S2 = {s['fi']: s for s in json.load(open(f'{G}/stream_{p}_v2.json'))}
    base = [s for s in S1.values() if s.get('trusted') and 'ext' in s and s['tip_h_mm'] >= 18 and s.get('seg_n', 0) >= 10]
    used = [s for s in base if S2.get(s['fi']) is not None and 'ext' in S2[s['fi']] and np.linalg.norm(np.subtract(s['ext'], S2[s['fi']]['ext'])) * 1e3 < 2.0]
    rows = []
    for s in used:
        t, zs = s['t'], s['zs']
        r = {}
        Rv, pv = trv(t); r['v2out'] = compare(s['path'], Rv @ Z['tip_p'] + pv, Rv, zs)
        Rc, pc = P.track(t)
        for k in tips:
            r[k] = compare(s['path'], Rc @ tips[k] + pc, Rc, zs, spouts[k])
        rows.append(r)
    R = dict(n_frames=len(used))
    for k in ('v2out',) + tuple(tips):
        R[k] = dict(horiz=pct([x[k]['h'] for x in rows if x[k]]), along_med=float(np.median([x[k]['along'] for x in rows if x[k]])),
                    across_med=float(np.median([x[k]['across'] for x in rows if x[k]])))
    res[p] = R
    print(p, R['n_frames'], ' | '.join(f"{k}: {R[k]['horiz']['median']:.1f} [{R[k]['horiz']['p90']:.1f}] along {R[k]['along_med']:+.1f} across {R[k]['across_med']:+.1f} (n {R[k]['horiz']['n']})" for k in ('v2out',) + tuple(tips)), flush=True)
json.dump(res, open(f'{V2}/audit/out/stream_audit.json', 'w'), indent=1)
