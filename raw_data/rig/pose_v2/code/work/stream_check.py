"""Stream-origin check (independent of every pose): the audit's stream centre lines (pose_audit/geom streamtop.py,
traced upwards from the measured stream foot with the 3-camera top-hat cue, stored per frame as 'path') are compared
with the spout tip of a pose source.  Frame set = the audit's rule (trusted stream_foot frame, >= 10 slices, the two
top-hat cues agree < 2 mm at the tip height), applied once with the CURRENT tip so that both sources are scored on the
same frames.  For each source: the upper part of the line (heights max(2, h_tip-18) .. h_tip-3 mm, >= 5 slices, mean cue
> 0.3) is fitted with straight lines x(z), y(z) and extrapolated to the source's tip height; horizontal distance to
the tip, components along / across the horizontal spout heading.
    python3 stream_check.py <v2.pkl> -> ../out/stream_check.json"""
import sys, json, pickle
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/work')
import gcm
from pv2lib import V2, POURS
from heldout_cam import v2_source
G = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_audit/geom/out'


def compare(path, tip, R, zs, spout=np.array([0, 1.0, 0])):
    A = np.asarray(path)
    h = (zs - A[:, 0]) * 1e3
    h_tip = (zs - tip[2]) * 1e3
    seg = (h >= max(2.0, h_tip - 18.0)) & (h <= h_tip - 3.0)
    if seg.sum() < 5 or A[seg, 3].mean() <= 0.3:
        return None
    up = A[seg]
    fx = np.polyfit(up[:, 0], up[:, 1], 1); fy = np.polyfit(up[:, 0], up[:, 2], 1)
    ext = np.array([np.polyval(fx, tip[2]), np.polyval(fy, tip[2]), tip[2]])
    sd = R @ spout; sd[2] = 0; sd /= np.linalg.norm(sd)
    ac = np.array([-sd[1], sd[0], 0.0])
    d = ext - tip
    return dict(h=float(np.hypot(d[0], d[1]) * 1e3), along=float(d @ sd * 1e3), across=float(d @ ac * 1e3), h_tip=float(h_tip))


def pct(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    return dict(n=int(len(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90))) if len(x) else dict(n=0)


if __name__ == '__main__':
    V = pickle.load(open(sys.argv[1], 'rb'))
    tagname = sys.argv[2] if len(sys.argv) > 2 else ''
    res = {}
    for p in POURS:
        P = gcm.Pour(p)
        pose, cad, rng = v2_source(p, V)
        tip_v2 = cad.tip_pf
        a_ = np.radians(V['place']['az']); spout_v2 = np.array(V['place']['extra'])[:3, :3] @ np.array([np.cos(a_), np.sin(a_), 0.0])
        S1 = {s['fi']: s for s in json.load(open(f'{G}/stream_{p}_v2.json'.replace('_v2.json', '.json')))}
        S2 = {s['fi']: s for s in json.load(open(f'{G}/stream_{p}_v2.json'))}
        base = [s for s in S1.values() if s.get('trusted') and 'ext' in s and s['tip_h_mm'] >= 18 and s.get('seg_n', 0) >= 10]
        used = [s for s in base if S2.get(s['fi']) is not None and 'ext' in S2[s['fi']]
                and np.linalg.norm(np.subtract(s['ext'], S2[s['fi']]['ext'])) * 1e3 < 2.0]
        rows = []
        for s in used:
            t = s['t']; zs = s['zs']
            Rc, pc = P.track(t); tip_c = Rc @ P.tip_p + pc
            Rv, pv = pose(t); tip_v = Rv @ tip_v2 + pv
            rows.append(dict(fi=s['fi'], t=t, cur=compare(s['path'], tip_c, Rc, zs), v2=compare(s['path'], tip_v, Rv, zs, spout_v2),
                             tip_diff_mm=float(np.linalg.norm(tip_v - tip_c) * 1e3)))
        R = dict(n_frames=len(used))
        for src in ('cur', 'v2'):
            R[src] = dict(horiz_mm=pct([r[src]['h'] for r in rows if r[src]]), along_mm=pct([r[src]['along'] for r in rows if r[src]]),
                          across_abs_mm=pct([abs(r[src]['across']) for r in rows if r[src]]),
                          along_signed_median=float(np.median([r[src]['along'] for r in rows if r[src]])) if any(r[src] for r in rows) else None,
                          across_signed_median=float(np.median([r[src]['across'] for r in rows if r[src]])) if any(r[src] for r in rows) else None)
        R['rows'] = rows
        res[p] = R
        print(p, 'frames', len(used), 'cur', R['cur']['horiz_mm'], 'v2', R['v2']['horiz_mm'], 'along/across med cur', R['cur']['along_signed_median'],
              R['cur']['across_signed_median'], 'v2', R['v2']['along_signed_median'], R['v2']['across_signed_median'], flush=True)
    json.dump(res, open(f'{V2}/out/stream_check{tagname}.json', 'w'), indent=1, default=float)
