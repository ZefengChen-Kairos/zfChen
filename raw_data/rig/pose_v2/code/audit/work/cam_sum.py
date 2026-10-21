import json, sys
import numpy as np
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
cam = int(sys.argv[1]) if len(sys.argv) > 1 else 2
suf = sys.argv[2] if len(sys.argv) > 2 else ''
res = {}
for p in ['tulip', 'leaf', 'heart', 'push_heart', 'swan']:
    try:
        R = json.load(open(f'{V2}/audit/out/cam{cam}_{p}{suf}.json'))
    except FileNotFoundError:
        continue
    srcs = [k for k in R[0] if k not in ('step', 't')]
    out = {}
    for s in srcs:
        def col(k):
            return np.array([r[s].get(k, np.nan) if s in r else np.nan for r in R], float)
        iou, res_, sil, edge, sils, edges = col('iou'), col('res_absmed'), col('sil_absmed'), col('edge_absmed'), col('sil_med'), col('edge_med')
        def spread(x):                       # calibration-robust: frame-to-frame scatter of a signed metric
            x = x[np.isfinite(x)]
            return float(np.median(np.abs(x - np.median(x)))) if len(x) > 5 else np.nan
        def jit(x):
            x = x[np.isfinite(x)]
            return float(np.median(np.abs(np.diff(x)))) if len(x) > 5 else np.nan
        rs = col('res_med')
        out[s] = dict(spread_res_signed=spread(rs), spread_sil=spread(sils), spread_edge=spread(edges), jit_res=jit(rs), jit_sil=jit(sils), n=int(np.isfinite(iou).sum()), iou_med=np.nanmedian(iou), iou_p10=np.nanpercentile(iou, 10), res_med=np.nanmedian(res_), res_p90=np.nanpercentile(res_, 90),
                      sil_absmed=np.nanmedian(sil), sil_signed=np.nanmedian(sils), edge_absmed=np.nanmedian(edge), edge_signed=np.nanmedian(edges))
    base = 'v2out' if 'v2out' in srcs else srcs[0]
    for s in srcs:
        if s == base: continue
        a = np.array([r[base].get('iou', np.nan) - r[s].get('iou', np.nan) for r in R]); b = np.array([r[s].get('res_absmed', np.nan) - r[base].get('res_absmed', np.nan) for r in R])
        fa, fb = np.isfinite(a), np.isfinite(b)
        out[f'{base}_better_than_{s}'] = dict(iou_frac=float(np.mean(a[fa] > 0)), iou_dmed=float(np.nanmedian(a)), res_frac=float(np.mean(b[fb] > 0)), res_dmed=float(np.nanmedian(b)), n_res=int(fb.sum()))
    res[p] = out
    print(p, len(R))
    for s in srcs:
        o = out[s]; print(f"   {s:8s} n {o['n']:3d} IoU {o['iou_med']:.3f} [p10 {o['iou_p10']:.3f}]  depth|r| {o['res_med']:.2f} [p90 {o['res_p90']:.2f}] mm  sil {o['sil_absmed']:.1f} (signed {o['sil_signed']:+.1f}) px  edge {o['edge_absmed']:.1f} (signed {o['edge_signed']:+.1f}) px | scatter: depth {o['spread_res_signed']:.2f} mm sil {o['spread_sil']:.2f} edge {o['spread_edge']:.2f} px; jitter depth {o['jit_res']:.2f} sil {o['jit_sil']:.2f}")
    for k, v in out.items():
        if 'better' in k: print('  ', k, {a: round(b, 3) for a, b in v.items()})
json.dump(res, open(f'{V2}/audit/out/cam{cam}{suf}_summary.json', 'w'), indent=1, default=float)
