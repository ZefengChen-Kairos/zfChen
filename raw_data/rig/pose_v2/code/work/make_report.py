"""Aggregate every pose-v2 result into ../out/summary.json and draw the summary figures.
    python3 make_report.py"""
import sys, json, pickle, glob, os
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import V2, POURS, SCR, Layout
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

O = f'{V2}/out'
H = f'{SCR}/pose_audit/holdout'


def q(v, qs=(50, 90)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return [round(float(np.percentile(v, x)), 3) for x in qs] + [int(len(v))] if len(v) else None


S = {}
# ---- step 1: layout
L1 = {}
for p in POURS:
    for v in ('flat', 'flat_cam0', 'flat_sync5', 'cyl426', 'cyl437', 'flat_s', 'flat_u'):
        f = f'{O}/layout_{p}_{v}_edge.pkl'
        if os.path.exists(f):
            o = pickle.load(open(f, 'rb'))
            L1[f'{p}/{v}'] = dict(rms_px=round(o['rms'], 4), status=o['status'], iters=o['iters'], tag_mm=round(o['layout']['tag_mm'], 3),
                                  layout=Layout.from_dict(o['layout']).pretty())
S['step1_layouts'] = L1
S['pair_check'] = json.load(open(f'{O}/pair_check.json'))
for name in ('ct_tags', 'ct_tags_lean', 'ct_tags_lean_e1'):
    T = pickle.load(open(f'{O}/{name}.pkl', 'rb'))
    rec = dict(layout=Layout.from_dict(T['layout']).pretty(), sig_t=T['sig_t'], pours={})
    for p, tr in T['pours'].items():
        r = tr['tag_res']
        rec['pours'][p] = dict(c1_ms=round(tr['nuis']['c1'] * 1e3, 2),
                               rms_px={c: round(float(np.sqrt(np.mean(r[tr['tag_cam'] == c] ** 2))), 3) for c in (0, 1)})
    S[f'tags_only_{name}'] = rec
S['placement'] = json.load(open(f'{O}/placement_v2k.json'))
S['placement_detail'] = json.load(open(f'{O}/placement_detail_t1.json'))
S['dyn_stats'] = json.load(open(f'{O}/dyn_stats.json'))
for f in ('depth_xcam_t0.json', 'profile_t0.json'):
    if os.path.exists(f'{O}/{f}'):
        S[f.replace('.json', '')] = json.load(open(f'{O}/{f}'))
# ---- full fits
FF = {}
for p in POURS:
    F = pickle.load(open(f'{O}/full_{p}_v2k.pkl', 'rb'))
    rt, rd = F['tag_res'], F['dep_res']
    fc = F['dep_fcam'][F['dep_pf']]
    rec = dict(c1_ms=round(F['nuis']['c1'] * 1e3, 2), neff=round(F['neff'], 1), cost={k: round(v, 1) for k, v in F['cost'].items()})
    for c in (0, 1):
        m = F['tag_cam'] == c
        md = (fc == c) & F['dep_assoc']
        rec[f'cam{c}'] = dict(tag_rms_px=round(float(np.sqrt(np.mean(rt[m] ** 2))), 3), tag_med_px=round(float(np.median(np.abs(rt[m]))), 3),
                              depth_med_abs_mm=round(float(np.median(np.abs(rd[md]))) * 1e3, 3),
                              depth_signed_med_mm=round(float(np.median(rd[md & (np.abs(rd) < 0.005)])) * 1e3, 3),
                              depth_inlier5mm=round(float((np.abs(rd[md]) < 0.005).mean()), 3))
    FF[p] = rec
S['full_fit'] = FF
# ---- held-out camera
HO = {}
for p in POURS:
    for cam in (2, 0, 1):
        f = f'{O}/heldout_{p}_cam{cam}.json'
        if not os.path.exists(f):
            continue
        R = json.load(open(f))
        rec = {}
        for src in ('v2', 'cur', 'v2t', 'v2tn'):
            rows = [r[src] for r in R if src in r and r[src].get('n_body', 0) >= 300]
            rec[src] = {k: q([r.get(k) for r in rows]) for k in ('iou', 'res_absmed', 'res_med', 'sil_absmed', 'sil_med', 'edge_absmed', 'edge_med', 'frac_unknown')}
            rec[src]['n_frames'] = len(rows)
        both = [r for r in R if 'v2' in r and 'cur' in r and r['v2'].get('iou') is not None and r['cur'].get('iou') is not None]
        rec['iou_v2_better_frac'] = round(float(np.mean([r['v2']['iou'] > r['cur']['iou'] for r in both])), 3) if both else None
        rec['res_v2_better_frac'] = round(float(np.mean([r['v2'].get('res_absmed', 99) < r['cur'].get('res_absmed', 99) for r in both])), 3) if both else None
        HO[f'{p}/cam{cam}'] = rec
S['heldout'] = HO
for f in ('stream_check.json', 'eval_misc.json', 'downstream.json'):
    if os.path.exists(f'{O}/{f}'):
        d = json.load(open(f'{O}/{f}'))
        if f == 'stream_check.json':
            d = {p: {k: v for k, v in r.items() if k != 'rows'} for p, r in d.items()}
        S[f.replace('.json', '')] = d
# reference numbers of the current chain from the hold-out audit
A = json.load(open(f'{H}/summary.json'))
S['audit_current'] = {p: dict(delivered=A[p]['delivered_layout'], converged=A[p]['converged_layout'], coverage=A[p]['coverage'])
                      for p in POURS if p in A}
json.dump(S, open(f'{O}/summary.json', 'w'), indent=1, default=float)
print('summary written')

# ---- figure: time series v2 vs current
fig, axs = plt.subplots(5, 3, figsize=(22, 20))
for ip, p in enumerate(POURS):
    Z = np.load(f'{O}/{p}_pose_v2.npz')
    C = np.load(f'{SCR}/pours/{p}/tip.npz')
    ax = axs[ip, 0]
    ax.plot(C['t'], C['tilt'], '.', ms=3, c='C3', label='current tip.npz samples')
    ax.plot(Z['t'], Z['tilt'], '-', lw=1, c='C2', label='v2 (10 ms)')
    ax.fill_between(Z['t'], Z['tilt'] - 2 * Z['sd_tilt_deg'], Z['tilt'] + 2 * Z['sd_tilt_deg'], color='C2', alpha=0.25, label='v2 formal 2 sd (pour tilt)')
    ax.set_ylabel('axis tilt [deg]'); ax.set_title(f'{p}: axis tilt from vertical'); ax.legend(fontsize=7)
    for j, lab in ((0, 'x'), (1, 'y'), (2, 'z')):
        axs[ip, 1].plot(C['t'], C['tip_board'][:, j] * 1e3, '.', ms=2, c=f'C{j}')
        axs[ip, 1].plot(Z['t'], Z['tip_board'][:, j] * 1e3, '-', lw=1, c=f'C{j}', label=f'v2 tip {lab}')
    axs[ip, 1].set_ylabel('spout tip, board frame [mm]'); axs[ip, 1].set_title('tip (dots: current samples, lines: v2)'); axs[ip, 1].legend(fontsize=7)
    axs[ip, 2].plot(Z['t'], Z['sd_tilt_deg'], label='formal sd pour tilt [deg]'); axs[ip, 2].plot(Z['t'], Z['sd_spin_deg'], label='formal sd spin about axis [deg]')
    axs[ip, 2].set_yscale('log'); axs[ip, 2].legend(fontsize=7); axs[ip, 2].set_title('v2 formal (noise-only) uncertainty')
for a in axs[-1]:
    a.set_xlabel('t [s]')
fig.tight_layout(); fig.savefig(f'{V2}/img/timeseries.jpg', dpi=60); plt.close(fig)
# ---- figure: held-out camera 2 metrics
if HO:
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
    for k, (key, lab) in enumerate((('iou', 'silhouette IoU'), ('res_absmed', 'depth |residual| median [mm]'), ('edge_absmed', 'colour-edge |offset| median [px]'))):
        for ip, p in enumerate(POURS):
            f = f'{O}/heldout_{p}_cam2.json'
            if not os.path.exists(f):
                continue
            R = json.load(open(f))
            for js, src in enumerate(('cur', 'v2')):
                v = [r[src].get(key) for r in R if src in r and r[src].get(key) is not None]
                if v:
                    axs[k].boxplot(v, positions=[ip * 3 + js], widths=0.8, showfliers=False, patch_artist=True,
                                   boxprops=dict(facecolor='C3' if src == 'cur' else 'C2', alpha=0.6))
        axs[k].set_xticks([ip * 3 + 0.5 for ip in range(5)]); axs[k].set_xticklabels(POURS); axs[k].set_title(f'camera 2 (held out): {lab}\nred current, green v2')
    fig.tight_layout(); fig.savefig(f'{V2}/img/heldout_cam2.jpg', dpi=70); plt.close(fig)
print('figures written')
