"""Markdown tables for README.md from the result files (no hand transcription).
    python3 readme_tables.py > ../out/tables.md"""
import json, pickle, os, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import V2, POURS, SCR, Layout

O = f'{V2}/out'
J = lambda f: json.load(open(f'{O}/{f}'))
f1 = lambda x: '-' if x is None else f'{x:.1f}'
f2 = lambda x: '-' if x is None else f'{x:.2f}'
f3 = lambda x: '-' if x is None else f'{x:.3f}'


def med_p(rows, key, q=(50, 90)):
    v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], float)
    return [float(np.percentile(v, x)) for x in q] if len(v) else [None] * len(q)


print('### Corner estimator (same-frame two-tag rigid fit, camera 0 tags 4+5, rms px)\n')
pc = J('pair_check.json')
print('| pour | integer (tags.pkl now) | aruco contour | edge-line subpixel (v2) |\n|---|---|---|---|')
for p in POURS:
    print(f"| {p} | " + ' | '.join(f2(pc.get(f'{p}/{c}/45', {}).get('rms_px')) for c in ('none', 'contour', 'edge')) + ' |')

print('\n### Step 1: per-sample bundle adjustment, converged (corner rms px; layout tag 5 phi / z, tag 1 phi / z)\n')
print('| pour | flat a=42.60 (12 ms sync) | flat, cam-0 only | flat, 5 ms sync | cylinder r=42.6 | cylinder r=43.7 | flat + free scale (tag mm) | iters / status (flat) |\n|---|---|---|---|---|---|---|---|')
for p in POURS:
    row = []
    for v in ('flat', 'flat_cam0', 'flat_sync5', 'cyl426', 'cyl437', 'flat_s'):
        o = pickle.load(open(f'{O}/layout_{p}_{v}_edge.pkl', 'rb'))
        s = f2(o['rms'])
        if v == 'flat_s':
            s += f" ({o['layout']['tag_mm']:.2f})"
        if v in ('flat', 'flat_cam0'):
            L = Layout.from_dict(o['layout']).pretty()
            s += f" [t5 {L[5][0]:.2f} deg / {L[5][1]:.2f} mm" + (f"; t1 {L[1][0]:.1f} / {L[1][1]:.1f}]" if 1 in L else ']')
        row.append(s)
    o = pickle.load(open(f'{O}/layout_{p}_flat_edge.pkl', 'rb'))
    print(f"| {p} | " + ' | '.join(row) + f" | {o['iters']} / {o['status']} |")

print('\n### Continuous-time tags-only fits (joint over the five pours; tag rms px cam0 / cam1)\n')
print('| model | ' + ' | '.join(POURS) + ' | shared parameters |\n|---|---|---|---|---|---|---|')
for name, lab in (('ct_tags', 'flat layout + c1'), ('ct_tags_lean', '+ common lean'), ('ct_tags_lean_e1', '+ camera-1 extrinsic')):
    T = pickle.load(open(f'{O}/{name}.pkl', 'rb'))
    cells = []
    for p in POURS:
        tr = T['pours'][p]; r = tr['tag_res']
        cells.append('/'.join(f2(float(np.sqrt(np.mean(r[tr['tag_cam'] == c] ** 2)))) for c in (0, 1)))
    Lp = Layout.from_dict(T['layout']).pretty()
    extra = f"lean {Lp[4][4]:.2f} deg" + (f", ext1 {Lp['ext1']}" if 'ext1' in Lp else '')
    print(f'| {lab} | ' + ' | '.join(cells) + f' | {extra} |')

print('\n### Final joint fit (v2): tags + depth + prior, cameras 0/1\n')
JJ = pickle.load(open(f'{O}/joint_v2k.pkl', 'rb'))
Lp = Layout.from_dict(JJ['layout']).pretty()
print(f"layout (phi deg, z mm, psi deg, u mm, tilt_x deg, tilt_y deg): {{k: v for k, v in Lp.items()}}\n".replace('{k: v for k, v in Lp.items()}', str({k: v for k, v in Lp.items()})))
print('| pour | c1 offset ms | tag rms px cam0 | tag rms px cam1 | depth med abs mm cam0 | cam1 | signed depth med mm cam0 | cam1 | N_eff |\n|---|---|---|---|---|---|---|---|---|')
for p in POURS:
    F = JJ['pours'][p]; rt, rd = F['tag_res'], F['dep_res']; fc = F['dep_fcam'][F['dep_pf']]
    c = []
    for cc in (0, 1):
        c.append(f2(float(np.sqrt(np.mean(rt[F['tag_cam'] == cc] ** 2)))))
    d = [f2(float(np.median(np.abs(rd[(fc == cc) & F['dep_assoc']]))) * 1e3) for cc in (0, 1)]
    s = [f2(float(np.median(rd[(fc == cc) & F['dep_assoc'] & (np.abs(rd) < 0.005)])) * 1e3) for cc in (0, 1)]
    print(f"| {p} | {F['nuis']['c1'] * 1e3:.1f} | {c[0]} | {c[1]} | {d[0]} | {d[1]} | {s[0]} | {s[1]} | {F['neff']:.0f} |")

print('\n### Held-out camera 2 (never used in the fit): median [p10 for IoU | p90 otherwise] per pour\n')
print('| pour | source | IoU | depth abs residual mm | silhouette offset px | colour-edge offset px | n frames |\n|---|---|---|---|---|---|---|')
for p in POURS:
    R = json.load(open(f'{O}/heldout_{p}_cam2.json'))
    for src, lab in (('cur', 'current'), ('v2', 'v2'), ('v2t', 'v2 tags only'), ('v2tn', 'tags only, no cam-1 corr.')):
        rows = [r[src] for r in R if src in r and r[src].get('n_body', 0) >= 300]
        if not rows:
            continue
        iou = med_p(rows, 'iou', (50, 10)); res = med_p(rows, 'res_absmed'); sil = med_p(rows, 'sil_absmed'); ed = med_p(rows, 'edge_absmed')
        print(f"| {p} | {lab} | {f3(iou[0])} [{f3(iou[1])}] | {f2(res[0])} [{f2(res[1])}] | {f1(sil[0])} [{f1(sil[1])}] | {f1(ed[0])} [{f1(ed[1])}] | {len(rows)} |")

print('\n### Held-out camera 2, calibration-robust: per-frame correction still needed after one constant camera-2 correction (median [p90])\n')
print('| pour | source | constant cam-2 correction rot deg / t mm | pouring-tilt corr deg | total rot corr deg | spout-tip corr mm | ICP failures |\n|---|---|---|---|---|---|---|')
for p in POURS:
    f = f'{O}/cam2icp_{p}.json'
    if not os.path.exists(f):
        continue
    C = json.load(open(f))
    for src, lab in (('cur', 'current'), ('v2', 'v2')):
        c = C[src]
        q = lambda k: f"{c[k][0]:.2f} [{c[k][1]:.2f}]" if c.get(k) else '-'
        print(f"| {p} | {lab} | {np.round(c['C2_rot_deg'], 2).tolist()} / {np.round(c['C2_t_mm'], 1).tolist()} | {q('abs_tilt_corr_deg_p50_p90')} | {q('rot_corr_deg_p50_p90')} | {q('tip_corr_mm_p50_p90')} | {c.get('n_icp_fail', 0)}/{c['n_frames']} |")

print('\n### In-sample cameras 0 / 1 (every 3rd frame): IoU median, depth abs residual median mm\n')
print('| pour | cam | current IoU | v2 IoU | current depth | v2 depth |\n|---|---|---|---|---|---|')
for p in POURS:
    for cam in (0, 1):
        R = json.load(open(f'{O}/heldout_{p}_cam{cam}.json'))
        v = {}
        for src in ('cur', 'v2'):
            rows = [r[src] for r in R if src in r and r[src].get('n_body', 0) >= 300]
            v[src] = (med_p(rows, 'iou')[0], med_p(rows, 'res_absmed')[0])
        print(f"| {p} | {cam} | {f3(v['cur'][0])} | {f3(v['v2'][0])} | {f2(v['cur'][1])} | {f2(v['v2'][1])} |")

if os.path.exists(f'{O}/stream_check.json'):
    S = J('stream_check.json')
    print('\n### Stream-origin check (independent 3-camera stream tracer; same frames for both): stream line at tip height vs spout tip, horizontal mm, median [p90]; along / across spout heading (signed medians)\n')
    print('| pour | frames | current | v2 | current along / across | v2 along / across |\n|---|---|---|---|---|---|')
    for p in POURS:
        s = S[p]
        g = lambda d: f"{d['median']:.1f} [{d['p90']:.1f}]" if d.get('n') else '-'
        print(f"| {p} | {s['n_frames']} | {g(s['cur']['horiz_mm'])} | {g(s['v2']['horiz_mm'])} | {f1(s['cur']['along_signed_median'])} / {f1(s['cur']['across_signed_median'])} | {f1(s['v2']['along_signed_median'])} / {f1(s['v2']['across_signed_median'])} |")

if os.path.exists(f'{O}/eval_misc.json'):
    E = J('eval_misc.json'); A = json.load(open(f'{SCR}/pose_audit/holdout/summary.json'))
    print('\n### Leave-one-camera-out (held-out camera tag corners predicted by the fit without that camera, px median [p90]) and cross-camera bias\n')
    print('| pour | v2: cam0 held out (all / joined <12 ms) | v2: cam1 held out (all / joined) | current (audit, joined samples): cam0 / cam1 | v2 cam0-only vs cam1-only tip bias mm (|median vector|) / p50 abs diff | current tip bias mm (audit) | v2 tilt bias deg | current tilt bias deg |\n|---|---|---|---|---|---|---|---|')
    for p in POURS:
        e = E[p]; lo = e.get('loco', {}); cx = e.get('cross_camera', {}); a = A[p]['delivered_layout']
        g = lambda v: f"{v[0]:.2f} [{v[1]:.2f}]" if v else '-'
        print(f"| {p} | {g(lo.get('cam0_heldout_px'))} / {g(lo.get('cam0_heldout_px_joined12ms'))} | {g(lo.get('cam1_heldout_px'))} / {g(lo.get('cam1_heldout_px_joined12ms'))} | "
              f"{a['heldout_reproj_px_cam0'][0]:.1f} [{a['heldout_reproj_px_cam0'][1]:.1f}] / {a['heldout_reproj_px_cam1'][0]:.1f} [{a['heldout_reproj_px_cam1'][1]:.1f}] | "
              f"{f1(cx.get('tip_bias_mm'))} / {f1((cx.get('tip_absdiff_p50_p90') or [None])[0])} | {a['tip_bias']:.1f} | {f2(cx.get('tilt_bias_deg'))} | {a['tilt_bias']:.2f} |")
    print('\n### Dynamics (5 ms grid; p50 / p95 / max)\n')
    print('| pour | source | body angular acceleration deg/s^2 | spout-tip acceleration m/s^2 | angular speed deg/s |\n|---|---|---|---|---|')
    for p in POURS:
        for src, lab in (('v2', 'v2'), ('cur_production_40ms', 'current, production 40 ms smoothing'), ('cur_raw', 'current, raw Slerp')):
            d = E[p]['dyn'][src]
            fm = lambda v: ' / '.join(f'{x:.0f}' if x > 20 else f'{x:.2f}' for x in v)
            print(f"| {p} | {lab} | {fm(d['alpha_deg_s2'])} | {fm(d['a_tip_ms2'])} | {fm(d['omega_deg_s'])} |")
    print('\n### Flips\n')
    print('| pour | current tip.npz samples > 10 deg from v2 (of n) | > 5 deg | current ba.pkl samples > 10 deg | v2 detections where another pose branch fits > 4x better |\n|---|---|---|---|---|')
    for p in POURS:
        f = E[p]['flips_current_vs_v2']; v = E[p]['v2_flip_suspects']
        print(f"| {p} | {f['tip_npz']['n_gt10deg']} / {f['tip_npz']['n']} | {f['tip_npz']['n_gt5deg']} | {f['ba_pkl']['n_gt10deg']} / {f['ba_pkl']['n']} | {v['n_alt_branch_fits_better']} / {v['n_detections']} |")

if os.path.exists(f'{O}/downstream.json'):
    D = J('downstream.json')
    print('\n### Downstream: production inlet chain (cup_poses -> ForwardPitcher kona700, fill at the cut), v2 vs current\n')
    print('| pour | flow rms diff mL/s | mean flow (current) mL/s | poured volume current / v2 mL | dV mL | on/off mismatch | landing diff mm p50 [p90] | landing mean shift mm (x, y) | axis tilt diff deg median / p90 abs |\n|---|---|---|---|---|---|---|---|---|')
    for p in POURS:
        d = D[p]
        print(f"| {p} | {d['dQ_rms']:.2f} | {d['Q_mean_cur']:.1f} | {d['V_cur_mL']:.1f} / {d['V_v2_mL']:.1f} | {d['dV_mL']:+.1f} | {d['on_mismatch_frac']:.3f} | {d['dx_hit_p50']:.1f} [{d['dx_hit_p90']:.1f}] | {d['dx_hit_mean_vec_mm']} | {d['dtilt_deg_median']:+.2f} / {d['dtilt_deg_p90abs']:.2f} |")
if os.path.exists(f'{O}/foot_check.json'):
    Fc = J('foot_check.json')
    print('\n### Landing point vs the MEASURED stream foot (trusted stream_foot.json frames)\n')
    print('| pour | n | current: abs diff p50 [p90] mm, bias (x, y) | v2: abs diff p50 [p90] mm, bias | debiased p50 cur / v2 | fast motion rms (x, y) mm: foot / current / v2 | fast-motion correlation with the foot (x, y): current / v2 |\n|---|---|---|---|---|---|---|')
    for p in POURS:
        c, v = Fc[p]['cur'], Fc[p]['v2']
        print(f"| {p} | {c['n']} | {c['absdiff_p50']:.1f} [{c['absdiff_p90']:.1f}], {c['bias_mm']} | {v['absdiff_p50']:.1f} [{v['absdiff_p90']:.1f}], {v['bias_mm']} | {c['absdiff_debiased_p50']:.1f} / {v['absdiff_debiased_p50']:.1f} | {c['fast_rms_foot_mm']} / {c['fast_rms_landing_mm']} / {v['fast_rms_landing_mm']} | {c['fast_corr_xy']} / {v['fast_corr_xy']} |")
