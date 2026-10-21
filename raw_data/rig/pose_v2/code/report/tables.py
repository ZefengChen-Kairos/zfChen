"""Markdown tables for REPORT.md from report/out/*.json and the builder / audit JSONs (no hand transcription)."""
import json
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
V2 = f'{SCR}/pose_v2'; R = f'{V2}/report/out'
NB = json.load(open(f'{R}/numbers.json')); DN = json.load(open(f'{R}/downstream_numbers.json'))
C2S = json.load(open(f'{V2}/audit/out/cam2_summary.json')); LOCO = json.load(open(f'{V2}/audit/out/loco_check.json'))
STR = json.load(open(f'{V2}/audit/out/stream_audit.json')); FOOT = json.load(open(f'{V2}/out/foot_check.json'))
AH = json.load(open(f'{SCR}/pose_audit/holdout/summary.json'))
P = ['tulip', 'leaf', 'heart', 'push_heart', 'swan']
CUR_LOCO = {'tulip': (15.2, 10.8), 'leaf': (16.3, 11.9), 'heart': (11.1, 4.6), 'push_heart': (13.1, 17.5), 'swan': (14.5, 10.1)}
C1 = {'tulip': -2.2, 'leaf': 8.5, 'heart': -13.9, 'push_heart': 9.4, 'swan': 10.6}
f2 = lambda x: f'{x:.2f}'; f1 = lambda x: f'{x:.1f}'
print('### A per-instant')
print('| pour | v2 tilt p50 / p90 [deg] | v2 heading p50 / p90 | v2 spin p50 / p90 | v2 tip p50 / p90 [mm] | same test, current: tilt / rotation / tip p50 | v2 raw x-cam: tilt / rotation / tip p50 | current tip.npz per sample (time hold-out p50): tilt / rotation / tip |')
print('|---|---|---|---|---|---|---|---|')
for p in P:
    n = NB[p]; i = n['inst']; r = i['raw_abs_diff']; cx = n['cur_xcam']; cb = n['cur_band']
    print(f"| {p} | {f2(i['tilt_p50'])} / {f2(i['tilt_p90'])} | {f2(i['head_p50'])} / {f2(i['head_p90'])} | {f2(i['spin_p50'])} / {f2(i['spin_p90'])} | {f2(i['tip_p50'])} / {f2(i['tip_p90'])} | "
          f"{f2(cx['tilt'][0])}° / {f1(cx['rot'][0])}° / {f1(cx['tip'][0])} mm | {f2(r['tilt'][0])}° / {f2(r['rot'][0])}° / {f1(r['tip'][0])} mm | {f2(cb['tilt'])}° / {f1(cb['rot'])}° / {f1(cb['tip'])} mm |")
print('\n### B systematic')
print('| pour | tilt median (max) [deg] | heading median [deg] | spin median [deg] (incl. ±2.5 spout az) | tip median (max) [mm] | camera-1 world: tilt / heading / spin / tip vector mm | camera-2 world: tilt / heading / spin / tip vector mm | camera-2 source |')
print('|---|---|---|---|---|---|---|---|')
for p in P:
    s = NB[p]['sys']; w1 = s['per_world']['cam1']; w2 = s['per_world']['cam2']
    print(f"| {p} | {f2(s['tilt_med'])} ({f2(s['tilt_max'])}) | {f2(s['head_med'])} | {f2(s['spin_med'])} | {f1(s['tip_med'])} ({f1(s['tip_max'])}) | "
          f"{w1['tilt_med']:+.2f} / {w1['head_med']:+.2f} / {w1['spin_med']:+.2f} / {w1['tip_med_mm']} | {w2['tilt_med']:+.2f} / {w2['head_med']:+.2f} / {w2['spin_med']:+.2f} / {w2['tip_med_mm']} | {'own pour' if 'this' in s['cam2_source'] else 'median of other 4'} |")
print('\n### C held-out')
print('| pour | cam-2 IoU med: current / current+own placement / v2 | cam-2 depth abs residual med [mm]: current / +placement / v2 | v2 better than current+placement (IoU / depth, share of frames) | LOCO held-out corners med [px] cam0/cam1: current / v2 raw calib / v2 with cam-1 corr. | stream origin med [mm]: current production / +placement / v2 (n) | landing vs measured foot p50 [mm]: current / v2 (n) |')
print('|---|---|---|---|---|---|---|')
for p in P:
    c = C2S[p]; lo = LOCO[p]; s_ = STR[p]; f_ = FOOT[p]; cl = CUR_LOCO[p]; b = c['v2out_better_than_cur_pl']
    print(f"| {p} | {c['cur']['iou_med']:.3f} / {c['cur_pl']['iou_med']:.3f} / {c['v2out']['iou_med']:.3f} | {f1(c['cur']['res_med'])} / {f1(c['cur_pl']['res_med'])} / {f1(c['v2out']['res_med'])} | "
          f"{b['iou_frac']*100:.0f}% / {b['res_frac']*100:.0f}% | {cl[0]:.1f}/{cl[1]:.1f} / {lo['cam0_from_1_noext1'][0]:.1f}/{lo['cam1_from_0_noext1'][0]:.1f} / {lo['cam0_from_1'][0]:.1f}/{lo['cam1_from_0'][0]:.1f} | "
          f"{f1(s_['cur_kona']['horiz']['median'])} / {f1(s_['cur_pl']['horiz']['median'])} / {f1(s_['v2out']['horiz']['median'])} ({s_['n_frames']}) | {f1(f_['cur']['absdiff_p50'])} / {f1(f_['v2']['absdiff_p50'])} ({f_['v2']['n']}) |")
print('\n### D time')
print('| pour | span s | fps cam0/1/2 | two-camera samples (<12 ms) | speed <10 / 10-60 / >60 °/s [% time] | tags: both cams / one cam ≥2 tags / one cam single tag (ambiguous) / none [% time] | longest no-tag gap [ms] | flip suspects (of detections) | ω p95 / max [°/s] |')
print('|---|---|---|---|---|---|---|---|---|')
for p in P:
    n = NB[p]; ft = n['frac_time']; cv = AH[p]['coverage']['samples']; tot = cv['cam0_only'] + cv['cam1_only'] + cv['cam0+cam1']
    print(f"| {p} | {n['span_s']:.1f} | {n['fps'][0]:.1f}/{n['fps'][1]:.1f}/{n['fps'][2]:.1f} | {cv['cam0+cam1']/tot*100:.0f}% | {ft['slow']*100:.1f} / {ft['mid']*100:.1f} / {ft['fast']*100:.1f} | "
          f"{ft['both']*100:.1f} / {ft['one_multi']*100:.1f} / {ft['one_single']*100:.1f} ({ft['one_single_ambiguous']*100:.1f}) / {ft['none']*100:.1f} | {n['gaps']['max_ms']:.0f} | {n['flips']['n_suspect']} ({n['flips']['n_det']}) | {n['omega']['p95']:.0f} / {n['omega']['max']:.0f} |")
print('\n### E by regime (raw x-cam |diff| p90)')
print('| pour | <10 °/s: s, tilt / heading / spin / tip | 10-60 °/s | >60 °/s |')
print('|---|---|---|---|')
for p in P:
    b = NB[p]['inst']['by_speed']
    cell = lambda d: '—' if d['tilt_p90'] is None else f"{d['n_s']:.1f} s: {d['tilt_p90']:.2f}° / {d['head_p90']:.2f}° / {d['spin_p90']:.2f}° / {d['tip_p90']:.1f} mm"
    print(f"| {p} | {cell(b['slow'])} | {cell(b['mid'])} | {cell(b['fast'])} |")
print('\n### F downstream')
print('| pour | flow sensitivity %/deg p10/p50/p90 | per-instant flow error p50 (p90) | fastest-bin flow error | systematic tilt -> flow bias / volume at fixed fill | landing per instant p50/p90 [mm] | landing systematic [mm] | v2 vs current now: flow rms mL/s, dV mL, landing p50 mm |')
print('|---|---|---|---|---|---|---|---|')
for p in P:
    d = DN[p]; r = d['rel_per_deg_p10_p50_p90']; v = d['v2_vs_cur']
    ff = '—' if d['flow_fast_pct'] is None else '%.0f%%' % d['flow_fast_pct']
    print(f"| {p} | {r[0]*100:.0f} / {r[1]*100:.0f} / {r[2]*100:.0f} | {d['flow_inst_p50_pct']:.0f}% ({d['flow_inst_p90_pct']:.0f}%) ≈ {d['flow_inst_p50_mLs']:.1f} mL/s | {ff} | "
          f"{d['flow_sys_pct']:.0f}% / ±{d['vol_sys_mL']:.1f} mL ({d['vol_sys_pct']:.1f}% of {d['V_total_mL']:.0f} mL) | {d['landing_inst_p50_mm']:.1f} / {d['landing_inst_p90_mm']:.1f} | {d['landing_sys_mm']:.1f} | {v['dQ_rms']:.1f}, {v['dV']:+.1f}, {v['landing_p50']:.1f} |")
print('\npooled', DN['pooled_frac_time'], DN['pooled_span_s'])
