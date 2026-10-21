"""Downstream translation of the v2 accuracy (per pour) using the audit's sensitivities (pose_audit/holdout/summary.json)
and pooled time fractions.  -> ../out/downstream_numbers.json"""
import json
import numpy as np
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
R = f'{SCR}/pose_v2/report/out'
NB = json.load(open(f'{R}/numbers.json'))
AH = json.load(open(f'{SCR}/pose_audit/holdout/summary.json'))
DS = json.load(open(f'{SCR}/pose_v2/audit/out/ds_check.json'))
out = {}
tot = {}
span = 0
for p, nb in NB.items():
    s = AH[p]['sensitivity']
    rel = s['dQdtilt']['rel_per_deg_p10_p50_p90']; dq = s['dQdtilt']['dQdtilt_p10_p50_p90']
    dV1 = abs(s['perturb']['tilt+1deg']['dV_total_mL']); V = s['perturb']['tilt+1deg']['V_total_mL']
    ti50, ti90 = nb['inst']['tilt_p50'], nb['inst']['tilt_p90']
    fast = nb['inst']['by_speed']['fast']['tilt_p90']
    sysm = nb['sys']['tilt_med']
    out[p] = dict(
        rel_per_deg_p10_p50_p90=rel, dQdtilt_p50_mLs=dq[1], dV_per_deg_mL=round(dV1, 2), V_total_mL=round(V, 1),
        flow_inst_p50_pct=round(100 * rel[1] * ti50, 1), flow_inst_p90_pct=round(100 * rel[2] * ti90, 1),
        flow_inst_p50_mLs=round(dq[1] * ti50, 2),
        flow_fast_pct=round(100 * rel[1] * fast, 0) if fast else None,
        flow_sys_pct=round(100 * rel[1] * sysm, 0), vol_sys_mL=round(dV1 * sysm, 1), vol_sys_pct=round(100 * dV1 * sysm / V, 1),
        landing_inst_p50_mm=nb['inst']['tip_p50'], landing_inst_p90_mm=nb['inst']['tip_p90'], landing_sys_mm=nb['sys']['tip_med'],
        landing_tilt_mm_per_deg=s['perturb']['tilt+1deg']['dx_hit_p50'], landing_yaw_mm_per_deg=s['perturb']['yaw+1deg_about_origin']['dx_hit_p50'],
        v2_vs_cur=dict(dQ_rms=round(DS[p]['dQ_rms'], 2), dV=round(DS[p]['dV_mL'], 1), landing_p50=round(DS[p]['dx_hit_p50'], 1)))
    sp = nb['span_s']; span += sp
    for k, v in nb['frac_time'].items():
        tot[k] = tot.get(k, 0) + v * sp
out['pooled_frac_time'] = {k: round(v / span, 4) for k, v in tot.items()}
out['pooled_span_s'] = round(span, 2)
json.dump(out, open(f'{R}/downstream_numbers.json', 'w'), indent=1)
for k, v in out.items():
    print(k, v)
