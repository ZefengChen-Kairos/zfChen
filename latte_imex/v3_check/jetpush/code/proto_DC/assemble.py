"""fill the {TAB_*} placeholders of ../test_Ccompres.md with tables generated from out/*.jsonl (in place)."""
import os, re, json, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
FN = os.path.join(os.path.dirname(HERE), 'test_Ccompres.md')
md = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'gen_md.py'), '128'], capture_output=True, text=True).stdout
blocks = ['**' + b for b in md.split('\n**')[1:]]


def pick(prefix):
    return '\n'.join(b.rstrip() + '\n' for b in blocks if b.startswith('**' + prefix))


TAB = {'TAB_T1': pick('T1 回归'), 'TAB_T2': pick('T2 定点') + '\n' + pick('T2 移动'), 'TAB_T3': pick('T3 切'), 'TAB_T4': pick('T4 摆动'),
       'TAB_T5': pick('T5 账本'), 'TAB_COSTB': pick('代价')}
# 192 vs 128
R = {}
for f in ('scan_A128.jsonl', 'scan_A192.jsonl'):
    for l in open(os.path.join(HERE, 'out', f)):
        r = json.loads(l); R[(r['test'], r['var'], r['arg'], r['N'])] = r['obs']


def ff(x, d=3):
    return '—' if x is None else (('%.' + str(d) + 'f') % x if isinstance(x, float) else str(x))


LAB = {'v2': 'v3-min-2', 'q': 'Q', 'sv1': 'C-V 1 cm', 'sv05': 'C-V 0.5 cm', 'cd1e4': 'C-D 10⁻⁴'}
rows = [
    ('T1 褐洞内切 (D)', 'T1', None, lambda o: ff(o['hole_ins'], 4)),
    ('T1 开口 3 s', 'T1', None, lambda o: ff(o['open_deg_3'], 0) + '°'),
    ('T1 白内褐斑 3 s (mm²)', 'T1', None, lambda o: ff(o['specks_mm2_3'], 1)),
    ('T1 0.30 D 白前位移', 'T1', None, lambda o: ff(o['disp_0.30_pre_white'], 4)),
    ('T1 0.45 D 6 s 位移', 'T1', None, lambda o: ff(o['disp_0.45_6'], 4)),
    ('T1 停倒后 max\\|u\\| (mm/s) / max\\|Δc\\|', 'T1', None, lambda o: ff(o['umax_after_tr_mm_s'], 2) + ' / ' + ff(o['max_dc_tr_end'], 2)),
    ('T2 定点 s₀=0.15：画布环 1.4 s', 'T2h', '0.15', lambda o: ff(o['loop_ratio_1.4'])),
    ('T2 定点 s₀=0.15：缝 2 / 4 / 10 s', 'T2h', '0.15', lambda o: ' / '.join(ff(o[f'gap_min_{t}']) for t in ('2', '4', '10'))),
    ('T2 定点 s₀=0.15：solidity / 包角', 'T2h', '0.15', lambda o: ff(o['sol_mat_4']) + ' / ' + ff(o['wrap_mat_4'], 0) + '°'),
    ('T2 定点 s₀=0.25：缝 4 / 6 / 10 s', 'T2h', '0.25', lambda o: ' / '.join(ff(o.get(f'gap_min_{t}')) for t in ('4', '6', '10'))),
    ('T2 移动 s₀=0.15：solidity / 厚 × 跨 / 面积比', 'T2m', '0.15', lambda o: ff(o['sol_mat_4']) + ' / ' + ff(0.33 + o['far_4'] - o['near_4']) + ' × ' + ff(o['span_y_4']) + ' / ' + ff(o['area_ratio_4'], 2)),
    ('T2 移动 s₀=0.25：缝 4 / 6 s', 'T2m', '0.25', lambda o: ff(o['gap_min_4']) + ' / ' + ff(o['gap_min_6'])),
    ('T3 切线终亏量', 'T3', None, lambda o: ff(o['line_def_final'], 4)),
    ('T3 驻留凹口 / 入口外鼓', 'T3d', None, lambda o: ff(o['notch_req'], 4) + ' / ' + ff(o['entry_bulge_max'], 4)),
    ('T4 新画布褐线中位数 / 褐岛 (mm²)', 'T4f', None, lambda o: ff(o['n_lines_median'], 0) + ' / ' + ff(o['enclosed_area_mm2'], 0)),
    ('T4 白底褐线中位数 / 褐岛 (mm²)', 'T4b', None, lambda o: ff(o['n_lines_median'], 0) + ' / ' + ff(o['enclosed_area_mm2'], 0)),
    ('界面宽（格，T1 末）', 'T1', None, lambda o: ff(o['iface_width_cells'], 2)),
]
vs = ['v2', 'q', 'sv1', 'sv05', 'cd1e4']
t = '| 量（128² → 192²） | ' + ' | '.join(LAB[v] for v in vs) + ' |\n|---' + '|---' * len(vs) + '|\n'
for lab, test, arg, fn in rows:
    cells = []
    for v in vs:
        a, b = R.get((test, v, arg, 128)), R.get((test, v, arg, 192))
        if a is None and b is None: cells.append('—'); continue
        sa = fn(a) if a else '—'; sb = fn(b) if b else None
        cells.append(f'{sa}（仅 128²）' if sb is None else (sa if sa == sb else f'{sa} → {sb}'))
    t += f'| {lab} | ' + ' | '.join(cells) + ' |\n'
TAB['TAB_192'] = t
# timing
tt = '| 变体 | T1 墙钟 / 模拟秒 | T1 ms/步（其中 Poisson/Helmholtz） | T1 步数 | T4f 墙钟 / 模拟秒 | T4f 步数 | PCG 迭代/次 | 相对 v3-min-2（T1） |\n|---|---|---|---|---|---|---|---|\n'
TL = {}
fnT = os.path.join(HERE, 'out', 'timing_128.jsonl')
if os.path.exists(fnT):
    for l in open(fnT):
        r = json.loads(l); TL[(r['var'], r['test'])] = r
LABT = {'v2': 'v3-min-2', 'q': 'Q', 'sv1': 'C-V 1 cm（名义）', 'sv05': 'C-V 0.5 cm', 'sv1n': 'C-V 1 cm 无 W', 'cd1e4': 'C-D 10⁻⁴', 'cp': 'C-P'}
base = TL.get(('v2', 'T1'), {}).get('wall_per_sim_s')
for v in LABT:
    a, b = TL.get((v, 'T1')), TL.get((v, 'T4f'))
    if not a: continue
    tt += (f"| {LABT[v]} | {a['wall_per_sim_s']:.2f} s | {a['ms_per_step']:.1f}（{a['ms_poisson']:.1f}） | {a['steps']} | "
           f"{(b['wall_per_sim_s'] if b else float('nan')):.2f} s | {b['steps'] if b else '—'} | {a['cg_its_per_solve']:.2f} | ×{a['wall_per_sim_s'] / base:.2f} |\n")
TAB['TAB_TIMING'] = tt
s = open(FN).read()
for k, v in TAB.items():
    s = s.replace('{' + k + '}', v.rstrip())
left = re.findall(r'\{TAB_[A-Z0-9_]+\}', s)
open(FN, 'w').write(s)
print('filled', list(TAB), 'left', left, 'size', len(s.encode()))
