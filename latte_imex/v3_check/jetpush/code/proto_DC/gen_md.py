"""markdown tables for test_Ccompres.md from out/scan_*.jsonl.  usage: python3 -I gen_md.py N > out/tables_N.md"""
import sys, json, glob, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
R = {}
for fn in sorted(glob.glob(os.path.join(HERE, 'out', 'scan_A*.jsonl'))):
    for l in open(fn):
        r = json.loads(l)
        if r['N'] == N: R[(r['test'], r['var'], r['arg'])] = r['obs']
LAB = {'v2': 'v3-min-2', 'q': 'Q', 'sv20': 'C-V 20 cm', 'sv4': 'C-V 4 cm', 'sv2': 'C-V 2 cm', 'sv1': 'C-V 1 cm', 'sv05': 'C-V 0.5 cm',
       'sv20n': 'C-V 20 cm 无 W', 'sv2n': 'C-V 2 cm 无 W', 'sv1n': 'C-V 1 cm 无 W', 'sv05n': 'C-V 0.5 cm 无 W', 'cd1e3': 'C-D 10⁻³',
       'cd1e4': 'C-D 10⁻⁴', 'cd3e5': 'C-D 3×10⁻⁵', 'cd1e4n': 'C-D 10⁻⁴ 无 W', 'cp': 'C-P', 'cpn': 'C-P 无 W'}
ORDER = list(LAB)


def f(x, d=3):
    if x is None: return '—'
    if isinstance(x, bool): return str(x)
    if isinstance(x, int): return str(x)
    if isinstance(x, float):
        if x == 0: return '0'
        if abs(x) < 10 ** (-d) or abs(x) >= 1e5: return '%.1e' % x
        return ('%.' + str(d) + 'f') % x
    return str(x)


def table(title, test, arg, cols, vars_=None):
    vs = [v for v in ORDER if (test, v, arg) in R and (vars_ is None or v in vars_)]
    print(f'\n**{title}**\n')
    print('| 变体 | ' + ' | '.join(c[0] for c in cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for v in vs:
        o = R[(test, v, arg)]
        cells = []
        for c in cols:
            fn_ = c[1]
            try:
                val = fn_(o) if callable(fn_) else o.get(fn_)
            except Exception:
                val = None
            cells.append(val if isinstance(val, str) else f(val, c[2] if len(c) > 2 else 3))
        print(f'| {LAB[v]} | ' + ' | '.join(cells) + ' |')


g = lambda k: (lambda o: o.get(k))
table(f'T1 回归（{N}²）', 'T1', None, [
    ('首白 (s)', 't_first_vis'), ('dA_r/dt (cm²/s)', 'dAdt_raft', 4), ('dA_vis/dt', 'dAdt_vis'), ('首白环内径 τ_w+0.1 (mm)', 'ring_rin_tw+0.1_mm', 1),
    ('褐洞 ray 3 s / 6 s', lambda o: f"{f(o['hole_r_3'],4)} / {f(o['hole_r_6'],4)}"), ('褐洞内切 (D)', 'hole_ins', 4),
    ('开口 2 / 3 / 6 s', lambda o: f"{f(o['open_deg_2'],0)} / {f(o['open_deg_3'],0)} / {f(o['open_deg_6'],0)}°"),
    ('白内褐斑 3 s', lambda o: f"{o['specks_n_3']} 个 {f(o['specks_mm2_3'],1)} mm²"),
    ('0.30 D 位移 白前 / 6 s', lambda o: f"{f(o['disp_0.30_pre_white'],4)} / {f(o['disp_0.30_6'],3)}"), ('0.45 D 6 s', 'disp_0.45_6', 4),
    ('停倒+t_r 后 max\\|u\\| (mm/s)', 'umax_after_tr_mm_s', 2), ('13 s max\\|u\\|', 'umax_end_mm_s', 3), ('停后 max\\|Δc\\|', 'max_dc_tr_end', 2),
    ('停后 0.30 D 漂移', 'drift_0.30_tr_end', 4)])
for test, nm in (('T2h', '定点'), ('T2m', '移动 0.1 D / 1.5 s')):
    for s0 in ('0.10', '0.15', '0.25'):
        table(f'T2 {nm}，s₀ = {s0} D（{N}²；位移以朝前（远离壶）为正，D）', test, s0, [
            ('近缘 1 / 1.4 / 2 / 4 s', lambda o: ' / '.join(f(o.get(f'near_{t}'), 3) for t in ('1', '1.4', '2', '4'))),
            ('远缘 1 / 2 / 4 s', lambda o: ' / '.join(f(o.get(f'far_{t}'), 3) for t in ('1', '2', '4'))),
            ('画布环面积比 1.4 s', 'loop_ratio_1.4'),
            ('缝 2 / 4 / 6 / 10 s', lambda o: ' / '.join(f(o.get(f'gap_min_{t}'), 3) for t in ('2', '4', '6', '10'))),
            ('solidity 4 s', 'sol_mat_4'), ('包角 4 s', 'wrap_mat_4', 0),
            ('轴向厚 × 横跨 4 s', lambda o: f"{f(0.33 + o['far_4'] - o['near_4'], 3)} × {f(o['span_y_4'], 3)}"),
            ('物质面积比 4 s', 'area_ratio_4'), ('停倒后近缘蠕变 4.6→10 s', 'creep_near_4.6_10', 4)])
for test, nm in (('T3', '不驻留'), ('T3d', '入口驻留 0.7 s')):
    table(f'T3 切，{nm}（{N}²；D）', test, None, [
        ('2r_d', 'two_rd', 4), ('切线亏量 即时 / 终', lambda o: f"{f(o['line_def_imm'],4)} / {f(o['line_def_final'],4)}"),
        ('终可见宽', 'line_vis_width_final', 4), ('凹口 r_eq', 'notch_req', 4), ('入口外鼓 max / 入口缘最小偏移', lambda o: f"{f(o['entry_bulge_max'],4)} / {f(o['entry_bdev_min'],4)}"),
        ('出口外鼓', 'exit_bulge_max', 4), ('茎 入 / 出', lambda o: f"{f(o['stem_width_entry'],4)} / {f(o['stem_width_exit'],4)}"),
        ('V：物质线前拖 \\|y\\| = 0.005 / 0.02 / 0.04 / 0.08 D（切后 3 s）', lambda o: ' / '.join(f(o['vshape']['cut+3'][k], 4) for k in ('0.005', '0.02', '0.04', '0.08'))),
        ('V 半宽', lambda o: f(o['vshape']['cut+3']['half_width'], 4)), ('盘面积变化 (cm²)', 'disc_area_change_cm2', 3)])
for test, nm in (('T4f', '新画布'), ('T4b', '白底')):
    table(f'T4 摆动，{nm}（{N}²）', test, None, [
        ('轴线条纹 停时 / 5 s 后', lambda o: f"{o['n_stripes']} / {o['n_stripes_final']}"),
        ('y = 0, ±0.02, ±0.04 D 褐线数（停时）', lambda o: ', '.join(str(o.get(f'end_y{y:+.2f}_n')) for y in (0.0, 0.02, -0.02, 0.04, -0.04))),
        ('中位数 停时 / 5 s 后', lambda o: f"{f(o['n_lines_median'],1)} / {f(o['n_lines_median_fin'],1)}"),
        ('y=+0.02 节距 / 线宽', lambda o: f"{f(o.get('end_y+0.02_pitch'),3)} / {f(o.get('end_y+0.02_w'),3)}"),
        ('白内封闭褐岛', lambda o: f"{o['n_enclosed']} 个 {f(o['enclosed_area_mm2'],0)} mm²"),
        ('灰格比', 'mixed_frac_of_white'), ('停后 max\\|Δc\\|', 'max_abs_dc_after', 2)])
# T5
print(f'\n**T5 账本、有界、锐度（{N}²，各测试中的最差值）**\n')
print('| 变体 | 账本 1 相对误差 | 账本 2（奶泡）相对 | 账本 3 (cm²) | max (∇·u − RHS)Δt | c 修正前 [min, max] | c 修正后 | 重分配搬动 (cm²) | 净裁剪 (cm²) | 界面宽（格） |')
print('|---|---|---|---|---|---|---|---|---|---|')
for v in ORDER:
    os_ = [o for (t, vv, a), o in R.items() if vv == v]
    if not os_: continue
    mx = lambda k: max(abs(o.get(k) or 0.0) for o in os_)
    mn = lambda k: min(o.get(k) for o in os_ if o.get(k) is not None)
    iw = [o.get('iface_width_cells') for o in os_ if o.get('iface_width_cells') is not None]
    print(f"| {LAB[v]} | {f(mx('ledger1_rel'))} | {f(mx('foam_rel_err'))} | {f(mx('surf_err_cm2'))} | {f(mx('max_divres'))} | [{f(mn('min_c'))}, {f(max(o.get('max_c') for o in os_),4)}] | "
          f"[{f(mn('c_min_after'))}, {f(max(o.get('c_max_after') for o in os_),4)}] | {f(mx('redist_cm2'))} | {f(mx('clip_cm2'))} | {f(min(iw),2)}–{f(max(iw),2)} |")
# cost
print(f'\n**代价（{N}²，批量运行中测得，4 进程并行、与其它会话共享 CPU；只作相对比较）**\n')
print('| 变体 | T1 墙钟/模拟秒 | T1 ms/步 | T1 步数 | T4f 墙钟/模拟秒 | T4f 步数 | Helmholtz PCG 迭代/次 | 分解次数 (T1) |')
print('|---|---|---|---|---|---|---|---|')
for v in ORDER:
    o1 = R.get(('T1', v, None)); o4 = R.get(('T4f', v, None))
    if not o1: continue
    print(f"| {LAB[v]} | {f(o1['wall_per_sim_s'],2)} | {f(o1['ms_per_step'],1)} | {o1['steps']} | {f(o4['wall_per_sim_s'],2) if o4 else '—'} | {o4['steps'] if o4 else '—'} | {f(o1['cg_its_per_solve'],2)} | {o1['n_fact']} |")
