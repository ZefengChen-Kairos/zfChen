import sys, json, time
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
VARS = {
 'v1': dict(geom='pxx', emit_canvas=True),
 'v2_noobst': dict(obst=None),
 'v2_rc': dict(obst='rc'),
 'v2_r0': dict(obst='r0'),
 'v2_rsub': dict(obst='rsub'),
 'v2_rc_a5': dict(obst='rc', alpha_j=5.0),
 'v2_rc_a12': dict(obst='rc', alpha_j=12.0),
 'v2_rc_de08': dict(obst='rc', delta_e=0.8e-3),
 'v2_rc_de12': dict(obst='rc', delta_e=1.2e-3),
 'v2_rc_de16': dict(obst='rc', delta_e=1.6e-3),
 'v2_rc_aF0': dict(obst='rc', alpha_F=1e-5),
 'v2_rc_noemit': dict(obst='rc', emit_canvas=False),
 'v2_rc_G-': dict(obst='rc', Gamma=0.0316),
 'v2_rc_G+': dict(obst='rc', Gamma=0.0474),
 'v2_re': dict(obst='re'),
 'v2': {},
 'v2_ring': dict(Ke_mode=None),
 'v2_a5': dict(alpha_j=5.0), 'v2_a12': dict(alpha_j=12.0), 'v2_de08': dict(delta_e=0.8e-3), 'v2_de12': dict(delta_e=1.2e-3),
 'v2_de16': dict(delta_e=1.6e-3), 'v2_aF0': dict(alpha_F=1e-5), 'v2_G-': dict(Gamma=0.0316), 'v2_G+': dict(Gamma=0.0474),
 'v2_noemit': dict(emit_canvas=False), 'v2_dc025': dict(delta_c=0.25e-3), 'v2_dc075': dict(delta_c=0.75e-3),
 'v2_rsub23': dict(obst='rsub', p_sub=23.0),
 'v2_re_de16': dict(obst='re', delta_e=1.6e-3),
}
def run(arg):
    name, N = arg
    import v2_tests as V
    p = dict(V.V2) if name.startswith('v2') else {}
    p.update(VARS[name])
    t0 = time.time()
    o = V.T1L(params=p, N=N)
    o['wall'] = time.time() - t0
    return name, N, o
if __name__ == '__main__':
    names = sys.argv[1].split(',')
    Ns = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [128]
    cases = [(n, N) for n in names for N in Ns]
    with Pool(4, maxtasksperchild=1) as pool:
        for name, N, o in pool.imap(run, cases):
            short = {k: o[k] for k in o if k.startswith('at+')}
            print('T1L', name, N, 't_first=%.3f' % o['t_first'], 'clip=%.2e redist=%.2e maxc=%.4f foam=%.1e surf=%.1e wall=%.0f' % (o['clip_cm2'], o['redist_cm2'], o['max_c'], o['foam_rel_err'], o['surf_err_cm2'], o['wall']), flush=True)
            for k, m in short.items():
                print('   ', k, json.dumps({kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in m.items()}), flush=True)
            json.dump(o, open(f'T1L_{name}_{N}.json', 'w'), default=float)
