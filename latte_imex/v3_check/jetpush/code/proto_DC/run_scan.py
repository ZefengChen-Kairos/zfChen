"""usage: OMP_NUM_THREADS=1 python3 -I run_scan.py N OUT_TAG NPROC CASE [CASE ...]
CASE = TEST:VAR[:ARG]   TEST in T1 | T2m | T2h | T3 | T3d | T4f | T4b | U0;  ARG = s0 for T2 (e.g. T2h:sv1:0.15)
writes out/scan_<OUT_TAG>.jsonl (one line per case) and out/snap_<TEST>_<VAR>[_ARG]_N<N>.npz"""
import sys, os, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from multiprocessing import Pool
import numpy as np


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x); return x if math.isfinite(x) else None
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.ndarray): return clean(x.tolist())
    return x


def run(case):
    N, spec = case
    import DC_tests as C
    parts = spec.split(':'); test, var = parts[0], parts[1]; arg = parts[2] if len(parts) > 2 else None
    t0 = time.time(); snapf = os.path.join(HERE, 'out', f'snap_{test}_{var}' + (f'_{arg}' if arg else '') + f'_N{N}.npz')
    if test == 'U0':
        o = dict(rows=C.unit_C0(N=N)); ex = None
    elif test == 'T1':
        o, ex = C.T1x(var, N=N, t_end=13.0, keep=True)
        np.savez_compressed(snapf, **{f'c{t}': v for t, v in ex['snaps'].items()}, **{f'm{t}': v for t, v in ex['mks'].items()},
                            mk0=ex['mk0'], t_fr=ex['t_fr'])
        o['series_umax'] = [ex['ser']['umax'][i] for i in range(0, len(ex['ser']['t']), 4)]
        o['series_t'] = [ex['ser']['t'][i] for i in range(0, len(ex['ser']['t']), 4)]
    elif test in ('T2m', 'T2h'):
        s0 = float(arg); travel = 0.10 if test == 'T2m' else 0.0
        o, ex = C.T2x(var, s0=s0, travel=travel, t_end=10.0, N=N, keep=True)
        d = {}
        for t, (c, oo, m, l) in ex['snaps'].items():
            d[f'c{t}'] = c; d[f'o{t}'] = oo; d[f'm{t}'] = m; d[f'l{t}'] = l
        np.savez_compressed(snapf, **d, cx=ex['cx'], rad=ex['rad'], xs=ex['xs'])
    elif test in ('T3', 'T3d'):
        o, ex = C.T3x(var, dwell=0.7 if test == 'T3d' else 0.0, N=N, keep=True)
        np.savez_compressed(snapf, **{f'c{t}': v for t, v in ex['snaps'].items()}, **{f'm{t}': v for t, v in ex['mks'].items()},
                            mk0=ex['mk0'], t_cut=ex['t_cut'], t_end=ex['t_end'])
    elif test in ('T4f', 'T4b'):
        o, ex = C.T4x(var, base=(test == 'T4b'), N=N, keep=True)
        np.savez_compressed(snapf, **{f'c{t}': v for t, v in ex['snaps'].items()}, path=np.array(ex['path']),
                            x_lo=ex['x_lo'], x_hi=ex['x_hi'], t_stop=ex['t_stop'], t_end=ex['t_end'])
    else:
        raise ValueError(test)
    o['_wall'] = time.time() - t0
    return dict(test=test, var=var, arg=arg, N=N, obs=clean(o))


if __name__ == '__main__':
    N = int(sys.argv[1]); tag = sys.argv[2]; nproc = int(sys.argv[3]); specs = sys.argv[4:]
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    fn = os.path.join(HERE, 'out', f'scan_{tag}.jsonl')
    t0 = time.time()
    with Pool(nproc, maxtasksperchild=1) as pool, open(fn, 'a') as f:
        for r in pool.imap_unordered(run, [(N, s) for s in specs]):
            f.write(json.dumps(r) + '\n'); f.flush()
            print(r['test'], r['var'], r['arg'], 'N', N, 'wall %.1f' % r['obs']['_wall'], flush=True)
    print('done', round(time.time() - t0, 1))
