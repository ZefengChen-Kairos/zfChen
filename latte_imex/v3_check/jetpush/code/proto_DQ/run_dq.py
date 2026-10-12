"""usage: python3 -I run_dq.py TESTS VARS N [NPROC]
TESTS: comma list of T1, T1L, T2m<s0>, T2h<s0> (e.g. T2m0.15, T2h0.25), T3, T3d, T4f, T4b
VARS : comma list of keys of VARS below (v2 = v3-min-2, q = variant Q; lamXX = diagnostic multiplier only).
One JSON line per case on stdout; snapshots to out/<test>_<var>_<N>.npz."""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from multiprocessing import Pool
import numpy as np

VARS = {'v2': {}, 'q': dict(jetU=1.0),
        'q_a5': dict(jetU=1.0, alpha_j=5.0), 'q_a12': dict(jetU=1.0, alpha_j=12.0),
        'v2_a5': dict(alpha_j=5.0), 'v2_a12': dict(alpha_j=12.0),
        'lam0.1': dict(jetU=0.1), 'lam0.25': dict(jetU=0.25), 'lam0.5': dict(jetU=0.5), 'lam2': dict(jetU=2.0)}
OUT = os.path.join(HERE, 'out')


def params(var):
    from px_model import V2
    p = dict(V2); p.update(VARS[var]); return p


def jsonable(x):
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return [jsonable(v) for v in x.tolist()]
    if isinstance(x, float): return float('%.7g' % x) if x == x else None
    return x


def run(arg):
    test, var, N = arg
    import dq_tests as Q
    p = params(var); t0 = time.time()
    tag = f'{test}_{var}_{N}'
    if test == 'T1':
        o, ex = Q.T1(params=p, N=N, keep=True)
        np.savez_compressed(os.path.join(OUT, tag + '.npz'), **{f'c{t}': v for t, v in ex['snaps'].items()},
                            markers0=ex['markers0'], markers=ex['markers'],
                            t=np.array(ex['ser']['t']), A_r=np.array(ex['ser']['A_r']), A_vis=np.array(ex['ser']['A_vis']),
                            umax=np.array(ex['ser']['umax']))
    elif test == 'T1L':
        o = Q.T1L(params=p, N=N)
        rows = o.pop('rows'); o['rows_sel'] = {str(k): v for k, v in rows.items() if abs(k * 4 - round(k * 4)) < 1e-6}
    elif test.startswith('T2'):
        travel = 0.10 if test[2] == 'm' else 0.0
        s0 = float(test[3:])
        o, ex = Q.T2Q(params=p, N=N, keep=True, s0=s0, travel=travel)
        np.savez_compressed(os.path.join(OUT, tag + '.npz'), **{f'c{t}': v[0] for t, v in ex['snaps'].items()},
                            **{f'o{t}': v[1] for t, v in ex['snaps'].items()}, **{f'm{t}': v[2] for t, v in ex['snaps'].items()},
                            xs=ex['xs'], cx=ex['cx'], rad=ex['rad'])
    elif test in ('T3', 'T3d'):
        o, ex = Q.T3Q(params=p, N=N, keep=True, gate='lift', dwell=0.7 if test == 'T3d' else 0.0)
        np.savez_compressed(os.path.join(OUT, tag + '.npz'), **{f'c{t}': v for t, v in ex['snaps'].items()},
                            **{f'm{t}': v for t, v in ex['msnaps'].items()}, mk0=ex['mk0'], nline=ex['nline'])
    elif test in ('T4f', 'T4b'):
        o, ex = Q.T4Q(params=p, N=N, keep=True, base=(test == 'T4b'))
        np.savez_compressed(os.path.join(OUT, tag + '.npz'), **{f'c{t}': v for t, v in ex['snaps'].items()},
                            path=np.array(ex['path']), x_lo=ex['x_lo'], x_hi=ex['x_hi'])
    else:
        raise ValueError(test)
    o['wall_case'] = time.time() - t0
    return test, var, N, jsonable(o)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    tests = sys.argv[1].split(','); vars_ = sys.argv[2].split(','); N = int(sys.argv[3])
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    cases = [(t, v, N) for t in tests for v in vars_]
    with Pool(min(nproc, len(cases))) as pool:
        for test, var, N, out in pool.imap_unordered(run, cases):
            print(json.dumps(dict(test=test, var=var, N=N, obs=out)), flush=True)
