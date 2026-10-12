"""Main eta_J scan for test_Iinertia.md.  usage: python3 -I run_scan.py N [only=T1,T2,...] [etas=...]"""
import sys, os, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from multiprocessing import Pool
import numpy as np
OUT = os.path.join(HERE, 'out')

def tag(kind, kw):
    return kind + '_' + '_'.join(f'{k}{v:g}' if isinstance(v, (int, float)) and not isinstance(v, bool) else f'{k}{v}'
                                 for k, v in sorted(kw.items()))

def run(arg):
    kind, kw, N, keep = arg
    import DI_tests as DT
    t0 = time.time()
    if kind == 'T1':
        o, ex = DT.T1x(N=N, keep=keep, **kw)
    elif kind == 'T2':
        o, ex = DT.T2x(N=N, keep=keep, **kw)
    elif kind == 'T3':
        o, ex = DT.T3x(N=N, keep=keep, **kw)
    elif kind == 'T4':
        o, ex = DT.T4x(N=N, keep=keep, **kw)
    o['wall'] = time.time() - t0; o['N'] = N
    name = tag(kind, kw) + f'_N{N}'
    if keep and ex is not None:
        arrs = {}
        if kind == 'T1':
            for t, c in ex['snaps'].items(): arrs[f'c_{t:g}'] = c
            for t, m in ex['mks'].items(): arrs[f'm_{t:g}'] = m
            arrs['mk0'] = ex['mk0']
        elif kind == 'T2':
            for t, (c, oo, m) in ex['snaps'].items():
                arrs[f'c_{t:g}'] = c; arrs[f'o_{t:g}'] = oo; arrs[f'm_{t:g}'] = m
        elif kind == 'T3':
            for t, c in ex['snaps'].items(): arrs[f'c_{t:g}'] = c
            for t, m in ex['mks'].items(): arrs[f'm_{t:g}'] = m
            arrs['mk0'] = ex['mk0']
        elif kind == 'T4':
            for t, c in ex['snaps'].items(): arrs[f'c_{t:g}'] = c
            arrs['path'] = np.array(ex['path'])
        np.savez_compressed(os.path.join(OUT, 'snap_' + name + '.npz'), **arrs)
    return kind, kw, N, o, name

def rnd(x):
    if isinstance(x, float): return round(x, 6)
    if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, list): return [rnd(v) for v in x]
    return x

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else ['T1', 'T2', 'T3', 'T4']
    etas = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.0, 1e-4, 3e-4, 1e-3, 2e-3]
    cases = []
    if 'T2' in only:
        for s0 in (0.10, 0.15, 0.25):
            for tr in (0.0, 0.10):
                for e in etas:
                    cases.append(('T2', dict(eta=e, s0=s0, travel=tr), N, True))
            if N == 128 and 2e-3 in etas:
                cases.append(('T2', dict(eta=5e-3, s0=s0, travel=0.0), N, True))
    if 'T1' in only:
        for e in etas: cases.append(('T1', dict(eta=e), N, True))
    if 'T4' in only:
        for b in (False, True):
            for e in etas: cases.append(('T4', dict(eta=e, base=b), N, True))
    if 'T3' in only:
        for d in (0.0, 0.7):
            for e in etas: cases.append(('T3', dict(eta=e, dwell=d), N, True))
    fn = os.path.join(OUT, f'scan_N{N}_' + '_'.join(only) + '.jsonl')
    t0 = time.time()
    with Pool(4, maxtasksperchild=4) as pool, open(fn, 'a') as f:
        for kind, kw, N_, o, name in pool.imap_unordered(run, cases):
            f.write(json.dumps(dict(kind=kind, kw=kw, N=N_, name=name, o=rnd(o)), default=float) + '\n'); f.flush()
            print(kind, json.dumps(kw), N_, 'wall %.0f s (%.2f s/sim-s, %d steps)' % (o['wall'], o.get('wall_per_sim_s', -1), o.get('steps', -1)),
                  'total %.0f' % (time.time() - t0), flush=True)
