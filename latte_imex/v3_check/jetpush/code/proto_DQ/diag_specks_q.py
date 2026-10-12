"""Brown specks inside the white in T1 (fixed low pour) under variant Q: dependence on time step, kernel sub-sampling and
grid.  Counts enclosed brown components (not the hole, not connected to the rim) at 2 and 3 s."""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool


def run(args):
    import px_tests as T
    from px_model import PXSim, VQ, R
    N = args.get('N', 128); g = T.grid(N)
    sim = PXSim(grid=g, params=dict(VQ, jetU=args.get('jetU', 1.0)), cfl=args.get('cfl', 0.5),
                src_cfl=args.get('src_cfl', 0.2), nsub=args.get('nsub', 4))
    inp = T.jet_inp(lambda t: 0.0, lambda t: 0.0, lambda t: T.Q_LOW if t < 3 else 0.0, T.V_LOW)
    out = {}
    def cb(s, t):
        h = T.hole_and_specks(s.c, g, 0.0, 0.0)
        out[str(t)] = dict(n=h['n_specks'], mm2=round(h['specks_mm2'], 2), grey=h['grey_cells'],
                           grey_mm2=round(h['grey_cells'] * g.cellA * 1e6, 1))
    t0 = time.time()
    sim.run(inp, 3.0, lambda t: 0.0, breakpoints=[3.0], callback=cb, cb_times=[2.0, 3.0])
    return args, out, sim.stats['steps'], time.time() - t0


cases = [dict(), dict(cfl=0.25, src_cfl=0.1), dict(cfl=0.125, src_cfl=0.05), dict(nsub=8), dict(N=192), dict(N=256),
         dict(jetU=0.0)]
with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
    for a, o, st, w in pool.imap(run, cases):
        print(json.dumps(dict(case=a, steps=st, wall=round(w, 1), specks=o)), flush=True)
