"""supplement to sens_C.py: the l_s anchor measured as designed for the video (DC.md 4):
canvas loop at a FIXED place (loop placement with the nominal r_e,ref, not moved by delta_e) and read just before the
first white (t = t_k + tau_w - 0.1 s), T2 held s0 = 0.15 D, all parameter points.  -> adds keys to sensC_<ells>.json"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from multiprocessing import Pool
import sens_C as SC


def run(arg):
    name, p = arg
    import DC_tests as DT, px_tests as T
    from px_model import V2, jet_derived, FIXED, NOMINAL
    q = dict(V2); q.update(dict(jetU=1.0, canvas='screen')); q.update(p)
    Pn = dict(FIXED); Pn.update(NOMINAL); Pn.update(q); Pn['delta_e'] = SC.NOM['delta_e']; Pn['Pi_coh'] = SC.NOM['Pi_coh']
    J = jet_derived(T.Q_LOW, T.V_LOW, Pn)
    o, _ = DT.T2x(params=q, s0=0.15, travel=0.0, t_end=1.6, loop_re=J['re'])
    return name, dict(T2s15_loop_fixed_pw=o['loop_ratio_pw'], T2s15_near_pw=o['near_pw'], T2s15_far_pw=o['far_pw'],
                      T2s15_loop_fixed_14=o['loop_ratio_1.4'])


if __name__ == '__main__':
    ells = [float(x) for x in sys.argv[1].split(',')]
    fn = os.path.join(HERE, 'sensC_%s.json' % '_'.join(f'{e:g}' for e in ells))
    R = json.load(open(fn))
    cases = []
    for e in ells:
        cases.append((f'{e:g}|nominal', SC.P(e)))
        for k in SC.NOM:
            for s, f in (('+', 1.2), ('-', 0.8)):
                cases.append((f'{e:g}|{k}{s}', SC.P(e, **{k: SC.NOM[k] * f})))
        for s, f in (('+', 1.2), ('-', 0.8)):
            cases.append((f'{e:g}|ell_s{s}', SC.P(e * f)))
    with Pool(4) as pool:
        for name, out in pool.imap_unordered(run, cases):
            R[name].update(out); print(name, out, flush=True)
    json.dump(R, open(fn, 'w'), indent=1)
