"""single-process timing at 128^2: T1 (13 s) and T4f (8.5 s) per variant.  usage: OMP_NUM_THREADS=1 python3 -I run_timing.py v2,q,sv1,...
prints wall per simulated second, ms/step (and the Poisson/Helmholtz part), steps; appends to out/timing_128.jsonl"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import DC_tests as C
for v in sys.argv[1].split(','):
    for test in ('T1', 'T4f'):
        t0 = time.time()
        o, _ = C.T1x(v, t_end=13.0) if test == 'T1' else C.T4x(v, base=False)
        rec = dict(var=v, test=test, wall_per_sim_s=o['wall_per_sim_s'], ms_per_step=o['ms_per_step'], ms_poisson=o['ms_poisson'],
                   steps=o['steps'], cg_its_per_solve=o['cg_its_per_solve'], n_fact=o['n_fact'], total_wall=time.time() - t0,
                   load=os.getloadavg()[0])
        print(json.dumps(rec), flush=True)
        open(os.path.join(HERE, 'out', 'timing_128.jsonl'), 'a').write(json.dumps(rec) + '\n')
