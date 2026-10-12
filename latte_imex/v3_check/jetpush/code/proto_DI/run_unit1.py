import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import DI_tests as DT
for eta in (1e-9, 1e-3, 1e-1):
    t0 = time.time()
    r = DT.unit_I1(eta=eta)
    print('Theorem I check:', json.dumps(r))
    json.dump(r, open('out/unit_I1_%g.json' % eta, 'w'), indent=1)
    print('wall', round(time.time() - t0, 1))
