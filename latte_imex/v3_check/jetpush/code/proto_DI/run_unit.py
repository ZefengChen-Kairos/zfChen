import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import DI_tests as DT
t0 = time.time()
r = DT.unit_I0(eta=1e-3)
print('T-I0 F=%.4e N impulse=%.6e expected=%.6e faces=%.6e' % (r['F_N'], r['impulse'], r['impulse_expected'], r['impulse_faces']))
for row in r['rows']:
    print('  t=%.2f r=%.2f mm ux=%.4f an=%.4f ratio=%.4f' % (row['t'], row['r_mm'], row['ux_mm_s'], row['an_mm_s'], row['ratio']))
json.dump(r, open('out/unit_I0.json', 'w'), indent=1)
print('wall', round(time.time() - t0, 1))
t0 = time.time()
r = DT.unit_I1()
print('Theorem I check:', json.dumps(r))
json.dump(r, open('out/unit_I1.json', 'w'), indent=1)
print('wall', round(time.time() - t0, 1))
