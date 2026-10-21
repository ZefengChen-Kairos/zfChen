"""collect the per-pour full fits + placement into one pickle for the validation scripts."""
import sys, pickle, json
from pv2lib import V2, POURS
tag = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else 'v2_all'
PLF = sys.argv[3] if len(sys.argv) > 3 else 'placement_joint.json'
V = dict(pours={p: pickle.load(open(f'{V2}/out/full_{p}_{tag}.pkl', 'rb')) for p in POURS}, place=json.load(open(f'{V2}/out/{PLF}')))
pickle.dump(V, open(f'{V2}/out/{name}.pkl', 'wb'))
print('wrote', name, {p: round(v['nuis']['c1'] * 1e3, 2) for p, v in V['pours'].items()})
