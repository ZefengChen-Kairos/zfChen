"""eta_J = 0 regression: proto_DI must reproduce min/spec_v2 bit for bit (T1 and T4).  usage: python3 -I bitcheck.py DIR OUTNPZ"""
import sys, os, numpy as np
d = sys.argv[1]; sys.path.insert(0, d)
import px_tests as T
from px_model import V2
o1, e1 = T.T1(params=dict(V2), keep=True)
o4, e4 = T.T4(params=dict(V2), keep=True)
np.savez(sys.argv[2], c1=e1['snaps'][6.0], c13=e1['snaps'][3.0], c4=e4['snaps'][round(e4.get('t_end', 8.5), 4)] if 't_end' in e4 else list(e4['snaps'].values())[-1],
         hole=o1['hole_r_3'], first=o1['t_first_vis'], dadt=o1['dAdt_raft'])
print(d, o1['hole_r_3'], o1['t_first_vis'], o1['dAdt_raft'], o4['n_enclosed'], o4['enclosed_area_mm2'])
