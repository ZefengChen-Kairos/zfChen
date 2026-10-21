"""A12: downstream re-run (production chain via pose_v2/work/downstream.run) installing the delivered
<p>_pitcher_pose_v2.npz exactly as a consumer would (load_take reads t, rotvec, pos, tip_p from that file; no TIP
override for v2), vs current production (pitcher_pose.npz + TIP_KONA)."""
import sys, json
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
import downstream as DS
import latte_imex.rig.real_program as RP
V2 = DS.V2
res = {}
for p in ['tulip', 'leaf', 'heart', 'push_heart', 'swan']:
    Z = np.load(f'{V2}/out/{p}_pitcher_pose_v2.npz')
    C0, E0 = DS.run(p)
    # consumer-style: the file's own tip_p; TIP_OVERRIDE None
    def lt(q, _o=DS._orig_load):
        T = dict(_o(q)); T.update({k: np.asarray(Z[k], float) for k in ('t', 'rotvec', 'pos', 'tip_p')}); return T
    RP.load_take = lt; RP.TIP_OVERRIDE = None
    try:
        C1 = RP.cup_poses(p)
    finally:
        RP.load_take = DS._orig_load
    fp = RP.forward_pitcher(); V0 = DS.FILL[p]["fill_at_cut_mL"] * 1e-6; s = slice(None, None, DS.SUB)
    E1 = fp.run(C1["t"][s], C1["tip"][s], C1["yaw"][s], C1["tilt"][s], C1["roll"][s], V0, C1["dt"] * DS.SUB)
    E1["t_abs"] = E1["t"] + C1["t0_abs"]; E1["axis_tilt"] = C1["axis_tilt_deg"][s]
    R = DS.compare(E0, E1, C0['dt'] * DS.SUB)
    R['t0_abs_cur_v2'] = [C0['t0_abs'], C1['t0_abs']]
    res[p] = R
    print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items()}, flush=True)
json.dump(res, open(f'{V2}/audit/out/ds_check.json', 'w'), indent=1)
