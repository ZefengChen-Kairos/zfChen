"""A1: format / consumer check of the delivered v2 files vs the current files."""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
V2 = f'{SCR}/pose_v2'; RES = '/home/user/zfChen/raw_data/rig/results'
UP = np.array([0, 0, -1.0])
def tilt_az(Rm):
    a = Rm[:, :, 2]; tilt = np.degrees(np.arccos(np.clip(a @ UP, -1, 1)))
    g = np.einsum('nji,j->ni', Rm, -UP); az = np.degrees(np.arctan2(g[:, 1], g[:, 0])); return tilt, az
def desc(z):
    return {k: (z[k].dtype.str, z[k].shape) for k in z.files}
for p in ['tulip', 'leaf', 'heart', 'push_heart', 'swan']:
    cur_tip = np.load(f'{SCR}/pours/{p}/tip.npz'); cur_pp = np.load(f'{RES}/{p}/pitcher_pose.npz')
    v_tip = np.load(f'{V2}/out/{p}_pose_v2.npz'); v_pp = np.load(f'{V2}/out/{p}_pitcher_pose_v2.npz')
    print('=====', p)
    print(' cur tip.npz keys', desc(cur_tip))
    print(' v2  pose keys   ', {k: v for k, v in desc(v_tip).items() if k in cur_tip.files})
    print('   missing in v2 tip:', set(cur_tip.files) - set(v_tip.files))
    print(' cur pitcher_pose', desc(cur_pp))
    print(' v2  pitcher_pose', desc(v_pp))
    print('   key sets equal:', set(cur_pp.files) == set(v_pp.files))
    # is pitcher_pose.npz == tip.npz currently?
    same = len(cur_pp['t']) == len(cur_tip['t']) and np.allclose(cur_pp['t'], cur_tip['t']) and np.allclose(cur_pp['rotvec'], cur_tip['poses'][:, :3])
    print(' current pitcher_pose.npz identical to tip.npz:', same, ' tip_p cur (mm)', (cur_tip['tip_p'] * 1e3).round(1), 'v2', (v_tip['tip_p'] * 1e3).round(1))
    for name, z in (('cur', cur_tip), ('v2', v_tip)):
        t = z['t']; print(f'  {name} t [{t[0]:.3f}, {t[-1]:.3f}] n {len(t)} dt med {np.median(np.diff(t))*1e3:.1f} ms monotonic {bool(np.all(np.diff(t) > 0))} nan {int(sum(np.isnan(z[k]).sum() for k in ("t","poses","tilt","az","tip_board")))}')
    # consistency of v2 fields
    R = Rot.from_rotvec(v_tip['poses'][:, :3]).as_matrix()
    tl, az = tilt_az(R)
    tipb = np.einsum('nij,j->ni', R, v_tip['tip_p']) + v_tip['poses'][:, 3:]
    print('  v2 internal: |tilt - recomputed| max', float(np.abs(tl - v_tip['tilt']).max()), ' az', float(np.abs(np.angle(np.exp(1j*np.radians(az - v_tip['az'])))).max()),
          ' tip_board max err mm', float(np.abs(tipb - v_tip['tip_board']).max() * 1e3),
          ' pp==tip:', all(np.allclose(v_pp[k], v_tip[k2]) for k, k2 in (('t','t'),('rotvec','rotvec'),('pos','pos'),('tilt_deg','tilt'),('az_deg','az'),('tip_board','tip_board'))),
          ' rotvec==poses[:, :3]', np.allclose(v_tip['rotvec'], v_tip['poses'][:, :3]))
    # compare to current on the overlapping time: tilt, az, origin, tip_board
    tc = cur_tip['t']; m = (tc >= v_tip['t'][0]) & (tc <= v_tip['t'][-1])
    def I(x): return np.stack([np.interp(tc[m], v_tip['t'], x[:, j]) for j in range(x.shape[1])], 1)
    dtilt = np.interp(tc[m], v_tip['t'], v_tip['tilt']) - cur_tip['tilt'][m]
    daz = np.degrees(np.angle(np.exp(1j*np.radians(np.interp(tc[m], v_tip['t'], np.unwrap(np.radians(v_tip['az']))*180/np.pi) - cur_tip['az'][m]))))
    dtip = (I(v_tip['tip_board']) - cur_tip['tip_board'][m]) * 1e3
    dorg = (I(v_tip['poses'][:, 3:]) - cur_tip['poses'][m, 3:]) * 1e3
    print(f'  v2 - cur: tilt med {np.median(dtilt):+.2f} p90|.| {np.percentile(np.abs(dtilt),90):.2f} deg; az med {np.median(daz):+.2f} deg; '
          f'tip_board med vec {np.median(dtip,0).round(1)} |.|p50 {np.median(np.linalg.norm(dtip,axis=1)):.1f} mm; origin med vec {np.median(dorg,0).round(1)} mm')
    # time-lag between v2 and current (tilt), using cross-correlation of tilt derivative on 5 ms grid
    tg = np.arange(max(tc[0], v_tip['t'][0]), min(tc[-1], v_tip['t'][-1]), 0.005)
    a = np.interp(tg, v_tip['t'], v_tip['tilt']); b = np.interp(tg, tc, cur_tip['tilt'])
    best = None
    for lag in np.arange(-0.1, 0.1001, 0.005):
        bb = np.interp(tg + lag, tc, cur_tip['tilt'])
        e = np.mean(np.abs((a - a.mean()) - (bb - bb.mean())))
        if best is None or e < best[1]: best = (lag, e)
    print(f'  best time shift cur(t+lag) ~ v2(t): lag {best[0]*1e3:+.0f} ms (mean abs tilt diff {best[1]:.2f} deg)')
    # coverage relative to disc.json / sim clock
    D = json.load(open(f'{RES}/{p}/disc.json')); dt0 = D[0]['t']; dt1 = D[-1]['t']
    print(f'  disc t [{dt0:.3f}, {dt1:.3f}] -> sim t0_abs cur {max(cur_pp["t"][0], dt0):.3f}  v2 {max(v_pp["t"][0], dt0):.3f};  end cur {min(cur_pp["t"][-1], dt1):.3f} v2 {min(v_pp["t"][-1], dt1):.3f}')
    print('  tip json', json.load(open(f'{V2}/out/{p}_tip_v2.json')))
