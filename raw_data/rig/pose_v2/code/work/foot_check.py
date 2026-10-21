"""Landing point vs the MEASURED stream foot (raw_data/rig/results/<p>/stream_foot.json, trusted frames; traced from the
images with the 3-camera top-hat cue, independent of either pose track except for its +-30 mm search grid).
Production chain (downstream.run: cup_poses + ForwardPitcher kona700) for the current track and v2; at every trusted
foot time: landing - foot in the cup frame [mm] -> median |d|, p90, mean vector (bias); fast component (deviation from
a 0.53 s running mean) of landing and foot: rms and correlation (does the track carry the real fast motion?).
    python3 foot_check.py -> ../out/foot_check.json, ../img/foot_<p>.jpg"""
import sys, json
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
import downstream as DS
import latte_imex.rig.real_program as RP
from scipy.ndimage import uniform_filter1d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

V2 = DS.V2
res = {}
for p in ["tulip", "leaf", "heart", "push_heart", "swan"]:
    F = json.load(open(f'/home/user/zfChen/raw_data/rig/results/{p}/stream_foot.json'))['frames']
    tr = [f for f in F if f.get('trusted')]
    T = RP.load_take(p)
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz')
    C0, E0 = DS.run(p)
    C1, E1 = DS.run(p, dict(t=Z['t'], rotvec=Z['poses'][:, :3], pos=Z['poses'][:, 3:]), tip=Z['tip_p'])
    # foot times: stream_foot.json 't' is on the sim clock (cup_poses t = 0 at C['t0_abs'])
    tf = np.array([f['t'] for f in tr]) + C0['t0_abs']
    fb = np.array([f['foot_board'] for f in tr])
    cx = np.interp(tf, T['disc_t'], T['disc_c'][:, 0]); cy = np.interp(tf, T['disc_t'], T['disc_c'][:, 1])
    foot = np.stack([fb[:, 0] - cx, -(fb[:, 1] - cy)], 1) * 1e3          # cup frame F (x, -y) [mm]
    rec = {}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, E in (('cur', E0), ('v2', E1)):
        on = E['Q_o'] > 1e-6
        ok = (tf >= E['t_abs'][on][0]) & (tf <= E['t_abs'][on][-1])
        xh = np.stack([np.interp(tf, E['t_abs'][on], E['x_hit'][on, j]) for j in range(2)], 1) * 1e3
        d = xh[ok] - foot[ok]
        # fast components on a 15 Hz grid
        tg = np.arange(tf[ok][0], tf[ok][-1], 1 / 15.)
        fg = np.stack([np.interp(tg, tf[ok], foot[ok, j]) for j in range(2)], 1)
        hg = np.stack([np.interp(tg, E['t_abs'][on], E['x_hit'][on, j]) * 1e3 for j in range(2)], 1)
        ff = fg - np.stack([uniform_filter1d(fg[:, j], 8, mode='nearest') for j in range(2)], 1)
        hf = hg - np.stack([uniform_filter1d(hg[:, j], 8, mode='nearest') for j in range(2)], 1)
        corr = [float(np.corrcoef(ff[:, j], hf[:, j])[0, 1]) for j in range(2)]
        rec[name] = dict(n=int(ok.sum()), absdiff_p50=float(np.median(np.linalg.norm(d, axis=1))), absdiff_p90=float(np.percentile(np.linalg.norm(d, axis=1), 90)),
                         bias_mm=d.mean(0).round(2).tolist(), absdiff_debiased_p50=float(np.median(np.linalg.norm(d - d.mean(0), axis=1))),
                         fast_rms_landing_mm=np.sqrt((hf ** 2).mean(0)).round(2).tolist(), fast_rms_foot_mm=np.sqrt((ff ** 2).mean(0)).round(2).tolist(),
                         fast_corr_xy=np.round(corr, 3).tolist())
        for j, c in ((0, 'C0'), (1, 'C1')):
            ax[j].plot(E['t_abs'][on], E['x_hit'][on, j] * 1e3, '-', lw=0.8, label=f'{name} landing', alpha=0.8)
    for j in range(2):
        ax[j].plot(tf, foot[:, j], 'k.', ms=4, label='measured stream foot (trusted)')
        ax[j].set_xlabel('t [s]'); ax[j].set_ylabel(['x', 'y'][j] + ' [mm] (cup frame)'); ax[j].legend(fontsize=7)
    ax[0].set_title(f'{p}: landing point vs measured stream foot')
    fig.tight_layout(); fig.savefig(f'{V2}/img/foot_{p}.jpg', dpi=70); plt.close(fig)
    res[p] = rec
    print(p, json.dumps(rec), flush=True)
json.dump(res, open(f'{V2}/out/foot_check.json', 'w'), indent=1)
