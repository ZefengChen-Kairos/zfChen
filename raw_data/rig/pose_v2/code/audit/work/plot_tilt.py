import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
V2 = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2'
SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
P = ['tulip', 'leaf', 'heart', 'push_heart', 'swan']
fig, ax = plt.subplots(5, 2, figsize=(18, 20), gridspec_kw=dict(width_ratios=[2.2, 1]))
win = {}
for i, p in enumerate(P):
    S = np.load(f'{V2}/audit/out/samples_{p}.npz')
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz'); zc = np.load(f'{SCR}/pours/{p}/tip.npz')
    # fastest 1.2 s window of the v2 trace
    w = S['w_g']; tg = S['tg']
    k = int(np.argmax(np.convolve(w, np.ones(240) / 240, 'same')))
    t0, t1 = tg[k] - 0.6, tg[k] + 0.6
    win[p] = [float(t0), float(t1)]
    for j, (a, lim) in enumerate(((ax[i, 0], None), (ax[i, 1], (t0, t1)))):
        a.plot(Z['t'], Z['tilt'], 'g-', lw=1.2, label='v2 (npz)')
        a.plot(zc['t'], zc['tilt'], 'r.-', lw=0.6, ms=3, alpha=0.7, label='current tip.npz')
        for kd, c, m in (('c0', 'b', 'x'), ('c1', 'm', '+'), ('j2', 'k', 'o')):
            s = (S['kind'] == kd) & (S['rms'] < 1.5)
            a.plot(S['t'][s], S['tilt'][s], m, color=c, ms=4 if kd != 'j2' else 3, mfc='none', label=f'per-sample {kd} (v2 model)')
        if lim:
            a.set_xlim(*lim); mm = (Z['t'] > lim[0]) & (Z['t'] < lim[1]); a.set_ylim(Z['tilt'][mm].min() - 3, Z['tilt'][mm].max() + 3)
        else:
            a.axvspan(t0, t1, color='y', alpha=0.2)
        a.set_ylabel(f'{p}: axis tilt [deg]'); a.grid(alpha=0.3)
    ax[i, 0].legend(fontsize=7, ncol=3)
ax[-1, 0].set_xlabel('t [s] (take clock)'); ax[-1, 1].set_xlabel('t [s] (fastest 1.2 s)')
fig.tight_layout(); fig.savefig(f'{V2}/audit/img/tilt_traces.jpg', dpi=60)
json.dump(win, open(f'{V2}/audit/out/fast_windows.json', 'w'))
print(win)
