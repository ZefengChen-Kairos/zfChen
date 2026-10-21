"""Data for the pose-recoverability report (read-only on the repo and on pose_v2/out, audit/out).
Per pour, on v2's 10 ms grid:
  angles  tilt (jug axis from vertical), heading (azimuth of the jug axis about the vertical, board frame),
          spin (downhill azimuth in the pitcher frame minus the spout's 90 deg: 0 = spout points straight downhill);
          spout tip in the board frame.  Same definitions for v2 and the current track (tip.npz samples; tip =
          production TIP_KONA (0, 50.2, 87.9) mm in the current pitcher frame).
  inst    per-instant band of v2: rolling rms (0.25 s) of (camera-0-only refit - camera-1-only refit) / sqrt(2), i.e.
          the error of one single-camera trajectory if the two are independent (conservative for the joint fit),
          floored by the formal sd.  Only where both refits are inside their own cameras' observation ranges.
  sys     systematic band of v2: v2 re-expressed in camera 1's world (the jointly fitted camera-1 extrinsic correction,
          inverted) and in camera 2's world (the constant camera-2 correction fitted by cam2_icp.py for v2; tulip: none
          -> median of the other four), max |change|; spin adds the spout-azimuth spread (+-2.5 deg) and the tip the
          resulting lateral +-2.2 mm (root sum square).
  classes tag coverage (cam 0 and/or cam 1 detection frame within +-0.6 frame interval; single tag), v2 angular
          speed bins, flip-ambiguous single-tag instants, flip suspects (flips_<p>.json), depth frames.
    python3 compute.py -> ../out/series_<p>.npz, ../out/numbers.json"""
import sys, json, pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from spline import Traj
from trimrec import Take

SCR = '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad'
V2 = f'{SCR}/pose_v2'
OUT = f'{V2}/report/out'
POURS = ['tulip', 'leaf', 'heart', 'push_heart', 'swan']
UP = np.array([0, 0, -1.0])
TIP_KONA = np.array([0.0, 0.0502, 0.0879])
CEN = np.array([0.15, 0.22, -0.03])
AZ_SPREAD = 2.5                     # deg, spout azimuth per pour 87.3-92.4 (builder) / 87.8-93.0 (audit) around 89.5-90


def angles(Rm):
    Rm = np.asarray(Rm)
    a = Rm[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(a @ UP, -1, 1)))
    head = np.degrees(np.unwrap(np.arctan2(a[:, 1], a[:, 0])))
    g = np.einsum('nji,j->ni', Rm, -UP)
    az = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    spin = (az - 90 + 180) % 360 - 180
    return tilt, head, spin


def wrapd(x):
    return (x + 180) % 360 - 180


def rolling_rms(x, n, valid):
    x = np.where(valid, x, 0.0); v = valid.astype(float)
    k = np.ones(n)
    s = np.convolve(x ** 2, k, 'same'); c = np.convolve(v, k, 'same')
    out = np.sqrt(s / np.maximum(c, 1)); out[c < n / 3] = np.nan
    return out


def runs(mask, t):
    """contiguous True runs -> list of (t_start, t_end)"""
    m = np.r_[False, mask, False].astype(int); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    dt = np.median(np.diff(t))
    return [(float(t[a] - dt / 2), float(t[b] + dt / 2)) for a, b in zip(s, e)]


AH = json.load(open(f'{SCR}/pose_audit/holdout/summary.json'))
C2 = {}
for p in POURS:
    try:
        C2[p] = json.load(open(f'{V2}/out/cam2icp_{p}.json'))['v2']
    except FileNotFoundError:
        pass
c2_med = (np.median([C2[q]['C2_rot_deg'] for q in C2], 0), np.median([C2[q]['C2_t_mm'] for q in C2], 0))
numbers = {}
for p in POURS:
    Z = np.load(f'{V2}/out/{p}_pose_v2.npz')
    t = Z['t']; R = Rot.from_rotvec(Z['rotvec']).as_matrix(); pos = Z['pos']; tip = Z['tip_board']
    B = Z['B_out_to_tag']; tip_tag = Z['tip_p_tagframe']; tip_p = Z['tip_p']; c1 = float(Z['offsets_s'][0])
    tilt, head, spin = angles(R)
    w = np.r_[np.degrees(np.linalg.norm((Rot.from_matrix(R[:-1]).inv() * Rot.from_matrix(R[1:])).as_rotvec(), axis=1)) / np.diff(t), 0]
    w[-1] = w[-2]
    # ---------------- per-instant band from the single-camera refits
    fits = {}
    for k in (0, 1):
        F = pickle.load(open(f'{V2}/out/full_{p}_v2k_cam{k}.pkl', 'rb'))
        fits[k] = Traj(F['T0'], F['dt'], F['R'], F['P'])
    tags = pickle.load(open(f'{V2}/data/tags_{p}.pkl', 'rb'))
    tk = Take(f'{SCR}/pours/ghdata/{p}')
    tags = [d for d in tags if d['q_edge'] is not None and d['cam'] in (0, 1) and d['tag'] in (0, 1, 4, 5)]
    fr = {}
    for d in tags:
        key = (d['cam'], d['step'])
        fr.setdefault(key, [d['t'] + (c1 if d['cam'] == 1 else 0.0), 0]); fr[key][1] += 1
    tf = {c: np.array(sorted(v[0] for k_, v in fr.items() if k_[0] == c)) for c in (0, 1)}
    nf = {c: np.array([v[1] for k_, v in sorted(fr.items(), key=lambda kv: kv[1][0]) if k_[0] == c]) for c in (0, 1)}
    rng = {c: (tf[c][0] - 0.05, tf[c][-1] + 0.05) for c in (0, 1)}
    ok = (t >= max(rng[0][0], rng[1][0], fits[0].T0, fits[1].T0)) & (t <= min(rng[0][1], rng[1][1], fits[0].t_end, fits[1].t_end))
    tt = np.clip(t, max(fits[0].T0, fits[1].T0), min(fits[0].t_end, fits[1].t_end) - 1e-6)
    Ra, pa = fits[0].eval(tt); Rb, pb = fits[1].eval(tt)
    Ra_o, Rb_o = Ra @ B[:3, :3], Rb @ B[:3, :3]
    ta, ha, sa = angles(Ra_o); tb, hb, sb = angles(Rb_o)
    tipa = np.einsum('nij,j->ni', Ra, tip_tag) + pa; tipb = np.einsum('nij,j->ni', Rb, tip_tag) + pb
    d_tilt, d_head, d_spin = ta - tb, wrapd(ha - hb), wrapd(sa - sb)
    d_tip = (tipa - tipb) * 1e3
    rotd = np.degrees(np.linalg.norm((Rot.from_matrix(Ra_o) * Rot.from_matrix(Rb_o).inv()).as_rotvec(), axis=1))
    NW = 25
    inst = {}
    for name, d, floor in (('tilt', d_tilt, Z['sd_tilt_deg']), ('head', d_head, Z['sd_spin_deg']), ('spin', d_spin, Z['sd_spin_deg'])):
        b = rolling_rms(d, NW, ok) / np.sqrt(2)
        inst[name] = np.fmax(np.where(np.isfinite(b), b, np.nan), floor)
    inst['tip'] = np.stack([rolling_rms(d_tip[:, j], NW, ok) / np.sqrt(2) for j in range(3)], 1)
    inst['tipn'] = rolling_rms(np.linalg.norm(d_tip, axis=1), NW, ok) / np.sqrt(2)
    # ---------------- systematic band: camera-1 world and camera-2 world
    worlds = {}
    e1 = np.array(json.loads(str(Z['layout_json']))['ext1']); ec = np.array(json.loads(str(Z['layout_json']))['ext_cen'])
    worlds['cam1'] = (Rot.from_rotvec(e1[:3]).as_matrix(), e1[3:], ec)
    if p in C2:
        rd, tm = np.array(C2[p]['C2_rot_deg']), np.array(C2[p]['C2_t_mm'])
        c2src = 'cam2icp v2 (this pour)'
    else:
        rd, tm = c2_med; c2src = 'median of the other four pours (tulip cam2_icp crashed)'
    worlds['cam2'] = (Rot.from_rotvec(np.radians(rd)).as_matrix(), tm * 1e-3, CEN)
    sysd = {}
    for wn, (Rw, tw, cw) in worlds.items():
        Rk = np.einsum('ij,njk->nik', Rw, R)
        tk_, hk, sk = angles(Rk)
        tipk = (tip - cw) @ Rw.T + cw + tw
        sysd[wn] = dict(tilt=tk_ - tilt, head=wrapd(hk - head), spin=wrapd(sk - spin), tip=(tipk - tip) * 1e3)
    sys_ = {}
    for k in ('tilt', 'head'):
        sys_[k] = np.max([np.abs(sysd[w][k]) for w in sysd], 0)
    sys_['spin'] = np.sqrt(np.max([np.abs(sysd[w]['spin']) for w in sysd], 0) ** 2 + AZ_SPREAD ** 2)
    lat = np.linalg.norm(tip_p[:2]) * 1e3 * np.radians(AZ_SPREAD)
    sys_['tip'] = np.max([np.abs(sysd[w]['tip']) for w in sysd], 0)                  # per component
    sys_['tip'] = np.sqrt(sys_['tip'] ** 2 + (lat ** 2) / 2)                            # lateral split over x / y
    sys_['tipn'] = np.sqrt(np.max([np.linalg.norm(sysd[w]['tip'], axis=1) for w in sysd], 0) ** 2 + lat ** 2)
    # ---------------- current track
    zc = np.load(f'{SCR}/pours/{p}/tip.npz')
    tc = zc['t']; o = np.argsort(tc); tc = tc[o]; Pc = zc['poses'][o]
    Rc = Rot.from_rotvec(Pc[:, :3]).as_matrix()
    tilt_c, head_c, spin_c = angles(Rc)
    tip_c = np.einsum('nij,j->ni', Rc, TIP_KONA) + Pc[:, 3:]
    # align unwrapped headings to the same branch
    head_c = head_c + 360 * np.round((np.interp(tc, t, head) - head_c) / 360)
    # v2 at the current sample times
    m_in = (tc >= t[0]) & (tc <= t[-1])
    iv = lambda y: np.interp(tc[m_in], t, y)
    dcur = dict(tilt=tilt_c[m_in] - iv(tilt), head=wrapd(head_c[m_in] - iv(head)), spin=wrapd(spin_c[m_in] - iv(spin)),
                tip=(tip_c[m_in] - np.stack([iv(tip[:, j]) for j in range(3)], 1)) * 1e3)
    th = AH[p]['time_holdout']
    cur_band = dict(tilt=th['tip_filtered']['tilt_abs_deg'][0], rot=th['tip_filtered']['rot_deg'][0], tip=th['tip_filtered']['tip_mm'][0])
    cur_x = AH[p]['time_holdout']['cam1_vs_cam0stream']
    # ---------------- tag / depth coverage classes on the grid
    fi = {c: np.median(np.diff(tk.ts[c])) for c in range(3)}
    h = {c: 0.6 * fi[c] for c in (0, 1)}
    near = {}
    for c in (0, 1):
        j = np.clip(np.searchsorted(tf[c], t), 1, len(tf[c]) - 1)
        jj = np.where(np.abs(tf[c][j - 1] - t) < np.abs(tf[c][j] - t), j - 1, j)
        near[c] = (np.abs(tf[c][jj] - t) <= h[c], nf[c][jj], tf[c][jj])
    has0, has1 = near[0][0], near[1][0]
    nt = np.where(has0 & ~has1, near[0][1], np.where(has1 & ~has0, near[1][1], 0))
    cls = np.where(has0 & has1, 3, np.where(has0 | has1, np.where(nt >= 2, 2, 1), 0))   # 3 both, 2 one multi-tag, 1 one single-tag, 0 none
    # depth frames used by the fit
    dz = np.load(f'{V2}/data/depth_{p}_t1.npz')
    dft = dz['ft'] + np.where(dz['fcam'] == 1, c1, 0.0)
    dsort = np.sort(dft)
    j = np.clip(np.searchsorted(dsort, t), 1, len(dsort) - 1)
    hasdep = np.minimum(np.abs(dsort[j] - t), np.abs(dsort[j - 1] - t)) <= 0.6 * min(fi[0], fi[1])
    # flips
    FL = json.load(open(f'{OUT}/flips_{p}.json'))
    sus_t = np.array([r['t'] for r in FL if r['suspect']])
    amb_single = {(r['cam'], round(r['t'], 4)) for r in FL if r['ambiguous'] and r['ntag_frame'] == 1}
    amb_inst = np.zeros(len(t), bool)
    for c in (0, 1):
        only = (cls == 1) & (near[c][0]) & ~(near[1 - c][0])
        ta_ = near[c][2]
        amb_inst |= only & np.array([(c, round(x, 4)) in amb_single for x in ta_])
    # union gaps (no tag in either camera)
    tu = np.sort(np.r_[tf[0], tf[1]]); tu = tu[(tu >= t[0]) & (tu <= t[-1])]
    gaps = np.diff(np.r_[t[0], tu, t[-1]])
    speed = np.where(w < 10, 0, np.where(w < 60, 1, 2))
    span = t[-1] - t[0]
    fr_ = lambda m: round(float(np.mean(m)), 4)
    NB = dict(
        span_s=round(float(span), 2), t_range=[float(t[0]), float(t[-1])],
        take_range_cams=[[float(tk.ts[c][0] - tk.t0), float(tk.ts[c][-1] - tk.t0)] for c in range(3)],
        fps=[round(1 / fi[c], 2) for c in range(3)],
        n_tag_frames=[int(len(tf[0])), int(len(tf[1]))], n_frames=[int(len(tk.ts[c])) for c in range(3)],
        frac_frames_with_tag=[round(len(tf[c]) / np.sum((tk.ts[c] - tk.t0 >= t[0]) & (tk.ts[c] - tk.t0 <= t[-1])), 3) for c in (0, 1)],
        frac_time=dict(both=fr_(cls == 3), one_multi=fr_(cls == 2), one_single=fr_(cls == 1), one_single_ambiguous=fr_(amb_inst),
                       none=fr_(cls == 0), none_with_depth=fr_((cls == 0) & hasdep), none_without_depth=fr_((cls == 0) & ~hasdep),
                       slow=fr_(speed == 0), mid=fr_(speed == 1), fast=fr_(speed == 2), cam0=fr_(has0), cam1=fr_(has1)),
        gaps=dict(n_gt100=int((gaps > 0.1).sum()), n_gt150=int((gaps > 0.15).sum()), max_ms=round(float(gaps.max() * 1e3), 1),
                  time_gt100_s=round(float(gaps[gaps > 0.1].sum()), 2), time_gt150_s=round(float(gaps[gaps > 0.15].sum()), 2),
                  frac_gt100=round(float(gaps[gaps > 0.1].sum() / span), 4)),
        flips=dict(n_det=len(FL), n_suspect=int(len(sus_t)), suspect_t=[round(float(x), 3) for x in sus_t],
                   n_amb_detections=int(sum(r['ambiguous'] for r in FL)), n_amb_single_tag_frames=len(amb_single)),
        inst=dict(tilt_p50=round(float(np.nanmedian(inst['tilt'][ok])), 3), tilt_p90=round(float(np.nanpercentile(inst['tilt'][ok], 90)), 3),
                  head_p50=round(float(np.nanmedian(inst['head'][ok])), 3), spin_p50=round(float(np.nanmedian(inst['spin'][ok])), 3),
                  spin_p90=round(float(np.nanpercentile(inst['spin'][ok], 90)), 3), head_p90=round(float(np.nanpercentile(inst['head'][ok], 90)), 3),
                  tip_p50=round(float(np.nanmedian(inst['tipn'][ok])), 3), tip_p90=round(float(np.nanpercentile(inst['tipn'][ok], 90)), 3),
                  frac_covered=fr_(ok),
                  raw_abs_diff=dict(tilt=[round(float(np.percentile(np.abs(d_tilt[ok]), q)), 3) for q in (50, 90)],
                                    head=[round(float(np.percentile(np.abs(d_head[ok]), q)), 3) for q in (50, 90)],
                                    spin=[round(float(np.percentile(np.abs(d_spin[ok]), q)), 3) for q in (50, 90)],
                                    rot=[round(float(np.percentile(rotd[ok], q)), 3) for q in (50, 90)],
                                    tip=[round(float(np.percentile(np.linalg.norm(d_tip[ok], axis=1), q)), 3) for q in (50, 90)]),
                  by_speed={nm: dict(tilt_p90=round(float(np.percentile(np.abs(d_tilt[ok & (speed == s)]), 90)), 2) if (ok & (speed == s)).sum() > 5 else None,
                                     head_p90=round(float(np.percentile(np.abs(d_head[ok & (speed == s)]), 90)), 2) if (ok & (speed == s)).sum() > 5 else None,
                                     spin_p90=round(float(np.percentile(np.abs(d_spin[ok & (speed == s)]), 90)), 2) if (ok & (speed == s)).sum() > 5 else None,
                                     tip_p90=round(float(np.percentile(np.linalg.norm(d_tip[ok & (speed == s)], axis=1), 90)), 2) if (ok & (speed == s)).sum() > 5 else None,
                                     n_s=round(float((ok & (speed == s)).sum() * 0.01), 2))
                            for nm, s in (('slow', 0), ('mid', 1), ('fast', 2))},
                  by_cov={nm: dict(tilt_p90=round(float(np.percentile(np.abs(d_tilt[ok & m]), 90)), 2) if (ok & m).sum() > 5 else None,
                                   tip_p90=round(float(np.percentile(np.linalg.norm(d_tip[ok & m], axis=1), 90)), 2) if (ok & m).sum() > 5 else None,
                                   spin_p90=round(float(np.percentile(np.abs(d_spin[ok & m]), 90)), 2) if (ok & m).sum() > 5 else None,
                                   n_s=round(float((ok & m).sum() * 0.01), 2))
                          for nm, m in (('both', cls == 3), ('one_multi', cls == 2), ('one_single', cls == 1), ('none', cls == 0))}),
        sys=dict(tilt_med=round(float(np.median(sys_['tilt'])), 2), tilt_max=round(float(sys_['tilt'].max()), 2),
                 head_med=round(float(np.median(sys_['head'])), 2), spin_med=round(float(np.median(sys_['spin'])), 2),
                 tip_med=round(float(np.median(sys_['tipn'])), 2), tip_max=round(float(sys_['tipn'].max()), 2),
                 per_world={wn: dict(tilt_med=round(float(np.median(sysd[wn]['tilt'])), 2), head_med=round(float(np.median(sysd[wn]['head'])), 2),
                                     spin_med=round(float(np.median(sysd[wn]['spin'])), 2),
                                     tip_med_mm=[round(float(x), 2) for x in np.median(sysd[wn]['tip'], 0)]) for wn in sysd},
                 cam2_source=c2src, lateral_tip_from_az_mm=round(float(lat), 2)),
        cur_vs_v2=dict(n=int(m_in.sum()),
                       tilt=[round(float(np.median(dcur['tilt'])), 2), round(float(np.percentile(np.abs(dcur['tilt'] - np.median(dcur['tilt'])), 90)), 2)],
                       head=[round(float(np.median(dcur['head'])), 2), round(float(np.percentile(np.abs(dcur['head'] - np.median(dcur['head'])), 90)), 2)],
                       spin=[round(float(np.median(dcur['spin'])), 2), round(float(np.percentile(np.abs(dcur['spin'] - np.median(dcur['spin'])), 90)), 2)],
                       tip_med_vec=[round(float(x), 2) for x in np.median(dcur['tip'], 0)],
                       tip_scatter_p90=round(float(np.percentile(np.linalg.norm(dcur['tip'] - np.median(dcur['tip'], 0), axis=1), 90)), 2)),
        cur_band=cur_band,
        cur_xcam=dict(tilt=cur_x['tilt_abs_deg'][:2], rot=cur_x['rot_deg'][:2], tip=cur_x['tip_mm'][:2]),
        omega=dict(p50=round(float(np.median(w)), 1), p95=round(float(np.percentile(w, 95)), 1), max=round(float(w.max()), 1)),
    )
    numbers[p] = NB
    print(p, json.dumps(NB['frac_time']), NB['gaps'], NB['inst']['tilt_p50'], NB['inst']['tip_p50'], NB['sys']['tilt_med'], NB['sys']['tip_med'])
    np.savez_compressed(f'{OUT}/series_{p}.npz', t=t, tilt=tilt, head=head, spin=spin, tip=tip * 1e3, w=w,
                        inst_tilt=inst['tilt'], inst_head=inst['head'], inst_spin=inst['spin'], inst_tip=inst['tip'], inst_tipn=inst['tipn'], ok=ok,
                        sys_tilt=sys_['tilt'], sys_head=sys_['head'], sys_spin=sys_['spin'], sys_tip=sys_['tip'], sys_tipn=sys_['tipn'],
                        tc=tc, tilt_c=tilt_c, head_c=head_c, spin_c=spin_c, tip_c=tip_c * 1e3, tc_in=tc[m_in], dtip_c=dcur['tip'],
                        cls=cls, speed=speed, hasdep=hasdep, amb=amb_inst, sus_t=sus_t, tf0=tf[0], tf1=tf[1])
json.dump(numbers, open(f'{OUT}/numbers.json', 'w'), indent=1)
