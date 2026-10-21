"""Step 1: converged tag layout per pour, apothem / radius FIXED to the CAD at the tag height, per-sample poses
(pitcher_ba sample rule, cameras 0/1), analytic Jacobians, LM to convergence, flip re-check of every sample with the
converged layout (multi-start), re-solve.
    python3 layout_fit.py <pour> <variant> [corners]
variants: flat (a = 42.60 mm CAD apothem, planar tags on the 14-gon flats), cyl426 / cyl437 (wrapped on a cylinder of
radius 42.6 / 43.7 = CAD apothem / circumradius), flat_s (flat + free tag scale), flat_u (flat + free offset of each
tag along its flat), flat_r (flat with the apothem free: the old free-r check)."""
import sys, time, pickle, json
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/pose_v2/work')
from pv2lib import *

A_CAD = 0.04260          # outer apothem of the 14-gon at CAD z 28.5 (tag 4 centre): circumradius 43.70 x cos(pi/14)
R_CIRC = 0.04370


def init_layout(p, a, flat, D, free_u=False):
    ba = pickle.load(open(f"{SCR}/pours/{p}/ba.pkl", "rb"))
    init = {t: list(v) for t, v in ba['layout'].items()}
    init = {t: v for t, v in init.items() if t in (1, 4, 5)}       # tag 0 (if any) is re-placed by add_tag
    L = Layout(a, sorted(init), flat, init=init, free_u=free_u)
    return L


def add_tag(D, L, new, ref, min_n=5):
    """place a tag never in the layout from same-frame IPPE relative transforms to a tag in the layout."""
    if new in L.tp:
        return True
    by = D.by_frame()
    rel = []
    for k, v in by.items():
        dd = {d['tag']: d for d in v}
        if new in dd and ref in dd:
            Ta = dd[ref]['T_board_alts'][0]; Tb = dd[new]['T_board_alts'][0]
            rel.append(np.linalg.inv(Ta) @ Tb)
    if len(rel) < min_n:
        return False
    t = np.median([T[:3, 3] for T in rel], 0)
    Tr = T_pt(L, ref)
    c = Tr[:3, :3] @ t + Tr[:3, 3]                     # centre of the new tag in the pitcher frame
    phi = float(np.arctan2(c[1], c[0]))
    L.tags = sorted(L.tags + [new]); L.tp[new] = np.array([phi, float(c[2]), 0.0, 0.0])
    return True


def solve(p, variant, corners='edge', verbose=False):
    D = PourData(p, corners)
    flat = variant.startswith('flat')
    a = {'cyl426': A_CAD, 'cyl437': R_CIRC}.get(variant, A_CAD)
    L = init_layout(p, a, flat, D, free_u=(variant == 'flat_u'))
    if not variant.endswith('_cam0') and add_tag(D, L, 0, 1):
        print(p, 'tag 0 added', np.round([np.degrees(L.tp[0][0]), L.tp[0][1] * 1e3], 1), flush=True)
    D.rows = [d for d in D.rows if d['tag'] in L.tp]
    if variant.endswith('_cam0'):                      # single-camera geometry: tags 4+5 of camera 0 only
        D.rows = [d for d in D.rows if d['cam'] == 0]
        L.tags = [t for t in L.tags if t in (4, 5)]; L.tp = {t: L.tp[t] for t in L.tags}
    sync = 0.005 if 'sync5' in variant else 0.012
    samples, tt, by = build_samples(D, sync=sync, cams=(0,) if variant.endswith('_cam0') else CAMS)
    # initial poses: best IPPE candidate (max corner error over the sample)
    P0 = []
    for g in samples:
        obs = [d for k in g for d in by[k]]
        sols = frame_candidates(D, L, obs, iters=15)
        P0.append(sols[0][1])
    P0 = np.array(P0)
    fix_scale = variant != 'flat_s'
    t0 = time.time()
    P, L, info = sample_ba(D, L, P0, samples, by, fix_scale=fix_scale, verbose=verbose)
    if variant == 'flat_r':
        # free apothem: outer loop on a (1-D golden search on the converged cost)
        from scipy.optimize import minimize_scalar
        cache = {}

        def cost_a(a_):
            L2 = L.copy(); L2.a = a_
            P2, L2, inf2 = sample_ba(D, L2, P, samples, by, fix_scale=True)
            cache[a_] = (P2, L2, inf2)
            return inf2['cost']
        res = minimize_scalar(cost_a, bounds=(0.038, 0.050), method='bounded', options=dict(xatol=2e-5))
        P, L, info = cache[min(cache, key=lambda k: cache[k][2]['cost'])]
    rounds = []
    for rnd in range(3):
        # flip re-check: every sample re-solved from all IPPE alternatives + its current pose, fixed layout
        changed = 0
        for si, g in enumerate(samples):
            obs = [d for k in g for d in by[k]]
            sols = frame_candidates(D, L, obs, extra=[P[si]], iters=30)
            best = sols[0][1]
            if rot_deg(best, P[si]) > 2.0 or np.linalg.norm(best[3:] - P[si][3:]) > 0.004:
                P[si] = best; changed += 1
        rounds.append(changed)
        if changed == 0:
            break
        P, L, info = sample_ba(D, L, P, samples, by, fix_scale=fix_scale)
    sd = np.sqrt(np.diag(info['layout_cov'])) if 'layout_cov' in info else None
    out = dict(pour=p, variant=variant, corners=corners, layout=L.as_dict(), poses=P, t=tt, samples=samples,
               rms=info['rms'], med=info['med'], cost=info['cost'], status=info['status'], iters=info['iters'],
               n_obs=info['n_obs'], flip_rounds=rounds, time_s=time.time() - t0,
               layout_sd={f'{t}.{j}': float(s) for (t, j), s in zip(info['layout_idx'], sd)} if sd is not None else None,
               resid=info['resid'], oi=info['oi'], obs_tags=info['tags'], obs_cams=info['cams'], obs_steps=info['steps'])
    print(p, variant, corners, 'rms', round(info['rms'], 4), 'med', round(info['med'], 4), 'cost', round(info['cost'], 3),
          info['status'], 'iters', info['iters'], 'flips fixed', rounds, 'a mm', round(L.a * 1e3, 2), 'tag mm', round(L.s * TAG * 1e3, 3),
          'layout', L.pretty(), f'{time.time() - t0:.0f}s', flush=True)
    return out


if __name__ == '__main__':
    p, variant = sys.argv[1], sys.argv[2]
    corners = sys.argv[3] if len(sys.argv) > 3 else 'edge'
    out = solve(p, variant, corners)
    pickle.dump(out, open(f"{V2}/out/layout_{p}_{variant}_{corners}.pkl", "wb"))
