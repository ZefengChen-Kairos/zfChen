"""Quick look: run a pattern at low resolution and save a grid of snapshots at phase changes + regular times."""
import sys, json, numpy as np
from .run import run
from . import render

def peek(pattern, N=192, params=None, out=None, every=1.0, T=None, script_kw=None):
    out = out or f"runs/peek_{pattern}"
    sol, met = run(pattern, N=N, out=out, params=params, snap_dt=0.1, video=False, quiet=True, T_max=T, script_kw=script_kw)
    z = np.load(f"{out}/snapshots_c.npz"); t = z["t"]; c = z["c"].astype(np.float32)
    ev = [e["t"] for e in met["events"]][1:]
    times = sorted(set([round(x, 1) for x in ev] + list(np.arange(0, t[-1] + 1e-6, every)) + [t[-1]]))
    items = []
    for tt in times:
        k = int(np.argmin(np.abs(t - tt)))
        items.append((f"t={t[k]:.1f}", c[k], sol.g.mask))
    render.overview(f"{out}/grid.png", items, ncols=4)
    print(f"deposited={met['ledger']['deposited']:.3f} white={met['white_area_c_gt_0p4']:.3f} lmax={sol.l.max():.2f} elapsed={met['elapsed_s']:.0f}s retries={sol.stats['retries']}")
    return sol, met

if __name__ == "__main__":
    pat = sys.argv[1]
    params, skw, out, N = {}, {}, None, 192
    for kv in sys.argv[2:]:
        k, v = kv.split("=")
        if k == "out": out = v
        elif k == "N": N = int(v)
        elif k.startswith("s."): skw[k[2:]] = float(v)
        else: params[k] = v if k == "return_law" else float(v)
    peek(pat, params=params, script_kw=skw, out=out, N=N)
