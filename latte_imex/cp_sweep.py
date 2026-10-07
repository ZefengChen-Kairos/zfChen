"""IMEX check: the sub-step count is set by the material speed, not by c_p.  Runs the first seconds of the
heart script for several c_p and reports steps, sub-steps, wall time, conservation and energy."""
import json, sys, time
import numpy as np
from .solver import Solver, Params, Numerics
from .actions import SCRIPTS
from .run import DEFAULT_PARAMS


def sweep(cps=(0.25, 0.5, 1.0, 2.0, 4.0), N=128, T=2.5, out="runs/cp_sweep.json"):
    rows = []
    for cp in cps:
        pd = dict(DEFAULT_PARAMS); pd["cp"] = cp
        sol = Solver(Params(**pd), Numerics(N=N))
        script = SCRIPTS["heart"]()
        t0 = time.time()
        while sol.t < T - 1e-9:
            inl, _ = script.sample(sol.t + 0.5 * sol.num.frame_dt)
            sol.advance_frame(inl)
        el = time.time() - t0
        L = sol.ledger(); ke, pe = sol.energy(); c = sol.concentration()
        row = dict(cp=cp, N=N, T=T, steps=sol.stats["steps"], max_substeps=sol.stats["max_substeps"],
                   mean_substeps=sol.stats["steps"] / sol.stats["frames"], retries=sol.stats["retries"],
                   cg_mass_iters_per_step=sol.stats["cg_mass"] / sol.stats["steps"], wall_s=el,
                   layer_error=L["layer_error"], milk_error=L["milk_error"], max_u=sol.stats["max_u"],
                   l_max=float(sol.l[sol.g.mask].max()), white_area=float(np.nansum(c > 0.4) * sol.g.area),
                   kinetic=ke, pressure_energy=pe)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    sweep()
