"""viz29: forward re-run of a fitted closure for the three-view videos -- the white fraction C saved densely (the solver
frame ending nearest every 1/fps s), plus everything viz/scene.py needs (measured poses, the pour model's emission and
arrivals, the schedule the solver used, the pitcher grid geometry), in one npz.

The run is diag.py's own forward re-run of a closure (diag.render -> render512.make_pour + render512.run_pour): DiffPour
tau 0.05, scales (0.3, 0.15, 0), the closure's switches (render512.load_closure), parameters D.s1 + surface-current
defaults + the closure's numbers; only the snapshot list differs (dense instead of every 4th frame), and every snapshot
is reduced to uint8 C at once (no device copies pile up).

Inlet: resolved exactly as the update21 fit resolved it (autodl_diag21.sh model_env + diag / render512):
  - the closure's key 'inlet' (0 record, 1 law, 2 replay).  An oracle run's best_closure.json (diag modes inlet / both
    / both_wide) has that key removed; then the calibration pours' mode is read from the sibling done.json
    ('inlet' -> 'cal' -> <pour> -> 'mode'); no key and no done.json: 'record' (the control model v05n_ctrl);
  - environment: LATTE_INLET unset; LATTE_FOOT=mi for a measured-inlet closure (law / replay), unset for a record
    closure (E0 'world' on the v1 feet) -- set here as the launcher's model_env does (--keep-env leaves the environment
    alone); LATTE_POSE=v2 unless set;
  - calibration pours (tulip, leaf) of an MI closure: MI-R replay (+ E0 world on the validated feet, LATTE_FOOT=mi);
    held-out pours (heart, push_heart, swan) of an MI closure: MI-L with e0 off (render512.heldout_inlet = diag's
    held-out test); record closures: every pour record + the closure's e0;
  - --oracle auto (default): when the closure sits next to a diag best.json with inlet corrections u_in for this pour
    (oracle modes), they are applied as diag.render applies them (InletCorr(D, box).apply + assign); 'off' shows the
    closure under the measured inlet only.  meta.oracle says which.

    python forward_fields.py --closure results_diag21/s1kq/params/best_closure.json --pour tulip --N 256 --fps 30 \\
        --out viz_fields/s1kq_params_tulip.npz [--device cuda:0] [--T secs] [--pkg ROOT]
    python forward_fields.py ... --inlet-only        # no solver: poses, emission, arrivals, schedule (C empty)
Output npz: C (n, N, N) uint8 (= round(255 m / l), 0 outside the cup), t (n) frame-end times [s], k (n) solver frames,
t_render (n) the 1/fps times they stand for, mask (N, N), S_* schedule, P_* poses, E_* emission, R_* arrivals,
G_* pitcher grid (mm), LAW (json), meta (json).  Clock: real_program.cup_poses t (= solver = video_obs clock).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time

import numpy as np

E_KEYS = ("t", "V_P", "z_f", "Q_o", "Q_spill", "H", "U", "origin", "v_o", "d_o", "v_lip", "A_b", "W_b", "P_b", "t_f",
          "law_t_f", "x_hit", "t_arr", "u_h", "U_perp", "replay_w", "kM", "Cvis", "width")
R_KEYS = ("t", "Q", "x_hit", "u_h", "U_perp", "V_cup", "V_P", "V_jet", "V_spill", "on")
P_KEYS = ("t", "tip", "yaw", "tilt", "roll", "axis_tilt_deg", "on_obs")
G_KEYS = ("zs", "phis", "r_in", "r_out", "z_rim", "r_rim", "spout_phi_residual", "height", "units", "name")


def _sha(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sibling(fn, name):
    p = os.path.join(os.path.dirname(os.path.abspath(fn)), name)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:                                      # noqa: BLE001
            return None
    return None


def resolve(closure_fn, inlet_cli="auto"):
    """(num, sw, cal_mode, source, done) of a closure file (module docstring)."""
    from latte_imex.diff.render512 import load_closure
    from latte_imex.rig import real_program as RP
    num, sw = load_closure(closure_fn)
    done = _sibling(closure_fn, "done.json")
    mode, src = sw.get("inlet"), "closure key 'inlet'"
    if mode is None:
        cal = ((done or {}).get("inlet") or {}).get("cal") or {}
        ms = sorted({v.get("mode") for v in cal.values() if isinstance(v, dict) and v.get("mode")})
        if len(ms) == 1:
            mode, src = ms[0], "done.json (an oracle best_closure.json carries no inlet key)"
        elif len(ms) > 1:
            raise SystemExit(f"done.json lists several calibration inlet modes {ms}; pass --inlet")
        else:
            mode, src = None, "none: record (no inlet key, no MI done.json)"
    if inlet_cli != "auto":
        mode, src = RP.inlet_mode(inlet_cli), "--inlet"
    if mode is not None:
        num["inlet"] = float(RP.inlet_code(mode))
        sw["inlet"] = RP.inlet_mode(mode)
    return num, sw, mode, src, done


def set_env(mi, keep=False, log=print):
    os.environ.setdefault("LATTE_POSE", "v2")
    if keep:
        return
    if os.environ.get("LATTE_INLET"):
        log(f"forward_fields: LATTE_INLET={os.environ['LATTE_INLET']} unset (the fits ran without it)")
    os.environ.pop("LATTE_INLET", None)
    want = "mi" if mi else None
    have = os.environ.get("LATTE_FOOT")
    if have != want:
        log(f"forward_fields: LATTE_FOOT {have} -> {want} (as autodl_diag21.sh model_env)")
    if want:
        os.environ["LATTE_FOOT"] = want
    else:
        os.environ.pop("LATTE_FOOT", None)


class _Capture:
    """Keeps the last real_program.build_inlet result (C, E, R, info) built inside the block."""

    def __enter__(self):
        from latte_imex.rig import real_program as RP
        self.RP, self.orig, self.last = RP, RP.build_inlet, None

        def cap(*a, **k):
            r = self.orig(*a, **k)
            self.last = r
            return r
        RP.build_inlet = cap
        return self

    def __exit__(self, *a):
        self.RP.build_inlet = self.orig
        return False


def law_dict(mode, exit_cal):
    """The ForwardLaw the inlet was built with (real_program.forward_pitcher: PITCHER_LAW + EXIT_CAL for record /
    EXIT_LAW for law / replay)."""
    from latte_imex.pitcher import ForwardLaw
    from latte_imex.rig import real_program as RP
    law = json.loads(os.environ.get("PITCHER_LAW", "{}"))
    if mode == "record":
        if exit_cal:
            law.update(RP.EXIT_CAL)
    else:
        law.update(RP.EXIT_LAW)
    return dataclasses.asdict(ForwardLaw(**law))


def snap_to_uint8(sol, msk, store):
    """Make sol.snaps a dict that turns every snapshot into uint8 C at once (the device clone is dropped right away);
    fast_forward's `self.snaps = {}` gets a fresh converting dict through the property."""
    m_np = np.asarray(msk, bool)

    class Snaps(dict):
        def __setitem__(self, k, st):
            m, l = st["m"].numpy(), st["l"].numpy()
            c = np.where(m_np, m / np.where(m_np, l, 1.0), 0.0)
            store[int(k)] = (np.clip(c, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    base = type(sol)

    def get(self):
        return self.__dict__.get("_viz_snaps")

    def put(self, v):
        self.__dict__["_viz_snaps"] = v if isinstance(v, Snaps) else Snaps()
    sol.__class__ = type("VizSolver", (base,), {"snaps": property(get, put)})


def render_frames(t_end, fps):
    """Solver frames whose end is nearest to each 1/fps time in [0, t_end[-1]]: (frames (unique, sorted), times)."""
    tr = np.arange(0.0, t_end[-1] + 1e-9, 1.0 / fps)
    k = np.clip(np.searchsorted(t_end, tr), 0, len(t_end) - 1)
    km = np.clip(k - 1, 0, len(t_end) - 1)
    k = np.where(np.abs(t_end[km] - tr) < np.abs(t_end[k] - tr), km, k)
    ks, first = np.unique(k, return_index=True)
    return ks, tr[first]


def run(a, log=print):
    t_wall = time.time()
    if a.pkg:                                              # a foreign / read-only package tree: no __pycache__ writes
        sys.dont_write_bytecode = True
        sys.path.insert(0, os.path.abspath(a.pkg))
    num, sw, cal_mode, src, done = resolve(a.closure, a.inlet)
    mi = cal_mode in ("law", "replay")
    set_env(mi, a.keep_env, log)
    import warp as wp
    wp.config.quiet = True
    wp.init()
    from latte_imex.diff import render512 as RR
    from latte_imex.diff import schedule as SC
    from latte_imex.rig import real_program as RP
    dev = a.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    p = a.pour
    oracle = None
    t_solve = 0.0
    with RR.heldout_inlet(p, num, sw) as swp, _Capture() as cap:
        if a.inlet_only:
            script, pour = SC.pour_script(p, a.N, e0=swp["e0"], exit_cal=swp["exit_cal"], inlet=swp.get("inlet"))
            S = SC.build(script, a.N, T_max=a.T)
            inlet_used, pinfo, src_pour = pour.inlet, pour.info, pour.inlet_source
            from latte_imex.diff import wsolver as W
            msk = W.Geo(a.N, device="cpu").mask_np
            Cs, tC, ks, tr, Pfull = np.zeros((0, a.N, a.N), np.uint8), np.zeros(0), np.zeros(0, int), np.zeros(0), {}
        else:
            D = RR.make_pour(p, swp, a.N, dev, a.T)
            inlet_used, pinfo, src_pour = D.inlet, D.pour.info, D.pour.inlet_source
            best = _sibling(a.closure, "best.json")
            u = ((best or {}).get("u_in") or {}).get(p)
            if a.oracle != "off" and u is not None:
                from latte_imex.diff import diag as DG
                box = ((done or {}).get("inlet") or {}).get("box")
                md = (done or {}).get("mode")
                if box is None and md in DG.MODE_SET and DG.MODE_SET[md][1]:
                    box = DG.BOX_FAMILY["meas" if mi else "err"][DG.MODE_SET[md][1]]
                if box is None:
                    raise SystemExit(f"{p}: best.json holds inlet corrections but the box is unknown (done.json mode "
                                     f"{md!r}); pass --oracle off")
                D.inl = DG.InletCorr(D, box)
                Sx = D.inl.apply(np.array(u, float))
                DG.assign(D, Sx)
                ser = D.inl.series(Sx)
                oracle = dict(box=box, mode=md, knots=int(D.inl.K),
                              shift_mm_p50=float(np.median(np.hypot(ser["dx"], ser["dy"]))) if len(ser["dx"]) else 0.0,
                              Q_ratio_p50=float(np.median(ser["Q"] / ser["Q0"])) if len(ser["Q"]) else 1.0,
                              note="diag oracle inlet corrections (best.json u_in): diagnostic only, not an inlet")
                log(f"{p}: oracle inlet corrections applied (box {box}, diag mode {md}) -- diagnostic only")
            elif a.oracle == "on":
                raise SystemExit(f"{p}: --oracle on but best.json next to the closure has no u_in for this pour")
            from latte_imex.diff.fit import CIRC_START
            P = dict(D.s1)
            P.update(CIRC_START if swp["circ"] else {"kappa_M": 0.0, "eps_f": 0.0, "kappa_w": 0.0})
            P.update(num)
            D._apply_foot(P)
            sol = D.sol
            sol.rec, sol.tape = False, None
            Pnp, Pw = D._P(P, False)
            t_end = np.cumsum(D.S["fdt"])
            ks, tr = render_frames(t_end, a.fps)
            store = {}
            msk = D.geo.mask_np
            snap_to_uint8(sol, msk, store)
            t0 = time.time()
            sol.fast_forward(Pw, Pnp, D.Sw, D.S, keep=False, snap=[int(k) for k in ks])
            t_solve = time.time() - t0
            Cs = np.stack([store[int(k)] for k in ks])
            tC = t_end[ks]
            S = D.S
            from latte_imex.diff import wsolver as W
            Pfull = {k: float(P.get(k, 0.0)) for k in W.PARAMS}
            log(f"{p}: N {a.N} on {dev}: {len(t_end)} solver frames, {len(ks)} saved at {a.fps:g} fps, forward "
                f"{t_solve:.1f} s")
        if cap.last is None:
            raise RuntimeError("the inlet build was not captured (real_program.build_inlet)")
        Cp, E, R, info = cap.last
        exit_cal = bool(swp.get("exit_cal"))
        e0 = SC.e0_mode(swp.get("e0", 0))
        geom_fn = os.environ.get("PITCHER_GEOM")
        tip_p = None if RP.TIP_OVERRIDE is None else np.asarray(RP.TIP_OVERRIDE, float)
    if inlet_used == "law":
        e0 = False
    out = {}
    out["C"] = Cs
    out["t"] = np.asarray(tC, float)
    out["k"] = np.asarray(ks, int)
    out["t_render"] = np.asarray(tr, float)
    out["mask"] = np.asarray(msk, bool)
    nS = int(S["n"])
    for k, v in S.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == nS:
            out["S_" + k] = v
    for k in P_KEYS:
        if k in Cp:
            out["P_" + k] = np.asarray(Cp[k])
    for k in E_KEYS:
        if k in E:
            out["E_" + k] = np.asarray(E[k])
    out["E_dt"] = np.asarray(float(E["dt"]))
    out["E_V0"] = np.asarray(float(E["V0"]))
    for k in R_KEYS:
        if k in R:
            out["R_" + k] = np.asarray(R[k])
    G = json.load(open(geom_fn)) if geom_fn else {}
    for k in G_KEYS:
        if k in G:
            out["G_" + k] = np.asarray(G[k])
    law = law_dict(inlet_used, exit_cal)
    out["LAW"] = np.asarray(json.dumps(law, default=float))
    meta = dict(pour=p, N=a.N, fps=a.fps, device=dev, T_max=a.T, closure=os.path.abspath(a.closure),
                closure_sha256=_sha(a.closure), closure_numbers={k: v for k, v in num.items()},
                switches={k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v)) for k, v in swp.items()},
                params=Pfull, cal_inlet=cal_mode, cal_inlet_source=src, inlet=inlet_used, inlet_source=src_pour,
                heldout=p not in RP.MI_POURS, e0=str(e0), exit_cal=exit_cal, oracle=oracle,
                L_unit_m=float(info["L_m"]), R_disc_m=float(info["R_disc_mm"]) * 1e-3, V0_mL=float(info["V0_mL"]),
                poured_mL=float(info["poured_mL"]), V_cup_mL=float(info["V_cup_mL"]), T_pour=float(info["T_pour"]),
                t0_abs=float(Cp["t0_abs"]), pitcher_geom=os.path.basename(geom_fn) if geom_fn else None,
                pitcher_geom_sha256=_sha(geom_fn) if geom_fn else None,
                tip_p=None if tip_p is None else tip_p.tolist(), law=law,
                env={k: os.environ.get(k) for k in ("LATTE_POSE", "LATTE_INLET", "LATTE_FOOT", "PITCHER_GEOM", "PITCHER_LAW")},
                inlet_info=info.get("inlet"), pour_info={k: v for k, v in pinfo.items() if k != "assumptions"},
                solver_frames=nS, saved_frames=int(len(ks)), forward_s=t_solve, wall_s=time.time() - t_wall,
                clock="real_program.cup_poses t (t = 0: first sample with pitcher pose and disc) = solver = video_obs",
                field="C[i, j] = m / l at x_j = (j + .5) / N - .5, y_i = (i + .5) / N - .5 [solver units]; F metres = x L",
                package=os.path.dirname(os.path.dirname(os.path.abspath(RP.__file__))))
    out["meta"] = np.asarray(json.dumps(meta, default=float))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, a.out)
    log(f"{p}: -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB; inlet {inlet_used} [{src_pour}], cal {cal_mode} "
        f"[{src}], e0 {e0}, oracle {'yes' if oracle else 'no'}, wall {time.time() - t_wall:.0f} s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--closure", required=True)
    ap.add_argument("--pour", required=True, choices=("tulip", "leaf", "heart", "push_heart", "swan"))
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, help="default cuda:0 if CUDA is available, else cpu")
    ap.add_argument("--T", type=float, default=None, help="simulate only the first T seconds (tests)")
    ap.add_argument("--pkg", default=None, help="root holding latte_imex/ (put first on sys.path)")
    ap.add_argument("--inlet", default="auto", choices=("auto", "record", "law", "replay", "0", "1", "2"))
    ap.add_argument("--oracle", default="auto", choices=("auto", "on", "off"))
    ap.add_argument("--keep-env", action="store_true", help="do not set LATTE_INLET / LATTE_FOOT")
    ap.add_argument("--inlet-only", action="store_true", help="no solver run (C empty)")
    a = ap.parse_args(argv)
    run(a)


if __name__ == "__main__":
    main()
