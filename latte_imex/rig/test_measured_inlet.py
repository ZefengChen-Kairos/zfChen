"""mi27 measured-inlet tests (SPEC27 rev 2 section 3.5 item 9; decision rule D1 of PREREG27_2: all must pass before any
probe run; on a failure the code is fixed, never a parameter, a prediction or a tolerance).

    python3 -m latte_imex.rig.test_measured_inlet                 all tests (CLI runner; also pytest functions)
    python3 -m latte_imex.rig.test_measured_inlet T1 T2 T6        selected tests

  T1   default bit-identity: LATTE_INLET / LATTE_FOOT unset, build_inlet E / R (five pours, tulip_feas.Pour 'kona' path)
       and schedule.build (tulip, leaf, N 48, e0 off and 'world') equal the sha256 fixtures computed in the pristine
       fc9f270 copy before any edit (latte_imex/data/mi27/t1_fixtures.json)                          [local only]
  T1b  diag step loss and gradient (bp256_trained, N 48, T 1, cpu, box 'err', default pset; tulip and leaf, params
       (box None) and err (box 'err', u rng 3) steps: L, dL/dP, the params-mode unit gradient, dL/du, dL/d(schedule))
       bitwise = the fixture computed in the pristine fc9f270 copy before diag.py was edited
       (latte_imex/data/mi27/t1b_fixture.json; closure copy data/mi27/bp256_trained.json, sha256 ea968f36...)
                                                                                                       [local only]
  T1c  EXIT_CAL build (LAW_OVERRIDE = EXIT_CAL, tulip / leaf) = its fixture                          [local only]
  T1d  a law build then a record build in one process: the record equals the T1 fixture (_FP keyed on the mode)
  T1e  LATTE_INLET=record equals unset; a closure without 'inlet' round-trips through render512 / validate
       load_closure (no inlet switch, numbers unchanged)
  T1f  precedence: per-pour override > closure key > env > default, all combinations; the render512 / validate CLIs run
       a held-out pour of an MI closure as MI-L with e0 off (render512.heldout_inlet; override removed afterwards)
  T2   port parity: ForwardPitcher(exit='wall', capillary c, clip_rise True) arrival records = M5's hashed
       pred_<p>.npz <law>_arr_* to <= 1e-9 relative, c in (round, sheet, none), tulip and leaf (1-in-50 fixture,
       latte_imex/data/mi27/t2_parity.npz; the full arrays too when MI27_SRC holds them); the production 'law' build
       equals the direct round build; clip_rise False runs and is finite                             [local only]
  T3   closed form: sigma = 0 jet_flight = ballistics, |dt_f| <= 2e-7 s, |dx| <= 5e-8 m on M5's check domain (B6: first
       500 steps with Q >= 1 mL/s, five pours); every parcel <= the analytic crossing error g dt^2 / (8 |v_z|); the
       full-pour maxima are reported (they exceed the literal numbers on a few low-|v_z| parcels, see the docstring)
  T4   energy and geometry on every flowing step (Q_o > 1 mL/s, real_program.Q_ON): U0 <= sqrt(2 g H),
       0 < k_M <= sqrt(3)/2, exit elevation <= 0 (clip_rise; every wall step), C_visc > 0.5 strictly; reported:
       k_M min / max, C_visc min, median Q_w / Q_c, guard hits below 1 mL/s and their volume; info['inlet'] reports the
       same k_M / C_visc range (flowing steps) and the same trickle-step guard-hit count
  T5   ledger: sum R['Q'] dt = emitted volume to 1e-12 relative, law and replay
  T6   replay: (a) outside the taper support the replay parcels = the law parcels exactly; (b) GP mean at the frame
       times = the frozen r_hat (stream_kin.json) to 1e-9 (and = the frozen prototype replay_proto_<p>.npz when
       available), parcel values = q_law + w r_hat to 1e-12; (c) binned record = parcel series at the frame times to
       <= 1e-4 m/s; (d) LOO z-sd of every residual GP in [0.7, 1.3]; (e) footprint cap inactive; (f) reported:
       agreement with M1's GP (fraction within its 1 sigma) and the implied-drop residual; (g) replay() and band()
       refuse (RuntimeError) a stream_kin.json whose paired frame set or r_hat is not the frozen one (no gated frame,
       a changed GP frame set, r_hat + 1e-6), and replay(law E) with the shipped file = the production replay build
  T7   gradient: diag gradcheck N 32, T 1.0, eps 1e-7, relative error <= 2e-3 (tulip, cpu): (i) v05n_AB_cmaes_N48,
       inlet replay, box 'meas'; (ii) mcd_S1kq_N96 (vj 1, mp5fct, xi_b, lam_b, tau_s), inlet replay, box 'meas', keys
       tau_s, p_dep, kappa_c, B_dep, with the V_j adjoint on the dx / dy knots (its share must be > 0); (iii) record,
       LATTE_INLET / LATTE_FOOT unset, box 'err' (the unchanged path).  Also: (iv) the V_j adjoint InletCorr.vj_T is the
       exact transpose of schedule.foot_velocity (dot-product test on the replay schedule, 1e-12); (v) the 'meas' band
       from the replay emission equals the band from the MI-law build (measured_inlet.law_emission, 1e-12); (vi) the
       V_j adjoint where it matters: (ii)'s setting over T 6.0 s (the high pours, where 1 - chi is large), the 4 dx / dy
       knots with the largest V_j share, relative error <= 2e-3 with the V_j adjoint (reported: without it)
  T8   landing in the solver: MI-law schedule (N 48) x_hit vs the validated foot on the gated frames,
       |median along| and |median across| <= 3 mm, tulip and leaf; (b) stream_foot_mi.json carries the SPEC27 1.1 gate
       (E0-usable rows only on gated frames, every other row tier none, the file = foot_file() on the shipped inputs)
  T9   allow-list: stream_kin.json holds allow-listed keys only, no diag / record_diag_v1 / record_prod_v2 / reference /
       oracle* key
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "mi27")
MI27_SRC = os.environ.get("MI27_SRC") or None   # the mi27 campaign directory: optional full-array / prototype comparisons


def src_file(*parts):
    """A file of the mi27 campaign directory (env MI27_SRC), or None when MI27_SRC is unset or the file is missing (the
    optional comparisons are then skipped)."""
    if not MI27_SRC:
        return None
    fn = os.path.join(MI27_SRC, *parts)
    return fn if os.path.exists(fn) else None
POURS5 = ("tulip", "leaf", "heart", "push_heart", "swan")
CAL = ("tulip", "leaf")
REPORT = {}                      # numbers reported by the tests (CLI prints them)


class Skip(Exception):
    pass


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg) from None


@contextlib.contextmanager
def env(**kv):
    """Set (str) or unset (None) environment variables for the block; restored afterwards."""
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def clean_env():
    return env(LATTE_INLET=None, LATTE_FOOT=None, LATTE_POSE=None, PITCHER_LAW=None, FEAS_FLOW=None)


# ----------------------------------------------------------------------------- hashing (verbatim from the fixture code)
def arrays_sha256(d, skip=()):
    """sha256 over the numeric / string entries of a dict, sorted by key: key, dtype, shape, C-order bytes.  Entries that
    are neither arrays, numbers nor strings (e.g. lists of Inlet objects) and keys in `skip` are left out.  Returns
    (combined hex, {key: hex})."""
    per = {}
    h = hashlib.sha256()
    for k in sorted(d):
        if k in skip:
            continue
        v = d[k]
        if isinstance(v, str):
            b = v.encode()
            hk = hashlib.sha256(b"str" + b).hexdigest()
        elif isinstance(v, (bool, int, float, np.number, np.ndarray, np.bool_)):
            a = np.ascontiguousarray(np.asarray(v))
            hh = hashlib.sha256()
            hh.update(str(a.dtype).encode()); hh.update(str(a.shape).encode()); hh.update(a.tobytes())
            hk = hh.hexdigest()
        else:
            continue
        per[k] = hk
        h.update(k.encode()); h.update(hk.encode())
    return h.hexdigest(), per


def fixtures():
    return json.load(open(os.path.join(DATA, "t1_fixtures.json")))["items"]


def capture_pour(p, law=None, inlet=None):
    """tulip_feas.Pour(p, law=law, inlet=inlet) (production 'kona' path); returns (Pour, (C, E, R, info)) with
    build_inlet's return value captured through a transparent wrapper."""
    from . import real_program as RP
    from .tulip_feas import Pour
    cap = {}
    orig = RP.build_inlet

    def wrap(*a, **k):
        r = orig(*a, **k)
        cap["r"] = r
        return r
    RP.build_inlet = wrap
    try:
        pour = Pour(p, law=law, inlet=inlet)
    finally:
        RP.build_inlet = orig
    return pour, cap["r"]


def _diff_keys(per, ref):
    return sorted(k for k in set(per) | set(ref) if per.get(k) != ref.get(k))


def check_record(tag, E, R, pourR=None):
    F = fixtures()[tag]
    hE, pE = arrays_sha256(E)
    hR, pR = arrays_sha256(R)
    assert hE == F["E"], f"{tag}: E differs from the wt0 fixture in {_diff_keys(pE, F['E_keys'])}"
    assert hR == F["R"], f"{tag}: R differs from the wt0 fixture in {_diff_keys(pR, F['R_keys'])}"
    if pourR is not None:
        hS, pS = arrays_sha256(pourR)
        assert hS == F["pourR"], f"{tag}: Pour.R differs in {_diff_keys(pS, F['pourR_keys'])}"


# ----------------------------------------------------------------------------- shared builds (cached per process)
_CACHE = {}


def kona_setup(p):
    """The tulip_feas.Pour('kona') setup without the build: geometry path, TIP_KONA, fill [m^3]."""
    from . import real_program as RP
    from .full_pour import TIP_KONA
    gp = os.path.join(RP.REPO, "latte_imex", "pitcher_geom", "pitcher_kona700.json")
    os.environ["PITCHER_GEOM"] = gp
    RP.TIP_OVERRIDE = TIP_KONA
    J = json.load(open(os.path.join(RP.RES, "full_pour_kona_poured165.json")))[p]
    fill = J["fill_at_cut_v1_mL"] if (RP.pose_version() == "v1" and "fill_at_cut_v1_mL" in J) else J["fill_at_cut_mL"]
    return gp, fill * 1e-6


def direct_wall(p, capillary="round", clip_rise=True, sub=1):
    """ForwardPitcher(exit='wall') run on pour p's measured poses (kona geometry, fill, TIP_KONA), cached."""
    key = ("direct", p, capillary, clip_rise, sub)
    if key not in _CACHE:
        from ..pitcher import ForwardLaw, ForwardPitcher, GridPitcherGeometry
        from . import real_program as RP
        gp, V0 = kona_setup(p)
        C = RP.cup_poses(p)
        geom = GridPitcherGeometry.load(gp, lip_slope=math.radians(RP.ALPHA_DEG))
        fp = ForwardPitcher(geom, ForwardLaw(exit="wall", capillary=capillary, clip_rise=clip_rise))
        s = slice(None, None, sub)
        t, dt = C["t"][s], C["dt"] * sub
        E = fp.run(t, C["tip"][s], C["yaw"][s], C["tilt"][s], C["roll"][s], V0, dt)
        n_out = int(round((t[-1] + 0.5 * dt + RP.TAIL) / dt))
        R = ForwardPitcher.arrivals(E, 0.0, n_out, dt)
        _CACHE[key] = (C, E, R, n_out)
    return _CACHE[key]


def mi_build(p, mode):
    """Production MI build through tulip_feas.Pour(p, inlet=mode) (env clean), cached: (Pour, C, E, R, info)."""
    key = ("pour", p, mode)
    if key not in _CACHE:
        with clean_env():
            pour, (C, E, R, info) = capture_pour(p, inlet=mode)
        _CACHE[key] = (pour, C, E, R, info)
    return _CACHE[key]


# ============================================================================= T1 family
def test_T1_default_bit_identity():
    from ..diff import schedule as SCH
    with clean_env():
        for p in POURS5:
            pour, (C, E, R, info) = capture_pour(p)
            assert info["inlet"] == dict(mode="record", source="default"), info["inlet"]
            check_record(f"T1/{p}", E, R, pour.R)
        F = fixtures()
        for p in CAL:
            for e0 in (False, "world"):
                script, pour = SCH.pour_script(p, 48, e0=e0)
                S = SCH.build(script, 48)
                h, per = arrays_sha256(S, skip=("inlets",))
                ref = F[f"T1/sched/{p}/N48/e0_{e0 or 'off'}"]
                assert h == ref["S"], f"{p} e0 {e0}: schedule differs in {_diff_keys(per, ref['S_keys'])}"
    REPORT["T1"] = "E / R of 5 pours, Pour.R, schedules (tulip, leaf x e0 off / world, N 48) bit-identical to wt0"


def _t1b_record(D, P, r, DG, W, u=None):
    """Verbatim from build/code/fixtures_t1b.py (the wt0 fixture)."""
    g = r["g"]
    gk = sorted(g)
    out = dict(L=float(r["L"]), g=np.array([g[k] for k in gk], float),
               gunit=np.array([g[k] for k in DG.PKEYS], float) * DG.p_dunit(P),
               pen=float(r["pen"]), vol=float(r["vol"]), scan=float(r["scan"]), gvj=float(r["gvj"]),
               viou=float(r["viou"]))
    if r["gu"] is not None:
        out["gu"] = np.asarray(r["gu"], float)
    if u is not None:
        out["u"] = np.asarray(u, float)
    for k in W.SCHED:
        out["G_" + k] = np.asarray(D.sched_grad[k], float)
    h, per = arrays_sha256(out)
    scal = {k: float(v).hex() for k, v in out.items() if np.ndim(v) == 0}
    return dict(sha=h, keys=per, scalars=scal, g_keys=gk, g_hex=[float(x).hex() for x in out["g"]])


def test_T1b_diag_path():
    """diag step loss and gradient on the default path (bp256_trained, N 48, T 1, cpu, box 'err', default pset) bitwise
    equal to the wt0 fixture (build/code/fixtures_t1b.py, run in the pristine copy before diag.py was edited)."""
    from ..diff import diag as DG
    from ..diff import wsolver as W
    from ..diff.render512 import load_closure
    J = json.load(open(os.path.join(DATA, "t1b_fixture.json")))
    cl = os.path.join(DATA, "bp256_trained.json")
    sha = hashlib.sha256(open(cl, "rb").read()).hexdigest()
    assert sha == J["closure_sha256"] == "ea968f3674a4de048f136474d40607a1f91fc3038230bf4650b3fdf0d049d5c2", sha
    rows = {}
    with clean_env():
        num, sw = load_closure(cl)
        assert sw["inlet"] is None
        for p in CAL:
            Dp = DG.make_pour(p, sw, dict(N=J["N"], device="cpu", T=J["T"], box=None))
            assert Dp.inlet == "record"
            base = DG.full_P_host(Dp.s1, num, sw)
            up = DG.p_to_unit(base)
            P = DG.p_from_unit(up, base)
            rp = DG.step(Dp, P, J["tau"], None, J["prior"])
            got = dict(_t1b_record(Dp, P, rp, DG, W), up=[float(x).hex() for x in up],
                       P=[float(P[k]).hex() for k in DG.PKEYS], pkeys=list(DG.PKEYS))
            De = DG.make_pour(p, sw, dict(N=J["N"], device="cpu", T=J["T"], box="err"))
            rng = np.random.default_rng(J["rng"])
            u = np.clip(0.5 + 0.15 * rng.standard_normal((6, De.inl.K)), 0.05, 0.95)
            re = DG.step(De, P, J["tau"], u, J["prior"])
            gote = dict(_t1b_record(De, P, re, DG, W, u), K=int(De.inl.K))
            for tag, g in (("params", got), ("err", gote)):
                ref = J["items"][f"T1b/{p}/{tag}"]
                bad = _diff_keys(g["keys"], ref["keys"])
                for k in ("up", "P", "pkeys", "K", "scalars", "g_keys", "g_hex"):
                    if k in ref and g.get(k) != ref[k]:
                        bad.append(k)
                assert g["sha"] == ref["sha"] and not bad, f"T1b {p} {tag}: differs from the wt0 fixture in {bad}"
                rows[f"{p}/{tag}"] = dict(L=float.fromhex(g["scalars"]["L"]), sha=g["sha"][:16], gvj=g["scalars"]["gvj"])
    REPORT["T1b"] = rows


def test_T1c_exit_cal():
    from . import real_program as RP
    with clean_env():
        for p in CAL:
            pour, (C, E, R, info) = capture_pour(p, law=RP.EXIT_CAL)
            check_record(f"T1c/{p}", E, R, pour.R)
    REPORT["T1c"] = "EXIT_CAL builds (tulip, leaf) bit-identical to wt0"


def test_T1d_in_process_switch():
    from . import real_program as RP
    with clean_env():
        # (i) production path: a law build, then a record build
        mi_build("tulip", "law")
        pour, (C, E, R, info) = capture_pour("tulip")
        check_record("T1/tulip", E, R, pour.R)
        # (ii) the _FP cache: law pitcher cached first, then the record build reuses the same cache without a clear
        gp, V0 = kona_setup("tulip")
        RP._FP.clear()
        fpl = RP.forward_pitcher("law")
        C2, E2, R2, info2 = RP.build_inlet("tulip", "pitcher", V0, log=lambda *a: None)
        assert info2["inlet"]["mode"] == "record"
        fpr = RP.forward_pitcher()
        assert fpl is not fpr and fpl.law.exit == "wall" and fpr.law.exit == "lip" and len(RP._FP) == 2, RP._FP.keys()
        check_record("T1/tulip", E2, R2)
        RP._FP.clear()
    REPORT["T1d"] = "record after law (Pour path and shared _FP cache) bit-identical to the T1 fixture"


def test_T1e_explicit_default():
    from ..diff import render512, validate
    from ..diff import schedule as SCH
    from . import real_program as RP
    with clean_env(), env(LATTE_INLET="record"):
        pour, (C, E, R, info) = capture_pour("leaf")
        assert info["inlet"] == dict(mode="record", source="env"), info["inlet"]
        check_record("T1/leaf", E, R, pour.R)
    closure = os.path.join(HERE, "..", "data", "closures", "mcd_S1kq_N96.json")
    raw = {k: float(v) for k, v in json.load(open(closure)).items() if isinstance(v, (int, float)) and not k.startswith("_")}
    for L in (render512.load_closure, validate.load_closure):
        num, sw = L(closure)
        assert "inlet" not in num and num == raw, "load_closure changed the closure's numbers"
        assert sw["inlet"] is None, sw["inlet"]
        with clean_env():
            assert RP.resolve_inlet("tulip", None if sw["inlet"] is None else {"inlet": sw["inlet"]}) == ("record", "default")
    for fn, mode in (("bp256_mi.json", "replay"), ("mcd_S1kq_mi_N96.json", "replay")):
        for L in (render512.load_closure, validate.load_closure):
            num, sw = L(os.path.join(HERE, "..", "data", "closures", fn))
            assert sw["inlet"] == mode and num["inlet"] == 2.0, (fn, sw["inlet"])
    assert SCH.inlet_switch({}) is None and SCH.inlet_switch({"inlet": 1.0}) == "law"
    REPORT["T1e"] = "LATTE_INLET=record == unset (leaf E / R); closures without 'inlet' round-trip through both load_closure"


def test_T1f_precedence():
    from . import real_program as RP
    modes = (None, "record", "law", "replay")
    n = 0
    saved = dict(RP.INLET_OVERRIDE)
    try:
        for ov in modes:
            for cl in modes:
                for ev in modes:
                    RP.INLET_OVERRIDE.clear()
                    if ov is not None:
                        RP.INLET_OVERRIDE["heart"] = ov
                    with env(LATTE_INLET=ev):
                        got = RP.resolve_inlet("heart", None if cl is None else {"inlet": cl})
                        want = ((ov, "override") if ov is not None else (cl, "closure") if cl is not None else
                                (ev, "env") if ev is not None else ("record", "default"))
                        assert got == want, (ov, cl, ev, got, want)
                        # the override is per pour: another pour does not see it
                        got2 = RP.resolve_inlet("tulip", None if cl is None else {"inlet": cl})
                        want2 = ((cl, "closure") if cl is not None else (ev, "env") if ev is not None else ("record", "default"))
                        assert got2 == want2, (ov, cl, ev, got2, want2)
                        n += 1
        # codes and normalisation
        for c, m in ((0, "record"), (1, "law"), (2, "replay"), (2.0, "replay"), ("1", "law")):
            assert RP.inlet_mode(c) == m
        for bad in (3, 1.5, "wall", -1):
            try:
                RP.inlet_mode(bad)
                raise AssertionError(f"inlet_mode({bad!r}) accepted")
            except ValueError:
                pass
        with env(LATTE_INLET=""):
            assert RP.resolve_inlet("tulip") == ("record", "default")
        # closure key {"inlet": None} is 'absent'
        with env(LATTE_INLET="law"):
            assert RP.resolve_inlet("tulip", {"inlet": None}) == ("law", "env")
    finally:
        RP.INLET_OVERRIDE.clear()
        RP.INLET_OVERRIDE.update(saved)
    # guards: replay on a held-out pour, EXIT_CAL under law
    try:
        with clean_env():
            capture_pour("heart", inlet="replay")
        raise AssertionError("replay on heart accepted")
    except ValueError:
        pass
    try:
        with clean_env():
            capture_pour("tulip", law=RP.EXIT_CAL, inlet="law")
        raise AssertionError("EXIT_CAL under the wall exit accepted")
    except ValueError:
        pass
    # the render512 / validate CLIs (REV_code finding 3): a held-out pour of an MI closure runs MI-L with e0 off through
    # the per-pour override, removed afterwards; calibration pours and record closures keep their switches
    from ..diff.render512 import heldout_inlet, load_closure
    cdir = os.path.join(HERE, "..", "data", "closures")
    nh = 0
    with clean_env():
        for cl, mi in (("bp256_mi.json", True), ("mcd_S1kq_mi_N96.json", True), ("mcd_S1kq_N96.json", False),
                       ("v05n_AB_cmaes_N48.json", False)):
            num, sw = load_closure(os.path.join(cdir, cl))
            for p in POURS5:
                assert p not in RP.INLET_OVERRIDE
                with heldout_inlet(p, num, sw) as swp:
                    if mi and p not in CAL:
                        assert swp is not sw and swp["e0"] is False and swp["inlet"] == sw["inlet"], (cl, p, swp)
                        assert RP.resolve_inlet(p, num) == ("law", "override"), (cl, p)
                        nh += 1
                    else:
                        assert swp is sw and p not in RP.INLET_OVERRIDE, (cl, p)
                assert p not in RP.INLET_OVERRIDE, (cl, p, "override left behind")
        with env(LATTE_INLET="replay"):                  # an MI inlet from the environment counts too (DEVIATIONS 16)
            num, sw = load_closure(os.path.join(cdir, "mcd_S1kq_N96.json"))
            with heldout_inlet("swan", num, sw) as swp:
                assert swp["e0"] is False and RP.resolve_inlet("swan", num) == ("law", "override")
            assert "swan" not in RP.INLET_OVERRIDE
    REPORT["T1f"] = (f"{n} override / closure / env combinations (x 2 pours), codes, guards; render512 / validate "
                     f"held-out switch: {nh} MI held-out cases -> law, e0 off, override removed")


# ============================================================================= T2-T5: the wall exit
LAWS = (("round", "phys_round"), ("sheet", "phys"), ("none", "phys_nocap"))


def t_f_record(E, n_out):
    from .measured_inlet import bin_parcels
    return bin_parcels(E, dict(t_f=np.nan_to_num(E["t_f"])), n_out)["t_f"]


def test_T2_port_parity():
    Z = np.load(os.path.join(DATA, "t2_parity.npz"))
    rows = []
    for p in CAL:
        for cap, law in LAWS:
            C, E, R, n_out = direct_wall(p, cap)
            tf = t_f_record(E, n_out)
            arr = dict(Q=R["Q"], u_h=R["u_h"], U_perp=R["U_perp"], x_hit=R["x_hit"], t_f=tf)
            assert np.allclose(R["t"][::50], Z[f"{p}__t_arr"], rtol=0, atol=1e-12)
            worst = 0.0
            on50 = R["on"][::50]
            for k, a in arr.items():
                b = Z[f"{p}__{law}__{k}"]
                a50 = a[::50]
                if k != "Q":                                 # M5 leaves empty bins nan (Q: 0)
                    fin = np.isfinite(b) if b.ndim == 1 else np.all(np.isfinite(b), axis=1)
                    assert np.array_equal(fin, on50), f"{p} {law} {k}: arrival mask differs"
                m = on50 if k != "Q" else np.ones(len(b), bool)
                d = float(np.max(np.abs(a50[m] - b[m])) / max(np.max(np.abs(b[m])), 1e-300))
                worst = max(worst, d)
                assert d <= 1e-9, f"{p} {law} {k}: max relative difference {d:.3e} > 1e-9"
            full = src_file("M5", f"pred_{p}.npz")
            wfull = None
            if full:
                Y = np.load(full)
                wfull = 0.0
                for k, a in arr.items():
                    b = Y[f"{law}_arr_{k}"]
                    m = R["on"] if k != "Q" else np.ones(len(b), bool)
                    d = float(np.max(np.abs(a[m] - b[m])) / max(np.max(np.abs(b[m])), 1e-300))
                    wfull = max(wfull, d)
                    assert d <= 1e-9, f"{p} {law} {k} (full arrays): {d:.3e} > 1e-9"
            rows.append(f"{p} {cap}: 1-in-50 max rel {worst:.1e}" + (f", full {wfull:.1e}" if wfull is not None else ""))
    # the production 'law' build is the direct round build
    for p in CAL:
        pour, C, E, R, info = mi_build(p, "law")
        C2, E2, R2, _ = direct_wall(p, "round")
        for k in ("Q_o", "t_arr", "u_h", "U_perp", "x_hit", "t_f", "origin", "kM", "Cvis"):
            assert np.array_equal(E[k], E2[k], equal_nan=True), f"{p}: production law E[{k}] != direct wall build"
        for k in ("Q", "u_h", "U_perp", "x_hit"):
            assert np.array_equal(R[k], R2[k]), f"{p}: production law R[{k}] != direct"
    # clip_rise False: runs and is finite
    for p in CAL:
        C, E, R, n_out = direct_wall(p, "round", clip_rise=False, sub=5)
        on = E["Q_o"] > 0
        for k in ("t_f", "U_perp", "x_hit", "u_h"):
            assert np.all(np.isfinite(E[k][on])), f"{p} clip_rise False: non-finite {k}"
        rows.append(f"{p} clip_rise False (sub 5): finite, raw exit elevation max {np.nanmax(E['elev_raw']):.1f} deg, "
                    f"{int(np.sum(E['d_o'][on, 2] > 0))} rising exits kept")
    REPORT["T2"] = rows


def test_T3_closed_form():
    """sigma = 0: jet_flight = closed-form ballistics.
    (i) The SPEC27 numbers (2e-7 s, 5e-8 m) come from M5's check B6 (M5/work/dbg.py: the first 500 steps with
        Q >= 1 mL/s of a pour); asserted on that domain for each of the five pours (capillary 'none').
    (ii) Every wall-exit parcel of the five pours and 400 synthetic parcels are asserted against the analytic error of
        the crossing: the midpoint steps are exact under constant gravity, the only error is the linear interpolation of
        z over the last 0.25 ms step, |dt_f| <= g dt^2 / (8 |v_z,hit|), |dx| <= |v_h| times that.
    (iii) Reported, flagged: over all parcels of the five pours the literal SPEC numbers are exceeded (tulip 2.04e-7 s on
        17 parcels, push_heart 6.1e-8 m), because some parcels land with |v_z| < 0.38 m/s.  jet_flight is M5's flight
        verbatim (T2, which cannot be waived, requires that), so the code is not changed (build/DEVIATIONS.md item 4)."""
    from ..pitcher import ballistic_flight, jet_flight
    from .real_program import Q_ON
    rows = []
    g, dt = 9.81, 2.5e-4
    for p in POURS5:
        C, E, R, n_out = direct_wall(p, "none")
        on = (E["Q_o"] > 0) & ~E["wall_miss"]
        tb, xb, vb = ballistic_flight(E["origin"][on], E["v_o"][on])
        et = np.abs(E["t_f"][on] - tb); ex = np.hypot(*(E["x_hit"][on] - xb).T)
        # (i) M5 B6 domain: first 500 steps with Q >= 1 mL/s
        b6 = np.nonzero(E["Q_o"][on] >= Q_ON)[0][:500]
        dtf6, dx6 = float(et[b6].max()), float(ex[b6].max())
        assert dtf6 <= 2e-7, f"{p} (B6 domain): |dt_f| {dtf6:.2e} > 2e-7 s"
        assert dx6 <= 5e-8, f"{p} (B6 domain): |dx| {dx6:.2e} > 5e-8 m"
        # (ii) every parcel against the analytic crossing bound
        bt = g * dt ** 2 / (8 * np.abs(vb[:, 2])) + 1e-12
        bx = np.hypot(vb[:, 0], vb[:, 1]) * bt + 1e-12
        assert np.all(et <= bt), f"{p}: |dt_f| above the crossing bound on {int(np.sum(et > bt))} parcels"
        assert np.all(ex <= bx), f"{p}: |dx| above the crossing bound on {int(np.sum(ex > bx))} parcels"
        rows.append(f"{p}: B6 domain (first 500 Q >= 1 mL/s) |dt_f| {dtf6:.2e} s, |dx| {dx6:.2e} m | all {int(on.sum())} "
                    f"parcels: |dt_f| max {et.max():.2e} s ({int(np.sum(et > 2e-7))} > 2e-7), |dx| max {ex.max():.2e} m "
                    f"({int(np.sum(ex > 5e-8))} > 5e-8), min U_perp {np.min(-vb[:, 2]):.3f} m/s, max ratio to the crossing "
                    f"bound {np.max(et / bt):.3f} (t) {np.max(ex / bx):.3f} (x)")
    rng = np.random.default_rng(27)
    n = 400
    origin = np.c_[rng.uniform(-0.03, 0.03, n), rng.uniform(-0.03, 0.03, n), rng.uniform(0.004, 0.06, n)]
    v_lip = rng.normal(0, 0.05, (n, 3))
    e = rng.normal(0, 1, (n, 3)); e[:, 2] = -np.abs(e[:, 2]) * rng.uniform(0, 1, n); e /= np.linalg.norm(e, axis=1)[:, None]
    U0 = rng.uniform(0.1, 0.5, n); Q = rng.uniform(2e-6, 30e-6, n)
    A_b = rng.uniform(1e-5, 5e-5, n); W_b = rng.uniform(0.01, 0.02, n); P_b = W_b * 1.07
    tf, xh, vh, T0, M0 = jet_flight(origin, v_lip, e, U0, Q, A_b, W_b, P_b, 800.0, 0.0, "round")
    tb, xb, vb = ballistic_flight(origin, v_lip + U0[:, None] * e)
    bound_t = 9.81 * 2.5e-4 ** 2 / (8 * np.abs(vb[:, 2])) + 1e-12
    bound_x = np.hypot(vb[:, 0], vb[:, 1]) * bound_t + 1e-12
    et, ex = np.abs(tf - tb), np.hypot(*(xh - xb).T)
    assert np.all(et <= bound_t), f"synthetic: |dt_f| above g dt^2 / 8|v_z| on {int(np.sum(et > bound_t))} parcels"
    assert np.all(ex <= bound_x), f"synthetic: |dx| above |v_h| g dt^2 / 8|v_z| on {int(np.sum(ex > bound_x))} parcels"
    tf2, xh2, vh2, _, _ = jet_flight(origin, v_lip, e, U0, Q, A_b, W_b, P_b, 800.0, 0.045, "none")
    assert np.array_equal(tf2, tf) and np.array_equal(xh2, xh), "capillary 'none' must ignore sigma"
    rows.append(f"synthetic 400 parcels (z0 4-60 mm, U_perp {np.min(-vb[:, 2]):.2f}-{np.max(-vb[:, 2]):.2f} m/s): "
                f"max |dt_f| {et.max():.2e} s, max |dx| {ex.max():.2e} m, max ratio to the crossing bound "
                f"{np.max(et / bound_t):.3f} (t), {np.max(ex / bound_x):.3f} (x)")
    REPORT["T3"] = rows


def test_T4_energy_geometry():
    from .real_program import Q_ON
    rows = []
    for p in CAL:
        pour, C, E, R, info = mi_build(p, "law")
        wall = (E["Q_o"] > 0) & ~E["wall_miss"]
        fl = wall & (E["Q_o"] > Q_ON)
        g = 9.81
        U0, H, kM, Cv = E["U"][fl], E["H"][fl], E["kM"][fl], E["Cvis"][fl]
        assert np.all(U0 <= np.sqrt(2 * g * H)), f"{p}: U0 > sqrt(2 g H) on {int(np.sum(U0 > np.sqrt(2 * g * H)))} steps"
        assert np.all(kM > 0) and np.all(kM <= math.sqrt(3) / 2), f"{p}: k_M out of (0, sqrt(3)/2]: {kM.min()} {kM.max()}"
        assert np.all(Cv > 0.5), f"{p}: C_visc guard active on {int(np.sum(Cv <= 0.5))} flowing steps"
        elev = np.degrees(np.arcsin(np.clip(E["d_o"][wall, 2], -1, 1)))
        assert np.all(elev <= 0.0), f"{p}: exit elevation > 0 on {int(np.sum(elev > 0))} steps with clip_rise"
        assert int(E["wall_miss"].sum()) == 0, f"{p}: {int(E['wall_miss'].sum())} flowing steps without a brink section"
        qwc = E["Q_o"][fl] / E["Qc"][fl]
        sub = wall & ~fl
        # info['inlet'] reports k_M / C_visc on the same flowing steps and counts the trickle-step guard hits
        ii = info["inlet"]
        assert ii["Cvis_min"] == float(Cv.min()) and ii["kM_range"] == [float(kM.min()), float(kM.max())], (p, ii)
        assert ii["trickle_steps"] == int(sub.sum()) and ii["Cvis_guard_hits"] == int(np.sum(E["Cvis"][sub] <= 0.5)), (p, ii)
        rows.append(dict(pour=p, n_flowing=int(fl.sum()), kM_min=float(kM.min()), kM_max=float(kM.max()),
                         Cvis_min=float(Cv.min()), Qw_over_Qc_med=float(np.median(qwc)),
                         U0_over_sqrt2gH_max=float(np.max(U0 / np.sqrt(2 * g * H))),
                         rising_raw_frac=float(np.mean(E["rising"][fl])),
                         below_1mLs=dict(n=int(sub.sum()), guard_hits=int(np.sum(E["Cvis"][sub] <= 0.5)),
                                         kM_max=float(np.max(E["kM"][sub])) if sub.any() else None,
                                         volume_uL=float(ii["trickle_volume_uL"]))))
    REPORT["T4"] = rows


def test_T5_ledger():
    rows = []
    for p in CAL:
        for mode in ("law", "replay"):
            pour, C, E, R, info = mi_build(p, mode)
            dt = E["dt"]
            emitted = float(np.sum(E["Q_o"]) * dt)
            arrived = float(np.sum(R["Q"]) * dt)
            rel = abs(arrived - emitted) / emitted
            assert rel <= 1e-12, f"{p} {mode}: arrived {arrived} vs emitted {emitted} (rel {rel:.2e})"
            assert abs(R["V_cup"][-1] - emitted) / emitted <= 1e-12
            rows.append(f"{p} {mode}: emitted {emitted * 1e6:.3f} mL, rel {rel:.1e}")
    REPORT["T5"] = rows


# ============================================================================= T6: the replay
def test_T6_replay():
    from . import measured_inlet as MI
    from .real_program import TAIL
    rows = {}
    for p in CAL:
        K = MI.load(p)
        pour, C, E, R, info = mi_build(p, "replay")
        _, CL, EL, RL, _ = mi_build(p, "law")
        wins = K["windows"]
        on = E["Q_o"] > 0
        w_law = MI.taper(EL["t_arr"], wins)
        w = E["replay_w"]
        # (a) outside the support: the law parcels exactly
        out = on & (w_law == 0)
        for k in ("t_arr", "t_f", "U_perp"):
            assert np.array_equal(E[k][out], EL[k][out]), f"{p} (a): {k} differs outside the taper support"
        assert np.array_equal(E["u_h"][out], EL["u_h"][out]), f"{p} (a): u_h differs outside the support"
        for k in ("Q_o", "x_hit", "origin"):
            assert np.array_equal(E[k], EL[k]), f"{p}: replay changed {k}"
        # (b) GP mean at the frame times = frozen r_hat; parcel values = q_law + w r_hat
        dt = E["dt"]
        n_out = int(round((E["t"][-1] + 0.5 * dt + TAIL) / dt))
        rec = MI.law_record(EL, n_out)
        pr = MI.pair_residuals(K, rec)
        gps = MI.residual_gps(K, pr)
        fr = [f for f in K["frames"] if f.get("gp")]
        assert [f["fi"] for f in fr] == pr["fi"], f"{p} (b): GP frame set differs from the export's"
        tfr = np.array([f["t"] for f in fr])
        db = 0.0
        for q in MI.QS:
            mu = gps[q].mean(tfr)
            ref = np.array([f["rhat"][q] for f in fr])
            db = max(db, float(np.max(np.abs(mu - ref))))
        assert db <= 1e-9, f"{p} (b): GP mean at the frame times vs frozen r_hat {db:.2e} > 1e-9"
        dproto = None
        fp = src_file("spec", "work2", f"replay_proto_{p}.npz")
        if fp:
            P = np.load(fp)
            dproto = max(float(np.max(np.abs(gps[q].mean(P["t"]) - P[f"rhat_{q}"]))) for q in MI.QS)
            assert dproto <= 1e-9, f"{p} (b): GP mean vs the frozen prototype {dproto:.2e}"
            assert float(np.max(np.abs(MI.taper(P["t"], wins) - P["w"]))) == 0.0
        idx = np.nonzero(on)[0]
        ta = E["t_arr"][idx]
        wi = MI.taper(ta, wins)
        assert np.array_equal(wi, w[idx]), f"{p} (b): stored taper differs"
        rh = np.stack([gps[q].mean(ta) for q in MI.QS], 1)
        e_u = np.max(np.abs(E["u_h"][idx] - (EL["u_h"][idx] + wi[:, None] * rh[:, :2])))
        e_U = np.max(np.abs(E["U_perp"][idx] - (EL["U_perp"][idx] + wi * rh[:, 2])))
        e_a = np.max(np.abs(E["t_f"][idx] - (EL["t_f"][idx] + wi * rh[:, 3])))      # fixed point after 2 passes
        assert e_u <= 1e-12 and e_U <= 1e-12, f"{p} (b): parcel values vs q_law + w r_hat: {e_u:.1e} {e_U:.1e}"
        # (c) binned record vs the parcel series at the frame times (bin of the frame time, parcel series at its centre)
        g_fr = [f for f in K["frames"] if f["gated"]]
        order = np.argsort(ta)
        dc = 0.0
        for f in g_fr:
            j = int(math.floor(f["t"] / dt + 1e-9))
            if not (0 <= j < len(R["Q"]) and R["on"][j]):
                continue
            tc = (j + 0.5) * dt
            ux = np.interp(tc, ta[order], E["u_h"][idx][order, 0])
            uy = np.interp(tc, ta[order], E["u_h"][idx][order, 1])
            U = np.interp(tc, ta[order], E["U_perp"][idx][order])
            dc = max(dc, abs(R["u_h"][j, 0] - ux), abs(R["u_h"][j, 1] - uy), abs(R["U_perp"][j] - U))
        assert dc <= 1e-4, f"{p} (c): binned record vs parcel series {dc:.2e} m/s > 1e-4"
        # (g) the runtime refuses a replay that is not the frozen one (REV_code finding 1): no paired frame, a changed GP
        #     frame set, a perturbed r_hat -> RuntimeError from replay() and band(), never a silent fallback to the law
        import copy
        refused = {}
        for case in ("no_gated_frame", "gp_set_changed", "rhat_perturbed"):
            K2 = copy.deepcopy(K)
            if case == "no_gated_frame":
                for f in K2["frames"]:
                    f["gated"] = False
            elif case == "gp_set_changed":
                fgp = [f for f in K2["frames"] if f.get("gp")]
                fgp[len(fgp) // 2]["gated"] = False          # its residual leaves the fit; the frozen set still lists it
            else:
                fgp = [f for f in K2["frames"] if f.get("gp")]
                fgp[0]["rhat"]["ux"] += 1e-6
            for fn_, call in (("replay", lambda: MI.replay(EL, p, K=K2)),
                              ("band", lambda: MI.band(p, np.array([0.5, 1.0]), K=K2, E=EL))):
                try:
                    call()
                    raise AssertionError(f"{p} (g): {fn_} accepted a stream_kin.json with {case}")
                except RuntimeError as e:
                    assert "refusing" in str(e), (p, case, fn_, str(e))
            refused[case] = True
        E3, _ = MI.replay(EL, p, K=K)                        # the frozen file itself replays (= the production build)
        for k in ("t_arr", "u_h", "U_perp", "t_f"):
            assert np.array_equal(E3[k], E[k], equal_nan=True), f"{p} (g): replay(law E) != production replay build ({k})"
        # (d) LOO z-sd
        loo = {q: gps[q].loo_z_sd() for q in MI.QS}
        assert all(0.7 <= v <= 1.3 for v in loo.values()), f"{p} (d): LOO z-sd {loo}"
        # (e) footprint cap inactive (drvf_record: |u_h| <= min(0.40 / D_M, U_perp) in D units = 0.40 L / 0.08 m/s)
        from ..v05_pitcher_check import _D_M
        Lm = float(info["L_m"])
        cap = 0.40 * Lm / _D_M
        m = R["on"] & (R["Q"] > 0)
        uh = np.hypot(R["u_h"][m, 0], R["u_h"][m, 1])
        assert np.all(uh < cap) and np.all(uh < R["U_perp"][m]), f"{p} (e): |u_h| max {uh.max():.4f} vs cap {cap:.4f}"
        # (f) reported: M1's GP agreement; implied-drop residual
        fM1 = {}
        sm = src_file("M1", f"stream_kin_{p}_smooth.npz")
        if sm:
            S1 = np.load(sm)
            tm, gap = S1["t"], S1["gap"].astype(bool)
            ok = (~gap) & (MI.taper(tm, wins) >= 1.0)
            bins = np.clip(np.floor(tm / dt + 1e-9).astype(int), 0, len(R["Q"]) - 1)
            for q, key, series in (("ux", "u_F_x", R["u_h"][:, 0]), ("uy", "u_F_y", R["u_h"][:, 1]), ("wf", "w_f", R["U_perp"])):
                v = np.where(R["on"][bins], series[bins], np.nan)
                fin = ok & np.isfinite(v)
                fM1[q] = round(float(np.mean(np.abs(v[fin] - S1[key][fin]) <= S1[f"{key}_sd"][fin])), 3)
        inw = idx[wi >= 1.0]
        af = E["t_f"][inw]; wf = E["U_perp"][inw]
        resid = wf * af - 0.5 * 9.81 * af ** 2 - E["origin"][inw, 2]
        rows[p] = dict(a_parcels_outside=int(out.sum()), b_gp_vs_frozen=db, b_gp_vs_proto=dproto, b_parcel_u=float(e_u),
                       b_parcel_U=float(e_U), b_parcel_af_after_2_passes=float(e_a), c_binned_vs_parcels=float(dc),
                       d_loo_z_sd={q: round(v, 3) for q, v in loo.items()}, e_uh_max=float(uh.max()), e_cap=cap,
                       g_refused=refused,
                       f_frac_within_M1_1sig=fM1,
                       f_implied_drop_resid_mm=dict(p16=float(np.percentile(resid, 16) * 1e3),
                                                    p50=float(np.median(resid) * 1e3),
                                                    p84=float(np.percentile(resid, 84) * 1e3), n=int(len(resid))),
                       replay_medians_w1=dict(uh=float(np.median(np.hypot(*E["u_h"][inw].T))),
                                              uh_law=float(np.median(np.hypot(*EL["u_h"][inw].T))),
                                              U=float(np.median(E["U_perp"][inw])), U_law=float(np.median(EL["U_perp"][inw])),
                                              af_ms=float(np.median(af) * 1e3), af_law_ms=float(np.median(EL["t_f"][inw]) * 1e3)))
    REPORT["T6"] = rows


T7_CASES = (("i", "v05n_AB_cmaes_N48.json", "replay", "meas", "kappa_Q,beta"),
            ("ii", "mcd_S1kq_N96.json", "replay", "meas", "tau_s,p_dep,kappa_c,B_dep"),
            ("iii", "v05n_AB_cmaes_N48.json", None, "err", "kappa_Q,beta"))


def test_T7_gradient():
    """diag gradcheck (SPEC27 3.5 item 9 T7) and two exact checks of the new adjoint / box pieces."""
    import argparse
    from ..diff import diag as DG
    from ..diff import schedule as SCH
    from . import measured_inlet as MI
    cdir = os.path.join(HERE, "..", "data", "closures")
    rows = {}
    for tag, init, inlet, box, keys in T7_CASES:
        a = argparse.Namespace(init=os.path.join(cdir, init), pour="tulip", N=32, T=1.0, device="cpu", eps=1e-7, tol=2e-3,
                               box=box, keys=keys, inlet=inlet)
        with clean_env():
            r = DG._gradcheck(a)
        rows[tag] = dict(init=init, inlet=r["inlet"], box=box, milk=r["milk"], vj=r["vj"], worst=r["worst"], ok=r["ok"],
                         zero_bitwise=r["same"], K=r["K"], scan=r["scan"], gvj=r["gvj"], vj_share=r["vj_share"],
                         rows=r["rows"])
        if r["box_info"]:
            bi = r["box_info"]
            rows[tag]["box_median_windows"] = bi["median_windows"]
            rows[tag]["box_median_law_only"] = bi["median_law_only"]
            rows[tag]["knots_in_windows"] = bi["knots_in_windows"]
        assert r["inlet"] == (inlet or "record"), (tag, r["inlet"])
        assert r["same"], f"T7 ({tag}): a zero correction does not reproduce the record bitwise"
        assert r["ok"], f"T7 ({tag}): worst relative error {r['worst']:.2e} > 2e-3"
    # (ii) exercises the V_j adjoint: S1kq has vj 1, the solver reads V_j, and the dx / dy knots carry a V_j part
    assert rows["ii"]["vj"] == 1.0 and rows["ii"]["milk"] == "mp5fct", rows["ii"]
    assert rows["ii"]["gvj"] > 0 and rows["ii"]["vj_share"] > 0, "T7 (ii): the V_j adjoint is not exercised"
    assert rows["i"]["gvj"] == 0.0 and rows["iii"]["gvj"] == 0.0, "vj 0 closures must not read V_j"
    # (iv) vj_T is the exact transpose of foot_velocity (linear): <G, J dx> = <J^T G, dx> on the replay schedule
    with clean_env():
        D = DG.make_pour("tulip", dict(e0="world", exit_cal=False, upwell=False, circ=False, circ_shape={}, milk=0,
                                       sink=0, visc_tr=0, inlet="replay"), dict(N=32, device="cpu", T=None, box="meas"))
    inl = D.inl
    rng = np.random.default_rng(11)
    n = len(inl.t)
    dx, dy, Gx, Gy = (rng.standard_normal(n) for _ in range(4))
    vx, vy = SCH.foot_velocity(dx, dy, inl.active.astype(float), inl.fdt)
    ax, ay = inl.vj_T(Gx, Gy)
    lhs = float(Gx @ vx + Gy @ vy)
    rhs = float(ax @ dx + ay @ dy)
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert rel <= 1e-12, f"T7 (iv): vj_T is not the transpose of foot_velocity (rel {rel:.2e})"
    rows["iv_vjT_dot"] = dict(n_frames=n, lhs=lhs, rhs=rhs, rel=rel)
    # (v) the band from the replay emission (Pour.E) = the band from the MI-law build (law_emission)
    _, _, EL, _, _ = mi_build("tulip", "law")
    tk = inl.knots
    b_rep = MI.band("tulip", tk, E=D.pour.E)
    b_law = MI.band("tulip", tk, E=EL)
    dmax = max(float(np.max(np.abs(b_rep[q] - b_law[q]))) for q in MI.QS)
    assert dmax <= 1e-12, f"T7 (v): band(replay E) differs from band(law E) by {dmax:.2e}"
    rows["v_band"] = dict(max_abs_diff=dmax, knots=int(len(tk)), knots_in_windows=int(np.sum(inl.kw >= 1.0)))
    # (vi) the V_j adjoint where V_j matters (S1kq, T 6 s: high pours, 1 - chi large), discriminating knots only
    a = argparse.Namespace(init=os.path.join(cdir, "mcd_S1kq_N96.json"), pour="tulip", N=32, T=6.0, device="cpu", eps=1e-7,
                           tol=2e-3, box="meas", keys="", inlet="replay", rand_knots=0, vj_knots=4)
    with clean_env():
        r = DG._gradcheck(a)
    vr = [x for x in r["rows"] if x["var"] in ("dx", "dy")]
    rows["vi_vj"] = dict(T=6.0, worst=r["worst"], vj_share=r["vj_share"], gvj=r["gvj"], rows=vr)
    assert len(vr) == 4 and r["ok"], f"T7 (vi): worst relative error {r['worst']:.2e} > 2e-3 at the V_j knots"
    assert max(x["vj_share"] for x in vr) > 2e-3, "T7 (vi): no knot where the V_j part exceeds the tolerance"
    REPORT["T7"] = rows


# ============================================================================= T8, T9
def test_T8_landing_in_solver():
    from ..diff import schedule as SCH
    from . import measured_inlet as MI
    from .real_program import RES
    rows = {}
    for p in CAL:
        K = MI.load(p)
        with clean_env():
            script, pour = SCH.pour_script(p, 48, e0="world", inlet="law")     # 'law' forces e0 off
        assert pour.inlet == "law"

        class Rec:
            def __init__(s):
                s.T, s.prog, s.params, s.ts = script.T, script.prog, script.params, []

            def sample(s, t):
                s.ts.append(t)
                return script.sample(t)
        rc = Rec()
        S = SCH.build(rc, 48)
        ts = np.array(rc.ts[1::2])                       # build samples each row at t + fdt / 2 (second call)
        assert len(ts) == S["n"]
        L = float(pour.info["L_m"])
        assert abs(L - K["L_solver_m"]) < 1e-9, (L, K["L_solver_m"])
        col = {r["fi"]: r for r in json.load(open(os.path.join(RES, p, "stream_foot_col.json")))["frames"]}
        da, dc = [], []
        for f in K["frames"]:
            if not f["gated"]:
                continue
            r = col.get(f["fi"])
            if r is None or r.get("validated_grade") not in ("A", "B") or abs(r["t"] - f["t"]) >= 0.02 or r.get("x") is None:
                continue
            m = S["active"] & (np.abs(ts - f["t"]) <= 0.020) & (S["Q"] > 0)
            if not m.any():
                continue
            wq = S["Q"][m] * S["fdt"][m]
            x = np.array([(wq * S["x"][m]).sum(), (wq * S["y"][m]).sum()]) / wq.sum()
            d = (x - np.array([r["x"], r["y"]])) * L
            uF = np.array(f["u_F"]); e = uF / np.linalg.norm(uF); n = np.array([-e[1], e[0]])
            da.append(float(d @ e) * 1e3); dc.append(float(d @ n) * 1e3)
        ma, mc = float(np.median(da)), float(np.median(dc))
        rows[p] = dict(n=len(da), along_med_mm=ma, across_med_mm=mc, along_sd_mm=float(np.std(da)),
                       across_sd_mm=float(np.std(dc)))
        assert abs(ma) <= 3.0 and abs(mc) <= 3.0, f"{p}: in-solver MI-law landing along {ma:+.2f} / across {mc:+.2f} mm"
        # (b) stream_foot_mi.json carries the SPEC27 1.1 frame gate (REV_spec F1): the rows the E0 estimator can use are
        #     exactly the gated frames' A / B col rows; every other row is tier none; the shipped file equals foot_file's
        #     output regenerated from the shipped stream_kin.json and stream_foot_col.json
        Fmi = json.load(open(os.path.join(RES, p, MI.FOOT_FILE)))
        gated = {int(f["fi"]) for f in K["frames"] if f["gated"]}
        usable = {int(r["fi"]) for r in Fmi["frames"] if SCH._e0_sel(r, SCH.E0_PLAUS)}
        flagged = {int(r["fi"]) for r in Fmi["frames"] if r["mi_window"]}
        assert usable <= gated, f"{p} (b): E0-usable foot rows of ungated frames {sorted(usable - gated)}"
        assert flagged <= gated, f"{p} (b): kept foot rows of ungated frames {sorted(flagged - gated)}"
        for r in Fmi["frames"]:
            if not r["mi_window"]:
                assert r["quality"] == 0.0 and not r.get("trusted"), (p, r["fi"])
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, p))
            for fn_ in (MI.KIN_FILE, "stream_foot_col.json"):
                shutil.copy(os.path.join(RES, p, fn_), os.path.join(td, p, fn_))
            MI._KIN.pop(MI.kin_path(p, td), None)
            fn2, _ = MI.foot_file(p, td)
            MI._KIN.pop(MI.kin_path(p, td), None)
            same = open(fn2, "rb").read() == open(os.path.join(RES, p, MI.FOOT_FILE), "rb").read()
        assert same, f"{p} (b): shipped {MI.FOOT_FILE} differs from foot_file() on the shipped inputs"
        rows[p].update(foot_rows_kept=len(flagged), foot_rows_e0_usable=len(usable), gated_frames=len(gated))
    REPORT["T8"] = rows


def test_T9_allow_list():
    from . import measured_inlet as MI
    rows = {}
    for p in CAL:
        J = json.load(open(MI.kin_path(p)))
        bad = MI.check_keys(J)
        assert not bad, f"{p}: keys outside the allow-list {bad[:10]}"
        txt = json.dumps(J)
        for k in MI.FORBIDDEN:
            assert f'"{k}"' not in txt, f"{p}: forbidden key {k}"
        assert '"oracle' not in txt.lower(), f"{p}: an oracle* key"
        for f in J["frames"]:
            assert set(f) <= set(MI.FRAME_KEYS), (p, f["fi"], set(f) - set(MI.FRAME_KEYS))
        rows[p] = dict(frames=len(J["frames"]), gated=sum(f["gated"] for f in J["frames"]),
                       gp_frames=sum(f.get("gp", False) for f in J["frames"]), keys=len(MI.allowed_keys()))
    REPORT["T9"] = rows


# ============================================================================= CLI
TESTS = dict(T1=test_T1_default_bit_identity, T1b=test_T1b_diag_path, T1c=test_T1c_exit_cal,
             T1d=test_T1d_in_process_switch, T1e=test_T1e_explicit_default, T1f=test_T1f_precedence,
             T2=test_T2_port_parity, T3=test_T3_closed_form, T4=test_T4_energy_geometry, T5=test_T5_ledger,
             T6=test_T6_replay, T7=test_T7_gradient, T8=test_T8_landing_in_solver, T9=test_T9_allow_list)


def main(argv=None):
    names = list(sys.argv[1:] if argv is None else argv) or list(TESTS)
    import latte_imex
    print(f"latte_imex from {os.path.dirname(latte_imex.__file__)}; env LATTE_INLET={os.environ.get('LATTE_INLET')} "
          f"LATTE_FOOT={os.environ.get('LATTE_FOOT')} LATTE_POSE={os.environ.get('LATTE_POSE')}", flush=True)
    res = {}
    for nm in names:
        t0 = time.time()
        try:
            TESTS[nm]()
            res[nm] = "PASS"
        except Skip as e:
            res[nm] = f"STUB ({e})"
        except Exception as e:                       # noqa: BLE001
            res[nm] = f"FAIL: {e}"
            traceback.print_exc()
        print(f"[{nm}] {res[nm]}  ({time.time() - t0:.0f} s)", flush=True)
        if nm in REPORT:
            print(f"[{nm}] report: " + json.dumps(REPORT[nm], default=float, indent=1), flush=True)
    print("SUMMARY " + json.dumps(res), flush=True)
    return 0 if all(v == "PASS" or v.startswith("STUB") for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
