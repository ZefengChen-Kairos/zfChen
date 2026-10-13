"""V0.5 pourable design, constraints A: a fast QUASI-STATIC surrogate of what the pitcher + an ideal hand can deliver.

Question it serves: which per-stroke inlet quantities (flow Q, spout height z, impact speed U_perp, exit horizontal speed
u_h) can one pitcher, one fill and one human hand give at the same time?  Built from latte_imex.pitcher (unmodified)
through v05_pitcher_check.FastPitcher (identical free surface, faster), at rest: horizontal free surface (no slosh),
g_eff = g, no carried hand velocity unless asked for.

FLOW MAP  (spec, fill V, flow Q, roll)  [grid V 100..400 mL step 25, Q 1..32 mL/s in sqrt(2) steps, roll -20..20 deg]
  tilt theta(Q, V, roll) = Pitcher.tilt_for_flow (steady weir law); lip head h_max, discharge-weighted head h_bar,
  exit speed u0 = sqrt(2 g max(h_bar, h_k)); jet origin (discharge-weighted crest point) relative to the spout tip in
  the spout frame (forward along the horizontal spout direction, side = left of it, up).
  Exit direction R(yaw, theta, roll) (cos a, 0, -sin a) for the lip slope a = alpha_lip in {0, 10, 20, 30} deg; exit
  horizontal speed u_h = u0 |e_h| (= u0 cos(theta + a) at roll 0; negative cos means the stream leaves backwards).
JET  (as Pitcher.jet at rest): drop z0 = z_tip + dz_origin; v0z = u0 e_z; t_hit = (v0z + sqrt(v0z^2 + 2 g z0)) / g;
  U_perp = sqrt(v0z^2 + 2 g z0) (Bernoulli on the vertical component with the real u0 and the origin height); stream
  foot = tip + origin_h + u0 e_h t_hit (+ v_hand t_hit if a hand velocity is given; U_perp then uses v0z + v_hand_z).
CLEARANCE MAP  lowest spout-tip height z_min = margin + Pitcher.required_lift (default Cup: inner radius 40 mm, tilted 20 deg
  towards the pitcher, 2 mm margin, coffee surface z = 0; cup 'table' = standing on the table 60 mm below the coffee
  (the pitcher bottom outside the cup can hit it at tilts <= 40 deg near the near rim), 'held' = no table) vs the tip position in the spout frame: a = coordinate towards
  the pitcher body (= towards the near rim), l = lateral (required_lift's 'perp'), tilt, roll.  required_lift tilts the
  cup towards the pitcher whatever the yaw, so the model is invariant under rotations about the cup axis: yaw enters only
  through (a, l) of the tip (checked numerically in selftest).  z_min of a STREAM FOOT position = the tip z_min at the
  tip that puts the jet on the foot (fixed point over the flight offset).
  wall 'mean'     = Pitcher.required_lift exactly (azimuth-mean outer wall; the ideal-hand runs used this).
  wall 'resolved' = the same test on the azimuth-resolved outer wall r_out(z, phi) of the CAD grid (includes the spout
                    flare under the lip and the handle).  For NX it is 1-7 mm stricter near the spout (the review's
                    "model is optimistic near the spout").
PITCHER SPECS  'nx': CAD NX as is (spout tip b = 4.86 mm beyond the mean outer wall at the top).
  'b10'..'b25': pointed-spout variants, a GEOMETRY WRAPPER on the NX CAD grid (pitcher.py untouched): the beak of the rim
  is extended radially by db w(phi) (w = 1 - |phi| / 15 deg, so the extension is a pointed V in plan view), and the
  inner and outer walls in the top L_s = 35 mm at those azimuths by db w(phi) (z - (H - L_s)) / L_s (a straight spout
  flare / channel); db is chosen so that tip radius - mean outer radius at the top = b.  z_rim stays flat (NX's rim is
  flat).  Consequences that come out of the model, not put in: the longer lip reaches the free surface at a smaller
  tilt (tilt for a flow drops), the jet origin moves out, and the clearance changes through both the tip position and
  the tilt.  The closed form h_min = m + D_max cot(theta) - b / sin(theta) (D_max = r_in / cos(cup tilt) - margin - a)
  is checked against the map for every spec in `closed_form_check`.
HAND  presets 'ideal' (the ideal hand of v05_pitcher_check: tilt rate 300 deg/s, hand acceleration 10 m/s^2 on a 0.15 m
  lever, tip speed 0.5 m/s, yaw rate 300 deg/s, planned lifts 3 m/s^2) and 'video' (what the barista in the clips does:
  60 deg/s smooth tilt, 2 m/s^2, path 0.07 m/s, lifts 0.11 m/s); smooth_yaw, plan_lift (running max + closing with the
  hand parabola, as the ideal hand), track_tilt (rate / acceleration limited), hand_check (peaks vs limits).
  5 Hz flow pulsation: NOT allowed by default (pulse=None -> depth 0).  pulse='tip' = the ideal hand's tilt about the
  spout tip (amplitude <= 0.95 min(acc / (L w^2), rate / w) = 3.7 deg at 5 Hz, depth ~0.41 at tau = 0.08 s, no tip
  bob, but the tilt troughs need extra lift); pulse='wrist' = tilt about a wrist ~19 cm behind the tip (review:
  +-2 deg -> 0.21 (zeta 0.3) / 0.41 (zeta 0.1), tip bobs 12.7 mm p-p; +-3.5 deg -> 0.62-0.68, 21.4 mm; the flow breaks
  every cycle).
VALIDATION  against the ideal-hand simulations best/runs/fs_<pattern>_ideal_f{250,350} (pitcher.py stepped at 1 ms with
  slosh, lag, lifts): per stroke U_perp, u_h (exit, relative to the pitcher; and cup frame incl. the hand velocity), z
  incl. the lift, Q, tilt, u0; per ms the clearance requirement where the hand had to lift.

usage (from /home/user/zfChen; NUMBA_NUM_THREADS=1):
  python3 -m latte_imex.v05_pourable selftest
  python3 -m latte_imex.v05_pourable build --spec nx --what flow|clear --wall mean|resolved --cup table|held
  python3 -m latte_imex.v05_pourable build_all --jobs 2 [--cups table,held]   # every spec, both walls
  python3 -m latte_imex.v05_pourable selftest                    # interpolation vs direct pitcher.py calls
  python3 -m latte_imex.v05_pourable validate                    # NX vs the ideal-hand runs (250 / 350 mL)
  python3 -m latte_imex.v05_pourable report                      # key constraint numbers -> surrogate/key_numbers.json
Outputs: <review>/v05pour/surrogate/ (flow_<spec>.npz, clear_<spec>_<wall>[_held].npz, jet_<spec>.npz, validation.json,
key_numbers.json, closed_form_check.json, selftest.json).
DESIGN (pitcher-constrained inverse design of the V0.5 inlet, prereg <review>/v05pour/design/prereg.json):
  python3 -m latte_imex.v05_pourable design --spec S1|S2|S3 [--alphas 0,10,20,30]   -> design/<spec>/alpha<a>.json
  python3 -m latte_imex.v05_pourable finalize --spec S1 [--alpha a]                  -> design/<spec>/final.json, ptab_*.npz
  python3 -m latte_imex.v05_pourable solve --spec S1 --pattern heart --N 128         -> design/runs/<tag>_<pattern>_N<N>/
  python3 -m latte_imex.v05_pourable fsrun --pattern heart --N 48 --via drvf|pourf   (driver reproduction check)
In code: S = Surrogate("b20", wall="resolved", cup="held"); S.tilt(Q, V); S.jet(Q, V, z_tip, roll, alpha);
S.z_min(foot_x, foot_y, yaw, Q, V, roll); D = deliver(S, sched, fill_mL, alpha=10, hand="ideal"); stroke_summary(D, strokes).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .pitcher import Cup, G, GridPitcherGeometry, OutflowLaw, Pose
from . import v05_pitcher_check as PC

ML = 1e-6
D = PC.D
OUT = os.path.join(PC.REVIEW, "v05pour", "surrogate")
NX_JSON = os.path.join(os.path.dirname(__file__), "pitcher_geom", "pitcher_nx.json")

V_GRID = np.arange(100.0, 400.1, 25.0)                     # fill [mL]
LQ_GRID = np.arange(0.0, 5.001, 0.5)                       # log2 Q, Q = 1 .. 32 mL/s
ROLLS = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])          # [deg]
ALPHAS = (0.0, 10.0, 20.0, 30.0)                           # lip slope alpha_lip [deg]
A_GRID = np.arange(-45.0, 50.1, 5.0)                       # tip coordinate towards the pitcher body [mm]
L_GRID = np.arange(-30.0, 30.1, 5.0)                       # lateral tip coordinate [mm]
T_GRID = np.arange(10.0, 90.1, 2.5)                        # tilt [deg]
Z_GRID = np.array([2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 80.0])   # tip height for the tabulated jet [mm]
Z0 = 0.002                                                 # floor of the tip height = cup margin [m]
SPECS = ("nx", "b10", "b15", "b20", "b25")
WALLS = ("mean", "resolved")
CUPS = ("table", "held")


def make_cup(cup="table"):
    """'table': the default Cup (stands on the table, whose plane is 60 mm below the coffee: at tilts <= 40 deg the
    pitcher bottom outside the cup can reach it when the tip is low near the near rim); 'held': the same cup held in the
    other hand (no table within reach), as the baristas in the clips do.
    't<deg>' (design extension): held cup tilted <deg> towards the pitcher with a CONSISTENT rim: the near rim sits 1 mm
    above the coffee (rim_z = r_out tan(tilt) + 1 mm; the default Cup's 20 deg tilt with rim_z 10 mm puts the near rim
    2.7 mm below the coffee).  A cup that is less full early in the pour can be tilted further without spilling."""
    if cup == "table":
        return Cup()
    if cup.startswith("t"):
        tl = math.radians(float(cup[1:]))
        c0 = Cup()
        return Cup(height=1.0, tilt=tl, rim_z=c0.r_out * math.tan(tl) + 1e-3)
    return Cup(height=1.0)


def _clear_file(name, wall, cup):
    return f"clear_{name}_{wall}" + ("" if cup == "table" else f"_{cup}") + ".npz"


# =========================================================================== pitcher specs (geometry wrapper)
@dataclass
class Spec:
    b: float = None          # spout tip beyond the mean outer wall at the top [mm]; None = NX as is
    L_s: float = 35.0        # flare length of the spout extension below the rim [mm]
    phi_s: float = 15.0      # half-width in azimuth of the beak extension [deg]
    nr: int = 6              # radial sampling of the cavity (the ideal-hand runs used 6)

    @property
    def name(self):
        if self.b is None:
            return "nx"
        tag = f"b{self.b:g}"
        if self.L_s != 35.0:
            tag += f"L{self.L_s:g}"
        return tag


def spec_from_name(name):
    if name == "nx":
        return Spec()
    s = name[1:]
    L = 35.0
    if "L" in s:
        s, L = s.split("L")
        L = float(L)
    return Spec(b=float(s), L_s=L)


_GEO = {}


def _nx_b(data):
    """b of a CAD grid: tip radius - azimuth-mean outer radius at the top [mm]."""
    return float(np.max(data["r_rim"]) - np.mean(np.asarray(data["r_out"], float)[-1]))


def spout_data(spec):
    """CAD data dict of the spec (NX JSON, beak extended for b specs)."""
    with open(NX_JSON) as f:
        data = json.load(f)
    data.setdefault("name", "pitcher_nx")
    if spec.b is None:
        return data
    zs = np.asarray(data["zs"], float)
    ph = np.asarray(data["phis"], float) - data.get("spout_phi_residual", 0.0)
    ph = (ph + math.pi) % (2 * math.pi) - math.pi
    w = np.maximum(0.0, 1.0 - np.abs(ph) / math.radians(spec.phi_s))
    H = float(zs[-1])
    flare = np.clip((zs - (H - spec.L_s)) / spec.L_s, 0.0, 1.0)
    b0 = _nx_b(data)
    i_tip = int(np.argmax(data["r_rim"]))
    # tip radius grows by db w(phi_tip); the mean outer radius at the top by db mean(w) flare(H)
    db = (spec.b - b0) / (w[i_tip] - flare[-1] * float(np.mean(w)))
    data = dict(data)
    data["r_rim"] = (np.asarray(data["r_rim"], float) + db * w).tolist()
    ext = db * flare[:, None] * w[None, :]
    data["r_in"] = (np.asarray(data["r_in"], float) + ext).tolist()
    data["r_out"] = (np.asarray(data["r_out"], float) + ext).tolist()
    data["name"] = "pitcher_nx_" + spec.name
    data["spout_ext_mm"] = db
    return data


def geometry(spec):
    key = (spec.name, spec.nr)
    if key not in _GEO:
        if spec.b is None:
            _GEO[key] = PC.nx_geometry(spec.nr)
        else:
            _GEO[key] = GridPitcherGeometry(spout_data(spec), nr=spec.nr)
    return _GEO[key]


def spec_b(spec):
    g = geometry(spec)
    return float((g.tip[0] - g.r_outer(g.height)) * 1e3)


class SpoutPitcher(PC.FastPitcher):
    """FastPitcher whose clearance test can use the azimuth-resolved outer wall (wall='resolved')."""

    def __init__(self, *a, wall="mean", nz_wall=60, **kw):
        super().__init__(*a, **kw)
        self.wall = wall
        self._wall_b = None
        self.nz_wall = nz_wall

    def wall_samples(self, pose, nz=40, nphi=36):
        if self.wall == "mean":
            return super().wall_samples(pose, nz, nphi)
        if self._wall_b is None:
            g = self.geom
            zz = np.linspace(g.zs[0], g.height, self.nz_wall)
            R = np.stack([np.interp(zz, g.zs, g.r_out[:, j]) for j in range(len(g.phis))], 1)   # (nz, nphi)
            PH = np.broadcast_to(g.phis[None, :], R.shape)
            Z = np.broadcast_to(zz[:, None], R.shape)
            self._wall_b = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1).reshape(-1, 3)
        return self.world(pose, self._wall_b)


def make_pitcher(spec, V_mL=250.0, wall="mean", law=None):
    return SpoutPitcher(geom=geometry(spec), V0=V_mL * ML, law=law or OutflowLaw(slosh=False), wall=wall)


# =========================================================================== map builders
def build_flow(spec, out=OUT, verbose=True, V_grid=None):
    """Steady flow map on (V, log2 Q, roll); spout direction +x (yaw 0), tip at (0, 0, 2 cm).  V_grid [mL] overrides
    V_GRID (the design uses 20-400 mL: a barista ends the pour with ~60 mL left at 72-77 deg tilt)."""
    V_GRID = np.asarray(V_grid, float) if V_grid is not None else globals()["V_GRID"]
    P = make_pitcher(spec)
    law = P.law
    shape = (len(V_GRID), len(LQ_GRID), len(ROLLS))
    F = {k: np.full(shape, np.nan) for k in ("tilt", "h_max", "h_bar", "Q_ss", "u0", "ox", "oy", "oz", "width")}
    onset = np.full((len(V_GRID), len(ROLLS)), np.nan)
    t0 = time.time()
    tip = (0.0, 0.0, 0.02)
    for iv, V in enumerate(V_GRID):
        for ir, ro in enumerate(ROLLS):
            rr = math.radians(ro)
            pose0 = Pose(tip, 0.0, 0.0, rr)
            onset[iv, ir] = math.degrees(PC.onset_tilt(P, pose0, V * ML))
            for iq, lq in enumerate(LQ_GRID):
                Q = 2.0 ** lq * ML
                th = P.tilt_for_flow(pose0, Q, V=V * ML, iters=32)
                pose = Pose(tip, 0.0, th, rr)
                c, _ = P.free_surface(pose, V * ML)
                Wr = P.weir(pose, c)
                F["tilt"][iv, iq, ir] = math.degrees(th)
                F["h_max"][iv, iq, ir] = Wr["h_max"]
                F["h_bar"][iv, iq, ir] = Wr["h_bar"]
                F["Q_ss"][iv, iq, ir] = Wr["Q_ss"] / ML
                F["u0"][iv, iq, ir] = math.sqrt(2 * G * max(Wr["h_bar"], law.h_k))
                o = np.asarray(Wr["origin"], float) - np.asarray(tip)
                F["ox"][iv, iq, ir], F["oy"][iv, iq, ir], F["oz"][iv, iq, ir] = o
                F["width"][iv, iq, ir] = Wr["width"]
        if verbose:
            print(f"  flow {spec.name} V={V:.0f}: tilt {F['tilt'][iv, 0, 2]:.1f}-{F['tilt'][iv, -1, 2]:.1f} deg, "
                  f"u0 {F['u0'][iv, 0, 2]:.3f}-{F['u0'][iv, -1, 2]:.3f} [{time.time() - t0:.0f}s]", flush=True)
    os.makedirs(out, exist_ok=True)
    meta = dict(spec=asdict(spec), name=spec.name, b_mm=spec_b(spec), capacity_mL=geometry(spec).capacity / ML,
                law={k: getattr(law, k) for k in ("C_d", "h_k", "h_on", "h_off", "tau", "tau_stop")},
                note="yaw 0 (spout along +x), tip at (0, 0, 2 cm); ox forward, oy left, oz up relative to the tip [m]; "
                     "tilt [deg]; heads [m]; u0 [m/s]; Q_ss check [mL/s]; onset = tilt where h_max = h_on [deg]",
                wall_s=time.time() - t0)
    np.savez_compressed(os.path.join(out, f"flow_{spec.name}.npz"), V=V_GRID, lQ=LQ_GRID, roll=ROLLS, onset=onset,
                        meta=json.dumps(meta), **F)
    return F


def build_clear(spec, wall="mean", cup_kind="table", out=OUT, verbose=True):
    """z_min_tip [m] on (a, l, tilt, roll) at yaw -90 deg (spout towards -y, pitcher body towards +y)."""
    P = make_pitcher(spec, wall=wall)
    cup = make_cup(cup_kind)
    yaw = -math.pi / 2
    e = np.array([math.cos(yaw), math.sin(yaw)])
    d = -e
    n = np.array([-d[1], d[0]])
    Z = np.full((len(A_GRID), len(L_GRID), len(T_GRID), len(ROLLS)), np.nan)
    t0 = time.time()
    for ia, a in enumerate(A_GRID):
        for il, l in enumerate(L_GRID):
            xy = (a * d + l * n) * 1e-3
            for it, th in enumerate(T_GRID):
                for ir, ro in enumerate(ROLLS):
                    pose = Pose((xy[0], xy[1], Z0), yaw, math.radians(th), math.radians(ro))
                    Z[ia, il, it, ir] = Z0 + P.required_lift(pose, cup)
        if verbose and ia % 4 == 0:
            print(f"  clear {spec.name}/{wall}/{cup_kind} a={a:+.0f} mm: z_min(l=0, roll 0) at 40/50/60 deg "
                  + "/".join(f"{np.interp(x, T_GRID, Z[ia, len(L_GRID) // 2, :, 2]) * 1e3:.1f}" for x in (40, 50, 60))
                  + f" mm [{time.time() - t0:.0f}s]", flush=True)
    os.makedirs(out, exist_ok=True)
    meta = dict(spec=asdict(spec), name=spec.name, wall=wall, cup_kind=cup_kind, b_mm=spec_b(spec), cup=cup.__dict__, z0=Z0,
                note="z_min of the spout tip [m] = z0 + Pitcher.required_lift; a: tip coordinate towards the pitcher body "
                     "(near rim) [mm], l: lateral (required_lift 'perp') [mm]; tilt, roll [deg]; built at yaw -90 deg",
                wall_s=time.time() - t0)
    np.savez_compressed(os.path.join(out, _clear_file(spec.name, wall, cup_kind)), a=A_GRID, l=L_GRID, tilt=T_GRID, roll=ROLLS,
                        zmin=Z, meta=json.dumps(meta))
    return Z


# =========================================================================== the surrogate
def _rgi(axes, vals):
    return RegularGridInterpolator(axes, vals, method="linear", bounds_error=False, fill_value=None)


def _clip(x, ax):
    return np.clip(np.asarray(x, float), ax[0], ax[-1])


class Surrogate:
    """Interpolated maps of one pitcher spec.  Units: Q [mL/s], V [mL], heights / positions [m], angles [deg] unless
    the argument name says rad; every method is vectorised over numpy arrays."""

    def __init__(self, spec="nx", wall="mean", out=OUT, cup="table"):
        self.spec = spec_from_name(spec) if isinstance(spec, str) else spec
        self.wall, self.cup = wall, cup
        fz = np.load(os.path.join(out, f"flow_{self.spec.name}.npz"))
        self.fmeta = json.loads(str(fz["meta"]))
        ax = (fz["V"], fz["lQ"], fz["roll"])
        self.fax = ax
        self.F = {k: _rgi(ax, fz[k]) for k in ("tilt", "h_max", "h_bar", "u0", "ox", "oy", "oz", "width")}
        self.onset = _rgi((fz["V"], fz["roll"]), fz["onset"])
        cz = np.load(os.path.join(out, _clear_file(self.spec.name, wall, cup)))
        self.cax = (cz["a"], cz["l"], cz["tilt"], cz["roll"])
        self.C = _rgi(self.cax, cz["zmin"])
        self.b_mm = self.fmeta["b_mm"]
        self.law = OutflowLaw()

    # ---- flow
    def _fx(self, Q, V, roll):
        Q, V, roll = np.broadcast_arrays(np.asarray(Q, float), np.asarray(V, float), np.asarray(roll, float))
        lq = np.log2(np.maximum(Q, 1e-6))
        return np.stack([_clip(V, self.fax[0]), _clip(lq, self.fax[1]), _clip(roll, self.fax[2])], -1)

    def flow(self, Q, V, roll=0.0):
        """dict: tilt [deg], h_max, h_bar [m], u0 [m/s], ox, oy, oz (origin - tip, spout frame) [m], width [m]."""
        X = self._fx(Q, V, roll)
        return {k: f(X).reshape(X.shape[:-1]) for k, f in self.F.items()}

    def tilt(self, Q, V, roll=0.0):
        X = self._fx(Q, V, roll)
        return self.F["tilt"](X).reshape(X.shape[:-1])

    @staticmethod
    def exit_dir(tilt_deg, roll_deg=0.0, alpha_deg=10.0):
        """Unit exit vector in the spout frame (forward, left, up): R(0, tilt, roll) (cos a, 0, -sin a)."""
        th, ro, al = (np.radians(np.asarray(x, float)) for x in (tilt_deg, roll_deg, alpha_deg))
        ct, st, cr, sr, ca, sa = np.cos(th), np.sin(th), np.cos(ro), np.sin(ro), np.cos(al), np.sin(al)
        return ct * ca - st * cr * sa, sr * sa, -st * ca - ct * cr * sa

    def jet(self, Q, V, z_tip, roll=0.0, alpha=10.0, v_hand=None, yaw=None):
        """Stream at rest from a tip at height z_tip [m].  v_hand (vx, vy, vz) [m/s] world + yaw [rad] adds the carried
        hand velocity.  Returns dict: tilt, u0, u_h (exit horizontal speed relative to the pitcher, signed along the
        spout), u_h_side, v0z, z0, t_hit, U_perp, foot offset (fwd, side) [m] from the tip, and with v_hand the cup-frame
        horizontal velocity (uhx, uhy) and |u_h_cup|."""
        f = self.flow(Q, V, roll)
        ef, es, ez = self.exit_dir(f["tilt"], roll, alpha)
        u0 = f["u0"]
        vf, vs, vz = u0 * ef, u0 * es, u0 * ez
        out = dict(tilt=f["tilt"], u0=u0, u_h=np.hypot(vf, vs) * np.sign(vf), u_h_fwd=vf, u_h_side=vs, v0z=vz,
                   dz_origin=f["oz"])
        if v_hand is not None:
            yaw = np.asarray(yaw, float)
            ex, ey = np.cos(yaw), np.sin(yaw)
            vhx, vhy, vhz = (np.asarray(c, float) for c in v_hand)
            vz = vz + vhz
        z0 = np.maximum(np.asarray(z_tip, float) + f["oz"], 0.0)
        th = (vz + np.sqrt(vz ** 2 + 2 * G * z0)) / G
        out.update(z0=z0, t_hit=th, U_perp=np.sqrt(vz ** 2 + 2 * G * z0),
                   off_fwd=f["ox"] + vf * th, off_side=f["oy"] + vs * th)
        if v_hand is not None:
            uhx = vf * ex - vs * ey + vhx
            uhy = vf * ey + vs * ex + vhy
            out.update(uhx=uhx, uhy=uhy, u_h_cup=np.hypot(uhx, uhy),
                       foot_dx=(f["ox"] * ex - f["oy"] * ey) + uhx * th, foot_dy=(f["ox"] * ey + f["oy"] * ex) + uhy * th)
        return out

    def z_min_best(self, foot_x, foot_y, Q, V, yaws_deg=tuple(range(-180, 180, 15)), rolls_deg=(0.0,), alpha=10.0):
        """Lowest tip height over a set of yaws (and rolls) for one stream-foot position: (z_min, yaw_deg, roll_deg)."""
        best = None
        for r in rolls_deg:
            for y in yaws_deg:
                z = float(self.z_min(foot_x, foot_y, math.radians(y), Q, V, r, alpha)[0])
                if best is None or z < best[0]:
                    best = (z, y, r)
        return best

    # ---- clearance
    def zmin_tip(self, a_mm, l_mm, tilt_deg, roll_deg=0.0):
        """Lowest tip height [m] at tip coordinates (a towards the pitcher body, l lateral) [mm]."""
        a, l, t, r = np.broadcast_arrays(*(np.asarray(x, float) for x in (a_mm, l_mm, tilt_deg, roll_deg)))
        X = np.stack([_clip(a, self.cax[0]), _clip(l, self.cax[1]), _clip(t, self.cax[2]), _clip(r, self.cax[3])], -1)
        return self.C(X).reshape(X.shape[:-1])

    @staticmethod
    def tip_frame(x, y, yaw):
        """World tip position [m] -> (a, l) [mm] (required_lift's along / perp with d = -spout direction)."""
        dx, dy = -np.cos(yaw), -np.sin(yaw)
        return (x * dx + y * dy) * 1e3, (-x * dy + y * dx) * 1e3

    def z_min(self, foot_x, foot_y, yaw, Q, V, roll=0.0, alpha=10.0, tilt_offset=0.0, iters=4, z_offset=None):
        """Lowest tip height [m] with the stream foot at (foot_x, foot_y) [m] for spout yaw [rad], flow Q, fill V
        (tilt from the flow map + tilt_offset [deg], e.g. minus a pulsation amplitude).  The tip sits behind the foot by
        the jet offset; by default the offset is taken at the answer itself (fixed point: the lifted tip has a longer
        jet, so it sits further towards the pitcher body); z_offset [m] instead fixes the height at which the offset is
        evaluated (z_offset = the wanted height reproduces the ideal hand's conservative lift pre-plan).
        Returns (z_min, tip_x, tip_y, tilt)."""
        f = self.flow(Q, V, roll)
        th = f["tilt"] + tilt_offset
        ef, es, ez = self.exit_dir(th, roll, alpha)
        u0 = f["u0"]
        ex, ey = np.cos(yaw), np.sin(yaw)
        z = np.full(np.broadcast(foot_x, Q).shape, 0.015)
        if z_offset is not None:
            z, iters = np.broadcast_to(np.asarray(z_offset, float), z.shape), 1
        for _ in range(iters):
            z0 = np.maximum(z + f["oz"], 0.0)
            vz = u0 * ez
            t = (vz + np.sqrt(vz ** 2 + 2 * G * z0)) / G
            of, os_ = f["ox"] + u0 * ef * t, f["oy"] + u0 * es * t
            tx = foot_x - (of * ex - os_ * ey)
            ty = foot_y - (of * ey + os_ * ex)
            a, l = self.tip_frame(tx, ty, yaw)
            z = self.zmin_tip(a, l, th, roll)
        return z, tx, ty, th


# =========================================================================== hand limits
HANDS = dict(
    ideal=dict(PC.HAND, lift_acc=PC.LIFT_ACC, lift_win=PC.LIFT_MAXWIN,
               src="v05_pitcher_check.HAND (the ideal hand; upper bound): " + "; ".join(PC.HAND_SRC.values())),
    video=dict(tilt_rate=math.radians(60.0), pivot_L=0.15, acc=2.0, speed=0.11, yaw_rate=math.radians(90.0), f_max=4.0,
               lift_acc=2.0, lift_win=PC.LIFT_MAXWIN,
               src="clips (envelope.md section 7, video_ranges.md): smooth tilt ~60 deg/s, wiggle peak accel 1.0-1.9 m/s^2, "
                   "path <= 0.071 m/s, lifts <= 0.11 m/s (3-5 cm in 0.3-0.5 s), wiggle 2.9-3.7 Hz; yaw rate not measured "
                   "(90 deg/s assumed)"))
# 5 Hz pulsation from the review (V05_PITCHER_CHECK.md 2.6, data/review_dyn_tilt.json: 300 mL, 12 mL/s, tau 0.08 s)
WRIST_TABLE = dict(amp_deg=[0.0, 2.0, 3.5], m_z03=[0.0, 0.21, 0.62], m_z01=[0.0, 0.41, 0.68], bob_mm=[0.0, 12.7, 21.4])
WRIST_PIVOT = 0.19          # [m] wrist behind the spout tip (review)


def _rate_limit(x, dt, rmax):
    """Forward and backward rate-limited copies of x, averaged (a hand that knows the plan: turns start early and end
    late symmetrically); |rate| <= rmax holds for the average."""
    def one(sig):
        y = np.empty_like(sig)
        y[0] = sig[0]
        step = rmax * dt
        for i in range(1, len(sig)):
            y[i] = y[i - 1] + min(step, max(-step, sig[i] - y[i - 1]))
        return y
    x = np.asarray(x, float)
    return 0.5 * (one(x) + one(x[::-1])[::-1])


def smooth_yaw(yaw, dt, hand="ideal"):
    """Wanted yaw schedule (unwrapped) -> the yaw the hand turns, |yaw'| <= yaw_rate.  The ideal hand keeps
    v05_pitcher_check.smooth_yaw (what the ideal-hand runs used); other hands use the symmetric rate limiter (robust
    when turns overlap)."""
    H = HANDS[hand] if isinstance(hand, str) else hand
    yaw = np.unwrap(np.asarray(yaw, float))
    if hand == "ideal":
        return PC.smooth_yaw(yaw, dt, H["yaw_rate"])
    return _rate_limit(yaw, dt, H["yaw_rate"])


def plan_lift(R, dt, hand="ideal"):
    """Lift the hand follows for a clearance requirement R(t) [m]: running max over lift_win (as the ideal hand), then the
    smallest upper envelope with |L'| <= speed and |L''| <= lift_acc (morphological closing), + 0.2 mm where needed."""
    H = HANDS[hand] if isinstance(hand, str) else hand
    R = np.asarray(R, float)
    Rf = PC.max_filter(np.where(R > 0, R + 2e-4, 0.0), max(1, int(round(H["lift_win"] / dt))))
    return PC.lift_closing(Rf, dt, H["lift_acc"], H["speed"])


def track_tilt(target_deg, dt, hand="ideal", preview=True):
    """Rate- and acceleration-limited tilt [deg] that follows target_deg.  preview=True: the hand knows the plan
    (forward-backward pass: the result is the closest feasible curve that does not lag the target systematically)."""
    H = HANDS[hand] if isinstance(hand, str) else hand
    rmax = math.degrees(H["tilt_rate"])
    amax = math.degrees(H["acc"] / H["pivot_L"])
    x = np.asarray(target_deg, float)

    def one(sig):
        y = np.empty_like(sig)
        y[0], v = sig[0], 0.0
        for i in range(1, len(sig)):
            e = sig[i] - y[i - 1]
            v_des = math.copysign(min(math.sqrt(2 * amax * abs(e)), rmax, abs(e) / dt), e)
            v += max(-amax * dt, min(amax * dt, v_des - v))
            y[i] = y[i - 1] + v * dt
        return y
    if not preview:
        return one(x)
    return 0.5 * (one(x) + one(x[::-1])[::-1])


def hand_check(t, tilt_deg=None, tip=None, yaw=None, hand="ideal", smooth_ms=20):
    """Peak tilt rate / angular acceleration (as hand acceleration on pivot_L), tip speed / acceleration, yaw rate of a
    schedule, after a smooth_ms moving average (the hand's bandwidth, ~6 Hz); dict with the values, the times of the
    peaks and 'viol' (which exceed the hand)."""
    H = HANDS[hand] if isinstance(hand, str) else hand
    t = np.asarray(t, float)
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(smooth_ms * 1e-3 / dt)))
    out, when = {}, {}
    if tilt_deg is not None:
        th = np.radians(np.asarray(tilt_deg, float))
        r = np.gradient(PC._movavg(th, k), dt)
        a = np.gradient(PC._movavg(r, k), dt)
        out["tilt_rate_deg_s"] = float(np.degrees(np.max(np.abs(r))))
        out["tilt_acc_hand_m_s2"] = float(H["pivot_L"] * np.max(np.abs(a)))
        when["tilt_rate_deg_s"], when["tilt_acc_hand_m_s2"] = float(t[np.argmax(np.abs(r))]), float(t[np.argmax(np.abs(a))])
    if tip is not None:
        P = np.asarray(tip, float)
        v = np.gradient(np.stack([PC._movavg(P[:, j], k) for j in range(P.shape[1])], 1), dt, axis=0)
        a = np.gradient(np.stack([PC._movavg(v[:, j], k) for j in range(v.shape[1])], 1), dt, axis=0)
        vn, an = np.linalg.norm(v, axis=1), np.linalg.norm(a, axis=1)
        out["tip_speed_m_s"], out["tip_acc_m_s2"] = float(np.max(vn)), float(np.max(an))
        when["tip_speed_m_s"], when["tip_acc_m_s2"] = float(t[np.argmax(vn)]), float(t[np.argmax(an)])
    if yaw is not None:
        yr = np.abs(np.gradient(PC._movavg(np.unwrap(np.asarray(yaw, float)), k), dt))
        out["yaw_rate_deg_s"] = float(np.degrees(np.max(yr)))
        when["yaw_rate_deg_s"] = float(t[np.argmax(yr)])
    lim = dict(tilt_rate_deg_s=math.degrees(H["tilt_rate"]), tilt_acc_hand_m_s2=H["acc"], tip_speed_m_s=H["speed"],
               tip_acc_m_s2=H["acc"], yaw_rate_deg_s=math.degrees(H["yaw_rate"]))
    out["limits"] = {k_: lim[k_] for k_ in out if k_ in lim}
    out["t_peak"] = when
    out["viol"] = [k_ for k_ in lim if k_ in out and out[k_] > lim[k_] * 1.001]
    return out


def pulsation(m_want, f=5.0, mode=None, hand="ideal", zeta=0.3, tau=0.08):
    """What a hand can add as a periodic flow pulsation of relative depth m_want at f Hz.
    mode None  : not allowed (default) -> depth 0.
    mode 'tip' : tilt about the spout tip (the ideal hand; the hand translates L theta''): amplitude bounded by
                 0.95 min(acc / (pivot_L w^2), tilt_rate / w); depth ~ 21 /rad x amplitude x lag attenuation (the ideal
                 hand reached 0.31-0.47 at +-3.3-3.7 deg, tau 0.08 s); no tip bob, but the tilt troughs (tilt - amp)
                 set the clearance.
    mode 'wrist': tilt about a wrist WRIST_PIVOT behind the tip (review numbers at 5 Hz, tau 0.08 s); the tip bobs, the
                 flow breaks every cycle (spike train, not 1 + m cos)."""
    H = HANDS[hand] if isinstance(hand, str) else hand
    w = 2 * math.pi * f
    if not mode or m_want <= 0:
        return dict(mode=mode, m=0.0, amp_deg=0.0, tip_bob_pp_mm=0.0, ok=(m_want <= 0), breaks=False)
    att = 1.0 / math.sqrt(1.0 + (w * tau) ** 2) / (1.0 / math.sqrt(1.0 + (w * 0.08) ** 2))
    if mode == "tip":
        amax = 0.95 * min(H["acc"] / (H["pivot_L"] * w * w), H["tilt_rate"] / w)
        gain = 0.41 / math.radians(3.5) * att      # calibrated on the ideal-hand result (0.41 at ~3.5 deg, tau 0.08 s)
        amp = min(m_want / gain, amax)
        return dict(mode=mode, m=float(gain * amp), amp_deg=math.degrees(amp), amp_max_deg=math.degrees(amax),
                    hand_acc_m_s2=H["pivot_L"] * amp * w * w, tip_bob_pp_mm=0.0, ok=bool(gain * amp >= 0.98 * m_want),
                    breaks=False)
    if mode == "wrist":
        mt = WRIST_TABLE["m_z03"] if zeta >= 0.2 else WRIST_TABLE["m_z01"]
        mt = np.asarray(mt) * att
        amax_rate = math.degrees(H["tilt_rate"] / w)
        a = float(np.interp(m_want, mt, WRIST_TABLE["amp_deg"], right=np.nan))
        if not np.isfinite(a):
            a = WRIST_TABLE["amp_deg"][-1]
        a = min(a, amax_rate)
        m = float(np.interp(a, WRIST_TABLE["amp_deg"], mt))
        return dict(mode=mode, m=m, amp_deg=a, tip_bob_pp_mm=float(np.interp(a, WRIST_TABLE["amp_deg"], WRIST_TABLE["bob_mm"])),
                    ok=bool(m >= 0.98 * m_want), breaks=a > 0.5,
                    note="review numbers (300 mL, 12 mL/s); the stroke height must exceed z_min + bob/2")
    raise ValueError(mode)


def bridge_leak(Q0_mLs, t_hit=0.04, law=None):
    """Milk that still arrives after the hand stops a stroke of flow Q0 (review closed form Q0 (tau + t_flight)) [mL]."""
    law = law or OutflowLaw()
    return Q0_mLs * (law.tau + t_hit)


# =========================================================================== delivering a schedule
def deliver(S, sched, fill_mL, alpha=10.0, roll=0.0, hand="ideal", pulse=None, pulse_mask=None, pulse_amp_deg=0.0,
            v_hand=True, planner="fixed"):
    """Quasi-static delivery of a 1 ms schedule.  sched: dict of arrays on the surface clock: t [s], x, y [D] (stream
    foot), z [m] (wanted tip height), Q [mL/s] (wanted at the surface; Q_env [mL/s] used for the tilt if present),
    yaw [rad]; optional vhx, vhy [m/s] (hand velocity in the cup frame).  The ideal hand delivers the wanted flow
    (it inverts lag and flight time); the fill falls with the poured volume.  pulse_mask: samples whose tilt oscillates
    by +-pulse_amp_deg (the troughs set the clearance).  planner 'fixed': clearance requirement with the jet offset of
    the lifted tip (what is actually needed); 'preplan': offset at the wanted height (the ideal hand's lift pre-plan,
    v05_pitcher_check.lift_preplan; 0-4 mm more lift).  Returns dict of arrays."""
    t = np.asarray(sched["t"], float)
    dt = float(np.median(np.diff(t)))
    Q = np.nan_to_num(np.asarray(sched["Q"], float))
    Qf = np.nan_to_num(np.asarray(sched.get("Q_env", sched["Q"]), float))
    V = fill_mL - np.cumsum(Q) * dt
    yaw = smooth_yaw(sched["yaw"], dt, hand)
    fx, fy = np.asarray(sched["x"], float) * D, np.asarray(sched["y"], float) * D
    zw = np.asarray(sched["z"], float)
    Qt = np.maximum(Qf, 1.0)                       # the tilt is held near the 1 mL/s level between strokes
    off = np.zeros_like(t)
    if pulse_mask is not None:
        off = np.where(pulse_mask, -abs(pulse_amp_deg), 0.0)
    zmin, tx, ty, th = S.z_min(fx, fy, yaw, Qt, V, roll, alpha, tilt_offset=off,
                               z_offset=(zw if planner == "preplan" else None))
    R = np.maximum(zmin - zw, 0.0)                 # everywhere, as the ideal hand (bridges at the ~1 mL/s tilt)
    L = plan_lift(R, dt, hand)
    z = zw + L
    vh = None
    if v_hand and "vhx" in sched:
        vz = np.gradient(z, dt)
        vh = (np.nan_to_num(np.asarray(sched["vhx"], float)), np.nan_to_num(np.asarray(sched["vhy"], float)), vz)
    J = S.jet(Qt, V, z, roll, alpha, v_hand=vh, yaw=yaw)
    tilt0 = J.pop("tilt")
    out = dict(t=t, V=V, Q=Q, yaw=yaw, tilt=tilt0, z_w=zw, z_min=zmin, R=R, lift=L, z=z, tip_x=tx, tip_y=ty, **J)
    # hand: the demanded tilt (steady tilt for the wanted flow), how far a rate / acceleration limited wrist lags it
    # (the implied flow error follows from the flow map), the tip path (foot path - jet offset, z incl. the lift), yaw
    tt = track_tilt(tilt0, dt, hand)
    out["tilt_tracked"] = tt
    out["hand"] = hand_check(t, tilt_deg=tilt0, tip=np.stack([tx, ty, z], 1), yaw=yaw, hand=hand)
    out["hand"]["tilt_track_err_max_deg"] = float(np.max(np.abs(tt - tilt0)))
    return out


def stroke_summary(D, strokes, lift_tol=1e-3):
    """Per-stroke flow-weighted summary of a deliver() result.  strokes: list of dict(t0, t1[, n, cls, z_design]) on the
    surface clock.  'reach' = False where the hand must lift more than lift_tol above the wanted height."""
    t, Q = D["t"], np.asarray(D["Q"], float)
    rows = []
    for s in strokes:
        m = (t >= s["t0"]) & (t < s["t1"])
        w = np.where(m, Q, 0.0)
        if w.sum() <= 0:
            continue
        r = dict(n=s.get("n"), cls=s.get("cls"), t0=s["t0"], t1=s["t1"], V_mL=float(np.sum(Q[m]) * (t[1] - t[0])))
        for k in ("V", "tilt", "u0", "u_h", "U_perp", "t_hit", "z_w", "z_min", "R", "lift", "z", "u_h_cup"):
            if k in D:
                r[k] = _fwm(np.asarray(D[k], float), w)
        r["lift_max"] = float(np.max(np.asarray(D["lift"])[m]))
        r["reach"] = bool(r["lift"] <= lift_tol)
        rows.append(r)
    return rows


# =========================================================================== validation against the ideal-hand runs
BEST_RUNS = os.path.join(PC.BEST, "runs")


def _fwm(x, w):
    m = np.isfinite(x) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else float("nan")


VAL_VARIANTS = dict(fixed=dict(planner="fixed"), preplan=dict(planner="preplan"),
                    preplan_tip=dict(planner="preplan", pulse=True))


def validate_run(name, S, alpha=10.0, verbose=True):
    """Surrogate vs one ideal-hand run.  Surrogate inputs = what a designer specifies: the FS wanted table (foot path,
    wanted height z_w, flow at the surface, its pulsation-free envelope, yaw, hand velocity) and the fill.  Variants:
    fixed (clearance with the lifted tip's jet offset), preplan (the ideal hand's lift pre-plan), preplan_tip (+ the
    5 Hz tilt troughs of the pulsed strokes, amplitude = the run's c_max)."""
    W, meta, A, info = PC.load_run(name, PC.BEST)
    dt = info["dt"]
    fill = info["fill_mL"]
    n = len(W["t"])
    t = np.asarray(W["t"], float)
    Qw = np.nan_to_num(np.asarray(W["Q"], float)) / ML
    Qe = np.nan_to_num(np.asarray(W["Q_env"], float)) / ML
    pulsed_strokes = [s for s in meta["strokes"] if s.get("pulse")]
    pmask = np.zeros(n, bool)
    for s in pulsed_strokes:
        pmask |= (t >= s["t0"]) & (t < s["t1"])
    amp = np.nanmax([v["c_max"] for v in info.get("pulse_final", {}).values()]) if info.get("pulse_final") else 0.0
    sched = dict(t=t, x=W["x"], y=W["y"], z=W["z"], Q=Qw, Q_env=Qe, yaw=W["yaw"], vhx=W["vhx"], vhy=W["vhy"])
    res = {}
    for tag, kw in VAL_VARIANTS.items():
        pm = pmask if kw.get("pulse") else None
        res[tag] = deliver(S, sched, fill, alpha=alpha, hand="ideal", pulse_mask=pm,
                           pulse_amp_deg=(math.degrees(amp) if (pm is not None and amp) else 0.0), planner=kw["planner"])
    # simulation arrays (step i <-> table row i; arrival window [t_i, t_i + dt))
    k = min(n, len(A["t"]))
    sl = slice(0, k)
    hit = A["h_on"][sl] > 0
    Qd = np.nan_to_num(A["Q_hit"][sl]) / ML
    jon = A["j_on"][sl] > 0
    Qr = np.nan_to_num(A["Q"][sl]) / ML
    Qf = np.nan_to_num(A["Q_feed"][sl]) / ML
    ta = t[sl] + np.where(jon, A["j_thit"][sl], 0.0)
    rows = []
    for s in meta["strokes"]:
        if "yaw" not in s:
            continue
        a_, b_ = s["t0"], s["t1"]
        m = (t[sl] >= a_) & (t[sl] < b_)
        mr = jon & (ta >= a_) & (ta < b_)
        wW, wD, wR = np.where(m, Qw[sl], 0.0), np.where(m & hit, Qd, 0.0), np.where(mr, Qr, 0.0)
        wF = np.where(mr, Qf, 0.0)
        o = dict(n=s["n"], cls=s["cls"], pulsed=bool(s.get("pulse")), t0=a_, t1=b_,
                 Q_w=_fwm(Qw[sl], np.where(m, 1.0, 0.0)), V_w=float(np.sum(Qw[sl][m]) * dt), V_d=float(np.sum(Qd[m]) * dt))
        o["Q_ratio"] = o["V_d"] / o["V_w"] if o["V_w"] > 0 else float("nan")
        # simulation (ideal hand): release quantities weighted by the released flow (exit speed of pitcher.py = the
        # instantaneous head), also by the weir feed (the head the milk actually crossed the lip with)
        o["sim"] = dict(tilt=_fwm(np.degrees(A["tilt"][sl]), wR), u0=_fwm(A["j_u0"][sl], wR),
                        exit_h=_fwm(A["j_exh"][sl], wR), u0_feedw=_fwm(A["j_u0"][sl], wF), exit_h_feedw=_fwm(A["j_exh"][sl], wF),
                        U=_fwm(A["h_U"][sl], wD), uh_cup=_fwm(np.hypot(A["h_uhx"][sl], A["h_uhy"][sl]), wD),
                        z=_fwm(A["tip_z"][sl], wW), lift=_fwm(A["lift"][sl], wW), R=_fwm(A["x_R"][sl], wW),
                        dz_origin=_fwm(A["j_oz"][sl] - A["tip_z"][sl], wR), t_hit=_fwm(A["j_thit"][sl], wR),
                        V=_fwm(A["V"][sl] / ML, wR))
        for tag, D_ in res.items():
            d = {kk: np.asarray(vv)[sl] if isinstance(vv, np.ndarray) and len(vv) == n else vv for kk, vv in D_.items()}
            o[tag] = dict(tilt=_fwm(d["tilt"], wW), u0=_fwm(d["u0"], wW), exit_h=_fwm(d["u_h"], wW), U=_fwm(d["U_perp"], wW),
                          uh_cup=_fwm(d["u_h_cup"], wW) if "u_h_cup" in d else float("nan"),
                          z=_fwm(d["z"], wW), lift=_fwm(d["lift"], wW), R=_fwm(d["R"], wW),
                          dz_origin=_fwm(d["dz_origin"], wW), t_hit=_fwm(d["t_hit"], wW), V=_fwm(d["V"], wW))
        # surrogate jet at the simulation's own tip height and fill (isolates the jet / flow map from the lift model)
        Jz = S.jet(np.maximum(Qe[sl], 1.0), A["V"][sl] / ML, A["tip_z"][sl], 0.0, alpha)
        o["at_sim_z"] = dict(U=_fwm(Jz["U_perp"], wW), exit_h=_fwm(Jz["u_h"], wW), u0=_fwm(Jz["u0"], wW),
                             tilt=_fwm(Jz["tilt"], wW))
        rows.append(o)
    # per-ms clearance requirement where the hand had to lift (R > 0.5 mm, stream on): the sim's z_w + required_lift at
    # its actual tip / yaw / tilt vs (map) the clearance map at that same pose and (full) the surrogate's own chain
    xR = A["x_R"][sl]
    zreq = np.asarray(W["z"], float)[sl] + xR
    on = (xR > 5e-4) & jon
    a_s, l_s = S.tip_frame(A["tip_x"][sl], A["tip_y"][sl], A["yaw"][sl])
    z_map = S.zmin_tip(a_s, l_s, np.degrees(A["tilt"][sl]), 0.0)
    D0 = res["fixed"]
    zc = dict(n=int(on.sum()), n_pulsed=int((on & pmask[sl]).sum()))
    for lab, msk in (("all", on), ("nonpulsed", on & ~pmask[sl]), ("pulsed", on & pmask[sl])):
        if msk.sum() == 0:
            continue
        e_map = (z_map - zreq)[msk] * 1e3
        e_full = (np.asarray(D0["z_min"])[sl] - zreq)[msk] * 1e3
        zc[lab] = dict(n=int(msk.sum()), map_mean_mm=float(np.mean(e_map)), map_mae_mm=float(np.mean(np.abs(e_map))),
                       map_p95_mm=float(np.percentile(np.abs(e_map), 95)),
                       full_mean_mm=float(np.mean(e_full)), full_mae_mm=float(np.mean(np.abs(e_full))),
                       full_p95_mm=float(np.percentile(np.abs(e_full), 95)))
    out = dict(run=name, pattern=meta["pattern"], fill=fill, strokes=rows, zreq=zc,
               hand_fixed=D0["hand"], pulse_amp_deg=math.degrees(amp) if amp else 0.0)
    if verbose:
        print(f"  {name}: " + " ".join(
            f"#{r['n']}{'p' if r['pulsed'] else ''} U {_pct(r['preplan']['U'], r['sim']['U'])} uh {_pct(r['fixed']['exit_h'], r['sim']['exit_h'])}"
            f" z {(r['preplan']['z'] - r['sim']['z']) * 1e3:+.1f}mm R {(r['fixed']['R'] - r['sim']['R']) * 1e3:+.1f}mm" for r in rows)
              + f" | zreq map MAE {zc.get('all', {}).get('map_mae_mm', float('nan')):.2f} full "
                f"{zc.get('nonpulsed', {}).get('full_mae_mm', float('nan')):.2f} mm", flush=True)
    return out


def _pct(a, b):
    return f"{(a / b - 1) * 100:+.0f}%" if b and np.isfinite(a) and np.isfinite(b) else "n/a"


def validate(patterns=PC.PATTERNS, fills=(250, 350), alpha=10.0, out=OUT):
    S = Surrogate("nx", "mean", out)
    runs = []
    for p in patterns:
        for f in fills:
            name = PC.best_name(p, "ideal", f)
            if os.path.exists(os.path.join(BEST_RUNS, name + ".npz")):
                runs.append(validate_run(name, S, alpha))
    summ = summarize_validation(runs)
    with open(os.path.join(out, "validation.json"), "w") as fh:
        json.dump(dict(runs=runs, summary=summ), fh, indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    return runs, summ


def summarize_validation(runs):
    """Error statistics per quantity and surrogate variant, split by non-pulsed / pulsed strokes.  Relative errors for
    speeds and the delivered-volume ratio; mm for heights; deg for the tilt.  'within' counts |rel| <= 10 % or <= 2 mm /
    <= 1 deg."""
    def rel(a, b):
        return a / b - 1 if (b and np.isfinite(a) and np.isfinite(b)) else np.nan
    acc = {}

    def put(g, k, v):
        for gg in (g, "all"):
            acc.setdefault(gg, {}).setdefault(k, []).append(v)
    for r in runs:
        for s in r["strokes"]:
            g = "pulsed" if s["pulsed"] else "nonpulsed"
            Sm, Z = s["sim"], s["at_sim_z"]
            put(g, "Q_ratio", s["Q_ratio"] - 1)
            put(g, "jet_at_sim_z.U", rel(Z["U"], Sm["U"]))
            put(g, "jet_at_sim_z.exit_h", rel(Z["exit_h"], Sm["exit_h"]))
            put(g, "jet_at_sim_z.exit_h_vs_feedw", rel(Z["exit_h"], Sm["exit_h_feedw"]))
            put(g, "jet_at_sim_z.u0", rel(Z["u0"], Sm["u0"]))
            put(g, "jet_at_sim_z.u0_vs_feedw", rel(Z["u0"], Sm["u0_feedw"]))
            for v in VAL_VARIANTS:
                N = s[v]
                put(g, v + ".U", rel(N["U"], Sm["U"]))
                put(g, v + ".exit_h", rel(N["exit_h"], Sm["exit_h"]))
                put(g, v + ".uh_cup", rel(N["uh_cup"], Sm["uh_cup"]))
                put(g, v + ".u0", rel(N["u0"], Sm["u0"]))
                put(g, v + ".tilt_deg", N["tilt"] - Sm["tilt"])
                put(g, v + ".z_mm", (N["z"] - Sm["z"]) * 1e3)
                put(g, v + ".R_mm", (N["R"] - Sm["R"]) * 1e3)
    out = {}
    for g, G_ in acc.items():
        out[g] = {}
        for k, v in G_.items():
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            tol = 2.0 if k.endswith("_mm") else (1.0 if k.endswith("_deg") else 0.10)
            out[g][k] = dict(n=int(len(v)), mean=float(np.mean(v)), mae=float(np.mean(np.abs(v))),
                             max_abs=float(np.max(np.abs(v))), within=int(np.sum(np.abs(v) <= tol)), tol=tol)
    zc = {}
    for r in runs:
        for lab in ("nonpulsed", "pulsed", "all"):
            if lab in r["zreq"]:
                zc.setdefault(lab, []).append(r["zreq"][lab])
    out["zreq_per_ms"] = {lab: dict(n=int(sum(x["n"] for x in v)),
                                    map_mae_mm=float(sum(x["map_mae_mm"] * x["n"] for x in v) / sum(x["n"] for x in v)),
                                    map_mean_mm=float(sum(x["map_mean_mm"] * x["n"] for x in v) / sum(x["n"] for x in v)),
                                    map_p95_max_mm=float(max(x["map_p95_mm"] for x in v)),
                                    full_mae_mm=float(sum(x["full_mae_mm"] * x["n"] for x in v) / sum(x["n"] for x in v)),
                                    full_mean_mm=float(sum(x["full_mean_mm"] * x["n"] for x in v) / sum(x["n"] for x in v)),
                                    full_p95_max_mm=float(max(x["full_p95_mm"] for x in v)))
                          for lab, v in zc.items()}
    return out


# =========================================================================== checks and key numbers
def closed_form(a_mm, tilt_deg, b_mm, cup=None):
    """h_min = m + D_max cot(theta) - b / sin(theta), D_max = r_in / cos(cup tilt) - margin - a (tip a towards the body)."""
    cup = cup or Cup()
    th = np.radians(tilt_deg)
    Dm = cup.r_in / math.cos(cup.tilt) * 1e3 - cup.margin * 1e3 - np.asarray(a_mm, float)
    return (cup.margin * 1e3 + Dm / np.tan(th) - b_mm / np.sin(th)) * 1e-3


def closed_form_check(out=OUT):
    """Clearance map at l = 0, roll 0 vs the closed form, for every spec / wall / cup built."""
    res = {}
    for sp in SPECS:
        for wall in WALLS:
            for cup in CUPS:
                if not os.path.exists(os.path.join(out, _clear_file(sp, wall, cup))):
                    continue
                S = Surrogate(sp, wall, out, cup)
                rows = []
                for a in (-20.0, -10.0, 0.0, 10.0, 20.0):
                    for th in (35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0):
                        zm = float(S.zmin_tip(a, 0.0, th, 0.0))
                        cf = max(float(closed_form(a, th, S.b_mm)), Z0)
                        rows.append(dict(a_mm=a, tilt=th, map_mm=zm * 1e3, cf_mm=cf * 1e3, diff_mm=(zm - cf) * 1e3))
                d = np.array([r["diff_mm"] for r in rows])
                act = np.array([r["cf_mm"] > Z0 * 1e3 + 0.5 for r in rows])
                cen = np.array([r["a_mm"] == 0.0 for r in rows]) & act
                res[f"{sp}_{wall}_{cup}"] = dict(
                    b_mm=S.b_mm, max_abs_mm=float(np.max(np.abs(d[act]))) if act.any() else None,
                    mean_mm=float(np.mean(d[act])) if act.any() else None,
                    centre_max_abs_mm=float(np.max(np.abs(d[cen]))) if cen.any() else None, rows=rows)
    with open(os.path.join(out, "closed_form_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        print(f"  {k}: b {v['b_mm']:.2f} mm, map - closed form where the closed form > 2.5 mm: mean {v['mean_mm']:+.2f}, "
              f"max |.| {v['max_abs_mm']:.2f} mm (at the centre a = 0: {v['centre_max_abs_mm']})")
    return res


def key_numbers(out=OUT, fills=(100, 150, 200, 250, 300, 350, 400), Qs=(2.0, 12.0)):
    """z_min of the STREAM FOOT at the cup centre and 0.2 / 0.3 D towards the near rim (the pitcher-body side) and the
    far rim, roll 0 and best roll; z_min of the TIP at the centre; u_h range per fill and alpha_lip; tilt range."""
    res = dict(specs={}, note="foot positions along the spout axis: + = towards the pitcher body / near rim; z in mm "
                              "[roll 0, best roll in -20..20]; u_h = exit horizontal speed relative to the pitcher [m/s] "
                              "over Q 1-32 mL/s, roll 0")
    pos = (-0.3, -0.2, 0.0, 0.2, 0.3)
    for sp in SPECS:
        if not os.path.exists(os.path.join(out, f"flow_{sp}.npz")):
            continue
        R = res["specs"][sp] = dict(clear={})
        for wall in WALLS:
            for cup in CUPS:
                if not os.path.exists(os.path.join(out, _clear_file(sp, wall, cup))):
                    continue
                S = Surrogate(sp, wall, out, cup)
                R["b_mm"] = S.b_mm
                Wt = R["clear"][f"{wall}_{cup}"] = {}
                for F in fills:
                    for Q in Qs:
                        row = dict(tilt=float(S.tilt(Q, F, 0.0)))
                        yaw = -math.pi / 2              # spout towards -y: the pitcher body (near rim) is on +y
                        for p in pos:
                            fy = p * D
                            z0 = float(S.z_min(0.0, fy, yaw, Q, F, 0.0)[0])
                            zb = min(float(S.z_min(0.0, fy, yaw, Q, F, r)[0]) for r in ROLLS)
                            row[f"foot{p:+.1f}D"] = [z0 * 1e3, zb * 1e3]
                        row["tip_centre"] = float(S.zmin_tip(0.0, 0.0, row["tilt"], 0.0)) * 1e3
                        row["tip_centre_bestroll"] = float(min(S.zmin_tip(0.0, 0.0, float(S.tilt(Q, F, r)), r)
                                                               for r in ROLLS)) * 1e3
                        Wt[f"V{F}_Q{Q:g}"] = row
        S = Surrogate(sp, "mean", out)
        uh = {}
        Qg = 2.0 ** LQ_GRID
        for F in fills:
            f = S.flow(Qg, F, 0.0)
            ent = dict(tilt=[float(f["tilt"].min()), float(f["tilt"].max())], u0=[float(f["u0"].min()), float(f["u0"].max())],
                       dz_origin_mm=[float(f["oz"].min() * 1e3), float(f["oz"].max() * 1e3)],
                       onset_tilt=float(S.onset(np.array([[F, 0.0]]))[0]))
            for al in ALPHAS:
                J = S.jet(Qg, F, 0.015, 0.0, al)
                ent[f"uh_a{al:g}"] = [float(J["u_h"].min()), float(J["u_h"].max())]
                ent[f"uh_a{al:g}_Q"] = {f"{q:g}": float(v) for q, v in zip(Qg, J["u_h"]) if q in (1, 2, 4, 8, 16, 32)}
            for z in (0.003, 0.01, 0.02, 0.04):
                J = S.jet(Qg, F, z, 0.0, 10.0)
                ent[f"U_z{z * 1e3:g}mm"] = [float(J["U_perp"].min()), float(J["U_perp"].max())]
                ent[f"t_hit_z{z * 1e3:g}mm"] = [float(J["t_hit"].min()), float(J["t_hit"].max())]
                ent[f"off_z{z * 1e3:g}mm"] = [float(J["off_fwd"].min()), float(J["off_fwd"].max())]
            uh[F] = ent
        R["flow"] = uh
    with open(os.path.join(out, "key_numbers.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def jet_tables(out=OUT):
    """Tabulated jet quantities per spec on (V, Q, roll, alpha, z): u_h, t_hit, U_perp, foot offset (saved in npz)."""
    for sp in SPECS:
        p = os.path.join(out, f"flow_{sp}.npz")
        if not os.path.exists(p) or not os.path.exists(os.path.join(out, _clear_file(sp, "mean", "table"))):
            continue
        S = Surrogate(sp, "mean", out)
        Vg, Qg, Rg, Ag, Zg = np.meshgrid(V_GRID, 2.0 ** LQ_GRID, ROLLS, np.array(ALPHAS), Z_GRID * 1e-3, indexing="ij")
        J = S.jet(Qg, Vg, Zg, Rg, Ag)
        np.savez_compressed(os.path.join(out, f"jet_{sp}.npz"), V=V_GRID, Q=2.0 ** LQ_GRID, roll=ROLLS, alpha=np.array(ALPHAS),
                            z_mm=Z_GRID, u_h=J["u_h"], u_h_side=J["u_h_side"], v0z=J["v0z"], t_hit=J["t_hit"],
                            U_perp=J["U_perp"], off_fwd=J["off_fwd"], off_side=J["off_side"], dz_origin=J["dz_origin"],
                            tilt=J["tilt"], u0=J["u0"],
                            note="axes (V mL, Q mL/s, roll deg, alpha_lip deg, tip z mm); m, s, m/s")


# =========================================================================== self test
def selftest(out=OUT):
    """(1) required_lift is invariant under a rotation about the cup axis; (2) the b of each spec; (3) flow-map
    interpolation vs direct tilt_for_flow / weir at off-grid points; (4) clearance-map interpolation vs direct
    required_lift at off-grid points."""
    cup = Cup()
    rng = np.random.default_rng(3)
    P = make_pitcher(Spec())
    err = 0.0
    for _ in range(40):
        a, l, th, ro = rng.uniform(-30, 30), rng.uniform(-20, 20), rng.uniform(35, 70), rng.uniform(-15, 15)
        zs = []
        for yaw in (-math.pi / 2, 0.3, 2.0):
            d = -np.array([math.cos(yaw), math.sin(yaw)])
            n = np.array([-d[1], d[0]])
            xy = (a * d + l * n) * 1e-3
            zs.append(P.required_lift(Pose((xy[0], xy[1], Z0), yaw, math.radians(th), math.radians(ro)), cup))
        err = max(err, max(zs) - min(zs))
    print(f"yaw invariance of required_lift: max spread {err * 1e3:.3f} mm")
    for sp in SPECS:
        print(f"  spec {sp}: b = {spec_b(spec_from_name(sp)):.2f} mm, tip r = {geometry(spec_from_name(sp)).tip[0] * 1e3:.2f} mm, "
              f"capacity {geometry(spec_from_name(sp)).capacity / ML:.0f} mL")
    res = dict(yaw_spread_mm=err * 1e3)
    for sp in SPECS:
        if not os.path.exists(os.path.join(out, f"flow_{sp}.npz")):
            continue
        for wall, cup in [(w, c) for w in WALLS for c in CUPS]:
            if not os.path.exists(os.path.join(out, _clear_file(sp, wall, cup))):
                continue
            S = Surrogate(sp, wall, out, cup)
            cupo = make_cup(cup)
            P = make_pitcher(S.spec, wall=wall)
            e_t, e_u, e_z, e_o = [], [], [], []
            for _ in range(25):
                V, Q, ro = rng.uniform(110, 390), 2 ** rng.uniform(0.1, 4.9), rng.uniform(-18, 18)
                pose = Pose((0.0, 0.0, 0.02), 0.0, 0.0, math.radians(ro))
                th = P.tilt_for_flow(pose, Q * ML, V=V * ML, iters=32)
                pose.tilt = th
                c, _ = P.free_surface(pose, V * ML)
                Wr = P.weir(pose, c)
                f = S.flow(Q, V, ro)
                e_t.append(float(f["tilt"]) - math.degrees(th))
                e_u.append(float(f["u0"]) / math.sqrt(2 * G * max(Wr["h_bar"], P.law.h_k)) - 1)
                e_o.append((float(f["oz"]) - (Wr["origin"][2] - 0.02)) * 1e3)
                a, l, tt = rng.uniform(-35, 40), rng.uniform(-25, 25), rng.uniform(30, 80)
                yaw = -math.pi / 2
                d = -np.array([math.cos(yaw), math.sin(yaw)])
                n = np.array([-d[1], d[0]])
                xy = (a * d + l * n) * 1e-3
                zd = Z0 + P.required_lift(Pose((xy[0], xy[1], Z0), yaw, math.radians(tt), math.radians(ro)), cupo)
                e_z.append((float(S.zmin_tip(a, l, tt, ro)) - zd) * 1e3)
            r = dict(tilt_deg=[float(np.mean(np.abs(e_t))), float(np.max(np.abs(e_t)))],
                     u0_rel=[float(np.mean(np.abs(e_u))), float(np.max(np.abs(e_u)))],
                     dz_origin_mm=[float(np.mean(np.abs(e_o))), float(np.max(np.abs(e_o)))],
                     zmin_mm=[float(np.mean(np.abs(e_z))), float(np.max(np.abs(e_z))), float(np.percentile(np.abs(e_z), 90))])
            res[f"{sp}_{wall}_{cup}"] = r
            print(f"  interp {sp}/{wall}/{cup}: tilt |err| mean/max {r['tilt_deg'][0]:.2f}/{r['tilt_deg'][1]:.2f} deg, u0 "
                  f"{r['u0_rel'][0] * 100:.2f}/{r['u0_rel'][1] * 100:.2f} %, dz_origin {r['dz_origin_mm'][0]:.2f}/"
                  f"{r['dz_origin_mm'][1]:.2f} mm, z_min {r['zmin_mm'][0]:.2f}/{r['zmin_mm'][1]:.2f} mm (p90 {r['zmin_mm'][2]:.2f})")
    with open(os.path.join(out, "selftest.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


# =========================================================================== DESIGN: pitcher-constrained inverse design
# Redo of the FS inverse design (v05allphys/final/fits.py, allphys.design / fit_inverse) with the inlet quantities taken
# from this surrogate instead of free envelope values.  Pre-registration: <review>/v05pour/design/prereg.json.
#   shared closure  B_dep, p_dep, kappa_t (grid + refinement), kappa_r (FS's log least squares, after); cp 1.36, beta 3,
#                   nu 1e-3, D 1e-7, kappa_Q 2, kappa_c 1 fixed (as FS)
#   pitcher const.  alpha_lip scanned over DS_ALPHAS (one value for all patterns)
#   per pattern     start fill [mL] (V(t) = fill - poured >= DS_VMIN)
#   per stroke      wanted tip height z_k (delivered z(t) = max(z_k, Zreq(t)), Zreq = 0.25 s running max of the surrogate
#                   z_min at the stream foot), flow scale s_k of Codex's F (before the spout lag; 1 for cuts), roll, yaw
#                   (Codex's jet direction where it is not degenerate, free on a 15 deg grid for FREE_YAW within the
#                   yaw-rate window of the neighbours)
#   no 5 Hz pulsation; spout lag 0.08 s, 0.1 s ramps, bridges >= 0.3 s, ztr transitions (drvf rules)
#   objective       J = J_chi + J_s + J_uh + LAM_OTH J_oth  (see prereg.json; per-stroke additive)
DSN = os.path.join(PC.REVIEW, "v05pour", "design")
DSURR = os.path.join(DSN, "surr")
D_M = 0.08                                   # cup diameter [m] (solver length unit)
ML_D3 = 512.0                                # mL per D^3
DS_FILLS = (100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0)
DS_S = (0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
DS_ROLLS = (-20.0, 0.0, 20.0)
DS_ALPHAS = (0.0, 10.0, 20.0, 30.0)
DS_ZLOW = (0.003, 0.025)
DS_ZCUT = (0.012, 0.07)
DS_NZ, DS_NQ = 24, 16
DS_VMIN = 20.0
DS_DT = 0.01
DS_REQWIN = 0.25
DS_CHI_CLIP = (0.02, 0.98)
LB_STROKES = (("heart", 1), ("heart", 2), ("tulip", 6), ("leaf", 4), ("swan", 8), ("swan", 9))
FREE_YAW = (("tulip", 6), ("leaf", 4), ("swan", 8), ("swan", 9))
LAM_OTH = 0.25
KT_GRID = tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2))
DS_SPECS = dict(S1=dict(pitcher="nx", wall="resolved", cup="held"),
                S2=dict(pitcher="b20", wall="resolved", cup="held"),
                S3=dict(pitcher="b20", wall="resolved", cup="video"))
CUP_MAPS = (5.0, 10.0, 20.0, 30.0)
FS_FIXED = dict(cp=1.36, beta=3.0, nu=1e-3, D=1e-7, kappa_Q=2.0, kappa_c=1.0)
DS_YAW_RATE = 300.0                          # [deg/s] ideal hand; windows use 0.9 of it
FLOWING = ("low", "cut")


def _prereg():
    with open(os.path.join(DSN, "prereg.json")) as f:
        return json.load(f)


def _refs():
    with open(os.path.join(DSN, "refs.json")) as f:
        return json.load(f)


def cup_tilt_fn(pattern):
    """S3: cup tilt [deg] vs Codex time from the video series (clip time = Codex time, nearest value held outside)."""
    ser = _prereg()["specs"]["S3"]["cup_tilt_series_deg"][pattern]
    ts, cs = (np.array(v, float) for v in zip(*ser))
    return lambda t: np.interp(np.asarray(t, float), ts, cs)


class SpecSurrogate:
    """A design spec: the flow maps of DSURR (V 20-400 mL) and the clearance map(s): one static cup, or (S3) maps at
    CUP_MAPS cup tilts interpolated linearly in the cup tilt c(t) (clamped to the map range)."""

    def __init__(self, name):
        d = DS_SPECS[name]
        self.name, self.pitcher, self.wall, self.cup = name, d["pitcher"], d["wall"], d["cup"]
        if d["cup"] == "video":
            self.maps = [Surrogate(d["pitcher"], d["wall"], DSURR, f"t{c:g}") for c in CUP_MAPS]
            self.S0 = self.maps[CUP_MAPS.index(20.0)]
        else:
            self.maps = None
            self.S0 = Surrogate(d["pitcher"], d["wall"], DSURR, d["cup"])
        self.b_mm = self.S0.b_mm

    def flow(self, Q, V, roll=0.0):
        return self.S0.flow(Q, V, roll)

    def z_min(self, fx, fy, yaw, Q, V, roll, alpha, ctilt=None):
        if self.maps is None:
            return self.S0.z_min(fx, fy, yaw, Q, V, roll, alpha)[0]
        Z = np.stack([S.z_min(fx, fy, yaw, Q, V, roll, alpha)[0] for S in self.maps])
        c = np.clip(np.broadcast_to(np.asarray(ctilt, float), Z.shape[1:]), CUP_MAPS[0], CUP_MAPS[-1])
        cm = np.asarray(CUP_MAPS)
        i = np.clip(np.searchsorted(cm, c, side="right") - 1, 0, len(cm) - 2)
        w = (c - cm[i]) / (cm[i + 1] - cm[i])
        z0 = np.take_along_axis(Z, i[None], 0)[0]
        z1 = np.take_along_axis(Z, (i + 1)[None], 0)[0]
        return z0 + w * (z1 - z0)


# ----------------------------------------------------------------------- the inlet program (drvf-compatible driver)
_POURF = None


def pourf_class():
    """PourF = drvf.PhysF + (a) per-stroke flow scale s_k applied to the command envelope BEFORE the spout lag,
    (b) ulaw 'pitcher': U_perp = sqrt(vz^2 + 2 g max(z + dz_origin, 0)) with vz, dz_origin tabulated from the pitcher
    (ulaw 'fs': drvf's sqrt(0.35^2 + 2 g z)), z = drvf's wanted z (ztr) + the hand's planned lift L(t) (table),
    (c) lat 'pitcher': cup-frame u_h = tabulated pitcher exit vector + hand velocity (drvf.hand_v), |u| <= min(0.40, U).
    Everything else (Q rules, ztr, footprint, Inlet fields, log columns) is drvf.PhysF.__call__ line for line; with
    s = 1, ulaw 'fs' or a constant vz = 0.35 / dz_origin = 0 table and lat 'exitmin' it reproduces drvf."""
    global _POURF
    if _POURF is not None:
        return _POURF
    drvf = PC._drvf()
    from .v05_program import Program
    from .solver import Inlet

    class PourF(drvf.PhysF):
        def __init__(self, pattern, U_ms, kQ, params, pulse_cap, flow_phase=None, s_scale=None, ulaw="fs", **kw):
            self.s_scale = {int(k): float(v) for k, v in (s_scale or {}).items()}
            self.ulaw = ulaw
            self.ptab = None
            super().__init__(pattern, U_ms, kQ, params, pulse_cap, flow_phase, **kw)

        def envelope(self, t):
            e = drvf.PhysF.envelope(self, t)
            if self.s_scale and e > 0:
                e = e * self.s_scale.get(self.phase_index(t), 1.0)
            return e

        def _ti(self, key, t):
            return float(np.interp(t, self.ptab["t"], self.ptab[key]))

        def __call__(self, t):
            self.ncall += 1
            k = self.phase_index(t)
            p = self.phases[k]
            tt = max(0.0, min(p["duration"], t - self.events[k]))
            inl0 = Program.__call__(self, t)
            if self.lag > 0:
                E = float(np.interp(t, self.tg, self.Ef))
                fac = 1.0
                if self.cls[k] in ("low", "cut") and p.get("flow_modulation", 0.0) > 0:
                    fac = 1 + p["flow_modulation"] * math.cos(2 * math.pi * p.get("flow_frequency", p["frequency"]) * tt
                                                               + p.get("flow_phase", 0.0))
                F = E * fac
            else:
                F = (inl0.S_eff if inl0.active else 0.0) * self.gain(k, tt) * self.s_scale.get(k, 1.0)
            off = Inlet(active=False, x_hit=inl0.x_hit, S_eff=0.0, scan_speed=inl0.scan_speed, r1=inl0.r1, r2=inl0.r2)
            if F <= 0:
                return off
            Qraw = F / self.kQ
            if Qraw < 1.0 / drvf.ML and (self.stop or self.cls[k] not in ("low", "cut")):
                return off
            Q = min(max(Qraw, 1.0 / drvf.ML), 32.0 / drvf.ML)
            ks = k if self.cls[k] in ("low", "cut") else (self.prev_fl[k] if self.prev_fl[k] is not None else self.next_fl[k])
            z = self.z_at(t, k, tt)
            tab = self.ptab
            if tab is not None and tab.get("L") is not None:
                z = z + self._ti("L", t)
            if self.ulaw == "fs":
                U = drvf.U_of_z(z) / drvf.D_M
            else:
                vz = self._ti("vz", t)
                oz = self._ti("oz", t)
                U = math.sqrt(vz * vz + 2 * drvf.G * max(z + oz, 0.0)) / drvf.D_M
            ps = self.phases[ks]
            bx, by = ps["jet"]
            hi = min(0.40 / drvf.D_M, 1.0 * U)
            vh = (0.0, 0.0)
            osc = 0.0
            if self.lat == "pitcher":
                vh = self.hand_v(t)
                jx = self._ti("ex", t) / drvf.D_M + vh[0]
                jy = self._ti("ey", t) / drvf.D_M + vh[1]
                v = math.hypot(jx, jy)
                if v > hi:
                    jx, jy, v = jx * hi / v, jy * hi / v, hi
            elif self.lat == "exitmin":
                vh = self.hand_v(t)
                nb = math.hypot(bx, by)
                ex, ey = (bx / nb, by / nb) if nb > 0 else (0.0, -1.0)
                ue = min(max(nb - (vh[0] * ex + vh[1] * ey), 0.10 / drvf.D_M), 0.27 / drvf.D_M)
                jx, jy = ue * ex + vh[0], ue * ey + vh[1]
                v = math.hypot(jx, jy)
                if v > hi:
                    jx, jy, v = jx * hi / v, jy * hi / v, hi
            else:
                raise ValueError(self.lat)
            v = math.hypot(jx, jy)
            spd = math.hypot(U, v)
            aspect = spd / U
            r = math.sqrt(Q / (math.pi * U))
            d = 2 * math.sqrt(Q / spd / math.pi)
            out = Inlet(active=True, x_hit=inl0.x_hit, S_eff=self.kQ * Q, u_in=(jx, jy), U_perp=U, d_jet=d,
                        r1=r * math.sqrt(aspect), r2=r / math.sqrt(aspect), phi=math.atan2(jy, jx), scan_speed=inl0.scan_speed)
            ex_speed = math.hypot(jx - vh[0], jy - vh[1]) * drvf.D_M
            self.log.append((t, self.ncall % 2, k, ks, Qraw * drvf.ML, Q * drvf.ML, U * drvf.D_M, z * 100, v * drvf.D_M,
                             jx * drvf.D_M, jy * drvf.D_M, vh[0] * drvf.D_M, vh[1] * drvf.D_M, ex_speed, r * drvf.D_M * 1e3,
                             d * drvf.D_M * 1e3, F, osc * drvf.D_M))
            return out

    _POURF = PourF
    return PourF


def make_program(pattern, z_cm=None, s_scale=None, closure=None, lat="exitmin", ulaw="fs", pulse=0.0, flow_phase=None,
                 lag=0.08):
    """PourF on Codex's control of `pattern` (drvf flags --lag 0.08 --ztr, bridges 0.3 s, ramps 0.1 s).  z_cm {phase index
    (0-based): wanted height [cm]} overrides FS's P2 heights; closure: solver parameters (FS's by default); pulse = cap of
    the 5 Hz flow modulation (0: none)."""
    PourF = pourf_class()
    with open(PC.EXPS) as f:
        e = json.load(f)["P2"]
    sp = e["inlet"][pattern]
    params = dict(kappa_Q=2.0, cp=1.36, beta=3.0, nu=1e-3)
    for key in ("B_dep", "p_dep", "kappa_r"):
        params[key] = e["params"][key]
    if closure:
        params.update(closure)
    return PourF(pattern, sp["U"], sp["kappa_Q"], params, pulse, flow_phase, s_scale=s_scale, ulaw=ulaw, ramp=0.1,
                 lat=lat, lag=lag, stop=False, ztr=True, zset=z_cm)


def flow_schedule(prog):
    """Delivered flow [mL/s] on the program's lag grid (drvf rules: F <= 0 off, < 1 mL/s off outside strokes, clip to
    1-32 mL/s; no 5 Hz factor: the design programs have none), the phase index and the poured volume [mL]."""
    tg = prog.tg
    k = np.array([prog.phase_index(t) for t in tg])
    fl = np.array([prog.cls[i] in FLOWING for i in range(len(prog.phases))])[k]
    Qraw = prog.Ef / prog.kQ * ML_D3
    on = (prog.Ef > 0) & ~((Qraw < 1.0) & ~fl)
    Q = np.where(on, np.clip(Qraw, 1.0, 32.0), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (Q[1:] + Q[:-1]) * np.diff(tg))])
    return dict(t=tg, k=k, Q=Q, Qraw=Qraw, cum=cum, on=on)


def design_samples(pattern, s_scale=None, dt=DS_DT):
    """Samples (dt, mid-interval) of the flowing strokes' own phase windows: Qraw [mL/s, before the 1-32 clip], foot x, y
    [m], hand velocity [m/s], poured volume before t [mL], Codex's jet b [m/s]; per 0-based phase index."""
    from .v05_program import Program
    prog = make_program(pattern, s_scale=s_scale)
    fs_ = flow_schedule(prog)
    out = {}
    ts = (np.arange(int(prog.duration / dt)) + 0.5) * dt
    for t in ts:
        k = prog.phase_index(t)
        if prog.cls[k] not in FLOWING:
            continue
        E = float(np.interp(t, prog.tg, prog.Ef))
        if E <= 0:
            continue
        x, y = Program.__call__(prog, t).x_hit
        vh = prog.hand_v(t)
        o = out.setdefault(k, dict(t=[], Qraw=[], x=[], y=[], vhx=[], vhy=[], cum=[]))
        o["t"].append(t); o["Qraw"].append(E / prog.kQ * ML_D3); o["x"].append(x * D_M); o["y"].append(y * D_M)
        o["vhx"].append(vh[0] * D_M); o["vhy"].append(vh[1] * D_M); o["cum"].append(float(np.interp(t, fs_["t"], fs_["cum"])))
    for k, o in out.items():
        for key in list(o):
            o[key] = np.asarray(o[key], float)
        p = prog.phases[k]
        o["b"] = (p["jet"][0] * D_M, p["jet"][1] * D_M)
        o["cls"] = prog.cls[k]
        o["t0"], o["t1"] = prog.events[k], prog.events[k + 1]
        o["vh_mean"] = float(np.mean(np.hypot(o["vhx"], o["vhy"])))
    return dict(strokes=out, total_mL=float(fs_["cum"][-1]), T=prog.duration, prog=prog, sched=fs_)


def _wquant(X, W, nq=DS_NQ):
    """Weighted quantiles (levels (i + 0.5) / nq) along the last axis."""
    W = np.broadcast_to(W, X.shape)
    o = np.argsort(X, axis=-1)
    Xs = np.take_along_axis(X, o, -1)
    Ws = np.take_along_axis(W, o, -1)
    c = np.cumsum(Ws, -1)
    c = c / c[..., -1:]
    lev = (np.arange(nq) + 0.5) / nq
    idx = (c[..., None, :] < lev[:, None]).sum(-1)
    idx = np.minimum(idx, X.shape[-1] - 1)
    return np.take_along_axis(Xs, idx, -1)


def _fw(x, w, axis=-1):
    w = np.broadcast_to(w, x.shape)
    return np.sum(x * w, axis) / np.sum(w, axis)


def stroke_combos(SS, smp, alpha, fills, s_list, rolls, yaws_deg, ctilt=None):
    """Closure-independent arrays of one stroke for every combination (fill, s, roll, yaw): ln of the 16 flow-weighted
    quantiles of X = U^2 / d [D/s^2] on the stroke's z grid (z(t) = max(z_k, Zreq(t))), and the flow-weighted cup-frame
    |u_h|, pitcher-frame exit speed, Zreq, tilt, fill range."""
    cls = smp["cls"]
    n = len(smp["t"])
    F_, S_, R_, Y_ = len(fills), len(s_list), len(rolls), len(yaws_deg)
    shp = (F_, S_, R_, Y_, n)
    Q = np.clip(np.asarray(s_list, float)[:, None] * smp["Qraw"][None, :], 1.0, 32.0)        # (S, n) mL/s
    V = np.maximum(np.asarray(fills, float)[:, None] - smp["cum"][None, :], DS_VMIN)         # (F, n) mL
    Qb = np.broadcast_to(Q[None, :, None, None, :], shp)
    Vb = np.broadcast_to(V[:, None, None, None, :], shp)
    rb = np.broadcast_to(np.asarray(rolls, float)[None, None, :, None, None], shp)
    yb = np.broadcast_to(np.radians(np.asarray(yaws_deg, float))[None, None, None, :, None], shp)
    fx = np.broadcast_to(smp["x"], shp)
    fy = np.broadcast_to(smp["y"], shp)
    f = SS.flow(Qb[:, :, :, :1, :], Vb[:, :, :, :1, :], rb[:, :, :, :1, :])
    ef, es, ez = Surrogate.exit_dir(f["tilt"], rb[:, :, :, :1, :], alpha)
    u0 = f["u0"]
    vz, oz = u0 * ez, f["oz"]
    cy, sy = np.cos(yb), np.sin(yb)
    exx = u0 * (ef * cy - es * sy)
    exy = u0 * (ef * sy + es * cy)
    uabs = np.hypot(exx + smp["vhx"], exy + smp["vhy"])
    ct = None if ctilt is None else np.broadcast_to(np.asarray(ctilt, float), shp)
    zmin = SS.z_min(fx, fy, yb, Qb, Vb, rb, alpha, ct)
    from scipy.ndimage import maximum_filter1d
    win = max(1, int(round(DS_REQWIN / DS_DT)))
    Zreq = maximum_filter1d(zmin, size=win, axis=-1, mode="nearest")
    W = np.broadcast_to(Q[None, :, None, None, :], shp)
    lo, hi = DS_ZCUT if cls == "cut" else DS_ZLOW
    zg = np.geomspace(lo, hi, DS_NZ)
    lnXq = np.empty(shp[:-1] + (DS_NZ, DS_NQ))
    for j, z0 in enumerate(zg):
        z = np.maximum(z0, Zreq)
        U = np.sqrt(vz * vz + 2 * G * np.maximum(z + oz, 0.0))
        uc = np.minimum(uabs, np.minimum(0.40, U))
        spd = np.hypot(U, uc)
        d = 2 * np.sqrt(Qb * ML / (spd * math.pi))
        X = (U / D_M) ** 2 / (d / D_M)
        lnXq[..., j, :] = np.log(_wquant(X, W))
    return dict(zg=zg, lnXq=lnXq, uw=_fw(uabs, W), exitw=_fw(np.broadcast_to(u0 * np.hypot(ef, es), shp), W),
                Zreq_w=_fw(Zreq, W), Zreq_max=Zreq.max(-1), zmin_w=_fw(zmin, W),
                tilt_w=_fw(np.broadcast_to(f["tilt"], shp), W), u0_w=_fw(np.broadcast_to(u0, shp), W),
                V_lo=Vb.min(-1), V_hi=Vb.max(-1), Qw=_fw(Qb, W), shape=shp[:-1])


def _logit(x):
    x = np.clip(x, *DS_CHI_CLIP)
    return np.log(x / (1 - x))


def solve_z(lnXq, zg, tgt, B, p):
    """Per combination: wanted z_k [m] with flow-weighted chi = clipped target (interpolated in log z), or the grid end
    if unreachable; returns z, chi achieved, logit error, chi(z_lo), chi(z_hi)."""
    a = np.clip(p * (lnXq - math.log(B)), -60.0, 60.0)
    chi = (1.0 / (1.0 + np.exp(a))).mean(-1)                     # (..., NZ) non-increasing in z
    tc = float(np.clip(tgt, *DS_CHI_CLIP))
    below = chi <= tc
    anyb = below.any(-1)
    j = np.argmax(below, -1)
    jm = np.maximum(j - 1, 0)
    L = _logit(chi)
    Lt = float(_logit(tc))
    L0 = np.take_along_axis(L, jm[..., None], -1)[..., 0]
    L1 = np.take_along_axis(L, j[..., None], -1)[..., 0]
    lz = np.log(zg)
    fr = np.where(L0 > L1, (L0 - Lt) / np.where(L0 > L1, L0 - L1, 1.0), 0.0)
    zi = np.exp(lz[jm] + np.clip(fr, 0, 1) * (lz[j] - lz[jm]))
    z = np.where(~anyb, zg[-1], np.where(j == 0, zg[0], zi))
    ch = np.where(~anyb, chi[..., -1], np.where(j == 0, chi[..., 0], tc))
    e = _logit(ch) - Lt
    return z, ch, e, chi[..., 0], chi[..., -1]


DS_TIE = 1e-5                                # tie-break regulariser (correction after the v1 screen, see prereg)
out_ref = {}


def yaw_windows(pattern, smp_all, step=15.0):
    """Yaw candidates [deg] per flowing stroke: Codex's jet direction (fixed) or, for FREE_YAW, a 15 deg grid inside the
    window reachable from the fixed neighbours at 0.9 x DS_YAW_RATE (bridge: its duration; directly adjacent: 0.1 s)."""
    st = smp_all["strokes"]
    prog = smp_all["prog"]
    ks = sorted(st)
    fixed = {k: math.degrees(math.atan2(st[k]["b"][1], st[k]["b"][0])) for k in ks}
    out, adj = {}, {}
    for i, k in enumerate(ks):
        if (pattern, k + 1) not in FREE_YAW:
            out[k] = [fixed[k]]
            continue
        lo, hi = -540.0, 540.0
        for nb in (ks[i - 1] if i > 0 else None, ks[i + 1] if i + 1 < len(ks) else None):
            if nb is None:
                continue
            gap = (prog.events[k] - prog.events[nb + 1]) if nb < k else (prog.events[nb] - prog.events[k + 1])
            gap = gap if gap > 1e-9 else 0.1
            if (pattern, nb + 1) in FREE_YAW:
                adj[k] = (nb, 0.9 * DS_YAW_RATE * gap)
                continue
            c, w = fixed[nb], 0.9 * DS_YAW_RATE * gap
            lo, hi = max(lo, c - w), min(hi, c + w)
        g = np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)
        out[k] = [float(v) for v in g] if len(g) else [fixed[k]]
    # tie-break reference of the free strokes: circular mean of the nearest fixed strokes before and after
    ref = {}
    for i, k in enumerate(ks):
        if (pattern, k + 1) not in FREE_YAW:
            ref[k] = fixed[k]
            continue
        nb = [kk for kk in reversed(ks[:i]) if (pattern, kk + 1) not in FREE_YAW][:1] + \
             [kk for kk in ks[i + 1:] if (pattern, kk + 1) not in FREE_YAW][:1]
        if nb:
            a = np.radians([fixed[kk] for kk in nb])
            ref[k] = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
        else:
            ref[k] = fixed[k]
    out_ref.clear(); out_ref.update(ref)
    return out, adj, fixed


class PatternDesign:
    """Closure-independent precomputation for one pattern / spec / alpha: combos of every flowing stroke."""

    def __init__(self, SS, pattern, alpha, smp_all, fills=DS_FILLS, fixed=None):
        """fixed: None (pass 1: enumerate fills and s) or dict(fill=, s={k: s}) (pass 2: one fill, chosen s)."""
        self.pattern, self.alpha = pattern, alpha
        refs = _refs()[pattern]
        self.smp = smp_all["strokes"]
        self.total_mL = smp_all["total_mL"]
        self.yaws, self.adj, self.fixed_yaw = yaw_windows(pattern, smp_all)
        self.yaw_ref = dict(out_ref)
        ctf = cup_tilt_fn(pattern) if SS.maps is not None else None
        self.fills = list(fills) if fixed is None else [fixed["fill"]]
        self.C = {}
        for k, s in sorted(self.smp.items()):
            if fixed is None:
                sl = list(DS_S) if s["cls"] != "cut" else [1.0]
            else:
                sl = [1.0]                      # the samples already carry the chosen s (exact lagged schedule)
            ct = None if ctf is None else ctf(s["t"])
            free = (pattern, k + 1) in FREE_YAW
            rolls = DS_ROLLS if free else (0.0,)          # fixed-direction strokes: no roll (keeps the jet direction)
            c = stroke_combos(SS, s, alpha, self.fills, sl, rolls, self.yaws[k], ct)
            c["rolls"] = list(rolls)
            dy = (np.asarray(self.yaws[k]) - self.yaw_ref[k] + 180.0) % 360.0 - 180.0
            c["reg"] = DS_TIE * ((dy[None, None, None, :] / 90.0) ** 2 + (np.asarray(rolls)[None, None, :, None] / 20.0) ** 2)
            c["s_list"] = sl if fixed is None else [fixed["s"].get(k, 1.0)]
            r = refs[str(k + 1)]
            c.update(cls=s["cls"], chi_t=r["chi_t"], dep=r["dep"], F_int=r["F_int"], FS_uh=r["FS_uh_ms"],
                     lb=(pattern, k + 1) in LB_STROKES, n=k + 1)
            self.C[k] = c
        # fill feasibility: V(t) >= DS_VMIN with s = 1 (pass 1) / exact (pass 2)
        self.fill_ok = np.array([f - self.total_mL >= DS_VMIN - 1e-9 for f in self.fills])


def stroke_cost(c, z, ch, e, kt, norms):
    """Per-combination additive contribution to J_total (and its parts)."""
    W_dep, F_oth = norms["W_dep"], norms["F_oth"]
    s = np.asarray(c["s_list"], float)[None, :, None, None]
    js = (c["dep"] / W_dep) * (0.4 * np.log(s) / math.log(1.25)) ** 2
    jc = (c["dep"] / W_dep) * e ** 2
    m = np.log(np.maximum(kt * c["uw"], 1e-9) / (0.7 * c["FS_uh"]))
    if c["lb"]:
        ju = np.maximum(m, 0.0) ** 2 / len(LB_STROKES)
        jo = 0.0 * m
    else:
        ju = 0.0 * m
        jo = LAM_OTH * (c["F_int"] / F_oth) * m ** 2
    jc, js = np.broadcast_to(jc, m.shape), np.broadcast_to(js, m.shape)
    return jc + js + ju + jo + c.get("reg", 0.0), dict(chi=jc, s=js, uh=ju, oth=jo)


def _norms(designs):
    W = sum(c["dep"] for d in designs for c in d.C.values())
    Fo = sum(c["F_int"] for d in designs for c in d.C.values() if not c["lb"])
    return dict(W_dep=W, F_oth=Fo)


def evaluate(designs, B, p, kts=KT_GRID, detail=False):
    """J_total(B, p) minimised over kappa_t in kts, per pattern over the fill, per stroke over (s, roll, yaw) (pairs of
    adjacent free-yaw strokes jointly within their yaw window).  detail=True returns the chosen design."""
    norms = _norms(designs)
    sol = {}
    for d in designs:
        for k, c in d.C.items():
            sol[(d.pattern, k)] = solve_z(c["lnXq"], c["zg"], c["chi_t"], B, p)
    best = None
    for kt in kts:
        tot = 0.0
        pick = {}
        for d in designs:
            per_fill = np.zeros(len(d.fills))
            ch_ = {}
            mins = {}
            for k, c in d.C.items():
                z, ch, e, _, _ = sol[(d.pattern, k)]
                cost, _ = stroke_cost(c, z, ch, e, kt, norms)                      # (F, S, R, Y)
                cy = cost.min(axis=(1, 2))                                          # (F, Y)
                mins[k] = cy
            done = set()
            for k, c in d.C.items():
                if k in done:
                    continue
                if k in d.adj and d.adj[k][0] in d.C:
                    k2, w = d.adj[k]
                    y1, y2 = np.asarray(d.yaws[k]), np.asarray(d.yaws[k2])
                    ok = np.abs(y1[:, None] - y2[None, :]) <= w + 1e-9
                    pair = mins[k][:, :, None] + mins[k2][:, None, :]
                    pair = np.where(ok[None], pair, np.inf)
                    per_fill += pair.reshape(len(d.fills), -1).min(-1)
                    done |= {k, k2}
                else:
                    per_fill += mins[k].min(-1)
                    done.add(k)
            per_fill = np.where(d.fill_ok, per_fill, np.inf)
            i = int(np.argmin(per_fill))
            tot += per_fill[i]
            pick[d.pattern] = (i, per_fill[i])
        if best is None or tot < best[0]:
            best = (tot, kt, pick)
    if not detail:
        return best[0], best[1]
    return best[0], best[1], choose(designs, B, p, best[1], best[2], sol, norms)


def choose(designs, B, p, kt, pick, sol, norms):
    """The chosen combination per stroke (given the fill picked per pattern), with its numbers."""
    out = {}
    for d in designs:
        i = pick[d.pattern][0]
        rows = {}
        mins = {}
        for k, c in d.C.items():
            z, ch, e, c_lo, c_hi = sol[(d.pattern, k)]
            cost, parts = stroke_cost(c, z, ch, e, kt, norms)
            mins[k] = (cost[i], parts, z[i], ch[i], e[i], c_lo[i], c_hi[i])
        ychoice = {}
        done = set()
        for k in d.C:
            if k in done:
                continue
            if k in d.adj and d.adj[k][0] in d.C:
                k2, w = d.adj[k]
                a = mins[k][0].min(axis=(0, 1))
                b = mins[k2][0].min(axis=(0, 1))
                y1, y2 = np.asarray(d.yaws[k]), np.asarray(d.yaws[k2])
                ok = np.abs(y1[:, None] - y2[None, :]) <= w + 1e-9
                pr = np.where(ok, a[:, None] + b[None, :], np.inf)
                j1, j2 = np.unravel_index(int(np.argmin(pr)), pr.shape)
                ychoice[k], ychoice[k2] = j1, j2
                done |= {k, k2}
            else:
                ychoice[k] = int(np.argmin(mins[k][0].min(axis=(0, 1))))
                done.add(k)
        for k, c in d.C.items():
            cost, parts, z, ch, e, c_lo, c_hi = mins[k]
            jy = ychoice[k]
            sub = cost[:, :, jy]
            js, jr = np.unravel_index(int(np.argmin(sub)), sub.shape)
            idx = (i, js, jr, jy)
            rows[k] = dict(n=k + 1, cls=c["cls"], lb=c["lb"], s=float(c["s_list"][js]), roll=float(c["rolls"][jr]),
                           yaw=float(d.yaws[k][jy]), yaw_free=(d.pattern, k + 1) in FREE_YAW, z_mm=float(z[js, jr, jy] * 1e3),
                           chi_t=c["chi_t"], chi=float(ch[js, jr, jy]), e=float(e[js, jr, jy]),
                           chi_zlo=float(c_lo[js, jr, jy]), chi_zhi=float(c_hi[js, jr, jy]),
                           Zreq_w_mm=float(c["Zreq_w"][idx] * 1e3), Zreq_max_mm=float(c["Zreq_max"][idx] * 1e3),
                           zmin_w_mm=float(c["zmin_w"][idx] * 1e3), uw=float(c["uw"][idx]), exitw=float(c["exitw"][idx]),
                           FS_uh=c["FS_uh"], mom_ratio=float(kt * c["uw"][idx] / (0.7 * c["FS_uh"])),
                           tilt_w=float(c["tilt_w"][idx]), u0_w=float(c["u0_w"][idx]), V_lo=float(c["V_lo"][idx]),
                           V_hi=float(c["V_hi"][idx]), Qw=float(c["Qw"][idx]), cost=float(sub[js, jr] - c["reg"][0, 0, jr, jy]),
                           parts={q: float(np.broadcast_to(v, c["uw"].shape)[idx]) for q, v in parts.items()},
                           zlo_mm=float(c["zg"][0] * 1e3), zhi_mm=float(c["zg"][-1] * 1e3))
        out[d.pattern] = dict(fill=float(d.fills[i]), J=float(pick[d.pattern][1]), strokes=rows, total_mL=d.total_mL)
    return dict(B=B, p=p, kt=kt, patterns=out)


def fit_closure(designs, B0=None, p0=None, grid=True, log=print):
    """(B, p, kappa_t): grid p in 1..8 step 0.5 x 25 log B in [300, 6000] (optional), then coordinate refinement."""
    t0 = time.time()
    best = None
    if grid:
        for p in np.arange(1.0, 8.0001, 0.5):
            for lB in np.linspace(math.log(300.0), math.log(6000.0), 25):
                J, kt = evaluate(designs, math.exp(lB), p)
                if best is None or J < best[0]:
                    best = (J, lB, p, kt)
        log(f"    grid best J {best[0]:.4f} B {math.exp(best[1]):.0f} p {best[2]} kt {best[3]} [{time.time() - t0:.0f}s]")
    else:
        J, kt = evaluate(designs, B0, p0)
        best = (J, math.log(B0), p0, kt)
    J, lB, p, kt = best
    sl, sp = (math.log(6000.0) - math.log(300.0)) / 24, 0.5
    for _ in range(40):
        imp = False
        for dl, dp in ((sl, 0), (-sl, 0), (0, sp), (0, -sp)):
            l2 = min(max(lB + dl, math.log(300.0)), math.log(6000.0))
            p2 = min(max(p + dp, 1.0), 8.0)
            J2, kt2 = evaluate(designs, math.exp(l2), p2)
            if J2 < J - 1e-10:
                J, lB, p, kt, imp = J2, l2, p2, kt2, True
        if not imp:
            sl *= 0.5
            sp *= 0.5
            if sl < 2e-3:
                break
    # kappa_t refinement on a 0.01 grid around the optimum
    kts = tuple(np.round(np.arange(max(0.01, kt - 0.05), min(1.0, kt + 0.05) + 1e-9, 0.01), 2))
    J, kt = evaluate(designs, math.exp(lB), p, kts)
    log(f"    refined J {J:.4f} B {math.exp(lB):.1f} p {p:.3f} kt {kt} [{time.time() - t0:.0f}s]")
    return dict(J=J, B=math.exp(lB), p=p, kt=kt)


def design_spec(spec, alphas=DS_ALPHAS, patterns=PC.PATTERNS, log=print, tag=None):
    """Pass 1 (fills x s enumerated, s = 1 depletion) and pass 2 (fill and s fixed, exact lagged schedule) per alpha;
    returns {alpha: result} and writes design/<spec>/alpha<a>.json."""
    SS = SpecSurrogate(spec)
    odir = os.path.join(DSN, tag or spec)
    os.makedirs(odir, exist_ok=True)
    smp1 = {p: design_samples(p) for p in patterns}
    res = {}
    for al in alphas:
        t0 = time.time()
        log(f"[{spec}] alpha {al:g}: pass 1 precompute ...")
        D1 = [PatternDesign(SS, p, al, smp1[p]) for p in patterns]
        log(f"    precomputed [{time.time() - t0:.0f}s]")
        f1 = fit_closure(D1, log=log)
        J1, kt1, ch1 = evaluate(D1, f1["B"], f1["p"], (f1["kt"],), detail=True)
        # pass 2: exact schedule with the chosen s, the chosen fill
        D2 = []
        for p in patterns:
            P1 = ch1["patterns"][p]
            s_sc = {k: r["s"] for k, r in P1["strokes"].items() if abs(r["s"] - 1.0) > 1e-9}
            smp2 = design_samples(p, s_scale=s_sc)
            fill = P1["fill"]
            if fill - smp2["total_mL"] < DS_VMIN:            # exact volume > s = 1 estimate: next feasible fill
                fill = min([f for f in DS_FILLS if f - smp2["total_mL"] >= DS_VMIN] or [DS_FILLS[-1]])
            D2.append(PatternDesign(SS, p, al, smp2, fixed=dict(fill=fill, s={k: r["s"] for k, r in P1["strokes"].items()})))
        f2 = fit_closure(D2, B0=f1["B"], p0=f1["p"], grid=False, log=log)
        J2, kt2, ch2 = evaluate(D2, f2["B"], f2["p"], (f2["kt"],), detail=True)
        for p in patterns:                                   # carry s (pass 2 enumerates s = [chosen] only)
            for k, r in ch2["patterns"][p]["strokes"].items():
                r["s"] = ch1["patterns"][p]["strokes"][k]["s"]
        out = dict(spec=spec, alpha=al, pass1=dict(J=J1, B=f1["B"], p=f1["p"], kt=f1["kt"]),
                   pass2=dict(J=J2, B=f2["B"], p=f2["p"], kt=f2["kt"]), design=_jsonable(ch2),
                   pass1_design=_jsonable(ch1), wall_s=time.time() - t0)
        with open(os.path.join(odir, f"alpha{al:g}.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        log(f"[{spec}] alpha {al:g}: J pass1 {J1:.4f} pass2 {J2:.4f} B {f2['B']:.0f} p {f2['p']:.2f} kt {f2['kt']} "
            f"fills {[ch2['patterns'][p]['fill'] for p in patterns]} [{time.time() - t0:.0f}s]")
        res[al] = out
    return res


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ----------------------------------------------------------------------- final inlet tables, exact chi, kappa_r
def pitcher_tables(prog, SS, alpha, fill, yaw_deg, roll_deg, ctilt_fn=None, hand="ideal"):
    """Tables on the program's lag grid for ulaw/lat 'pitcher': delivered Q, fill V, foot, yaw (ideal-hand smoothed),
    roll, tilt, u0, vz, dz_origin, exit vector (world) [m/s], z_min, wanted z_w and the planned lift L (plan_lift of
    max(z_min - z_w, 0); 0 in 'stop' phases)."""
    from .v05_program import Program
    fs_ = flow_schedule(prog)
    tg = fs_["t"]
    dt = float(tg[1] - tg[0])
    V = np.maximum(fill - fs_["cum"], DS_VMIN)
    XY = np.array([Program.__call__(prog, t).x_hit for t in tg], float) * D_M
    segs_y = [(prog.events[k], prog.events[k + 1], math.radians(yaw_deg[k])) for k in sorted(yaw_deg)]
    segs_r = [(prog.events[k], prog.events[k + 1], math.radians(roll_deg[k])) for k in sorted(roll_deg)]
    yaw = smooth_yaw(PC._yaw_schedule(tg, segs_y, prog.duration), dt, hand)
    roll = np.degrees(PC._yaw_schedule(tg, segs_r, prog.duration))
    kk = fs_["k"]
    zw = np.array([prog.z_at(t, k, max(0.0, min(prog.phases[k]["duration"], t - prog.events[k]))) for t, k in zip(tg, kk)])
    Qt = np.maximum(fs_["Q"], 1.0)
    f = SS.flow(Qt, V, roll)
    ef, es, ez = Surrogate.exit_dir(f["tilt"], roll, alpha)
    u0 = f["u0"]
    ex = u0 * (ef * np.cos(yaw) - es * np.sin(yaw))
    ey = u0 * (ef * np.sin(yaw) + es * np.cos(yaw))
    ct = None if ctilt_fn is None else ctilt_fn(tg)
    zmin = SS.z_min(XY[:, 0], XY[:, 1], yaw, Qt, V, roll, alpha, ct)
    stop = np.array([prog.cls[k] == "stop" for k in range(len(prog.phases))])[kk]
    R = np.where(stop, 0.0, np.maximum(zmin - zw, 0.0))
    L = plan_lift(R, dt, hand)
    return dict(t=tg, Q=fs_["Q"], V=V, x=XY[:, 0], y=XY[:, 1], yaw=yaw, roll=roll, tilt=f["tilt"], u0=u0, vz=u0 * ez,
                oz=f["oz"], ex=ex, ey=ey, zmin=zmin, zw=zw, R=R, L=L, k=kk, ctilt=(ct if ct is not None else np.full_like(tg, np.nan)))


def build_design_program(spec, D, pattern, kappa_r=None, SS=None):
    """PourF (lat/ulaw 'pitcher') of one pattern from a design dict D (alpha<a>.json 'design' + closure)."""
    SS = SS or SpecSurrogate(spec)
    P = D["patterns"][pattern]
    st = {int(r["n"]) - 1: r for r in P["strokes"].values()}
    z_cm = {k: r["z_mm"] / 10.0 for k, r in st.items()}
    s_sc = {k: r["s"] for k, r in st.items() if abs(r["s"] - 1.0) > 1e-12}
    clo = dict(FS_FIXED, B_dep=D["B"], p_dep=D["p"], kappa_t=D["kt"])
    if kappa_r is not None:
        clo["kappa_r"] = kappa_r
    prog = make_program(pattern, z_cm=z_cm, s_scale=s_sc, closure=clo, lat="pitcher", ulaw="pitcher", pulse=0.0)
    ctf = cup_tilt_fn(pattern) if SS.maps is not None else None
    prog.ptab = pitcher_tables(prog, SS, D["alpha"], P["fill"], {k: r["yaw"] for k, r in st.items()},
                               {k: r["roll"] for k, r in st.items()}, ctf)
    return prog


def inlet_strokes(prog, B, p, dt=1e-3):
    """Exact per-stroke numbers of the inlet the solver receives (1 ms samples of prog(t), stroke's own phase window):
    flow-weighted chi, radial V_R (for kappa_r), U, |u_h|, z, Q, and lifts."""
    n0 = len(prog.log)
    acc = {}
    for t in (np.arange(int(prog.duration / dt)) + 0.5) * dt:
        k = prog.phase_index(t)
        if prog.cls[k] not in FLOWING:
            prog(t)
            continue
        inl = prog(t)
        if not inl.active:
            continue
        zc = inl.U_perp ** 2 / (B * max(inl.d_jet, 1e-9))
        chi = 1.0 / (1.0 + zc ** p)
        VR = min(inl.U_perp * chi ** (1.0 / (2.0 * p)), inl.U_perp, math.sqrt(B * inl.d_jet))
        row = prog.log[-1]
        o = acc.setdefault(k, dict(S=0.0, chi=0.0, VR=0.0, U=0.0, uh=0.0, z=0.0, Q=0.0, L=0.0, Lmax=0.0, vol=0.0,
                                   zmin=0.0, V=0.0, tilt=0.0, exit=0.0))
        S = inl.S_eff
        o["S"] += S; o["chi"] += S * chi; o["VR"] += S * VR; o["U"] += S * row[6]; o["uh"] += S * row[8]
        o["z"] += S * row[7] * 10.0; o["Q"] += S * row[5]; o["vol"] += row[5] * dt; o["exit"] += S * row[13]
        if prog.ptab is not None:
            L = float(np.interp(t, prog.ptab["t"], prog.ptab["L"]))
            o["L"] += S * L * 1e3; o["Lmax"] = max(o["Lmax"], L * 1e3)
            o["zmin"] += S * float(np.interp(t, prog.ptab["t"], prog.ptab["zmin"])) * 1e3
            o["V"] += S * float(np.interp(t, prog.ptab["t"], prog.ptab["V"]))
            o["tilt"] += S * float(np.interp(t, prog.ptab["t"], prog.ptab["tilt"]))
    del prog.log[n0:]
    out = {}
    for k, o in acc.items():
        W = o["S"]
        out[k + 1] = dict(chi=o["chi"] / W, VR=o["VR"] / W, U_ms=o["U"] / W, uh_ms=o["uh"] / W, z_mm=o["z"] / W,
                          Q_mLs=o["Q"] / W, lift_mm=o["L"] / W, lift_max_mm=o["Lmax"], vol_mL=o["vol"], zmin_mm=o["zmin"] / W,
                          V_mL=o["V"] / W, tilt=o["tilt"] / W, exit_ms=o["exit"] / W)
    return out


def finalize(spec, alpha=None, patterns=PC.PATTERNS, log=print, tag=None):
    """Pick the alpha with the lowest pass-2 J (or the given one), build the six programs, fit kappa_r (FS's rule on the
    exact inlets), recompute chi / J from the exact inlets, hand checks; writes design/<spec>/final.json and the tables."""
    odir = os.path.join(DSN, tag or spec)
    scan = {}
    for al in DS_ALPHAS:
        f = os.path.join(odir, f"alpha{al:g}.json")
        if os.path.exists(f):
            with open(f) as fh:
                scan[al] = json.load(fh)
    if alpha is None:
        alpha = min(scan, key=lambda a: scan[a]["pass2"]["J"])
    R_ = scan[alpha]
    D = dict(R_["design"], alpha=alpha)
    SS = SpecSurrogate(spec)
    refs = _refs()
    progs, ex = {}, {}
    for p in patterns:
        progs[p] = build_design_program(spec, D, p, SS=SS)
        ex[p] = inlet_strokes(progs[p], D["B"], D["p"])
    num = den = 0.0
    for p in patterns:
        for n, r in ex[p].items():
            w = refs[p][str(n)]["F_int"]
            num += w * math.log(refs[p][str(n)]["radial_t"] / (r["VR"])); den += w
    kr = math.exp(num / den)
    kr = min(max(kr, 0.005), 0.2)
    # exact J with the same per-stroke terms
    W = sum(refs[p][str(n)]["dep"] for p in patterns for n in ex[p])
    Fo = sum(refs[p][str(n)]["F_int"] for p in patterns for n in ex[p] if (p, n) not in LB_STROKES)
    Jp, rows = {}, {}
    hand = {}
    for p in patterns:
        P = D["patterns"][p]
        Jp[p] = dict(chi=0.0, s=0.0, uh=0.0, oth=0.0)
        rows[p] = []
        for n, r in sorted(ex[p].items()):
            rf = refs[p][str(n)]
            dsg = P["strokes"][str(n - 1)] if str(n - 1) in P["strokes"] else P["strokes"][n - 1]
            e = float(_logit(r["chi"]) - _logit(rf["chi_t"]))
            js = rf["dep"] / W * (0.4 * math.log(dsg["s"]) / math.log(1.25)) ** 2
            m = math.log(max(D["kt"] * r["uh_ms"], 1e-9) / (0.7 * rf["FS_uh_ms"]))
            lb = (p, n) in LB_STROKES
            ju = max(m, 0.0) ** 2 / len(LB_STROKES) if lb else 0.0
            jo = 0.0 if lb else LAM_OTH * rf["F_int"] / Fo * m * m
            jc = rf["dep"] / W * e * e
            Jp[p]["chi"] += jc; Jp[p]["s"] += js; Jp[p]["uh"] += ju; Jp[p]["oth"] += jo
            unreach = abs(e) > 0.4
            why = ""
            if unreach and abs(dsg["e"]) <= 0.4:
                why = ("matched in the design (chi %.3f) but not in the inlet: the 0.08 s spout lag / ztr height transition "
                       "carries the previous stroke's flow and height into this one" % dsg["chi"])
            elif unreach:
                if e < 0:
                    why = ("clearance: z pinned at the pitcher's z_min (Zreq %.1f mm > design %.1f mm)" % (dsg["Zreq_w_mm"], dsg["z_mm"])
                           if dsg["Zreq_w_mm"] > dsg["zlo_mm"] + 0.5 and dsg["Zreq_w_mm"] >= dsg["z_mm"] - 0.3
                           else "U too high even at the lowest allowed z (%.1f mm)" % dsg["zlo_mm"])
                else:
                    why = "U too low even at the highest allowed z (%.0f mm)" % dsg["zhi_mm"]
            rows[p].append(dict(n=n, cls=dsg["cls"], lb=lb, chi_t=rf["chi_t"], chi_design=dsg["chi"], chi_inlet=r["chi"], e=e,
                                unreachable=unreach, why=why, z_design_mm=dsg["z_mm"], z_inlet_mm=r["z_mm"],
                                Zreq_w_mm=dsg["Zreq_w_mm"], zmin_inlet_mm=r["zmin_mm"], lift_mm=r["lift_mm"],
                                lift_max_mm=r["lift_max_mm"], s=dsg["s"], yaw=dsg["yaw"], yaw_free=dsg["yaw_free"],
                                roll=dsg["roll"], U_ms=r["U_ms"], uh_ms=r["uh_ms"], exit_ms=r["exit_ms"], FS_uh_ms=rf["FS_uh_ms"],
                                mom_ratio=D["kt"] * r["uh_ms"] / (0.7 * rf["FS_uh_ms"]), Q_mLs=r["Q_mLs"], vol_mL=r["vol_mL"],
                                V_mL=r["V_mL"], tilt=r["tilt"], J=jc + js + ju + jo, J_parts=dict(chi=jc, s=js, uh=ju, oth=jo)))
        Jp[p]["total"] = sum(Jp[p].values())
        tab = progs[p].ptab
        on = tab["Q"] > 0
        t_on = tab["t"][on]
        trk = track_tilt(tab["tilt"], float(tab["t"][1] - tab["t"][0]), "video")
        hand[p] = dict(ideal=hand_check(tab["t"], tilt_deg=tab["tilt"], tip=None, yaw=tab["yaw"], hand="ideal"),
                       video_tilt_track_err_deg=dict(max=float(np.max(np.abs(trk - tab["tilt"])[on])) if on.any() else 0.0,
                                                     p95=float(np.percentile(np.abs(trk - tab["tilt"])[on], 95)) if on.any() else 0.0),
                       video=hand_check(tab["t"], tilt_deg=tab["tilt"], tip=None, yaw=tab["yaw"], hand="video"),
                       lift_max_mm=float(tab["L"].max() * 1e3), V_end=float(tab["V"][-1]),
                       tilt_range=[float(tab["tilt"][on].min()), float(tab["tilt"][on].max())] if on.any() else None,
                       t_on=[float(t_on[0]), float(t_on[-1])] if on.any() else None)
        np.savez_compressed(os.path.join(odir, f"ptab_{p}.npz"), **{k: v for k, v in tab.items()})
    Jtot = sum(v["total"] for v in Jp.values())
    out = dict(spec=spec, tag=tag or spec, vmin_mL=DS_VMIN, def_=DS_SPECS[spec], b_mm=SS.b_mm, alpha=alpha,
               alpha_scan={str(a): dict(J=v["pass2"]["J"], J_pass1=v["pass1"]["J"], B=v["pass2"]["B"], p=v["pass2"]["p"],
                                        kt=v["pass2"]["kt"], fills={q: v["design"]["patterns"][q]["fill"] for q in patterns})
                           for a, v in scan.items()},
               closure=dict(FS_FIXED, B_dep=D["B"], p_dep=D["p"], kappa_t=D["kt"], kappa_r=kr),
               J_design=R_["pass2"]["J"], J_inlet=Jtot, J_pattern=Jp,
               patterns={p: dict(fill=D["patterns"][p]["fill"], total_mL=D["patterns"][p]["total_mL"], strokes=rows[p],
                                 hand=hand[p]) for p in patterns}, design=D)
    with open(os.path.join(odir, "final.json"), "w") as fh:
        json.dump(_jsonable(out), fh, indent=1, default=float)
    log(f"[{spec}] final alpha {alpha:g}: B {D['B']:.0f} p {D['p']:.2f} kt {D['kt']} kr {kr:.4f}  J design "
        f"{R_['pass2']['J']:.4f} inlet {Jtot:.4f}")
    return out


def load_final(spec):
    with open(os.path.join(DSN, spec, "final.json")) as fh:
        return json.load(fh)


def solve_design(spec, pattern, N=128, tag=None, snaps=False, prog=None, extra=None):
    """Run the solver (latte_imex.run.run, unmodified) on the design inlet of one pattern; output like drvf:
    design/runs/<tag>_<pattern>_N<N>/{c.npz, res.json, trajectory.json, inlet_log.npy, ptab.npz}."""
    import latte_imex.run as RR
    drvf = PC._drvf()
    F = load_final(tag or spec)
    global DS_VMIN
    DS_VMIN = F.get("vmin_mL", DS_VMIN)
    if prog is None:
        D = F["design"]
        prog = build_design_program(F["spec"], D, pattern, kappa_r=F["closure"]["kappa_r"])
    tag = tag or spec
    RR.v05_program = lambda name: prog
    out = os.path.join(DSN, "runs", f"{tag}_{pattern}_N{N}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sol, m = RR.run(pattern, N=N, out=out, video=False, quiet=True, control="v05")
    wall = time.time() - t0
    c = sol.concentration()
    mask = sol.g.mask.copy()
    np.savez_compressed(os.path.join(out, "c.npz"), c=c.astype(np.float32), mask=mask)
    np.save(os.path.join(out, "inlet_log.npy"), np.array(prog.log))
    if prog.ptab is not None:
        np.savez_compressed(os.path.join(out, "ptab.npz"), **prog.ptab)
    for fn in (("snapshots_c.npz",) if not snaps else ()) + ("final_state.npz",):
        try:
            os.remove(os.path.join(out, fn))
        except OSError:
            pass
    res = dict(tag=tag, spec=spec, pattern=pattern, N=N, params=dict(prog.config["parameters"]), wall_s=wall,
               stats=m["stats"], ledger=m["ledger"], T=m["T"], extra=extra or {})
    with open(os.path.join(out, "res.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(dict(tag=tag, pattern=pattern, N=N, wall_s=round(wall, 1), deposited=m["ledger"].get("deposited"))),
          flush=True)
    return out


def solve_fs(pattern, N=128, tag="FSr", via="pourf", pulse=0.41):
    """FS reference / reproduction runs: via 'drvf' = drvf.PhysF exactly as runs/FS_<p> (flags of PC.fs_program);
    via 'pourf' = PourF through the pitcher U path with FS's quantities (vz = 0.35 m/s, dz_origin = 0, no lift),
    lat 'exitmin', s = 1."""
    import latte_imex.run as RR
    if via == "drvf":
        prog = PC.fs_program(pattern)
    else:
        with open(PC.EXPS) as f:
            e = json.load(f)["P2"]
        sp = e["inlet"][pattern]
        params = dict(kappa_Q=2.0, cp=1.36, beta=3.0, nu=1e-3)
        for key in ("B_dep", "p_dep", "kappa_r"):
            params[key] = e["params"][key]
        PourF = pourf_class()
        prog = PourF(pattern, sp["U"], sp["kappa_Q"], params, pulse, PC.FS_FLOW_PHASE[pattern], s_scale={}, ulaw="pitcher",
                     ramp=0.1, lat="exitmin", lag=0.08, stop=False, ztr=True)
        prog.ptab = dict(t=prog.tg, vz=np.full_like(prog.tg, 0.35), oz=np.zeros_like(prog.tg), L=None)
    RR.v05_program = lambda name: prog
    out = os.path.join(DSN, "runs", f"{tag}_{via}_{pattern}_N{N}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sol, m = RR.run(pattern, N=N, out=out, video=False, quiet=True, control="v05")
    c = sol.concentration()
    np.savez_compressed(os.path.join(out, "c.npz"), c=c.astype(np.float32), mask=sol.g.mask.copy())
    np.save(os.path.join(out, "inlet_log.npy"), np.array(prog.log))
    for fn in ("snapshots_c.npz", "final_state.npz"):
        try:
            os.remove(os.path.join(out, fn))
        except OSError:
            pass
    with open(os.path.join(out, "res.json"), "w") as fh:
        json.dump(dict(tag=tag, via=via, pattern=pattern, N=N, wall_s=time.time() - t0, stats=m["stats"], ledger=m["ledger"],
                       T=m["T"], params=dict(prog.config["parameters"])), fh, indent=1, default=float)
    print(json.dumps(dict(tag=tag, via=via, pattern=pattern, N=N, wall_s=round(time.time() - t0, 1))), flush=True)
    return out


# =========================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--spec", default="nx")
    ap.add_argument("--what", default="flow")
    ap.add_argument("--wall", default="mean")
    ap.add_argument("--cup", default="table")
    ap.add_argument("--cups", default="table")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--specs", default=",".join(SPECS))
    ap.add_argument("--alphas", default="0,10,20,30")
    ap.add_argument("--alpha", default=None)
    ap.add_argument("--pattern", default="heart")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--via", default="pourf")
    ap.add_argument("--vmin", default=None)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "build":
        sp = spec_from_name(a.spec)
        if a.what == "flow":
            build_flow(sp)
        else:
            build_clear(sp, a.wall, a.cup)
    elif a.cmd == "build_all":
        jobs = []
        for sp in a.specs.split(","):
            if "table" in a.cups.split(","):
                jobs.append(["build", "--spec", sp, "--what", "flow"])
            for c in a.cups.split(","):
                for w in WALLS:
                    jobs.append(["build", "--spec", sp, "--what", "clear", "--wall", w, "--cup", c])
        env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        running = []
        logd = os.path.join(OUT, "logs")
        os.makedirs(logd, exist_ok=True)
        while jobs or running:
            while jobs and len(running) < a.jobs:
                j = jobs.pop(0)
                lg = open(os.path.join(logd, "_".join(j[1:]).replace("--", "") + ".log"), "w")
                running.append((subprocess.Popen([sys.executable, "-m", "latte_imex.v05_pourable"] + j, stdout=lg,
                                                 stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(os.path.dirname(__file__))), j, lg))
            time.sleep(2)
            for r in list(running):
                if r[0].poll() is not None:
                    r[2].close()
                    print("done", r[1], r[0].returncode, flush=True)
                    running.remove(r)
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "validate":
        validate()
    elif a.cmd == "design":                       # design --spec S1 [--alphas 0,10,20,30] [--vmin 50 --tag S1_V50]
        if a.vmin is not None:
            globals()["DS_VMIN"] = float(a.vmin)
        design_spec(a.spec, tuple(float(x) for x in a.alphas.split(",")), tag=a.tag)
    elif a.cmd == "finalize":                     # finalize --spec S1 [--alpha 10] [--vmin 50 --tag S1_V50]
        if a.vmin is not None:
            globals()["DS_VMIN"] = float(a.vmin)
        finalize(a.spec, None if a.alpha is None else float(a.alpha), tag=a.tag)
    elif a.cmd == "solve":                        # solve --spec S1 --pattern heart --N 128 [--tag S1]
        solve_design(a.spec, a.pattern, a.N, a.tag)
    elif a.cmd == "fsrun":                        # fsrun --pattern heart --N 48 --via drvf|pourf [--tag FSr]
        solve_fs(a.pattern, a.N, a.tag or "FSr", a.via)
    elif a.cmd == "report":
        closed_form_check()
        jet_tables()
        k = key_numbers()
        print(json.dumps(k, indent=1)[:6000])
    else:
        raise SystemExit(a.cmd)


if __name__ == "__main__":
    main()
