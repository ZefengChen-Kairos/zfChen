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
    other hand (no table within reach), as the baristas in the clips do."""
    return Cup() if cup == "table" else Cup(height=1.0)


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
def build_flow(spec, out=OUT, verbose=True):
    """Steady flow map on (V, log2 Q, roll); spout direction +x (yaw 0), tip at (0, 0, 2 cm)."""
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
    elif a.cmd == "report":
        closed_form_check()
        jet_tables()
        k = key_numbers()
        print(json.dumps(k, indent=1)[:6000])
    else:
        raise SystemExit(a.cmd)


if __name__ == "__main__":
    main()
