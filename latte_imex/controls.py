"""Zefeng's parametric pouring programs (own phase structure + bounded parameter vectors).

Each pattern is a fixed phase *structure* (what the hand does, in which order) with a bounded
parameter vector (where, how long, how much, how wide).  The parameter values are NOT taken from
the V0.5 controls: they are found by CMA-ES against targets extracted from the reference clips
(optimize.py).  The phase list is turned into Inlet records by the same V0.5 `Program` semantics
(continuity-ellipse inlet geometry, zero-flow 0.18 s bridges), so the control *interface* is shared
with Codex while the control *values* are ours.

Cup frame: x right, y up, barista at y<0.  The stream leans towards the far side (+y), the cut
runs from the barista side to the far side, so the heart tip / stem exit is at +y.
"""
import math
from collections import OrderedDict
import numpy as np
from .v05_program import Program

# physics: the shared V0.5 parameter set (guessed by Codex, nothing fitted) so that IMEX vs HLLC
# comparisons differ only in numerics and controls.  Over-ridable from the CLI.
V05_PHYSICS = dict(cp=0.3, beta=3.0, nu=1e-3, D=1e-7, kappa_Q=125.0, kappa_c=1.0, kappa_t=0.7, kappa_r=0.3,
                   B_dep=240.0, p_dep=2.0)


class Space:
    """Ordered box; x in [0,1]^n <-> physical dict."""

    def __init__(self, **bounds):
        self.b = OrderedDict(bounds)

    @property
    def n(self):
        return len(self.b)

    @property
    def names(self):
        return list(self.b)

    def decode(self, x):
        x = np.clip(np.asarray(x, float), 0, 1)
        return {k: lo + (hi - lo) * xi for (k, (lo, hi)), xi in zip(self.b.items(), x)}

    def encode(self, d):
        return np.array([(d[k] - lo) / (hi - lo) for k, (lo, hi) in self.b.items()])


def phase(duration, start, end, Q, radius, jet=(0.0, 1.2), amplitude=0.0, frequency=0.0, **extra):
    return dict(duration=float(duration), start=[float(start[0]), float(start[1])], end=[float(end[0]), float(end[1])],
                Q=float(Q), radius=float(radius), amplitude=float(amplitude), frequency=float(frequency),
                jet=[float(jet[0]), float(jet[1])], U_perp=0.4,
                **{k: (v if isinstance(v, list) else float(v)) for k, v in extra.items()})


def snap(T, freq):
    """Round a wiggle duration to whole half-periods so the phase ends on the axis (no spurious bridge)."""
    if freq <= 0:
        return T
    return max(1, round(T * 2 * freq)) / (2 * freq)


def cut(y0, y1, T, Q, r0=0.004, r1=0.0015, taper=0.12, jet=1.8):
    return phase(T, (0, y0), (0, y1), Q, r0, jet=(0, jet), Q_end=taper * Q, radius_end=r1)


def stop(y, x=0.0, T=0.7):
    return phase(T, (x, y), (x, y), 0.0, 0.035, jet=(0, 0))


def config(name, phases, physics=None):
    return dict(name=name, action=name, parameters=dict(physics or V05_PHYSICS), phases=phases,
                inlet_geometry="continuity_ellipse", continuous_actions=True, max_jet_angle_degrees=60.0,
                reposition_duration=0.18, return_law={"mode": "return_speed"},
                source="zefeng_parametric_cmaes", phase_flow_units="effective_input_rate_before_chi")


# ----------------------------------------------------------------------------- patterns
HEART = Space(y0=(-0.25, 0.10), dy1=(-0.15, 0.15), T1=(2.5, 5.0), Q1=(0.03, 0.09), r1=(0.015, 0.035), j1=(0.3, 1.5),
              T2=(0.5, 2.0), Q2=(0.015, 0.06), dy2=(0.0, 0.3), r2=(0.012, 0.03),
              T3=(0.5, 1.2), Q3=(0.006, 0.03), yc=(0.20, 0.44))


def heart(p):
    y1 = p["y0"] + p["dy1"]; y2 = y1 + p["dy2"]
    return config("heart", [
        phase(p["T1"], (0, p["y0"]), (0, y1), p["Q1"], p["r1"], jet=(0, p["j1"])),
        phase(p["T2"], (0, y1), (0, y2), p["Q2"], p["r2"], jet=(0, p["j1"])),
        cut(y2, p["yc"], p["T3"], p["Q3"]),
        stop(p["yc"])])


PUSH_HEART = Space(ya=(-0.30, 0.0), push1=(0.0, 0.25), Tb1=(0.6, 1.8), Qb1=(0.04, 0.10), rb=(0.010, 0.025), j=(0.5, 1.5),
                   gap2=(0.02, 0.15), push2=(0.0, 0.20), Tb2=(0.5, 1.5), Qb2=(0.03, 0.09),
                   gap3=(0.02, 0.15), push3=(0.0, 0.15), Tb3=(0.4, 1.2), Qb3=(0.02, 0.08),
                   Tc=(0.6, 1.4), Qc=(0.006, 0.03), yc=(0.20, 0.44))


def push_heart(p):
    y1 = p["ya"]; y2 = y1 - p["gap2"]; y3 = y2 - p["gap3"]
    return config("push_heart", [
        phase(p["Tb1"], (0, y1), (0, y1 + p["push1"]), p["Qb1"], p["rb"], jet=(0, p["j"])),
        phase(p["Tb2"], (0, y2), (0, y2 + p["push2"]), p["Qb2"], 0.95 * p["rb"], jet=(0, p["j"])),
        phase(p["Tb3"], (0, y3), (0, y3 + p["push3"]), p["Qb3"], 0.85 * p["rb"], jet=(0, p["j"])),
        cut(y3, p["yc"], p["Tc"], p["Qc"]),
        stop(p["yc"])])


_WIGGLE = dict(ya=(-0.20, 0.10), dy=(-0.15, 0.10), Tw=(2.5, 6.0), amp0=(0.004, 0.03), amp1=(0.02, 0.07),
               freq=(1.5, 4.0), Q0=(0.01, 0.04), Q1=(0.03, 0.08), r=(0.02, 0.05), fmod=(0.0, 1.0),
               jamp=(0.0, 2.0), jy=(0.3, 1.5), Tw2=(0.0, 4.0))


def wiggle_base(p):
    """Growing wiggle while drifting, then a steady wiggle at the end point.  Returns phases and the end y."""
    Tw = snap(p["Tw"], p["freq"]); Tw2 = snap(p["Tw2"], p["freq"]) if p["Tw2"] > 0.3 else 0.0
    ye = p["ya"] + p["dy"]
    ph = [phase(Tw, (0, p["ya"]), (0, ye), p["Q0"], p["r"], jet=(0, p["jy"]), amplitude=p["amp0"], amplitude_end=p["amp1"],
                frequency=p["freq"], Q_end=p["Q1"], flow_modulation=p["fmod"], jet_amplitude_x=p["jamp"])]
    if Tw2 > 0:
        ph.append(phase(Tw2, (0, ye), (0, ye), p["Q1"], p["r"], jet=(0, p["jy"]), amplitude=p["amp1"], frequency=p["freq"],
                        flow_modulation=p["fmod"], jet_amplitude_x=p["jamp"]))
    return ph, ye


LAYERED_HEART = Space(**_WIGGLE, Tc=(0.6, 1.4), Qc=(0.006, 0.03), yc=(0.20, 0.44))


def layered_heart(p):
    ph, ye = wiggle_base(p)
    return config("layered_heart", ph + [cut(ye, p["yc"], p["Tc"], p["Qc"]), stop(p["yc"])])


TULIP = Space(**_WIGGLE, gap2=(0.03, 0.15), push2=(0.0, 0.10), T2=(0.5, 1.8), Qp2=(0.02, 0.07), rp2=(0.015, 0.04),
              amp2=(0.0, 0.04), gap3=(0.03, 0.15), push3=(0.0, 0.10), T3=(0.3, 1.2), Qp3=(0.01, 0.05),
              rp3=(0.010, 0.025), amp3=(0.0, 0.03), Tc=(0.6, 1.4), Qc=(0.006, 0.03), yc=(0.20, 0.44))


def tulip(p):
    ph, ye = wiggle_base(p)
    y2 = ye - p["gap2"]; y3 = y2 - p["gap3"]
    T2 = snap(p["T2"], p["freq"]) if p["amp2"] > 0.003 else p["T2"]
    T3 = snap(p["T3"], p["freq"]) if p["amp3"] > 0.003 else p["T3"]
    ph += [phase(T2, (0, y2), (0, y2 + p["push2"]), p["Qp2"], p["rp2"], jet=(0, p["jy"]), amplitude=p["amp2"],
                 frequency=p["freq"], flow_modulation=p["fmod"], jet_amplitude_x=p["jamp"]),
           phase(T3, (0, y3), (0, y3 + p["push3"]), p["Qp3"], p["rp3"], jet=(0, 0.25), amplitude=p["amp3"],
                 frequency=p["freq"]),
           cut(y3 - 0.03, p["yc"], p["Tc"], p["Qc"]), stop(p["yc"])]
    return config("tulip", ph)


LEAF = Space(ya=(-0.10, 0.20), Tw1=(1.5, 5.0), ampb=(0.01, 0.05), freq=(1.5, 4.0), Qb=(0.02, 0.06), rb=(0.02, 0.05),
             jy=(0.3, 1.5), fmod=(0.0, 1.0), jamp=(0.0, 2.0),
             Tr=(1.5, 4.0), dist=(0.10, 0.40), ampr=(0.03, 0.10), Qr0=(0.02, 0.06), Qr1=(0.005, 0.03),
             rr0=(0.005, 0.020), rr1=(0.004, 0.012), jr=(0.2, 1.2), fmodr=(0.0, 1.0),
             Te=(0.1, 0.8), Qe=(0.01, 0.04), Tc=(0.6, 1.4), Qc=(0.006, 0.03), yc=(0.20, 0.44))


def leaf(p):
    Tw1 = snap(p["Tw1"], p["freq"]); Tr = snap(p["Tr"], p["freq"]); Te = snap(p["Te"], p["freq"])
    ye = p["ya"] - p["dist"]
    return config("leaf", [
        phase(Tw1, (0, p["ya"]), (0, p["ya"]), 0.5 * p["Qb"], p["rb"], jet=(0, p["jy"]), amplitude=0.3 * p["ampb"],
              amplitude_end=p["ampb"], frequency=p["freq"], Q_end=p["Qb"], flow_modulation=p["fmod"], jet_amplitude_x=p["jamp"]),
        phase(Tr, (0, p["ya"]), (0, ye), p["Qr0"], p["rr0"], jet=(0, p["jr"]), amplitude=p["ampr"], frequency=p["freq"],
              Q_end=p["Qr1"], radius_end=p["rr1"], flow_modulation=p["fmodr"]),
        phase(Te, (0, ye), (0, ye - 0.01), p["Qe"], 0.018, jet=(0, 0.35), amplitude=0.4 * p["ampr"], frequency=p["freq"]),
        cut(ye - 0.04, p["yc"], p["Tc"], p["Qc"], r0=0.0035),
        stop(p["yc"])])


SWAN = Space(xb=(-0.15, 0.15), yb=(-0.20, 0.20), Tw1=(1.5, 4.0), ampb=(0.01, 0.05), Qb=(0.02, 0.06), rb=(0.02, 0.05),
             dx=(-0.40, 0.40), dy=(-0.40, 0.10), Tr=(1.5, 4.0), ampr=(0.03, 0.10), Qr0=(0.015, 0.05), Qr1=(0.005, 0.02),
             rr=(0.006, 0.016), jr=(0.2, 1.2),
             Twe=(0.3, 1.0), Qwe=(0.005, 0.02),
             xh=(-0.40, 0.40), yh=(0.0, 0.40), a1=(-0.3, 0.3), b1=(-0.3, 0.3), a2=(-0.3, 0.3), b2=(-0.3, 0.3),
             Tn=(0.6, 1.6), Qn=(0.005, 0.02), rn=(0.006, 0.015),
             Th=(0.1, 0.4), Qh=(0.006, 0.02), thb=(-math.pi, math.pi), lb=(0.03, 0.12), Qbk=(0.004, 0.015))


def swan(p):
    f = 2.5
    Tw1 = snap(p["Tw1"], f); Tr = snap(p["Tr"], f)
    b0 = (p["xb"], p["yb"]); be = (p["xb"] + p["dx"], p["yb"] + p["dy"])
    n0 = (p["xb"] - 0.5 * p["dx"] * 0.3, p["yb"] - 0.5 * p["dy"] * 0.3)  # neck base: just behind the body start
    head = (p["xh"], p["yh"])
    c1 = (n0[0] + p["a1"], n0[1] + p["b1"]); c2 = (head[0] + p["a2"], head[1] + p["b2"])
    beak_end = (head[0] + p["lb"] * math.cos(p["thb"]), head[1] + p["lb"] * math.sin(p["thb"]))
    jr = p["jr"]; nrm = math.hypot(p["dx"], p["dy"]) + 1e-9
    jet_r = (-jr * p["dx"] / nrm, -jr * p["dy"] / nrm)        # lean against the retreat (pushes the arcs forward)
    jet_we = (1.2 * (n0[0] - be[0]) / (math.hypot(n0[0] - be[0], n0[1] - be[1]) + 1e-9),
              1.2 * (n0[1] - be[1]) / (math.hypot(n0[0] - be[0], n0[1] - be[1]) + 1e-9))
    return config("swan", [
        phase(Tw1, b0, b0, 0.5 * p["Qb"], p["rb"], jet=(0, 1.2), amplitude=0.3 * p["ampb"], amplitude_end=p["ampb"],
              frequency=f, Q_end=p["Qb"], flow_modulation=0.85, jet_amplitude_x=1.4),
        phase(Tr, b0, be, p["Qr0"], p["rr"], jet=jet_r, amplitude=p["ampr"], frequency=f, Q_end=p["Qr1"],
              flow_modulation=0.6),
        phase(p["Twe"], be, n0, p["Qwe"], 0.0019, jet=jet_we),
        phase(p["Tn"], n0, head, p["Qn"], p["rn"], jet=(0, 0.3), bezier_controls=[list(c1), list(c2)]),
        phase(p["Th"], head, (head[0] + 0.01, head[1] + 0.01), p["Qh"], 0.012, jet=(0.1, 0.1)),
        phase(0.22, head, beak_end, p["Qbk"], 0.0019, jet=(2.0 * math.cos(p["thb"]), 2.0 * math.sin(p["thb"])),
              Q_end=0.2 * p["Qbk"], radius_end=0.0012),
        stop(beak_end[1], x=beak_end[0])])


SPACES = dict(heart=HEART, push_heart=PUSH_HEART, layered_heart=LAYERED_HEART, tulip=TULIP, leaf=LEAF, swan=SWAN)
BUILDERS = dict(heart=heart, push_heart=push_heart, layered_heart=layered_heart, tulip=tulip, leaf=leaf, swan=swan)


def build(name, x=None, params=None, physics=None):
    """x in [0,1]^n (or a physical parameter dict) -> (Program, config, physical dict)."""
    sp = SPACES[name]
    p = sp.decode(x) if params is None else dict(params)
    cfg = BUILDERS[name](p)
    if physics:
        cfg["parameters"] = dict(physics)
    return Program(cfg), cfg, p


class ZScript:
    """(Inlet, label) sampler for run.py."""

    def __init__(self, name, x=None, params=None, physics=None):
        self.prog, self.config, self.p = build(name, x, params, physics)
        self.name, self.T = name, self.prog.duration
        self.params = dict(self.config["parameters"]); self.params["return_law"] = "v05"

    def sample(self, t):
        from .solver import Inlet
        if t >= self.T:
            return Inlet(active=False), "settle"
        return self.prog(t), self.prog.phase_label(t)


# ----------------------------------------------------------------------------- physics as search variables
# Own search box for the ten shared coefficients (+ the return-law switch); log scale where the range spans decades.
# Nothing here is taken from V0.5: the optimiser starts at the geometric/arithmetic middle of each range.
PHYSICS_BOUNDS = OrderedDict(
    cp=(0.1, 1.5, "log"), beta=(0.5, 10.0, "log"), nu=(1e-4, 1e-2, "log"), D=(1e-8, 1e-5, "log"),
    kappa_Q=(20.0, 400.0, "log"), kappa_c=(0.2, 3.0, "log"), kappa_t=(0.1, 1.5, "lin"), kappa_r=(0.0, 1.0, "lin"),
    B_dep=(50.0, 1500.0, "log"), p_dep=(1.0, 4.0, "lin"), return_law=(0.0, 1.0, "switch"))


class PhysicsSpace:
    n = len(PHYSICS_BOUNDS)
    names = list(PHYSICS_BOUNDS)

    def decode(self, x):
        x = np.clip(np.asarray(x, float), 0, 1)
        out = {}
        for (k, (lo, hi, kind)), xi in zip(PHYSICS_BOUNDS.items(), x):
            if kind == "log":
                out[k] = float(lo * (hi / lo) ** xi)
            elif kind == "switch":
                out[k] = "v1" if xi < 0.5 else "v05"
            else:
                out[k] = float(lo + (hi - lo) * xi)
        return out

    def encode(self, d):
        x = []
        for k, (lo, hi, kind) in PHYSICS_BOUNDS.items():
            v = d[k]
            if kind == "log":
                x.append(math.log(v / lo) / math.log(hi / lo))
            elif kind == "switch":
                x.append(0.25 if v == "v1" else 0.75)
            else:
                x.append((v - lo) / (hi - lo))
        return np.clip(np.array(x), 0, 1)


PHYSICS = PhysicsSpace()
