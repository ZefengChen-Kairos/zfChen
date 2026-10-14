"""Checks for the forward pose-driven pitcher (pitcher.ForwardPitcher, V1 mode B) and the emission-time inlet law
(v05_pourable.emission_kinematics / PourF vtime):  python -m latte_imex.test_forward
  1. hydrostatic: NX pitcher, tilt just below the lip-wetting tilt -> Q = 0; just above -> Q > 0
  2. straight lip: Q = 2/3 C_d sqrt(2 g) b h^1.5 (exact), Q(h2)/Q(h1) = (h2/h1)^1.5
  3. projected width of a straight lip perpendicular to the flow = its chord (any yaw); arc width = chord too, a
     V-shaped lip's projected width < its arc length
  4. volume ledger V_P + V_jet + V_cup + V_spill = V0 to round-off over a moving, tilting pour; all released milk arrives
  5. omega x r: pure rotation about the tip -> zero tip velocity, nonzero origin velocity = omega x (p_o - tip)
  6. jet arrival time vs the closed form t_f = (v_z + sqrt(v_z^2 + 2 g z0)) / g; first arrival of a steady stream
  7. emission-time law: uniform hand motion -> same velocity and landing point as the landing-time law; 2.5 Hz wiggle
     -> phase lag of the carried velocity = omega t_flight
  8. (if the S1 design data exist) PourF vtime 'emission' vs 'landing' on the layered-heart wiggle: measured phase lag
"""
import math
import os

import numpy as np

from .pitcher import ForwardLaw, ForwardPitcher, Pose, G, rotation


class BoxGeometry:
    """Rectangular box [-a, 0] x [-b/2, b/2] x [0, H] with a straight front lip along x = 0 (the 'spout tip' at its
    middle).  Rim polyline counter-clockwise; front mask = the points on x = 0."""

    def __init__(self, a=0.06, b=0.03, H=0.08, nf=61, n=24):
        self.lip_slope = 0.0
        xs = (np.arange(n) + 0.5) / n * a - a
        ys = (np.arange(n) + 0.5) / n * b - b / 2
        zs = (np.arange(n) + 0.5) / n * H
        X, Y, Zz = np.meshgrid(xs, ys, zs, indexing="ij")
        self.points = np.stack([X, Y, Zz], -1).reshape(-1, 3)
        self.dv = np.full(len(self.points), a * b * H / n ** 3)
        fy = np.linspace(-b / 2, b / 2, nf)                       # front edge x = 0, y increasing (counter-clockwise)
        bx = np.linspace(0, -a, n + 1)[1:]
        by = np.linspace(b / 2, -b / 2, nf)[1:]
        rx = np.linspace(-a, 0, n + 1)[1:-1]
        rim = np.concatenate([np.c_[np.zeros(nf), fy], np.c_[bx, np.full(n, b / 2)], np.c_[np.full(nf - 1, -a), by],
                              np.c_[rx, np.full(n - 1, -b / 2)]])
        self.rim = np.c_[rim, np.full(len(rim), H)]
        self.front = np.zeros(len(rim), bool)
        self.front[:nf] = True
        self.tip = np.array([0.0, 0.0, H])
        self.b, self.H = b, H


def _lockin(t, x, f):
    w = 2 * math.pi * f
    return complex(np.sum(x * np.cos(w * t)), np.sum(x * np.sin(w * t)))


def main():
    ok = True
    from .v05_pourable import loop_geometry
    nx = loop_geometry("nx", 0.0)
    FP = ForwardPitcher(nx)
    print(f"NX front edge: {len(FP.seg_a)} segments, beak axis {math.degrees(FP.beak_phi):+.2f} deg")
    ok &= abs(math.degrees(FP.beak_phi) - 2.5) < 0.5
    # 1. hydrostatic
    V = 100e-6
    pose = Pose((0.0, 0.0, 0.015), -math.pi / 2, 0.0, 0.0)
    th = FP.onset_tilt(pose, V)
    res = []
    for d in (-0.05, 0.3):
        pose.tilt = th + math.radians(d)
        z, _ = FP.free_surface(pose, V)
        W = FP.weir(pose, z)
        res.append(W["Q"])
    print(f"[1] lip-wetting tilt {math.degrees(th):.3f} deg at 100 mL; Q at -0.05 deg {res[0]*1e6:.3g} mL/s, at +0.3 deg "
          f"{res[1]*1e6:.3f} mL/s")
    ok &= res[0] == 0.0 and res[1] > 0
    # 2. straight lip, h^1.5
    g = BoxGeometry()
    B = ForwardPitcher(g, ForwardLaw(), front=g.front)
    K = (2.0 / 3.0) * B.law.C_d * math.sqrt(2 * G)
    Qs = []
    for h in (0.002, 0.004, 0.008):
        W = B.weir(Pose(tuple(g.tip), 0.0, 0.0, 0.0), g.H + h)
        Qs.append(W["Q"])
        rel = W["Q"] / (K * g.b * h ** 1.5) - 1
        print(f"[2] h {h*1e3:.0f} mm: Q {W['Q']*1e6:.3f} mL/s, closed form {K*g.b*h**1.5*1e6:.3f}, rel {rel:+.1e}")
        ok &= abs(rel) < 1e-12
    r1, r2 = Qs[1] / Qs[0], Qs[2] / Qs[1]
    print(f"    Q ratios per head doubling {r1:.6f}, {r2:.6f} (2^1.5 = {2**1.5:.6f})")
    ok &= abs(r1 - 2 ** 1.5) < 1e-9 and abs(r2 - 2 ** 1.5) < 1e-9
    # 3. projected width
    for yaw in (0.0, math.radians(37.0)):
        W = B.weir(Pose((0.01, -0.02, 0.0), yaw, 0.0, 0.0), 0.0 + 0.004)
        print(f"[3] straight lip yaw {math.degrees(yaw):.0f} deg: projected width {W['width']*1e3:.6f} mm, chord {g.b*1e3:.6f} mm,"
              f" e_main {np.round(W['e_main'], 6)}")
        ok &= abs(W["width"] - g.b) < 1e-12
    Barc = ForwardPitcher(g, ForwardLaw(width="arc"), front=g.front)
    ok &= abs(Barc.weir(Pose(tuple(g.tip), 0.0, 0.0, 0.0), g.H + 0.004)["width"] - g.b) < 1e-12
    # V lip (NX beak): projected < arc
    pose = Pose((0.0, 0.0, 0.015), -math.pi / 2, th + math.radians(3.0), 0.0)
    z, _ = FP.free_surface(pose, V)
    wp = FP.weir(pose, z)
    wa = ForwardPitcher(nx, ForwardLaw(width="arc")).weir(pose, z)
    print(f"    NX beak at +3 deg: projected width {wp['width']*1e3:.2f} mm < arc {wa['width']*1e3:.2f} mm; "
          f"Q {wp['Q']*1e6:.2f} vs {wa['Q']*1e6:.2f} mL/s")
    ok &= wp["width"] < wa["width"]
    # 4. ledger
    dt = 1e-3
    t = np.arange(-0.2, 1.5, dt)
    tilt = np.radians(np.clip(62.0 + 14.0 * (t + 0.2) / 1.0, 0, 80.0))
    tip = np.c_[0.004 * np.sin(2 * math.pi * 2.5 * t), 0.02 - 0.01 * t, 0.012 + 0.006 * np.clip(t - 0.9, 0, None)]
    yaw = -math.pi / 2 + 0.2 * np.sin(2 * math.pi * t)
    E = FP.run(t, tip, yaw, tilt, 0.0, V, dt)
    n_out = int((t[-1] + 0.3 - t[0]) / dt)
    R = ForwardPitcher.arrivals(E, t[0], n_out, dt)
    led = R["V_P"] + R["V_jet"] + R["V_cup"] + R["V_spill"] - V
    rel_tot = float(np.sum(E["Q_o"]) * dt)
    print(f"[4] ledger max |V_P+V_jet+V_cup+V_spill-V0| = {np.max(np.abs(led)):.1e} m3 (V0 {V:.0e}); released {rel_tot*1e6:.3f} mL,"
          f" arrived {R['V_cup'][-1]*1e6:.3f} mL, V_P(end) {E['V_P'][-1]*1e6:.3f} mL, spill {R['V_spill'][-1]*1e6:.3g} mL")
    ok &= np.max(np.abs(led)) < 1e-15 and abs(R["V_cup"][-1] - rel_tot) < 1e-15
    ok &= abs(E["V_P"][-1] + rel_tot + float(np.sum(E["Q_spill"]) * dt) - V) < 1e-15
    # 5. omega x r: rotate about the tip at 2 rad/s tilt rate, tip fixed
    t5 = np.arange(0, 0.05, dt)
    th5 = th + math.radians(3.0) + 2.0 * t5
    v5, om5 = ForwardPitcher.pose_rates(t5, np.tile([0.0, 0.0, 0.015], (len(t5), 1)), -math.pi / 2, th5, 0.0)
    i = len(t5) // 2
    pose = Pose((0.0, 0.0, 0.015), -math.pi / 2, float(th5[i]), 0.0)
    z, _ = FP.free_surface(pose, V)
    W0 = FP.weir(pose, z)
    W1 = FP.weir(pose, z, v5[i], om5[i])
    dv = W1["v_o"] - W0["v_o"]
    om_exp = 2.0 * (rotation(-math.pi / 2, 0, 0) @ np.array([0.0, 1.0, 0.0]))
    # flux-weighted omega x (m_i - tip) == omega x (p_o - tip) (linear in m_i)
    exp = np.cross(om5[i], W0["origin"] - np.array(pose.tip))
    print(f"[5] pure rotation: |v_tip| {np.linalg.norm(v5[i]):.1e} m/s, omega {np.round(om5[i], 6)} (expect {np.round(om_exp, 6)}),"
          f" origin velocity {np.round(dv * 1e3, 4)} mm/s = omega x r {np.round(exp * 1e3, 4)} mm/s")
    ok &= np.linalg.norm(v5[i]) < 1e-12 and np.allclose(om5[i], om_exp, rtol=0, atol=1e-5) and np.allclose(dv, exp, atol=1e-12)
    ok &= np.linalg.norm(dv) > 1e-4
    # 6. arrival time
    t6 = np.arange(0.0, 0.3, dt)
    th6 = th + math.radians(3.0)
    E6 = FP.run(t6, np.tile([0.0, 0.0, 0.02], (len(t6), 1)), -math.pi / 2, th6, 0.0, V, dt)
    vz, z0 = E6["v_o"][5, 2], E6["origin"][5, 2]
    tf = (vz + math.sqrt(vz * vz + 2 * G * z0)) / G
    zz = z0 + vz * E6["t_f"][5] - 0.5 * G * E6["t_f"][5] ** 2
    R6 = ForwardPitcher.arrivals(E6, -0.5 * dt, len(t6) + 100, dt)
    first = R6["t"][np.argmax(R6["Q"] > 0)] - dt               # start of the first bin with arrivals
    print(f"[6] t_f {E6['t_f'][5]*1e3:.4f} ms vs closed form {tf*1e3:.4f} ms; z(t_f) {zz:.1e} m; first arrival bin starts "
          f"{(first + 0.5*dt)*1e3:.1f} ms after the stream starts (t_f {tf*1e3:.1f} ms); x_hit - (p_o + v_o t_f) "
          f"{np.max(np.abs(E6['x_hit'][5] - (E6['origin'][5, :2] + E6['v_o'][5, :2] * tf))):.1e} m")
    ok &= abs(E6["t_f"][5] - tf) < 1e-15 and abs(zz) < 1e-15 and abs(first + 0.5 * dt - tf) <= dt
    # 7. emission-time law (synthetic tables)
    from .v05_pourable import emission_kinematics
    tg = np.arange(0.0, 3.0, 5e-4)
    n = len(tg)
    base = dict(t=tg, yaw=np.full(n, -math.pi / 2), tilt=np.full(n, 70.0), roll=np.zeros(n), ox=np.full(n, 0.0017),
                oy=np.full(n, 0.0002), oz=np.full(n, 0.0004), ex=np.zeros(n), ey=np.full(n, -0.1), vz=np.full(n, -0.27),
                zw=np.full(n, 0.012), L=np.zeros(n))
    vx = 0.05
    tab = dict(base, x=vx * tg, y=0.01 - 0.02 * tg)
    Ek = emission_kinematics(tab)
    m = (Ek["t_arr"] > 0.5) & (Ek["t_arr"] < 2.5)
    ex_ = np.max(np.abs(Ek["x"][m] - vx * Ek["t_arr"][m]))
    ev_ = max(np.max(np.abs(Ek["vcx"][m] - vx)), np.max(np.abs(Ek["vcy"][m] + 0.02)))
    print(f"[7] uniform motion: landing point - path(t_arrival) max {ex_:.1e} m; carried velocity - hand velocity max {ev_:.1e} m/s")
    ok &= ex_ < 1e-12 and ev_ < 1e-12
    f, A = 2.5, 0.004
    tab = dict(base, x=A * np.sin(2 * math.pi * f * tg), y=np.full(n, 0.01))
    Ek = emission_kinematics(tab)
    ta = np.arange(0.4, 2.8, 1e-3)
    vc = np.interp(ta, Ek["t_arr"], Ek["vcx"])                 # carried velocity of what lands at ta (emission law)
    vl = 2 * math.pi * f * A * np.cos(2 * math.pi * f * ta)   # hand velocity at the landing time (old law)
    lag = math.degrees(np.angle(_lockin(ta, vc, f)) - np.angle(_lockin(ta, vl, f)))
    lag = (lag + 180) % 360 - 180
    tf7 = float(np.mean(Ek["t_f"]))
    exp7 = math.degrees(2 * math.pi * f * tf7)
    print(f"    2.5 Hz wiggle: phase lag {lag:.3f} deg vs omega t_f {exp7:.3f} deg (t_f {tf7*1e3:.2f} ms)")
    ok &= abs(lag - exp7) < 0.2
    # 8. on the S1 design program (layered heart, stroke #2 wiggle 2.5 Hz)
    try:
        from . import v05_pourable as VP
        F = VP._final_setup("S1")
        pe = VP.build_design_program(F["spec"], F["design"], "layered_heart", kappa_r=F["closure"]["kappa_r"])
        pl = VP.build_design_program(F["spec"], F["design"], "layered_heart", kappa_r=F["closure"]["kappa_r"], vtime="landing")
    except Exception as e:                                         # design data not available
        print(f"[8] skipped ({type(e).__name__}: {e})")
        pe = None
    if pe is not None:
        ts = np.arange(3.6, 6.2, 1e-3)
        ue = np.array([pe(t).u_in[0] for t in ts])
        ul = np.array([pl(t).u_in[0] for t in ts])
        lag8 = (math.degrees(np.angle(_lockin(ts, ue, 2.5)) - np.angle(_lockin(ts, ul, 2.5))) + 180) % 360 - 180
        Ek = VP.emission_kinematics(pe.ptab)
        mm = (Ek["t"] > 3.6) & (Ek["t"] < 6.2)
        tf8 = float(np.mean(Ek["t_f"][mm]))
        amp = abs(_lockin(ts, ue, 2.5)) / abs(_lockin(ts, ul, 2.5))
        print(f"[8] S1 layered heart #2: lateral u_in phase lag {lag8:.2f} deg vs omega t_f {math.degrees(2*math.pi*2.5*tf8):.2f}"
              f" deg (t_f {tf8*1e3:.1f} ms), amplitude ratio {amp:.3f}")
        ok &= abs(lag8 - math.degrees(2 * math.pi * 2.5 * tf8)) < 3.0
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
