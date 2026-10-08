"""Checks for the level-0 pitcher model:  python -m latte_imex.test_pitcher
  1. capacity of the parametric pitcher vs the analytic frustum volume (within the sampling error)
  2. weir integral vs the closed-form V-notch law at a prescribed head
  3. volume bookkeeping: poured volume == integral of Q dt
  4. ballistic hit: free-fall time and continuity thinning
  5. steady tilt-to-flow inversion round trip
"""
import math
import numpy as np
from .pitcher import Pitcher, Pose, PitcherGeometry, BaristaScript, records_table, G


def frustum(r0, r1, h):
    return math.pi * h / 3 * (r0 * r0 + r0 * r1 + r1 * r1)


def main():
    ok = True
    P = Pitcher()
    g = P.geom
    # 1. capacity
    V_an = frustum(g.r_bottom, g.r_belly, g.z_belly) + frustum(g.r_belly, g.r_top, g.height - g.z_belly)
    # minus the notch opening (small): accept 1.5 % error
    err = abs(g.capacity - V_an) / V_an
    print(f"[1] capacity {g.capacity*1e6:.1f} mL vs frustum {V_an*1e6:.1f} mL  err {err*100:.2f}%")
    ok &= err < 0.015
    # 2. weir vs V-notch closed form: pitcher upright, plane at height c above the apex
    pose = Pose(tip=(0.0, 0.0, 0.0), yaw=0.0, tilt=0.0)
    for h in (0.004, 0.008, 0.012):
        W = P.weir(pose, h)   # tip is the apex, at world z = 0
        theta2 = math.atan(g.r_top * g.notch_half_angle / g.notch_depth)   # half opening angle in the unrolled rim
        Q_cf = (8 / 15) * P.law.C_d * math.sqrt(2 * G) * math.tan(theta2) * max(h - P.law.h_k, 0) ** 2.5
        rel = W["Q_ss"] / Q_cf - 1
        print(f"[2] h {h*1e3:4.1f} mm  weir {W['Q_ss']*1e6:6.2f} mL/s  V-notch {Q_cf*1e6:6.2f}  rel {rel:+.3f}")
        ok &= abs(rel) < 0.25      # the beak widens the notch and the rim is curved: same law, not the same number
    # 3. bookkeeping
    S = BaristaScript(); S.sample(S.T)
    T = records_table(S.records)
    dt = np.diff(T["t"], prepend=0.0)
    poured = float((T["Q"] * dt).sum())
    V0 = S.pitcher.state.V + poured
    print(f"[3] poured {poured*1e6:.2f} mL  == V0 - V_end {(200e-6 - T['V'][-1])*1e6:.2f} mL")
    ok &= abs(poured - (200e-6 - T["V"][-1])) < 1e-9
    # 4. ballistic
    pose = Pose(tip=(0.0, 0.0, 0.05), yaw=0.0, tilt=math.radians(60))
    J = P.jet(pose, 15e-6, 0.007, np.array([0.0, 0.0, 0.05]))
    t_ff = math.sqrt(2 * 0.05 / G)
    print(f"[4] u0 {J['u0']:.3f} m/s  t_hit {J['t_hit']:.3f} s (free fall {t_ff:.3f})  d0 {J['d0']*1e3:.2f} -> d_hit {J['d_hit']*1e3:.2f} mm  U_perp {J['U_perp']:.2f}")
    ok &= J["t_hit"] < t_ff and abs(J["d_hit"] / J["d0"] - math.sqrt(J["u0"] / J["speed"])) < 1e-9
    ok &= abs(J["U_perp"] ** 2 - ((J["v0"][2]) ** 2 + 2 * G * 0.05)) < 1e-6
    # 5. inversion
    pose = Pose(tip=(0.0, 0.0, 0.03), yaw=-math.pi / 2, tilt=0.0)
    th = P.tilt_for_flow(pose, 15e-6, V=200e-6)
    pose.tilt = th
    c, _ = P.free_surface(pose, 200e-6)
    Q = P.weir(pose, c)["Q_ss"]
    print(f"[5] tilt for 15 mL/s at 200 mL: {math.degrees(th):.2f} deg -> Q {Q*1e6:.2f} mL/s")
    ok &= abs(Q - 15e-6) < 0.05e-6
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
