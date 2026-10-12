"""drvf.py -- final all-physical driver (built on ref2/drv.py; nothing in latte_imex / v05audit / allphys edited).

Physical inlet for every flowing stroke k of Codex's control (paths, timings, wiggles kept):
  heights / closure  per-stroke U_perp = sqrt(0.35^2 + 2 g z_k) from an inverse design (allphys_experiments.json or fits.json)
  flow               Codex flow F (taper, pulsation depth capped, bridges >= 0.3 s, 0.1 s on/off ramps)
                     --lag tau : the stroke-level envelope (taper x ramps, bridges = 0) passes a first-order spout lag tau
                                 (pulsation is applied after the lag, at its capped depth = what pulse_tau says is reachable);
                                 the flow therefore leaks into bridges and changes smoothly between strokes
                     --stop    : F/kappa_Q < 1 mL/s -> no coherent stream (inlet off) instead of raising it to 1 mL/s
                                 (without --stop a commanded stroke keeps >= 1 mL/s, as in allphys; a lag leak into a
                                  bridge/stop is always cut below 1 mL/s)
  Q                  = clip(F/kappa_Q, 1, 32 mL/s), S_eff = kappa_Q Q
  height transitions --ztr : between two flowing strokes without a bridge the spout height moves linearly at <= 0.11 m/s
                     (window >= 0.05 s, centred on the boundary); across a bridge z moves linearly; leaks use these z
  horizontal speed   --lat scale    : allphys (|Codex jet + hand-speed wiggle| clipped to [0.10, min(0.40, U)], whole vector scaled)
                     --lat latfix   : ref2 (only the forward part raised to the floor; lateral = hand speed)
                     --lat exithand : u_h = e_b clip(|b|, 0.10, 0.27) + v_hand  (exit part along Codex's jet direction b,
                                      v_hand = d x_hit/dt, full hand velocity), |u_h| <= min(0.40, U) (tan alpha <= 1)
                     --lat exitmin  : (final) u_h = v_hand + e_b clip(|b| - v_hand.e_b, 0.10, 0.27): Codex's cup-frame jet kept
                                      unless the pitcher-frame exit part along e_b would be < 0.10 m/s; lateral = hand speed
  footprint          continuity ellipse r = sqrt(Q/(pi U)), aspect |v|/U, d = 2 sqrt(Q/(|v| pi))
Diagnostics (NOT physical, labelled diag_*): --diag_uh k:m/s (force |u_h| of stroke k), --diag_r k:mm (force footprint),
  --kr_scale f (global kappa_r x f).
usage: python3 -I drvf.py <tag> <pattern> [--exp P2 | --fit F_p4] [--cp 1.36] [--beta 3] [--nu 1e-3] [--pulse 0.41]
                          [--flow_phase 0|keep] [--lat exithand] [--lag 0.08] [--stop] [--ztr] [--zset k:cm,...] [--snaps]
"""
import argparse, copy, json, math, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
REPO = "/home/user/zfChen"
sys.path.insert(0, REPO)
import numpy as np
from latte_imex.v05_program import Program, load_control
from latte_imex.solver import Inlet
import latte_imex.run as R

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
D_M, G, ML, V0 = 0.08, 9.81, 512.0, 0.35


def U_of_z(z):
    return math.sqrt(V0 * V0 + 2 * G * z)


def z_of_U(U):
    return (U * U - V0 * V0) / (2 * G)


def pclass(p):
    if p.get("purpose"):
        return "bridge"
    if max(p.get("Q", 0.0), p.get("Q_end", p.get("Q", 0.0))) <= 0:
        return "stop"
    return "cut" if p["radius"] <= 0.006 else "low"


def wiggly(p):
    return p.get("frequency", 0.0) > 0 and max(abs(p.get("amplitude", 0.0)), abs(p.get("amplitude_end", p.get("amplitude", 0.0)))) > 0


def kv(s, f=float):
    out = {}
    for it in (s or "").split(","):
        if it:
            a, b = it.split(":")
            out[int(a)] = f(b)
    return out


class PhysF(Program):
    def __init__(self, pattern, U_ms, kQ, params, pulse_cap, flow_phase=None, bridge=0.3, ramp=0.1, lat="exithand",
                 lag=0.0, stop=False, ztr=False, zset=None, diag_uh=None, diag_r=None):
        cfg = copy.deepcopy(load_control(pattern))
        for p in cfg["phases"]:
            if p.get("purpose"):
                p["duration"] = max(p["duration"], bridge)
            if "flow_modulation" in p:
                p["flow_modulation"] = min(p["flow_modulation"], pulse_cap)
                if flow_phase is not None:
                    p["flow_phase"] = flow_phase
        cfg["parameters"].update(params)
        super().__init__(cfg)
        self.kQ = kQ
        self.ramp_s = ramp
        self.lat, self.lag, self.stop, self.ztr = lat, lag, stop, ztr
        self.cls = [pclass(p) for p in self.phases]
        self.z = {}
        for k, c in enumerate(self.cls):
            if c in ("low", "cut"):
                self.z[k] = z_of_U(U_ms[str(k)])
        for k, zc in (zset or {}).items():
            self.z[k] = zc / 100.0
        self.diag_uh = diag_uh or {}
        self.diag_r = diag_r or {}
        fl = [k for k, c in enumerate(self.cls) if c in ("low", "cut")]
        self.prev_fl = {k: max([j for j in fl if j <= k], default=None) for k in range(len(self.phases))}
        self.next_fl = {k: min([j for j in fl if j >= k], default=None) for k in range(len(self.phases))}
        if lag > 0:
            dt = 5e-4
            tg = np.arange(0.0, self.duration + dt, dt)
            E = np.array([self.envelope(t) for t in tg])
            a = 1.0 - math.exp(-dt / lag)
            Ef = np.empty_like(E); y = 0.0
            for i, e in enumerate(E):
                y += a * (e - y); Ef[i] = y
            self.tg, self.Ef, self.Ecmd = tg, Ef, E
        self.ncall = 0
        self.log = []

    def gain(self, k, tt):
        if self.cls[k] not in ("low", "cut"):
            return 0.0
        g, dur = 1.0, self.phases[k]["duration"]
        if k == 0 or self.cls[k - 1] in ("bridge", "stop"):
            g *= min(1.0, tt / self.ramp_s)
        if k == len(self.phases) - 1 or self.cls[k + 1] in ("bridge", "stop"):
            g *= min(1.0, max(dur - tt, 0.0) / self.ramp_s)
        return g

    def envelope(self, t):
        """commanded stroke-level flow (taper x ramps), no pulsation; 0 in bridges/stops"""
        k = self.phase_index(t); p = self.phases[k]
        if self.cls[k] not in ("low", "cut"):
            return 0.0
        tt = max(0.0, min(p["duration"], t - self.events[k]))
        f = tt / p["duration"]
        return (p["Q"] * (1 - f) + p.get("Q_end", p["Q"]) * f) * self.gain(k, tt)

    def z_at(self, t, k, tt):
        c = self.cls[k]
        if c not in ("low", "cut"):
            a, b = self.prev_fl[k], self.next_fl[k]
            if a is None:
                return self.z[b]
            if b is None or not self.ztr:
                return self.z[a]
            return self.z[a] + (self.z[b] - self.z[a]) * tt / self.phases[k]["duration"]
        z = self.z[k]
        if not self.ztr:
            return z
        dur = self.phases[k]["duration"]
        for nb, tb in ((k - 1, 0.0), (k + 1, dur)):
            if 0 <= nb < len(self.phases) and self.cls[nb] in ("low", "cut"):
                za, zb = (self.z[nb], z) if nb < k else (z, self.z[nb])
                hw = 0.5 * max(abs(zb - za) / 0.11, 0.05)
                if abs(tt - tb) < hw:
                    s = ((tt - tb) + hw) / (2 * hw)          # 0 at tb - hw, 1 at tb + hw
                    return za + (zb - za) * s
        return z

    def hand_v(self, t):
        d = 1e-3
        t0, t1 = max(t - d, 0.0), min(t + d, self.duration - 1e-9)
        x0 = Program.__call__(self, t0).x_hit; x1 = Program.__call__(self, t1).x_hit
        return (x1[0] - x0[0]) / (t1 - t0), (x1[1] - x0[1]) / (t1 - t0)

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
                fac = 1 + p["flow_modulation"] * math.cos(2 * math.pi * p.get("flow_frequency", p["frequency"]) * tt + p.get("flow_phase", 0.0))
            F = E * fac
        else:
            F = (inl0.S_eff if inl0.active else 0.0) * self.gain(k, tt)
        off = Inlet(active=False, x_hit=inl0.x_hit, S_eff=0.0, scan_speed=inl0.scan_speed, r1=inl0.r1, r2=inl0.r2)
        if F <= 0:
            return off
        Qraw = F / self.kQ
        if Qraw < 1.0 / ML and (self.stop or self.cls[k] not in ("low", "cut")):
            return off                     # --stop: no coherent stream below 1 mL/s; a lag leak into a bridge/stop always breaks
        Q = min(max(Qraw, 1.0 / ML), 32.0 / ML)
        ks = k if self.cls[k] in ("low", "cut") else (self.prev_fl[k] if self.prev_fl[k] is not None else self.next_fl[k])
        z = self.z_at(t, k, tt)
        U = U_of_z(z) / D_M
        ps = self.phases[ks]
        bx, by = ps["jet"]
        hi = min(0.40 / D_M, 1.0 * U)
        vh = (0.0, 0.0); osc = 0.0
        if self.lat == "exitmin":
            # minimal change: Codex's jet b is the cup-frame material speed; the spout exit part (pitcher frame) is
            # b - v_hand projected on Codex's jet direction (the spout does not swing sideways), floored at 0.10 and capped
            # at 0.27 m/s; cup frame u = v_hand + exit  (lateral = hand speed exactly; |u| = |b| whenever the floor is inactive)
            vh = self.hand_v(t)
            nb = math.hypot(bx, by)
            ex, ey = (bx / nb, by / nb) if nb > 0 else (0.0, -1.0)
            ue = min(max(nb - (vh[0] * ex + vh[1] * ey), 0.10 / D_M), 0.27 / D_M)
            jx, jy = ue * ex + vh[0], ue * ey + vh[1]
            v = math.hypot(jx, jy)
            if v > hi:
                jx, jy, v = jx * hi / v, jy * hi / v, hi
        elif self.lat == "exithand":
            vh = self.hand_v(t)
            nb = math.hypot(bx, by)
            ex, ey = (bx / nb, by / nb) if nb > 0 else (0.0, -1.0)
            ue = min(max(nb, 0.10 / D_M), 0.27 / D_M)
            jx, jy = ue * ex + vh[0], ue * ey + vh[1]
            v = math.hypot(jx, jy)
            if v > hi:
                jx, jy, v = jx * hi / v, jy * hi / v, hi
        else:
            if ks == k and wiggly(p):
                f = tt / p["duration"]
                amp = p["amplitude"] * (1 - f) + p.get("amplitude_end", p["amplitude"]) * f
                osc = 2 * math.pi * p["frequency"] * amp * math.cos(2 * math.pi * p["frequency"] * tt)
            jx, jy = bx + osc, by
            vraw = math.hypot(jx, jy)
            lo = 0.10 / D_M
            v = min(max(vraw, lo), hi)
            if self.lat == "latfix" and vraw < lo and abs(osc) > 0 and math.hypot(bx, by) > 0:
                bb = bx * bx + by * by; bo = bx * osc
                s = (-bo + math.sqrt(bo * bo - bb * (osc * osc - lo * lo))) / bb
                jx, jy = s * bx + osc, s * by
            elif vraw > 0:
                jx, jy = jx * v / vraw, jy * v / vraw
            else:
                jx, jy = 0.0, -v
        if ks in self.diag_uh:
            v0 = math.hypot(jx, jy); v = self.diag_uh[ks] / D_M
            jx, jy = (jx * v / v0, jy * v / v0) if v0 > 0 else (0.0, -v)
        v = math.hypot(jx, jy)
        spd = math.hypot(U, v)
        aspect = spd / U
        r = math.sqrt(Q / (math.pi * U))
        d = 2 * math.sqrt(Q / spd / math.pi)
        if ks in self.diag_r:
            r = self.diag_r[ks] / 1e3 / D_M; d = 2 * r
        out = Inlet(active=True, x_hit=inl0.x_hit, S_eff=self.kQ * Q, u_in=(jx, jy), U_perp=U, d_jet=d,
                    r1=r * math.sqrt(aspect), r2=r / math.sqrt(aspect), phi=math.atan2(jy, jx), scan_speed=inl0.scan_speed)
        ex_speed = math.hypot(jx - vh[0], jy - vh[1]) * D_M if self.lat in ("exithand", "exitmin") else float("nan")
        self.log.append((t, self.ncall % 2, k, ks, Qraw * ML, Q * ML, U * D_M, z * 100, v * D_M, jx * D_M, jy * D_M,
                         vh[0] * D_M, vh[1] * D_M, ex_speed, r * D_M * 1e3, d * D_M * 1e3, F, osc * D_M))
        return out


LOGCOLS = "t parity k ks Qraw_mLs Q_mLs U_ms z_cm uh_ms ux_ms uy_ms vhx_ms vhy_ms exit_ms r_mm d_mm F osc_ms".split()


def load_ref(p):
    z = np.load(os.path.join(REPO, "runs", "v05_256", p, "final_state.npz"))
    mask = z["mask"]
    return np.where(mask, z["m"] / np.where(mask, z["l"], 1.0), np.nan), mask


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("pattern")
    ap.add_argument("--exp", default="P2")
    ap.add_argument("--fit", default=None, help="take U and (B, p, kappa_r) from fits.json[<name>]")
    ap.add_argument("--cp", type=float, default=1.36)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--nu", type=float, default=1e-3)
    ap.add_argument("--pulse", type=float, default=0.41)
    ap.add_argument("--flow_phase", default="keep")
    ap.add_argument("--lat", default="exitmin", choices=["scale", "latfix", "exithand", "exitmin"])
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--ramp", type=float, default=0.1)
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--ztr", action="store_true")
    ap.add_argument("--zset", default="")
    ap.add_argument("--diag_uh", default="")
    ap.add_argument("--diag_r", default="")
    ap.add_argument("--kr_scale", type=float, default=1.0)
    ap.add_argument("--snaps", action="store_true")
    ap.add_argument("--N", type=int, default=256)
    a = ap.parse_args()
    if a.fit:
        e = json.load(open(os.path.join(HERE, "fits.json")))[a.fit]
    else:
        e = json.load(open(os.path.join(TOP, "allphys_experiments.json")))[a.exp]
    params = dict(kappa_Q=2.0, cp=a.cp, beta=a.beta, nu=a.nu)
    for key in ("B_dep", "p_dep", "kappa_r"):
        params[key] = e["params"][key]
    params["kappa_r"] *= a.kr_scale
    fp = None if a.flow_phase == "keep" else float(a.flow_phase)
    spec = e["inlet"][a.pattern]
    prog = PhysF(a.pattern, spec["U"], spec["kappa_Q"], params, a.pulse, fp, ramp=a.ramp, lat=a.lat, lag=a.lag, stop=a.stop,
                 ztr=a.ztr, zset=kv(a.zset), diag_uh=kv(a.diag_uh), diag_r=kv(a.diag_r))
    R.v05_program = lambda name: prog
    out = os.path.join(HERE, "runs", f"{a.tag}_{a.pattern}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sol, m = R.run(a.pattern, N=a.N, out=out, video=False, quiet=True, control="v05")
    wall = time.time() - t0
    c = sol.concentration(); mask = sol.g.mask.copy()
    np.savez_compressed(os.path.join(out, "c.npz"), c=c.astype(np.float32), mask=mask)
    np.save(os.path.join(out, "inlet_log.npy"), np.array(prog.log))
    if not a.snaps:
        try:
            os.remove(os.path.join(out, "snapshots_c.npz"))
        except OSError:
            pass
    try:
        os.remove(os.path.join(out, "final_state.npz"))
    except OSError:
        pass
    cr, mr = load_ref(a.pattern)
    res = dict(tag=a.tag, pattern=a.pattern, src=a.fit or a.exp, params=params, args=vars(a),
               wall_s=wall, stats=m["stats"], ledger=m["ledger"], T=m["T"],
               iou04=iou((c > 0.4) & mask, (cr > 0.4) & mr), iou05=iou((c > 0.5) & mask, (cr > 0.5) & mr),
               white=float(((c > 0.4) & mask).sum() / mask.sum()), white_ref=float(((cr > 0.4) & mr).sum() / mr.sum()))
    json.dump(res, open(os.path.join(out, "res.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("params", "stats", "ledger", "args")}, default=float))


if __name__ == "__main__":
    main()
