"""Port of the V0.5 (Codex) control program: phase list -> Inlet record.

Semantics follow experiments/v1_unified/selected/*.json and the V0.5 `scenarios.Program`:
  * phase fields: duration, start, end, Q (effective input rate before chi, S = Q),
    radius (geometric-mean footprint radius), amplitude/frequency (x wiggle), jet (material
    horizontal velocity), Q_end/radius_end/amplitude_end (linear taper), flow_modulation /
    flow_frequency / flow_phase (flow pulsation), jet_amplitude_x / jet_frequency / jet_phase,
    bezier_controls.
  * continuous_actions: zero-flow bridges (0.18 s) are inserted between disconnected phases.
  * inlet_geometry = continuity_ellipse:  Q_rec = Q/kappa_Q,  U_perp = Q_rec/(pi r^2),
    tangential jet capped at U_perp*tan(60 deg), footprint ellipse aligned with the jet with
    aspect sqrt(|v_hit|/U_perp), A_perp = Q_rec/|v_hit|.
The controls are Codex's guesses (manual_synthetic_observed_inlet), not video measurements.
"""
import copy, json, math, os
from .solver import Inlet

HERE = os.path.join(os.path.dirname(__file__), "v05_controls")
NAMES = ("heart", "push_heart", "layered_heart", "tulip", "leaf", "swan")


def load_control(name):
    with open(os.path.join(HERE, f"{name}.json")) as f:
        return json.load(f)


def _phase(duration, start, end=None, Q=.07, radius=.035, amplitude=0., frequency=0.,
           jet=(0., -.25), U_perp=.4, **extra):
    return dict(duration=duration, start=list(start), end=list(end or start), Q=Q, radius=radius,
                amplitude=amplitude, frequency=frequency, jet=list(jet), U_perp=U_perp, **extra)


class Program:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        phases = self.config["phases"]
        # the selected JSONs already contain the bridges; only add when missing
        if self.config.get("continuous_actions") and not any(p.get("purpose") for p in phases):
            connected = []
            for p in phases:
                if connected:
                    prev = connected[-1]
                    end = list(prev["end"])
                    end[0] += prev.get("amplitude_end", prev["amplitude"]) * math.sin(
                        2 * math.pi * prev["frequency"] * prev["duration"])
                    gap = math.hypot(p["start"][0] - end[0], p["start"][1] - end[1])
                    if gap > 1e-12:
                        bridge = _phase(self.config.get("reposition_duration", .18), end, p["start"], Q=0, jet=(0., 0.))
                        bridge["purpose"] = "assumed_zero_flow_reposition"
                        connected.append(bridge)
                connected.append(p)
            self.config["phases"] = connected
        self.phases = self.config["phases"]
        self.events = [0.]
        for p in self.phases:
            self.events.append(self.events[-1] + p["duration"])
        self.duration = self.events[-1]
        self.kappa_Q = self.config["parameters"]["kappa_Q"]
        self.name = self.config.get("name", "")

    def phase_index(self, t):
        return min(len(self.phases) - 1, next((i for i in range(len(self.phases)) if t < self.events[i + 1] - 1e-12),
                                              len(self.phases) - 1))

    def phase_label(self, t):
        k = self.phase_index(t)
        p = self.phases[k]
        if p.get("purpose"):
            return f"phase {k+1}: reposition (no flow)"
        if p["Q"] == 0 and p.get("Q_end", 0) == 0:
            return f"phase {k+1}: stop"
        kind = "wiggle" if p.get("frequency", 0) > 0 and p.get("amplitude", 0) > 0 else "pour"
        if p["radius"] <= 0.006:
            kind = "fine cut"
        return f"phase {k+1}: {kind}"

    def __call__(self, t):
        k = self.phase_index(t)
        p = self.phases[k]
        tt = max(0., min(p["duration"], t - self.events[k]))
        f = tt / p["duration"]
        amp0 = p["amplitude"]; amp1 = p.get("amplitude_end", amp0)
        amp = amp0 * (1 - f) + amp1 * f
        freq = p["frequency"]
        x = (1 - f) * p["start"][0] + f * p["end"][0] + amp * math.sin(2 * math.pi * freq * tt)
        y = (1 - f) * p["start"][1] + f * p["end"][1]
        controls = p.get("bezier_controls")
        if controls is not None:
            a, b = controls
            x = ((1 - f) ** 3 * p["start"][0] + 3 * (1 - f) ** 2 * f * a[0] + 3 * (1 - f) * f * f * b[0]
                 + f ** 3 * p["end"][0] + amp * math.sin(2 * math.pi * freq * tt))
            y = (1 - f) ** 3 * p["start"][1] + 3 * (1 - f) ** 2 * f * a[1] + 3 * (1 - f) * f * f * b[1] + f ** 3 * p["end"][1]
        speed = math.hypot(p["end"][0] - p["start"][0], p["end"][1] - p["start"][1]) / p["duration"]
        if controls is not None:
            poly = [p["start"], *controls, p["end"]]
            speed = 3 * max(math.hypot(q[0] - o[0], q[1] - o[1]) for o, q in zip(poly, poly[1:])) / p["duration"]
        speed += 2 * math.pi * freq * max(abs(amp0), abs(amp1)) + abs(amp1 - amp0) / p["duration"]
        flow = p["Q"] * (1 - f) + p.get("Q_end", p["Q"]) * f
        flow_freq = p.get("flow_frequency", freq)
        flow *= 1 + p.get("flow_modulation", 0.) * math.cos(2 * math.pi * flow_freq * tt + p.get("flow_phase", 0.))
        r = p["radius"] * (1 - f) + p.get("radius_end", p["radius"]) * f
        Q = flow / self.kappa_Q
        jet_x, jet_y = p["jet"]
        jet_frequency = p.get("jet_frequency", freq)
        jet_wave = math.sin(2 * math.pi * jet_frequency * tt + p.get("jet_phase", 0.))
        jet_x += p.get("jet_amplitude_x", 0.) * jet_wave
        jet_y += p.get("jet_amplitude_y", 0.) * jet_wave
        if Q <= 0:
            return Inlet(active=False, x_hit=(x, y), S_eff=0.0, scan_speed=speed, r1=r, r2=r)
        if self.config.get("inlet_geometry") == "continuity_ellipse":
            U_perp = Q / (math.pi * r * r)
            vt = math.hypot(jet_x, jet_y)
            max_angle = self.config.get("max_jet_angle_degrees")
            if max_angle is not None and vt > 0:
                scale = min(1., U_perp * math.tan(math.radians(max_angle)) / vt)
                jet_x *= scale; jet_y *= scale; vt *= scale
            speed_material = math.hypot(U_perp, vt)
            aspect = speed_material / U_perp
            r1, r2 = r * math.sqrt(aspect), r / math.sqrt(aspect)
            area = Q / speed_material
            angle = math.atan2(jet_y, jet_x)
        else:
            U_perp, r1, r2 = p["U_perp"], r, r * p.get("aspect", 1.)
            area, angle = p.get("A_perp", math.pi * r * r), p.get("angle", 0.)
        d = 2 * math.sqrt(area / math.pi)
        return Inlet(active=True, x_hit=(x, y), S_eff=flow, u_in=(jet_x, jet_y), U_perp=U_perp,
                     d_jet=d, r1=r1, r2=r2, phi=angle, scan_speed=speed)


def program(name):
    return Program(load_control(name))
