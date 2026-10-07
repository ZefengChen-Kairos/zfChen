"""Guessed pouring scripts for six patterns (observation-style inlet, mode A).

Cup frame: x to the right, y up; the barista stands at y<0 (start side B), the push goes to y>0 (D).
All numbers are guesses in cup-diameter units; none are measured from the video.
"""
import math
import numpy as np
from .solver import Inlet

G_REF = 122.0  # D_L/s^2 for an 8 cm cup


def U_from_height(h_cm, cup_cm=8.0):
    return math.sqrt(2.0 * G_REF * (h_cm / cup_cm))


class Seg:
    """One pouring segment.  pos(tau) in [0,1] -> (x,y); tau = local time / duration."""

    def __init__(self, dur, p0, p1, S=1.0, height=1.0, d=0.07, foot=None, wobble=0.0, wfreq=2.5,
                 pulse=0.0, lean=1.0, lean_vec=(0.0, 0.0), ramp=0.15, path=None, name="", wobble_axis="x",
                 ease=True):
        self.dur, self.p0, self.p1 = dur, np.array(p0, float), np.array(p1, float)
        self.S, self.height, self.d = S, height, d
        self.foot = foot if foot is not None else 0.65 * d
        self.wobble, self.wfreq, self.pulse = wobble, wfreq, pulse
        self.lean, self.lean_vec, self.ramp = lean, np.array(lean_vec, float), ramp
        self.path, self.name, self.wobble_axis, self.ease = path, name, wobble_axis, ease

    def base_pos(self, tau):
        if self.path is not None:
            return np.array(self.path(tau), float)
        e = tau
        if self.ease:
            e = 0.5 - 0.5 * math.cos(math.pi * tau)
        return self.p0 + (self.p1 - self.p0) * e

    def pos(self, tl):
        tau = min(max(tl / self.dur, 0.0), 1.0)
        p = self.base_pos(tau)
        if self.wobble > 0:
            w = self.wobble * math.sin(2 * math.pi * self.wfreq * tl)
            if self.wobble_axis == "x":
                p = p + np.array([w, 0.0])
            elif self.wobble_axis == "y":
                p = p + np.array([0.0, w])
            else:  # perpendicular to the base direction
                dirv = self.base_pos(min(tau + 1e-3, 1.0)) - self.base_pos(max(tau - 1e-3, 0.0))
                n = np.linalg.norm(dirv)
                perp = np.array([-dirv[1], dirv[0]]) / n if n > 1e-9 else np.array([1.0, 0.0])
                p = p + w * perp
        return p

    def flow(self, tl):
        tau = tl / self.dur
        r = 1.0
        if self.ramp > 0:
            r = min(1.0, tau / self.ramp, (1.0 - tau) / self.ramp)
            r = max(r, 0.0)
        pulse = 1.0 + self.pulse * math.cos(2 * math.pi * 2 * self.wfreq * tl + math.pi) if self.pulse > 0 else 1.0
        return self.S * r * pulse

    def inlet(self, tl):
        p = self.pos(tl)
        eps = 1e-3
        v = (self.pos(min(tl + eps, self.dur)) - self.pos(max(tl - eps, 0.0))) / (2 * eps if 0 < tl < self.dur else eps)
        if tl + eps > self.dur or tl - eps < 0:
            v = (self.pos(min(tl + eps, self.dur)) - self.pos(max(tl - eps, 0.0))) / (min(tl + eps, self.dur) - max(tl - eps, 0.0) + 1e-12)
        u_in = self.lean * v + self.lean_vec
        return Inlet(active=True, x_hit=(float(p[0]), float(p[1])), S_eff=self.flow(tl),
                     u_in=(float(u_in[0]), float(u_in[1])), U_perp=U_from_height(self.height),
                     d_jet=self.d, r1=self.foot, r2=self.foot, phi=0.0)


class Pause:
    def __init__(self, dur, name="pause"):
        self.dur, self.name = dur, name


class Script:
    def __init__(self, name, segs, tail=1.0):
        self.name, self.segs, self.tail = name, segs, tail
        self.T = sum(s.dur for s in segs) + tail

    def sample(self, t):
        acc = 0.0
        for s in self.segs:
            if t < acc + s.dur:
                if isinstance(s, Pause):
                    return Inlet(active=False), s.name
                return s.inlet(t - acc), s.name
            acc += s.dur
        return Inlet(active=False), "settle"


# ----------------------------------------------------------------------------- patterns
def heart(disc_S=0.3, cut_start=-0.46, cut_end=0.34, cut_foot=0.08, cut_lean=1.3, cut_S=0.3, cut_dur=1.4):
    return Script("heart", [
        Seg(0.6, (0.0, -0.12), (0.0, -0.12), S=0.25, height=1.5, d=0.07, name="start (low, slow)", ramp=0.4),
        Seg(2.8, (0.0, -0.12), (0.0, -0.16), S=disc_S, height=0.8, d=0.075, wobble=0.012, wfreq=3.0,
            name="spread white disc", ramp=0.05),
        Seg(0.7, (0.0, -0.16), (0.0, -0.02), S=0.8 * disc_S, height=1.0, d=0.07, lean=1.2, name="push forward", ramp=0.1),
        Seg(0.3, (0.0, -0.02), (0.0, cut_start), S=0.1, height=4.0, d=0.045, name="lift & move back", ramp=0.3),
        Seg(cut_dur, (0.0, cut_start), (0.0, cut_end), S=cut_S, height=8.0, d=0.03, foot=cut_foot, lean=cut_lean,
            name="cut through", ramp=0.1),
    ], tail=1.2)


def push_heart(S=0.25, fwd=0.6, blobs=((0.2, 1.2), (-0.05, 1.0), (-0.27, 0.8)), move=0.06,
               cut_start=-0.46, cut_end=0.36, cut_foot=0.08, cut_lean=1.3):
    """Three blobs deposited from the far side towards the barista, each started in brown just behind the previous
    one; the stream leans forward so every new blob pushes the earlier ones outward and folds the brown gap into a U."""
    segs = []
    for i, (y0, dur) in enumerate(blobs):
        segs.append(Seg(dur, (0.0, y0 - 0.5 * move), (0.0, y0 + 0.5 * move), S=S, height=0.9, d=0.07,
                        lean_vec=(0.0, fwd), name=f"blob {i+1}: pour low, lean forward", ramp=0.2))
        segs.append(Pause(0.35, name="stop, lift, move back"))
    segs.append(Seg(1.3, (0.0, cut_start), (0.0, cut_end), S=0.3, height=8.0, d=0.03, foot=cut_foot, lean=cut_lean,
                    name="cut through", ramp=0.1))
    return Script("push_heart", segs, tail=1.2)


def layered_heart(amp=0.1, S=0.12, retreat=0.12, pulse=0.7, freq=2.5, y_start=0.1, y_end=-0.2, lean=-1.0,
                  fwd=0.1, cut_foot=0.08, cut_lean=1.3):
    """Wiggle while slowly backing up so each arc lands in brown behind the previous one; then push and cut."""
    dur = (y_start - y_end) / retreat
    return Script("layered_heart", [
        Seg(0.3, (0.0, y_start), (0.0, y_start), S=0.1, height=1.2, d=0.07, name="start", ramp=0.4),
        Seg(dur, (0.0, y_start), (0.0, y_end), S=S, height=0.9, d=0.065, wobble=amp, wfreq=freq, pulse=pulse,
            lean=lean, lean_vec=(0.0, fwd), ease=False, name="wiggle & back up (arcs)", ramp=0.03),
        Seg(0.7, (0.0, y_end), (0.0, y_end + 0.12), S=0.8 * S, height=1.0, d=0.065, wobble=0.6 * amp, wfreq=freq,
            lean=1.0, name="wiggle + push forward", ramp=0.1),
        Seg(0.3, (0.0, y_end + 0.12), (0.0, -0.46), S=0.1, height=4.0, d=0.045, name="lift & move back", ramp=0.3),
        Seg(1.4, (0.0, -0.46), (0.0, 0.36), S=0.3, height=8.0, d=0.03, foot=cut_foot, lean=cut_lean,
            name="cut through", ramp=0.1),
    ], tail=1.2)


def tulip(S=0.25, fwd=0.6, amp=0.06, base_S=0.12, base_retreat=0.16, pulse=0.7, freq=2.5,
          petals=((-0.1, 0.8), (-0.3, 0.6)), move=0.06, cut_foot=0.08, cut_lean=1.3):
    """Base: wiggle while backing up a little on the far side (U arcs); then two petals deposited behind it,
    each leaning forward so it pushes into the previous one; then the cut."""
    y0, y1 = 0.3, 0.3 - base_retreat * 1.6
    return Script("tulip", [
        Seg(0.3, (0.0, y0), (0.0, y0), S=0.1, height=1.2, d=0.07, name="base: start", ramp=0.4),
        Seg(1.6, (0.0, y0), (0.0, y1), S=base_S, height=0.9, d=0.065, wobble=amp, wfreq=freq, pulse=pulse,
            lean=-1.0, lean_vec=(0.0, 0.1), ease=False, name="base: wiggle & back up (arcs)", ramp=0.03),
        Pause(0.35, name="stop, lift, move back"),
        Seg(petals[0][1], (0.0, petals[0][0] - 0.5 * move), (0.0, petals[0][0] + 0.5 * move), S=S, height=0.9, d=0.07,
            lean_vec=(0.0, fwd), name="petal 2: pour low, lean forward", ramp=0.2),
        Pause(0.35, name="stop, lift, move back"),
        Seg(petals[1][1], (0.0, petals[1][0] - 0.5 * move), (0.0, petals[1][0] + 0.5 * move), S=S, height=0.9, d=0.07,
            lean_vec=(0.0, fwd), name="petal 3: pour low, lean forward", ramp=0.2),
        Pause(0.35, name="stop, lift, move back"),
        Seg(1.3, (0.0, -0.46), (0.0, 0.4), S=0.3, height=8.0, d=0.03, foot=cut_foot, lean=cut_lean,
            name="cut through", ramp=0.1),
    ], tail=1.2)


def leaf(amp=0.08, S=0.12, retreat=0.2, pulse=0.7, freq=2.5, y_start=0.12, y_end=-0.36, base_S=0.1,
         cut_lean=1.3, cut_foot=0.08, lean=-1.0, fwd=0.1):
    dur = (y_start - y_end) / retreat
    return Script("leaf", [
        Seg(0.3, (0.0, y_start), (0.0, y_start), S=base_S, height=1.2, d=0.07, name="start (small base)", ramp=0.4),
        Seg(dur, (0.0, y_start), (0.0, y_end), S=S, height=0.9, d=0.065, wobble=amp, wfreq=freq, pulse=pulse,
            lean=lean, lean_vec=(0.0, fwd), ease=False, name="wiggle & retreat (leaflets)", ramp=0.03),
        Seg(0.25, (0.0, y_end), (0.0, y_end - 0.06), S=0.1, height=4.0, d=0.045, name="lift", ramp=0.3),
        Seg(1.4, (0.0, y_end - 0.06), (0.0, 0.40), S=0.3, height=8.0, d=0.03, foot=cut_foot, lean=cut_lean,
            name="cut through (stem)", ramp=0.1),
    ], tail=1.2)


def swan(S=0.12, amp=0.07, pulse=0.7, freq=2.5, neck_S=0.06, head_S=0.12):
    """Body/wing: wiggle while retreating diagonally (leaflets); thin stream back along the wing edge;
    neck: thin low pour along a curve; head: small blob; beak: short cut."""
    p0, p1 = (0.02, 0.0), (0.3, -0.36)

    def neck(tau):  # from the body top, arching up and to the left
        a = math.pi * 0.5 * (1.0 - tau) - math.pi * 0.5 * tau
        return (-0.08 - 0.2 * math.sin(math.pi * tau) * 0.0 + (-0.22) * tau, 0.1 + 0.22 * math.sin(math.pi * tau))
    return Script("swan", [
        Seg(0.3, p0, p0, S=0.1, height=1.2, d=0.07, name="body: start", ramp=0.4),
        Seg(2.4, p0, p1, S=S, height=0.9, d=0.065, wobble=amp, wfreq=freq, pulse=pulse, wobble_axis="perp",
            lean=-1.0, lean_vec=(-0.08, 0.08), ease=False, name="wing: wiggle & retreat diagonally", ramp=0.03),
        Seg(0.9, (0.26, -0.32), (-0.02, 0.06), S=0.2, height=7.0, d=0.03, foot=0.07, lean=1.0,
            name="wing edge: thin stream back", ramp=0.1),
        Pause(0.3, name="lift, move to neck base"),
        Seg(1.6, (0.0, 0.0), (0.0, 0.0), S=neck_S, height=1.0, d=0.04, foot=0.03, path=neck, lean=0.8,
            name="neck: thin low pour along a curve", ramp=0.1),
        Seg(0.35, (-0.3, 0.1), (-0.32, 0.08), S=head_S, height=0.9, d=0.06, name="head: small blob", ramp=0.3),
        Seg(0.35, (-0.32, 0.08), (-0.42, 0.0), S=0.25, height=6.0, d=0.03, foot=0.05, lean=1.2, name="beak: cut", ramp=0.2),
    ], tail=1.2)


SCRIPTS = dict(heart=heart, push_heart=push_heart, layered_heart=layered_heart, tulip=tulip, leaf=leaf, swan=swan)
