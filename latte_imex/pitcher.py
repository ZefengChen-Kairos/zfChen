"""Level-0 pitcher model: quasi-static free surface + sharp-crested weir outflow + ballistic jet.

Chain:  pitcher pose (tip position, yaw, tilt, roll) + remaining volume V
        -> horizontal free surface z = c  (volume below the plane inside the pitcher == V)
        -> outflow Q over the rim (sharp-crested weir integral, Kindsvater head correction,
           surface-tension start/stop hysteresis, first-order spout lag)
        -> jet: exit speed from the discharge-weighted head, exit direction along the spout,
           free fall to the coffee surface, continuity thinning
        -> impact record (x_hit, Q, U_perp, u_h, d_jet) in SI
        -> Coupling: SI record -> solver.Inlet in cup units (D_L, s) with three scale constants.

Everything in SI internally: metres, seconds, m^3.  World frame: z up, coffee surface z = 0,
cup centre at the origin.  Body frame: Z along the pitcher axis, X towards the spout.
Derivation and the numbers behind the defaults: latte_imex/PITCHER_MODEL.md
"""
import math
from dataclasses import dataclass, field
import numpy as np
from .solver import Inlet

G = 9.81


# --------------------------------------------------------------------------- geometry
@dataclass
class PitcherGeometry:
    """Body of revolution (piecewise-linear profile r(z)) with a V notch cut into the rim at azimuth 0."""
    height: float = 0.095           # H [m]
    r_bottom: float = 0.034
    r_belly: float = 0.040
    z_belly: float = 0.030
    r_top: float = 0.030
    notch_depth: float = 0.015      # V notch depth below the rim [m]
    notch_half_angle: float = math.radians(25.0)   # half-width of the notch in azimuth [rad]
    beak: float = 0.12              # radial protrusion of the rim at the spout (fraction of r_top)
    lip_slope: float = math.radians(10.0)          # exit direction below the body-frame horizontal
    nz: int = 48
    nr: int = 20
    nphi: int = 36
    n_rim: int = 720

    def r_body(self, z):
        z = np.asarray(z, float)
        return np.interp(z, [0.0, self.z_belly, self.height], [self.r_bottom, self.r_belly, self.r_top])

    def z_rim(self, phi):
        phi = np.abs(np.asarray(phi, float))
        return self.height - self.notch_depth * np.maximum(0.0, 1.0 - phi / self.notch_half_angle)

    def r_rim(self, phi):
        phi = np.abs(np.asarray(phi, float))
        return self.r_top * (1.0 + self.beak * np.maximum(0.0, 1.0 - phi / self.notch_half_angle) ** 2)

    def __post_init__(self):
        # interior sample points with volume weights (uniform in z, phi and in r^2)
        zc = (np.arange(self.nz) + 0.5) / self.nz * self.height
        dz = self.height / self.nz
        u = (np.arange(self.nr) + 0.5) / self.nr            # r^2 fraction
        ph = (np.arange(self.nphi) + 0.5) / self.nphi * 2 * math.pi - math.pi
        Z, U, PH = np.meshgrid(zc, u, ph, indexing="ij")
        Rb = self.r_body(Z)
        R = Rb * np.sqrt(U)
        keep = Z < self.z_rim(PH)                           # no liquid in the notch opening
        pts = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1)[keep]
        dv = (math.pi * Rb ** 2 / self.nr) * dz / self.nphi
        self.points = pts.astype(np.float64)                # (n, 3) body frame
        self.dv = dv[keep].astype(np.float64)
        self.capacity = float(self.dv.sum())
        # rim polyline (closed), body frame
        phi = (np.arange(self.n_rim) + 0.5) / self.n_rim * 2 * math.pi - math.pi
        self.rim = np.stack([self.r_rim(phi) * np.cos(phi), self.r_rim(phi) * np.sin(phi), self.z_rim(phi)], -1)
        self.tip = np.array([self.r_rim(0.0), 0.0, self.z_rim(0.0)])  # notch apex: the pose reference point
        # volume below the crest when upright (fill above this spills even at zero tilt)
        self.capacity_to_crest = float(self.dv[self.points[:, 2] < self.tip[2]].sum())


class GridPitcherGeometry:
    """Pitcher cavity from a CAD model (latte_imex/pitcher_cad.py JSON): inner/outer wall radius on a (z, phi) grid,
    rim height and radius per azimuth.  Same interface as PitcherGeometry (points, dv, rim, tip, capacity,
    capacity_to_crest, height, r_body, lip_slope)."""

    def __init__(self, data, nr=16, n_rim=720, lip_slope=math.radians(10.0), scale=1e-3):
        self.name = data.get("name", "cad")
        zs = np.asarray(data["zs"], float) * scale
        phis = np.asarray(data["phis"], float) - data.get("spout_phi_residual", 0.0)   # spout exactly at phi = 0
        r_in = np.asarray(data["r_in"], float) * scale
        r_out = np.asarray(data["r_out"], float) * scale
        z_rim = np.asarray(data["z_rim"], float) * scale
        r_rim = np.asarray(data["r_rim"], float) * scale
        self.zs, self.phis, self.r_in, self.r_out, self.z_rim_tab, self.r_rim_tab = zs, phis, r_in, r_out, z_rim, r_rim
        self.height = float(zs[-1])
        self.lip_slope = lip_slope
        self.r_top = float(r_in[-1].mean()); self.r_belly = float(r_in.max(1).mean()); self.r_bottom = float(r_in[0].mean())
        self.notch_half_angle = math.radians(25.0)
        # interior sample points: polar cells, equal area in r^2
        nz, nphi = r_in.shape
        dphi = 2 * math.pi / nphi
        edges = np.concatenate([[zs[0] - 0.5 * (zs[1] - zs[0])], 0.5 * (zs[1:] + zs[:-1]), [zs[-1] + 0.5 * (zs[-1] - zs[-2])]])
        dz = np.diff(edges)
        u = (np.arange(nr) + 0.5) / nr
        pts, dv = [], []
        for iz in range(nz):
            keep = zs[iz] < z_rim                       # no liquid in the open notch / above the rim
            for ip in np.nonzero(keep)[0]:
                rr = r_in[iz, ip] * np.sqrt(u)
                pts.append(np.stack([rr * math.cos(phis[ip]), rr * math.sin(phis[ip]), np.full(nr, zs[iz])], -1))
                dv.append(np.full(nr, 0.5 * r_in[iz, ip] ** 2 * dphi * dz[iz] / nr))
        self.points = np.concatenate(pts); self.dv = np.concatenate(dv)
        self.capacity = float(self.dv.sum())
        # rim polyline (dense, periodic linear interpolation)
        ph = (np.arange(n_rim) + 0.5) / n_rim * 2 * math.pi - math.pi
        self.rim = np.stack([self.r_rim(ph) * np.cos(ph), self.r_rim(ph) * np.sin(ph), self.z_rim(ph)], -1)
        ip_tip = int(np.argmax(self.r_rim_tab))
        self.tip = np.array([self.r_rim_tab[ip_tip] * math.cos(phis[ip_tip]), self.r_rim_tab[ip_tip] * math.sin(phis[ip_tip]), z_rim[ip_tip]])
        self.capacity_to_crest = float(self.dv[self.points[:, 2] < z_rim.min()].sum())

    def _interp_phi(self, tab, phi):
        phi = (np.asarray(phi, float) - self.phis[0]) % (2 * math.pi) + self.phis[0]
        xp = np.concatenate([self.phis, [self.phis[0] + 2 * math.pi]])
        fp = np.concatenate([tab, [tab[0]]])
        return np.interp(phi, xp, fp)

    def z_rim(self, phi):
        return self._interp_phi(self.z_rim_tab, phi)

    def r_rim(self, phi):
        return self._interp_phi(self.r_rim_tab, phi)

    def r_body(self, z):
        """Azimuth-mean inner radius (for drawing and the clearance check)."""
        return np.interp(np.asarray(z, float), self.zs, self.r_in.mean(1))

    def r_outer(self, z):
        return np.interp(np.asarray(z, float), self.zs, self.r_out.mean(1))

    @classmethod
    def load(cls, path=None, **kw):
        import json, os
        path = path or os.path.join(os.path.dirname(__file__), "pitcher_geom", "pitcher_nx.json")
        with open(path) as f:
            data = json.load(f)
        data.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        return cls(data, **kw)


def rotation(yaw, tilt, roll):
    """R = Rz(yaw) Ry(tilt) Rx(roll); tilt > 0 brings the spout (body +X) down."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    ct, st = math.cos(tilt), math.sin(tilt)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[ct, 0, st], [0, 1.0, 0], [-st, 0, ct]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass
class Pose:
    tip: tuple = (0.0, 0.0, 0.03)   # world position of the notch apex [m]
    yaw: float = 0.0                # azimuth of the spout direction [rad]; 0 = +x
    tilt: float = 0.0               # [rad]
    roll: float = 0.0               # [rad]


@dataclass
class Milk:
    rho: float = 1030.0
    sigma: float = 0.045
    r_min: float = 0.0005   # smallest footprint radius [m]: a thinner dribble breaks into drops (capillary limit)


@dataclass
class OutflowLaw:
    C_d: float = 0.60       # sharp-crested weir discharge coefficient (V notch 0.58-0.60)
    h_k: float = 0.0008     # Kindsvater-Shen head correction [m]
    h_on: float = 0.0025    # surface tension: head needed to start the flow [m]
    h_off: float = 0.0005   # flow stops below this head [m]
    tau: float = 0.08       # spout filling / emptying lag [s]
    tau_stop: float = 0.15  # trailing-off lag after the head drops below h_off [s]
    # level 1: the free surface follows the effective gravity g - a (pitcher acceleration) through the first
    # sloshing mode, a damped oscillator forced by the quasi-static slope; slosh=False keeps the surface horizontal
    slosh: bool = True
    zeta: float = 0.30      # sloshing damping ratio (foamed milk is strongly damped; guess, to be calibrated)
    slope_max: float = 0.36 # cap on the free-surface slope (tan 20 deg): beyond this the single-mode model is meaningless
    acc_tau: float = 0.02   # low-pass on the finite-difference acceleration [s]
    carry_velocity: bool = True   # the jet leaves with the pitcher's velocity added


@dataclass
class Cup:
    """Cup as an obstacle: coffee surface at world z = 0 (centre at the origin), open cylinder tilted by `tilt`
    towards the pitcher body (the barista tilts the cup towards the pitcher and pours over its near rim)."""
    r_in: float = 0.040
    r_out: float = 0.043
    height: float = 0.07
    rim_z: float = 0.010     # rim above the coffee surface when upright [m]
    tilt: float = math.radians(20.0)
    margin: float = 0.002


@dataclass
class PitcherState:
    V: float = 2.0e-4       # bulk milk in the cavity [m^3]
    Q: float = 0.0          # outflow leaving the spout, Q_out [m^3/s]
    flowing: bool = False
    t: float = 0.0
    # conservation ledger: V + V_lip + V_jet + V_cup == V0 at all times
    V_lip: float = 0.0      # milk in the spout channel (the lip buffer: Q_out = V_lip / tau)
    V_jet: float = 0.0      # milk in flight
    V_cup: float = 0.0      # milk that has arrived on the coffee
    Q_feed: float = 0.0     # bulk -> lip (weir law)
    Q_hit: float = 0.0      # arriving on the coffee this frame
    parcels: list = field(default_factory=list)   # in-flight parcels (dicts), FIFO by arrival time
    tip_prev: tuple = None          # last tip position (for the finite-difference velocity)
    vel: tuple = (0.0, 0.0, 0.0)    # pitcher (tip) velocity [m/s]
    acc: tuple = (0.0, 0.0, 0.0)    # low-passed pitcher acceleration [m/s^2]
    slope: tuple = (0.0, 0.0)       # free-surface slope (dz/dx, dz/dy) in the world frame
    slope_rate: tuple = (0.0, 0.0)
    omega: float = 0.0              # first sloshing mode [rad/s] at the current fill


# --------------------------------------------------------------------------- the model
class Pitcher:
    def __init__(self, geom=None, milk=None, law=None, V0=2.0e-4):
        self.geom = geom or PitcherGeometry()
        self.milk = milk or Milk()
        self.law = law or OutflowLaw()
        self.state = PitcherState(V=V0)
        self.n = np.array([0.0, 0.0, 1.0])     # current free-surface normal (unit, up)
        self.g_eff = G                         # magnitude of the effective gravity driving the outflow

    # ---- kinematics
    def world(self, pose, P):
        """Body-frame points (n,3) -> world, with the notch apex at pose.tip."""
        R = rotation(pose.yaw, pose.tilt, pose.roll)
        return (P - self.geom.tip) @ R.T + np.asarray(pose.tip, float)

    def free_surface(self, pose, V):
        """Height c of the horizontal plane with volume V below it inside the pitcher.
        Returns (c, overflow) where overflow > 0 is the volume that cannot be held at this pose."""
        zw = self.world(pose, self.geom.points) @ self.n     # height along the free-surface normal
        order = np.argsort(zw)
        zs = zw[order]
        cum = np.cumsum(self.geom.dv[order])
        if V <= 0:
            return zs[0], 0.0
        if V >= cum[-1]:
            return zs[-1], V - cum[-1]
        k = int(np.searchsorted(cum, V))
        c0 = zs[k - 1] if k > 0 else zs[0]
        v0 = cum[k - 1] if k > 0 else 0.0
        f = (V - v0) / max(cum[k] - v0, 1e-18)
        return c0 + f * (zs[k] - c0), 0.0

    # ---- outflow
    def weir(self, pose, c):
        """Sharp-crested weir integral over the rim below the free surface.
        Q = C_d (2/3) sqrt(2g) sum h_eff^{3/2} dl   (dl = horizontal rim element)
        Returns dict(Q_ss, h_max, h_bar, origin, A_wet)."""
        rim = self.world(pose, self.geom.rim)
        nxt = np.roll(rim, -1, axis=0)
        mid = 0.5 * (rim + nxt)
        seg = nxt - rim
        seg_n = seg @ self.n
        dl = np.linalg.norm(seg - seg_n[:, None] * self.n[None, :], axis=1)   # rim element in the surface plane
        h = c - mid @ self.n                                                   # head along the surface normal
        h_max = float(h.max())
        he = np.maximum(h - self.law.h_k, 0.0)
        w32 = he ** 1.5 * dl
        S32 = float(w32.sum())
        Q_ss = self.law.C_d * (2.0 / 3.0) * math.sqrt(2 * self.g_eff) * S32
        width = 0.0; tangent = np.array([1.0, 0.0])
        if S32 > 0:
            h_bar = float((he ** 2.5 * dl).sum() / S32)          # discharge-weighted head
            origin = (w32[:, None] * mid).sum(0) / S32            # discharge-weighted crest point
            # second moment of the lip flux in the horizontal plane: wetted lip extent and its direction
            d2 = mid[:, :2] - origin[:2]
            cov = (w32[:, None, None] * d2[:, :, None] * d2[:, None, :]).sum(0) / S32
            evals, evecs = np.linalg.eigh(cov)
            width = float(math.sqrt(12.0 * max(evals[-1], 0.0)))  # uniform segment of length L has variance L^2/12
            tangent = evecs[:, -1]
        else:
            h_bar = 0.0
            origin = rim[np.argmin(rim @ self.n)]
        A_wet = float((np.maximum(h, 0.0) * dl).sum())
        return dict(Q_ss=Q_ss, h_max=h_max, h_bar=h_bar, origin=origin, A_wet=A_wet, width=width, tangent=tangent)

    def exit_direction(self, pose):
        R = rotation(pose.yaw, pose.tilt, pose.roll)
        e = np.array([math.cos(self.geom.lip_slope), 0.0, -math.sin(self.geom.lip_slope)])
        return R @ e

    # ---- jet
    def jet(self, pose, Q, h_bar, origin, z_surface=0.0, width=0.0, tangent=(1.0, 0.0)):
        """Free jet from the crest to the coffee surface.  Exit speed sqrt(2 g_eff h_bar) relative to the pitcher
        (Torricelli at the discharge-weighted head), direction along the spout, plus the pitcher's own velocity;
        then free flight under g, continuity thinning d ~ |v|^{-1/2}."""
        if Q <= 0:
            return None
        u0 = math.sqrt(2 * self.g_eff * max(h_bar, self.law.h_k))   # the lip buffer may drain after the head is gone   # floor: the trailing dribble leaves at the h_k head
        A0 = Q / u0
        d0 = 2 * math.sqrt(A0 / math.pi)
        v0 = u0 * self.exit_direction(pose)
        if self.law.carry_velocity:
            v0 = v0 + np.asarray(self.state.vel, float)
        z0 = origin[2] - z_surface
        if z0 <= 0:
            t_hit = 0.0
        else:
            t_hit = (v0[2] + math.sqrt(v0[2] ** 2 + 2 * G * z0)) / G
        x_hit = origin[:2] + v0[:2] * t_hit
        vz = v0[2] - G * t_hit
        speed = math.sqrt(v0[0] ** 2 + v0[1] ** 2 + vz ** 2)
        d_hit = max(d0 * math.sqrt(u0 / speed), 2 * self.milk.r_min)
        # exit sheet: wetted lip width L and thickness A0/L; surface tension rounds it up on the capillary time
        # t_cap = sqrt(rho t^3 / sigma); what remains at impact is an ellipse of the continuity area A0*u0/|v|
        L = max(float(width), d0)
        thick = A0 / L
        aspect0 = L / thick
        t_cap = math.sqrt(self.milk.rho * thick ** 3 / self.milk.sigma)
        rounding = 1.0 - math.exp(-t_hit / max(t_cap, 1e-6))
        aspect = aspect0 ** (1.0 - rounding)
        A_hit = A0 * u0 / speed
        r1 = max(math.sqrt(A_hit * aspect / math.pi), self.milk.r_min); r2 = max(math.sqrt(A_hit / aspect / math.pi), self.milk.r_min)
        phi_lip = math.atan2(float(tangent[1]), float(tangent[0]))
        We = self.milk.rho * u0 ** 2 * d0 / self.milk.sigma
        L_break = 19.5 * d0 * We ** 0.325 if We > 0 else 0.0       # Grant & Middleman, laminar jet
        path = abs(v0[2]) * t_hit + 0.5 * G * t_hit ** 2 + np.hypot(*(v0[:2] * t_hit))
        return dict(u0=u0, d0=d0, v0=v0, origin=np.asarray(origin, float), t_hit=t_hit, x_hit=np.asarray(x_hit, float),
                    U_perp=max(-vz, 0.0), u_h=v0[:2].copy(), speed=speed, d_hit=d_hit, We=We,
                    L_break=L_break, coherent=bool(path < L_break), drop=z0,
                    r1=r1, r2=r2, phi=phi_lip, aspect=aspect, width=L, rounding=rounding)

    def jet_polyline(self, J, n=24):
        """World-frame points along the jet for drawing."""
        if J is None:
            return np.zeros((0, 3))
        ts = np.linspace(0.0, J["t_hit"], n)
        return J["origin"][None, :] + J["v0"][None, :] * ts[:, None] - 0.5 * G * (ts ** 2)[:, None] * np.array([0, 0, 1.0])

    def clearance(self, pose, cup_radius=0.04, z_surface=0.0):
        """Lowest point of the pitcher wall above the cup footprint (z - z_surface, [m]); negative means the
        pitcher body dips into the coffee.  Level 0 does not resolve this: the barista avoids it by tilting the
        cup towards the pitcher and pouring over its near rim."""
        zz = np.linspace(0, self.geom.height, 40)
        ph = np.linspace(-math.pi, math.pi, 72, endpoint=False)
        Z, PH = np.meshgrid(zz, ph, indexing="ij")
        Rb = self.geom.r_body(Z)
        wall = np.stack([Rb * np.cos(PH), Rb * np.sin(PH), Z], -1).reshape(-1, 3)
        W = self.world(pose, wall)
        inside = np.hypot(W[:, 0], W[:, 1]) < cup_radius
        if not inside.any():
            return float("inf")
        return float(W[inside, 2].min() - z_surface)

    def wall_samples(self, pose, nz=40, nphi=36):
        """Outer-wall sample points in the world frame (n, 3)."""
        zz = np.linspace(0, self.geom.height, nz)
        ph = np.linspace(-math.pi, math.pi, nphi, endpoint=False)
        Z, PH = np.meshgrid(zz, ph, indexing="ij")
        if hasattr(self.geom, "r_outer"):
            Rb = self.geom.r_outer(Z)
        else:
            Rb = self.geom.r_body(Z) + 0.0015
        return self.world(pose, np.stack([Rb * np.cos(PH), Rb * np.sin(PH), Z], -1).reshape(-1, 3))

    def required_lift(self, pose, cup: "Cup", iters=14):
        """Smallest vertical lift of the pitcher that keeps its wall clear of the cup (tilted towards the pitcher
        body) and of the coffee surface.  The cup is tilted about the horizontal axis perpendicular to the spout
        direction so that its rim on the pitcher's side goes down."""
        W = self.wall_samples(pose)
        R = rotation(pose.yaw, 0.0, 0.0)
        e = R @ np.array([1.0, 0.0, 0.0])          # horizontal spout direction (pitcher body is on the -e side)
        d = -e[:2] / max(np.linalg.norm(e[:2]), 1e-9)
        ct, st = math.cos(cup.tilt), math.sin(cup.tilt)
        # cup frame: z' along the tilted cup axis (the +d rim goes down), coordinates of a world point p:
        #   along = p.d (horizontal, towards the pitcher), z' = ct*z - st*along ... rotation about the axis z x d
        def ok(L):
            x, y, z = W[:, 0], W[:, 1], W[:, 2] + L
            along = x * d[0] + y * d[1]
            perp = -x * d[1] + y * d[0]
            zc = ct * z + st * along            # cup-frame height: the rim on the pitcher's side (+along) is lower
            ac = ct * along - st * z
            r = np.hypot(ac, perp)
            inside = r < cup.r_in - cup.margin
            onrim = (r >= cup.r_in - cup.margin) & (r <= cup.r_out + cup.margin)
            outside = r > cup.r_out + cup.margin
            table = -(cup.height - cup.rim_z)          # the cup stands on the table (world z of its base, upright)
            good = (inside & (z > cup.margin)) | (onrim & (zc > cup.rim_z + cup.margin)) | (outside & (z > table))
            return bool(good.all())
        if ok(0.0):
            return 0.0
        lo, hi = 0.0, 0.15
        if not ok(hi):
            return hi
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            if ok(m):
                hi = m
            else:
                lo = m
        return hi

    # ---- level 1: pitcher acceleration -> effective gravity -> free-surface slope (first sloshing mode)
    def kinematics(self, pose, dt):
        st, law = self.state, self.law
        tip = np.asarray(pose.tip, float)
        if st.tip_prev is None:
            v_new = np.zeros(3)
            a_raw = np.zeros(3)
        else:
            v_new = (tip - np.asarray(st.tip_prev)) / dt
            vn = np.linalg.norm(v_new)
            if vn > 1.5:                                   # a hand: at most ~1.5 m/s, ~30 m/s^2
                v_new = v_new * (1.5 / vn)
            a_raw = (v_new - np.asarray(st.vel)) / dt
            an = np.linalg.norm(a_raw)
            if an > 30.0:
                a_raw = a_raw * (30.0 / an)
        acc = np.asarray(st.acc) + (a_raw - np.asarray(st.acc)) * (1.0 - math.exp(-dt / law.acc_tau))
        st.tip_prev, st.vel, st.acc = tuple(tip), tuple(v_new), tuple(acc)
        # quasi-static slope of the free surface in the accelerated frame: perpendicular to g - a
        gz = max(G + acc[2], 0.2 * G)
        s_qs = np.array([-acc[0] / gz, -acc[1] / gz])
        if law.slosh:
            R = self.geom.r_belly
            depth = max(st.V / (math.pi * R * R), 0.005)
            k = 1.841 / R
            omega = math.sqrt(G * k * math.tanh(k * depth))
            st.omega = omega
            sl, sr = np.asarray(st.slope), np.asarray(st.slope_rate)
            # semi-implicit Euler for s'' + 2 zeta w s' + w^2 s = w^2 s_qs
            sr = sr + dt * (omega * omega * (s_qs - sl) - 2 * law.zeta * omega * sr)
            sl = sl + dt * sr
            sn = float(np.linalg.norm(sl))
            if sn > law.slope_max:
                sl = sl * (law.slope_max / sn); sr = sr * 0.0
            st.slope, st.slope_rate = tuple(sl), tuple(sr)
        else:
            st.slope = tuple(s_qs)
            st.slope_rate = (0.0, 0.0)
        n = np.array([-st.slope[0], -st.slope[1], 1.0])
        self.n = n / np.linalg.norm(n)
        self.g_eff = float(np.linalg.norm(np.array([0.0, 0.0, -G]) - acc))

    # ---- time step
    def step(self, pose, dt, z_surface=0.0):
        """Advance the pitcher by dt at the given pose.  Returns the impact record (SI) for this frame."""
        st, law = self.state, self.law
        self.kinematics(pose, dt)
        c, overflow = self.free_surface(pose, st.V)
        W = self.weir(pose, c)
        # surface-tension hysteresis on the start / stop of the flow
        if not st.flowing and W["h_max"] > law.h_on:
            st.flowing = True
        if st.flowing and W["h_max"] < law.h_off:
            st.flowing = False
        Q_feed = W["Q_ss"] if st.flowing else 0.0
        if overflow > 0:
            Q_feed += overflow / dt                     # spill of whatever cannot be held
        feed = min(Q_feed * dt, st.V)                   # bulk -> lip buffer
        st.V -= feed
        st.V_lip += feed
        st.Q_feed = feed / dt
        # lip buffer drains on the spout time constant: Q_out = V_lip / tau (== first-order lag of Q_feed);
        # tau_stop (slower) only once the head is gone and the channel merely trails off (comparing Q_feed with the
        # previous Q_out instead flips tau every step and makes Q_out alternate)
        tau = law.tau if st.Q_feed > 0 else law.tau_stop
        Q_out = st.V_lip / tau
        release = min(st.V_lip, Q_out * dt)
        st.V_lip -= release
        st.Q = release / dt
        st.t += dt
        J = None
        if release > 0:
            J = self.jet(pose, st.Q, W["h_bar"], W["origin"], z_surface, W["width"], W["tangent"])
            # the volume released during [t-dt, t] arrives during [t-dt+t_hit, t+t_hit]: a parcel with an arrival window,
            # so that Q_hit is a proper rate whatever the sampling step (run.py shortens frames during fast scans)
            st.parcels.append(dict(ta0=st.t - dt + J["t_hit"], ta1=st.t + J["t_hit"], vol=release, x_hit=J["x_hit"].copy(),
                                   U_perp=J["U_perp"], u_h=J["u_h"].copy(), d_hit=J["d_hit"], speed=J["speed"],
                                   r1=J["r1"], r2=J["r2"], phi=J["phi"], coherent=J["coherent"]))
            st.V_jet += release
        # arrivals during this step [t-dt, t]: the overlap of each parcel's arrival window with the step
        t0, t1 = st.t - dt, st.t
        vol = 0.0; acc = dict(x=0.0, y=0.0, U_perp=0.0, ux=0.0, uy=0.0, d_hit=0.0, speed=0.0, r1=0.0, r2=0.0)
        phi = 0.0; coherent = True; keep = []
        for pk in st.parcels:
            ov = max(0.0, min(pk["ta1"], t1) - max(pk["ta0"], t0))
            if ov > 0:
                v = pk["vol"] * ov / max(pk["ta1"] - pk["ta0"], 1e-12)
                vol += v
                acc["x"] += v * pk["x_hit"][0]; acc["y"] += v * pk["x_hit"][1]; acc["U_perp"] += v * pk["U_perp"]
                acc["ux"] += v * pk["u_h"][0]; acc["uy"] += v * pk["u_h"][1]; acc["d_hit"] += v * pk["d_hit"]
                acc["speed"] += v * pk["speed"]; acc["r1"] += v * pk["r1"]; acc["r2"] += v * pk["r2"]
                phi = pk["phi"]; coherent = coherent and pk["coherent"]
            if pk["ta1"] > t1:
                keep.append(pk)
        st.parcels = keep
        hit = None
        if vol > 0:
            st.V_jet -= vol; st.V_cup += vol
            hit = dict(Q=vol / dt, x_hit=np.array([acc["x"], acc["y"]]) / vol, U_perp=acc["U_perp"] / vol,
                       u_h=np.array([acc["ux"], acc["uy"]]) / vol, d_hit=acc["d_hit"] / vol, speed=acc["speed"] / vol,
                       r1=acc["r1"] / vol, r2=acc["r2"] / vol, phi=phi, coherent=coherent)
        st.Q_hit = hit["Q"] if hit else 0.0
        return dict(t=st.t, V=st.V, Q=st.Q, Q_feed=st.Q_feed, Q_hit=st.Q_hit, V_lip=st.V_lip, V_jet=st.V_jet, V_cup=st.V_cup,
                    Q_ss=W["Q_ss"], h_max=W["h_max"], h_bar=W["h_bar"], c=c, width=W["width"],
                    flowing=st.flowing, overflow=overflow, tilt=pose.tilt, tip=np.asarray(pose.tip, float),
                    jet=J, hit=hit, acc=np.asarray(st.acc, float), slope=np.asarray(st.slope, float), n=self.n.copy(),
                    g_eff=self.g_eff, vel=np.asarray(st.vel, float))

    def tilt_for_flow(self, pose, Q_want, V=None, lo=0.0, hi=math.radians(150), iters=30):
        """Steady tilt that gives Q_want at the pose's yaw/roll/tip and volume V (bisection on the weir law)."""
        V = self.state.V if V is None else V
        p = Pose(pose.tip, pose.yaw, lo, pose.roll)

        def q_of(t):
            p.tilt = t
            c, _ = self.free_surface(p, V)
            return self.weir(p, c)["Q_ss"]
        if q_of(hi) < Q_want:
            return hi
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            if q_of(m) < Q_want:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- coupling to the 2-D model
@dataclass
class Coupling:
    """SI impact record -> solver.Inlet (cup units).  Positions are exact (D_L); the three rates carry
    one scale constant each (the V0.5 controls are in effective units, not SI):
        S_eff  = c_S * Q           (0.05 at 15 mL/s, the V0.5 heart disc)
        U_perp = c_U * U_perp_SI   (0.263 D_L/s at 0.70 m/s, the low pour)
        u_in   = c_u * u_h_SI      (0.5  D_L/s at 0.15 m/s)
    Footprint: V0.5 continuity ellipse from (S_eff/kappa_Q, U_perp, u_in)."""
    D_L: float = 0.08
    c_S: float = 0.05 / 15e-6
    c_U: float = 0.263 / 0.70
    c_u: float = 0.5 / 0.15
    kappa_Q: float = 125.0
    max_jet_angle: float = math.radians(60.0)
    axes: tuple = (0, 1)            # world (x, y) -> cup (x, y)
    footprint: str = "physical"     # "physical": exit-sheet ellipse with capillary rounding; "v05": continuity ellipse
    use_arrivals: bool = True       # feed the solver with what arrives this frame (flight delay), not what leaves

    def inlet(self, rec, scan_speed=0.0):
        J = rec.get("hit") if self.use_arrivals else rec.get("jet")
        Q = (J["Q"] if self.use_arrivals else rec["Q"]) if J else 0.0
        if J is None or Q <= 0:
            return Inlet(active=False)
        x = J["x_hit"][self.axes[0]] / self.D_L
        y = J["x_hit"][self.axes[1]] / self.D_L
        S = self.c_S * Q
        U = self.c_U * J["U_perp"]
        ux, uy = self.c_u * J["u_h"][self.axes[0]], self.c_u * J["u_h"][self.axes[1]]
        vt = math.hypot(ux, uy)
        if vt > 0:
            sc = min(1.0, U * math.tan(self.max_jet_angle) / vt)
            ux, uy, vt = ux * sc, uy * sc, vt * sc
        Q_rec = S / self.kappa_Q
        speed = math.hypot(U, vt)
        if self.footprint == "physical" and "r1" in J:
            r1, r2 = J["r1"] / self.D_L, J["r2"] / self.D_L
            phi = J["phi"]
            d = 2 * math.sqrt(r1 * r2)
        else:
            aspect = speed / U
            r = math.sqrt(Q_rec / (math.pi * U))
            r1, r2 = r * math.sqrt(aspect), r / math.sqrt(aspect)
            d = 2 * math.sqrt(Q_rec / speed / math.pi)
            phi = math.atan2(uy, ux)
        return Inlet(active=True, x_hit=(float(x), float(y)), S_eff=float(S), u_in=(float(ux), float(uy)),
                     U_perp=float(U), d_jet=float(d), r1=float(r1), r2=float(r2),
                     phi=float(phi), scan_speed=float(scan_speed))


# physics for the physical-velocity inlet: same model, B_dep rescaled so that chi ~ 0.85 at the low pour and ~ 0.14
# at the high cut.  The deposition scale is tied to how d_jet is defined: with the physical footprint
# (d = 2 sqrt(r1 r2): low pour 5 mm = 0.063 D_L, high cut 2.6 mm = 0.033 D_L) B_dep = 2.5; with the V0.5 continuity
# footprint (about half these diameters) it was 4.0.
PHYSICAL_PARAMS = dict(cp=0.3, beta=3.0, nu=1e-3, D=1e-7, kappa_Q=125.0, kappa_c=1.0, kappa_t=0.7, kappa_r=0.3,
                       B_dep=2.5, p_dep=2.0, return_law="v05")


# --------------------------------------------------------------------------- virtual barista
@dataclass
class Move:
    """One segment of the barista's intent: tip path in cup units, tip height [m], wanted flow [m^3/s]."""
    dur: float
    p0: tuple
    p1: tuple
    z0: float
    z1: float = None
    Q0: float = 0.0
    Q1: float = None
    wobble: float = 0.0     # lateral (x) wobble amplitude in cup units (start)
    wfreq: float = 0.0
    wobble1: float = None   # wobble amplitude at the end of the move (linear ramp)
    wobble_axis: str = "x"  # "x": lateral (across the spout direction), "y": fore-aft (along the spout direction)
    wobble_dir: float = None  # wiggle direction in the cup plane [rad] (0 = lateral x, pi/2 = fore-aft y); overrides wobble_axis
    bezier: tuple = None    # optional cubic Bezier control points ((c1x, c1y), (c2x, c2y)) between p0 and p1
    name: str = ""

    def __post_init__(self):
        self.z1 = self.z0 if self.z1 is None else self.z1
        self.Q1 = self.Q0 if self.Q1 is None else self.Q1
        self.wobble1 = self.wobble if self.wobble1 is None else self.wobble1


class Barista:
    """Turns a list of Moves into poses.  The stream (not the tip) follows the path: the tip is placed so that
    the previous frame's jet offset lands the stream on the intended point.  The wrist tracks the wanted
    flow with a rate-limited proportional controller on the tilt rate (the barista reads the stream, not the
    angle); before the milk reaches the lip the wrist swings at the full rate."""

    def __init__(self, pitcher, moves, coupling, yaw=-math.pi / 2, gain=math.radians(0.8) / 1e-6,
                 gain_i=math.radians(1.5) / 1e-6, max_rate=math.radians(60.0), swing_rate=math.radians(60.0),
                 tail=1.0, cup=None, tilt0=math.radians(40.0), hold_rate=math.radians(4.0), feedforward=True,
                 tau_ff=0.1):
        self.pitcher, self.moves, self.cp = pitcher, moves, coupling
        self.cup = Cup() if cup is None else cup           # None-safe: pass cup=False to disable the collision lift
        self.lift = 0.0
        self.tilt0 = tilt0                                 # the barista arrives already tilted, just below the flow onset
        self.yaw, self.gain, self.gain_i, self.max_rate, self.swing_rate = yaw, gain, gain_i, max_rate, swing_rate
        self.hold_rate = hold_rate      # slow tilt-back once the stream has stopped (stays just below the onset head)
        self.feedforward, self.tau_ff = feedforward, tau_ff
        self.z_cmd, self.z_rate, self.tau_z = None, 0.25, 0.04   # smoothed spout-height command [m], [m/s], [s]
        self._ff = (None, None, 0.0)    # cache: (Q_want, V, tilt) of the last feed-forward solve

        self.T = sum(m.dur for m in moves) + tail
        self.tilt = tilt0
        self.ierr = 0.0                 # integrated flow error [m^3]
        self.offset = np.zeros(2)       # world xy: hit point - tip, low-passed (the barista corrects the drift, not the wiggle)
        self.Q_seen = 0.0               # low-passed flow the barista reacts to
        self.tau_see = 0.25             # [s]

    def _tilt_ff(self, pose, Q_want):
        """Steady tilt for Q_want at the current fill (bisection on the weir law in a bracket around the current
        tilt); recomputed only when the wanted flow or the fill changed by more than 3 %."""
        q0, v0, t0 = self._ff
        V = self.pitcher.state.V
        if q0 is not None and abs(Q_want - q0) <= 0.03 * q0 and abs(V - v0) <= 0.03 * v0:
            return t0
        lo, hi = max(0.0, self.tilt - math.radians(12)), min(math.radians(150), self.tilt + math.radians(12))
        t = self.pitcher.tilt_for_flow(pose, Q_want, V=V, lo=lo, hi=hi, iters=10)
        self._ff = (Q_want, V, t)
        return t

    def intent(self, t):
        acc = 0.0
        for m in self.moves:
            if t < acc + m.dur:
                f = (t - acc) / m.dur
                e = 0.5 - 0.5 * math.cos(math.pi * f)
                amp = m.wobble + (m.wobble1 - m.wobble) * f
                amp *= min(1.0, (t - acc) / 0.25, (acc + m.dur - t) / 0.25)   # ease the wiggle in and out (no impulsive start)
                wob = amp * math.sin(2 * math.pi * m.wfreq * (t - acc))
                if m.bezier is not None:
                    (ax, ay), (bx, by) = m.bezier
                    x = (1 - e) ** 3 * m.p0[0] + 3 * (1 - e) ** 2 * e * ax + 3 * (1 - e) * e * e * bx + e ** 3 * m.p1[0]
                    y = (1 - e) ** 3 * m.p0[1] + 3 * (1 - e) ** 2 * e * ay + 3 * (1 - e) * e * e * by + e ** 3 * m.p1[1]
                else:
                    x = m.p0[0] + (m.p1[0] - m.p0[0]) * e
                    y = m.p0[1] + (m.p1[1] - m.p0[1]) * e
                if m.wobble_dir is not None:
                    x += wob * math.cos(m.wobble_dir); y += wob * math.sin(m.wobble_dir)
                else:
                    x += wob if m.wobble_axis == "x" else 0.0
                    y += wob if m.wobble_axis == "y" else 0.0
                z = m.z0 + (m.z1 - m.z0) * e
                Q = m.Q0 + (m.Q1 - m.Q0) * f
                return (x, y), z, Q, m.name
            acc += m.dur
        last = self.moves[-1]
        return last.p1, last.z1, 0.0, "settle"

    def pose_at(self, t, dt=None):
        (x, y), z, Q_want, name = self.intent(t)
        if dt is not None:
            # the hand cannot jump: the commanded spout height follows the intent with a rate limit (0.25 m/s, a 5 cm
            # lift takes ~0.25 s) and a short low-pass.  An instantaneous jump between moves kicked the slosh mode and
            # produced a 20-30 mL/s burst at the start of every high cut.
            if self.z_cmd is None:
                self.z_cmd = z
            dz = max(-self.z_rate * dt, min(self.z_rate * dt, z - self.z_cmd))
            self.z_cmd += dz * (1.0 - math.exp(-dt / self.tau_z)) if abs(z - self.z_cmd) < self.z_rate * dt else dz
            z = self.z_cmd
        tip = (x * self.cp.D_L - self.offset[0], y * self.cp.D_L - self.offset[1], z)
        return Pose(tip=tip, yaw=self.yaw, tilt=self.tilt, roll=0.0), Q_want, name

    def step(self, dt):
        """One control frame: update the tilt from the flow error, advance the pitcher.  Returns (record, label)."""
        t = self.pitcher.state.t
        pose, Q_want, name = self.pose_at(t, dt)
        st = self.pitcher.state
        self.Q_seen += (st.Q - self.Q_seen) * (1.0 - math.exp(-dt / self.tau_see))
        if Q_want <= 0:
            # stop: swing back only until the head is off the lip, then hold just below it (a barista does not
            # straighten the pitcher between two petals; the next phase must restart the stream in ~0.1 s)
            rate = -self.swing_rate if st.flowing else -self.hold_rate
            self.ierr = 0.0
        elif not st.flowing:
            rate = self.swing_rate                       # milk not at the lip (or just left it): tilt until it is
        else:
            # feed-forward wrist: the barista knows roughly how far to tilt for the flow they want (the steady weir
            # law at the current fill) and goes there within ~0.1 s; the PI term only trims what they see.
            # Without it a 0.25 s head or beak pour never reaches its flow (PI alone moves 6 deg/s).
            if self.feedforward:
                ff = self._tilt_ff(pose, Q_want)
                rate_ff = max(-self.max_rate, min(self.max_rate, (ff - self.tilt) / self.tau_ff))
            else:
                rate_ff = 0.0
            err = Q_want - self.Q_seen
            kp, ki = (self.gain * 0.5, self.gain_i * 0.3) if self.feedforward else (self.gain, self.gain_i)
            self.ierr = max(-2e-6, min(2e-6, self.ierr + err * dt))
            rate = max(-self.max_rate, min(self.max_rate, rate_ff + kp * err + ki * self.ierr))
        self.tilt = max(0.0, min(math.radians(150), self.tilt + rate * dt))
        pose.tilt = self.tilt
        if self.cup:
            self.lift = self.pitcher.required_lift(pose, self.cup)
            if self.lift > 0:
                pose.tip = (pose.tip[0], pose.tip[1], pose.tip[2] + self.lift)
        (x1, y1), _, _, _ = self.intent(t + dt)
        (x0, y0), _, _, _ = self.intent(t)
        scan = math.hypot(x1 - x0, y1 - y0) / dt
        rec = self.pitcher.step(pose, dt)
        if rec["jet"] is not None:
            off = rec["jet"]["x_hit"] - np.asarray(pose.tip[:2])
            self.offset = self.offset + (off - self.offset) * (1.0 - math.exp(-dt / 0.3))
        rec["Q_want"] = Q_want
        rec["label"] = name
        rec["lift"] = self.lift
        rec["inlet"] = self.cp.inlet(rec, scan_speed=scan)
        rec["off_cup"] = bool(rec["inlet"].active and math.hypot(*rec["inlet"].x_hit) > 0.47)   # stream at/outside the rim
        if rec["off_cup"] and getattr(self, "drop_off_cup", False):    # old rule; default: the solver accounts the
            rec["inlet"] = Inlet(active=False)                         # off-cup part (Params.offcup)
        return rec, name


def heart_moves():
    """Heart as the V0.5 script does it, written as barista intent instead of inlet records:
    low disc at the far side, approach the centre, lift and cut back through."""
    return [
        Move(4.0, (0.0, 0.20), (0.0, 0.16), z0=0.015, Q0=15e-6, name="disc: low, 15 mL/s"),
        Move(1.6, (0.0, 0.16), (0.0, 0.00), z0=0.015, z1=0.02, Q0=9e-6, name="approach centre, 9 mL/s"),
        Move(0.35, (0.0, 0.00), (0.0, 0.00), z0=0.02, z1=0.07, Q0=5e-6, Q1=4e-6, name="lift"),
        Move(0.8, (0.0, 0.00), (0.0, -0.27), z0=0.07, Q0=4e-6, Q1=1e-6, name="cut through, high"),
        Move(0.5, (0.0, -0.27), (0.0, -0.27), z0=0.07, z1=0.09, Q0=0.0, name="stop"),
        Move(1.0, (0.0, -0.27), (0.0, -1.8), z0=0.09, z1=0.14, Q0=0.0, name="move away"),
    ]


def layered_heart_moves(axis="x", amp=1.0, freq=2.5):
    """V0.5 layered heart as intent: low wiggle pour building the base (amplitude ramp), a held wiggle, then a high
    cut.  Flow in V0.5 effective units 0.02->0.045 and 0.045 ~ 6->13.5 and 13.5 mL/s.  axis "x" = lateral wiggle
    (V0.5), "y" = fore-aft; amp scales the V0.5 amplitudes (0.008->0.025, then 0.045 cup diameters)."""
    return [
        Move(3.2, (0.0, 0.08), (0.0, 0.14), z0=0.015, Q0=6e-6, Q1=13.5e-6, wobble=0.008 * amp, wobble1=0.025 * amp,
             wfreq=freq, wobble_axis=axis, name="wiggle base, 6->13.5 mL/s"),
        Move(3.2, (0.0, 0.14), (0.0, 0.14), z0=0.015, Q0=13.5e-6, wobble=0.045 * amp, wfreq=freq, wobble_axis=axis,
             name="held wiggle 13.5 mL/s"),
        Move(0.3, (0.0, 0.14), (0.0, 0.14), z0=0.015, z1=0.07, Q0=5e-6, Q1=4e-6, name="lift"),
        Move(1.0, (0.0, 0.14), (0.0, -0.32), z0=0.07, Q0=4e-6, Q1=1e-6, name="cut through, high"),
        Move(0.5, (0.0, -0.32), (0.0, -0.32), z0=0.07, z1=0.09, Q0=0.0, name="stop"),
        Move(1.0, (0.0, -0.32), (0.0, -1.8), z0=0.09, z1=0.14, Q0=0.0, name="move away"),
    ]


# V0.5 effective flow S -> physical flow: S = c_S Q with c_S = 0.05 / 15 mL/s
S_TO_Q = 15e-6 / 0.05
# V0.5 footprint radius -> pitcher height above the surface (the radius encodes the pouring height through the
# continuity rule U_perp = Q/(pi r^2)); log-linear table, cup-diameter units -> metres
_R_Z = [(0.035, 0.015), (0.022, 0.015), (0.015, 0.02), (0.008, 0.035), (0.004, 0.07), (0.0015, 0.09)]


def height_from_radius(r):
    import numpy as _np
    rs = _np.log([a for a, _ in _R_Z][::-1]); zs = [b for _, b in _R_Z][::-1]
    return float(_np.interp(math.log(max(r, 1e-4)), rs, zs))


def away_point(p, dist=1.8):
    """Point `dist` cup diameters from the cup centre, away along the barista's side (-y), keeping x."""
    return (p[0], -math.sqrt(max(dist * dist - p[0] * p[0], 0.0)))


def v05_moves(name):
    """Barista intent from a V0.5 control file: stream path, height (from the footprint radius), wanted flow
    (from S), lateral wiggle (amplitude/frequency, with ramps); flow pulsation and jet wiggle are dropped
    (the pitcher physics has to produce them); zero-flow bridges become Q = 0 moves; a move-away is appended."""
    from .v05_program import load_control, Program
    prog = Program(load_control(name))
    moves = []
    for p in prog.phases:
        Q0 = p["Q"] * S_TO_Q; Q1 = p.get("Q_end", p["Q"]) * S_TO_Q
        z0 = height_from_radius(p["radius"]); z1 = height_from_radius(p.get("radius_end", p["radius"]))
        if Q0 <= 0 and Q1 <= 0:                      # bridges and stops: hold at least 3 cm, or the previous height
            z0 = z1 = max(moves[-1].z1 if moves else 0.0, 0.03)
        bez = tuple(tuple(c) for c in p["bezier_controls"]) if p.get("bezier_controls") else None
        label = "reposition" if p.get("purpose") else ("wiggle" if p.get("amplitude", 0) > 0 and p.get("frequency", 0) > 0 else ("stop" if Q0 <= 0 else "pour"))
        moves.append(Move(p["duration"], tuple(p["start"]), tuple(p["end"]), z0=z0, z1=z1, Q0=Q0, Q1=Q1,
                          wobble=p.get("amplitude", 0.0), wobble1=p.get("amplitude_end", p.get("amplitude", 0.0)),
                          wfreq=p.get("frequency", 0.0), bezier=bez,
                          name=f"{label} {Q0*1e6:.0f}->{Q1*1e6:.0f} mL/s, h {z0*100:.1f}->{z1*100:.1f} cm"))
    last = moves[-1]
    moves.append(Move(1.0, last.p1, away_point(last.p1), z0=max(last.z1, 0.09), z1=0.14, Q0=0.0, name="move away"))
    return moves


def moves_to_json(moves):
    return [dict(dur=m.dur, p0=list(m.p0), p1=list(m.p1), z0=m.z0, z1=m.z1, Q0=m.Q0, Q1=m.Q1, wob=m.wobble, wob1=m.wobble1,
                 wf=m.wfreq, axis=m.wobble_axis, bez=([list(m.bezier[0]), list(m.bezier[1])] if m.bezier else None), name=m.name)
            for m in moves]


def v05_moves_low(name, z_low=0.015):
    """Same as v05_moves but every pouring move below 5 cm is put at z_low (tests the height map's effect)."""
    mv = v05_moves(name)
    for m in mv:
        if m.Q0 > 0 and m.z0 < 0.05:
            m.z0 = z_low; m.z1 = min(m.z1, z_low) if m.z1 < 0.05 else m.z1
    return mv


MOVES = dict(heart=heart_moves, layered_heart=layered_heart_moves, tulip_low=lambda: v05_moves_low("tulip"),
             v05_heart=lambda: v05_moves("heart"), push_heart=lambda: v05_moves("push_heart"),
             v05_layered_heart=lambda: v05_moves("layered_heart"), tulip=lambda: v05_moves("tulip"),
             leaf=lambda: v05_moves("leaf"), swan=lambda: v05_moves("swan"),
             layered_heart_y=lambda: layered_heart_moves(axis="y"),
             layered_heart_big=lambda: layered_heart_moves(axis="x", amp=2.2, freq=4.0))


class BaristaScript:
    """Adapter with the run.py sampler interface (sample(t) -> (Inlet, label)) driven frame by frame.
    The pitcher is stateful, so sample() must be called with non-decreasing t; it advances the pitcher
    to t and returns the latest record."""

    def __init__(self, moves=None, pitcher=None, coupling=None, dt=1.0 / 120.0, V0=2.0e-4, pattern="heart",
                 law=None, geometry="param", **kw):
        if pitcher is None:
            geom = None if geometry == "param" else GridPitcherGeometry.load(None if geometry == "nx" else geometry)
            pitcher = Pitcher(geom=geom, V0=V0, law=law)
        self.pitcher = pitcher
        self.cp = coupling or Coupling()
        if moves is None:
            moves = MOVES[pattern]()
        self.barista = Barista(self.pitcher, moves, self.cp, **kw)
        self.dt = dt
        self.T = self.barista.T
        self.name = f"pitcher_{pattern}"
        self.records = []
        self.params = dict(PHYSICAL_PARAMS)
        self._last = (Inlet(active=False), "start")

    def sample(self, t):
        """Advance the pitcher on its own fixed clock (dt = 1/120 s) up to t and return the latest record.  Fixed
        steps keep the finite-difference kinematics clean: irregular steps (run.py shortens frames during fast
        scans) made the acceleration estimate jump, the sloshing surface tilt, and the flow surge."""
        while self.pitcher.state.t + self.dt <= t + 1e-9:
            rec, name = self.barista.step(self.dt)
            self.records.append(rec)
            self._last = (rec["inlet"], name)
        return self._last


def records_table(records):
    """Flatten the records into float arrays for plotting / CSV."""
    keys = ["t", "V", "Q", "Q_want", "Q_ss", "h_max", "h_bar", "tilt"]
    out = {k: np.array([r[k] for r in records], float) for k in keys}
    out["z_tip"] = np.array([r["tip"][2] for r in records])
    J = [r["jet"] for r in records]
    out["U_perp"] = np.array([j["U_perp"] if j else np.nan for j in J])
    out["u_h"] = np.array([np.hypot(*j["u_h"]) if j else np.nan for j in J])
    out["d_hit"] = np.array([j["d_hit"] if j else np.nan for j in J])
    out["d0"] = np.array([j["d0"] if j else np.nan for j in J])
    out["x_hit"] = np.array([j["x_hit"][0] if j else np.nan for j in J])
    out["y_hit"] = np.array([j["x_hit"][1] if j else np.nan for j in J])
    out["drop"] = np.array([j["drop"] if j else np.nan for j in J])
    out["coherent"] = np.array([j["coherent"] if j else True for j in J])
    out["acc_x"] = np.array([r["acc"][0] if "acc" in r else 0.0 for r in records])
    out["slope_x"] = np.array([r["slope"][0] if "slope" in r else 0.0 for r in records])
    out["g_eff"] = np.array([r.get("g_eff", G) for r in records])
    out["lift"] = np.array([r.get("lift", 0.0) for r in records])
    for k in ("Q_feed", "Q_hit", "V_lip", "V_jet", "V_cup"):
        out[k] = np.array([r.get(k, 0.0) for r in records])
    out["label"] = [r.get("label", "") for r in records]
    return out


# =========================================================================== forward, pose-driven pitcher (V1 mode B)
# Input: a continuous pose series T_WP(t) (tip position, yaw, tilt, roll) and the initial milk volume.  Output: the
# arrival record at the coffee surface (z = 0): received flow, hit position, hit material velocity, A_perp, footprint.
#   step 1  effective spout front edge = the contiguous beak arc of the rim (r_rim above the plain rim radius); segment
#           widths are PROJECTED perpendicular to the main outflow direction (not the rim arc length)
#   step 2  rigid-body lip velocity v_lip,i = v_tip + omega x (m_i - tip), from the pose series (central differences)
#   step 3  quasi-static horizontal free surface z_f from V_P (no slosh); head h_i = [z_f - z_lip,i]_+ (no Kindsvater
#           subtraction, no start / stop hysteresis; dry lip: Q = 0 when max_i h_i <= 0)
#   step 4  Q_i = 2/3 C_d sqrt(2 g) dw_i h_i^1.5 over the front edge
#   step 5  (optional, off) one shared outflow-speed state U: L_e dU/dt = g H - U|U| / (2 C_v^2), Q_o = A_* [U]_+
#   step 6  dV_P/dt = -Q_o; ledger V_P + V_jet + V_cup + V_spill = V_P(0)
#   step 7  v_o,i = v_lip,i + U_i d_o,i, U = C_v sqrt(2 g H) (H = max head), d_o,i = local outward lip normal tilted
#           down by the lip slope; flux-weighted origin, velocity and direction; A_o = sum Q_i / U_i
#   step 7b (optional, calibrated from the measured stream, off by default) the exit direction is lifted by exit_lift
#           towards the spout heading (horizontal part of e_main) and the jet origin moved by anchor along it: the
#           milk leaves the beak after running down the spout channel, flatter and further forward than the lip normal
#   step 8  ballistic parcels (emission time, volume, origin, velocity) to z = 0; arrivals binned on the output grid
# exit 'wall' (mi27, SPEC27 rev 2 section 3.1; port of mi27/M5 exitlaw.py, sha256 b7c3a8ba...) REPLACES steps 7 / 7b / 8
# (it is not added on top: exit_lift and anchor must be 0):
#   P1 direction  e_o = flux-weighted unit tangent of the INNER wall at the rim, perpendicular to the rim edge, out of the
#                 opening (an ASSUMED rule, not derived from the momentum balance and not tested by the measurement);
#                 rising exits clipped to horizontal when clip_rise (switch; True reproduces M5)
#   P2 speed      U_o = C_visc k_M sqrt(2 g H): k_M from the momentum balance between a hydrostatic broad-crested critical
#                 section (H = y_c + A / 2B, Q_c^2 = g A^3 / B) and the free jet (u_j = Q_c / A + g S1 / Q_c), with the
#                 rim profile of the CAD; C_visc = 1 - theta_m P_b / A_b (laminar boundary layer, Thwaites, over c_L H).
#                 Mass stays the sharp-crested weir's (C_d), the speed uses the broad-crested section: two declared crest
#                 idealisations (Q_w / Q_c ~ 1.10 exported as E['Qc'])
#   P3 origin     centroid of the brink section at depth kappa_b y_c; lip velocity v_tip + omega x (origin - tip)
#   P4 flight     capillary jet (jet_flight): M = rho Q v_rel - T t_hat, dM/dt = -rho Q g e_z in the frame translating with
#                 the lip velocity at emission; capillary 'round' (pi sigma a, the declared rule-2 choice), 'sheet'
#                 (sigma (W_b + P_b) relaxing to the round value on the Taylor-Culick time) or 'none' (ballistic)
# Parameters rho, sigma, nu, c_L, kappa_b: independent estimates (M5 report section C), none fitted.
@dataclass
class ForwardLaw:
    C_d: float = 0.60           # weir discharge coefficient
    C_v: float = 1.0            # velocity coefficient: U = C_v sqrt(2 g H)
    U_head: str = "max"         # 'max': H = max_i h_i (V1); 'local': U_i = C_v sqrt(2 g h_i); 'bar': discharge-weighted head
    width: str = "projected"    # 'projected': dw_i perpendicular to the main outflow direction (V1); 'arc': horizontal arc length
    h_k: float = 0.0            # dead head subtracted from h_i (0 = V1; the old Pitcher uses 0.8 mm) -- ablation only
    front_dr: float = 0.5e-3    # front edge: rim points with r_rim > median rim radius + front_dr, contiguous around the beak
    side_spill: bool = True     # rim outside the front edge under the surface overflows: counted as V_spill, not delivered
    inertia: bool = False       # step 5 (off by default)
    L_e: float = 0.02           # effective inertial length [m] (step 5 only)
    smooth_s: float = 0.0       # gaussian sigma [s] on the pose before differentiation (0: none)
    exit_lift: float = 0.0      # [rad] step 7b: exit direction lifted towards the spout heading (0: lip normal, V1)
    anchor: float = 0.0         # [m] step 7b: jet origin moved along the spout heading (0: V1)
    g: float = G
    # mi27 wall exit (see the comment above); 'lip' keeps steps 7 / 7b / 8 exactly as before
    exit: str = "lip"           # 'lip' (steps 7, 7b, 8) | 'wall' (P1-P4; needs a GridPitcherGeometry)
    capillary: str = "round"    # 'round' | 'sheet' | 'none' (exit 'wall' only)
    rho: float = 800.0          # [kg/m^3] steamed microfoam milk (range 600-1030)
    sigma: float = 0.045        # [N/m] milk surface tension at 50-65 C (0.040-0.052)
    nu: float = 1.5e-6          # [m^2/s] bubbly milk (1e-6 - 5e-6)
    c_L: float = 4.0            # wetted acceleration length L_w = c_L H (2-8)
    kappa_b: float = 0.75       # brink depth / critical depth (0.70-0.80)
    clip_rise: bool = True      # rising wall tangent at the edge -> horizontal exit


CAPILLARY = ("round", "sheet", "none")


def jet_flight(origin, v_lip, e, U0, Q, A_b, W_b, P_b, rho, sigma, capillary, g=G, dt=2.5e-4, t_max=0.6):
    """P4 + step 8 of the wall exit (port of mi27/M5 exitlaw.flight, arithmetic unchanged): vectorised steady-jet flight
    of n parcels to z = 0 with the capillary tension law, in the frame translating with the lip velocity at emission.
    Arrays (n,3) / (n,).  M = rho Q v_rel - T t_hat with dM/dt = -rho Q g e_z, |v_rel| = (|M| + T) / (rho Q),
    T = pi sigma a(|v|) (+ the sheet excess relaxing on the Taylor-Culick time); M_0 = max(rho Q U0 - T_0, 0) e (teapot
    limit).  Midpoint steps of dt to z = 0 (linear interpolation of the crossing); with sigma = 0 it equals closed-form
    ballistics to ~1e-7 s.  Returns t_f (n,), x_hit (n,2), v_hit (n,3), T0 (n,), |M0| (n,); parcels that do not reach
    z = 0 within t_max keep t_f = nan."""
    n = len(U0)
    rho, sig = float(rho), float(sigma)
    mode = capillary
    if mode not in CAPILLARY:
        raise ValueError(f"capillary must be one of {CAPILLARY}, got {mode!r}")
    if mode == "none":
        sig = 0.0
    rQ = rho * Q
    a0 = np.sqrt(Q / (math.pi * np.maximum(U0, 1e-6)))   # continuity with the emitted flow
    if mode == "sheet":
        T0 = sig * (W_b + P_b)
    else:
        T0 = math.pi * sig * a0
    T0 = np.minimum(T0, rQ * U0)                       # teapot limit: the wall takes the excess tension
    Mmag0 = rQ * U0 - T0
    M0 = Mmag0[:, None] * e
    hbar = Q / np.maximum(U0, 1e-6) / np.maximum(W_b, 1e-6)   # mean sheet thickness at detachment
    uTC = np.sqrt(2 * sig / (rho * hbar)) if sig > 0 else np.full(n, np.inf)
    tcap = np.where(np.isfinite(uTC), 0.5 * W_b / np.maximum(uTC, 1e-9), 0.0)
    Tex0 = T0 - math.pi * sig * a0                    # excess over the round-jet tension (relaxes on t_cap)

    p = origin.copy()
    done = p[:, 2] <= 0
    tf = np.where(done, 0.0, np.nan)
    xh = np.where(done[:, None], p[:, :2], np.nan)
    vh = np.full((n, 3), np.nan)

    def vrel(t):
        M = M0.copy(); M[:, 2] -= rQ * g * t
        Mm = np.linalg.norm(M, axis=1)
        Mh = np.where(Mm[:, None] > 1e-15, M / np.maximum(Mm, 1e-30)[:, None], np.array([0.0, 0.0, -1.0]))
        Tex = Tex0 * (np.exp(-t / np.maximum(tcap, 1e-9)) if np.ndim(t) == 0 else 1.0)
        sp = (Mm + T0) / rQ                           # first guess
        for _ in range(4):                             # T = pi sigma a(|v|) + excess, a from continuity
            T = math.pi * sig * np.sqrt(Q / (math.pi * np.maximum(sp, 1e-6))) + Tex
            sp = (Mm + T) / rQ
        return sp[:, None] * Mh

    if done.all():
        v0 = v_lip + vrel(0.0)
        return tf, xh, v0, T0, Mmag0
    t = 0.0
    v_prev = v_lip + vrel(0.0)
    while t < t_max and not done.all():
        v_mid = v_lip + vrel(t + 0.5 * dt)
        v_new = v_lip + vrel(t + dt)
        p_new = p + v_mid * dt
        cross = (~done) & (p_new[:, 2] <= 0)
        if cross.any():
            f = p[cross, 2] / np.maximum(p[cross, 2] - p_new[cross, 2], 1e-12)
            tf[cross] = t + f * dt
            xh[cross] = p[cross, :2] + f[:, None] * (p_new[cross, :2] - p[cross, :2])
            vh[cross] = v_prev[cross] + f[:, None] * (v_new[cross] - v_prev[cross])
            done |= cross
        p, v_prev, t = p_new, v_new, t + dt
    v0 = v_lip + vrel(0.0)
    vh[np.isnan(vh[:, 0])] = v0[np.isnan(vh[:, 0])]
    return tf, xh, vh, T0, Mmag0


def ballistic_flight(origin, v0, g=G):
    """Closed-form ballistics to z = 0 (step 8 / the reference of jet_flight): t_f, x_hit (n,2), v_hit (n,3)."""
    vz, z0 = v0[:, 2], origin[:, 2]
    tf = np.where(z0 > 0, (vz + np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))) / g, 0.0)
    xh = origin[:, :2] + v0[:, :2] * tf[:, None]
    vh = np.c_[v0[:, :2], -np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))]
    return tf, xh, vh


class ForwardPitcher:
    """Pose-driven pitcher (V1 mode B).  Geometry: any object with the GridPitcherGeometry interface (points, dv, rim,
    tip, lip_slope); `front` (bool mask over rim points) overrides the beak detection.  The old Pitcher is untouched."""

    NB = 256                                           # bins of the fast free-surface quantile

    def __init__(self, geom=None, law=None, front=None):
        self.geom = geom if geom is not None else GridPitcherGeometry.load()
        self.law = law or ForwardLaw()
        self.n = np.array([0.0, 0.0, 1.0])
        rim = np.asarray(self.geom.rim, float)
        nr = len(rim)
        r = np.hypot(rim[:, 0], rim[:, 1])
        if front is None:
            hi = r > np.median(r) + self.law.front_dr
            i0 = int(np.argmax(r))
            front = np.zeros(nr, bool)
            for step, i in ((1, i0), (-1, i0 - 1)):   # contiguous run (periodic) through the beak maximum
                while hi[i % nr] and not front[i % nr]:
                    front[i % nr] = True
                    i += step
        self.front_pts = np.asarray(front, bool)
        # segments (i, i+1) with both ends on the front edge, ordered along the rim
        idx = np.nonzero(self.front_pts & np.roll(self.front_pts, -1))[0]
        if len(idx) < nr:                              # start the ordering after a gap
            gap = np.nonzero(~self.front_pts)[0]
            start = (gap[-1] + 1) % nr if len(gap) else 0
            idx = idx[np.argsort((idx - start) % nr)]
        self.seg_a, self.seg_b = idx, (idx + 1) % nr
        self.side_a = np.nonzero(~(self.front_pts & np.roll(self.front_pts, -1)))[0]
        self.side_b = (self.side_a + 1) % nr
        A, B = rim[self.seg_a], rim[self.seg_b]
        self.mid_P = 0.5 * (A + B)
        self.seg_P = B - A
        Z = np.array([0.0, 0.0, 1.0])
        nrm = np.cross(self.seg_P, Z)                  # tangent x up = outward normal for a counter-clockwise rim
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        if np.mean(np.sum(nrm[:, :2] * (self.mid_P[:, :2] - rim[:, :2].mean(0)), 1)) < 0:
            nrm = -nrm
        a = float(getattr(self.geom, "lip_slope", 0.0))
        self.dir_P = math.cos(a) * nrm - math.sin(a) * Z[None, :]
        w = np.maximum(r - np.median(r), 0.0) * self.front_pts
        self.beak_phi = float(math.atan2((w * np.sin(np.arctan2(rim[:, 1], rim[:, 0]))).sum(),
                                         (w * np.cos(np.arctan2(rim[:, 1], rim[:, 0]))).sum()))
        self.side_mid_P = 0.5 * (rim[self.side_a] + rim[self.side_b])
        self.side_seg_P = rim[self.side_b] - rim[self.side_a]
        exit_ = getattr(self.law, "exit", "lip")
        if exit_ not in ("lip", "wall"):
            raise ValueError(f"ForwardLaw.exit must be 'lip' or 'wall', got {exit_!r}")
        if exit_ == "wall":
            if self.law.exit_lift or self.law.anchor:
                raise ValueError("ForwardLaw(exit='wall') replaces steps 7 / 7b: exit_lift and anchor must be 0 "
                                 "(EXIT_CAL is refused under the wall exit)")
            if self.law.capillary not in CAPILLARY:
                raise ValueError(f"ForwardLaw.capillary must be one of {CAPILLARY}, got {self.law.capillary!r}")
            self._wall_dirs()

    # ---- wall exit (mi27; port of M5 exitlaw.ExitPitcher)
    def _wall_dirs(self):
        """P1 geometry: body-frame exit tangent of every front segment from the CAD inner wall r_in(z, phi) (central
        differences, dz 0.5 mm, dphi 1 deg), the outward wall normal, dr/dz at the rim point, the 3-D rim length per
        segment.  Requires a GridPitcherGeometry (zs, phis, r_in)."""
        g = self.geom
        if not all(hasattr(g, k) for k in ("zs", "phis", "r_in")):
            raise TypeError("ForwardLaw(exit='wall') needs a GridPitcherGeometry (CAD r_in(z, phi)), got "
                            f"{type(g).__name__}")
        from scipy.interpolate import RegularGridInterpolator
        ph = np.concatenate([g.phis - 2 * math.pi, g.phis, g.phis + 2 * math.pi])
        rr = np.concatenate([g.r_in, g.r_in, g.r_in], axis=1)
        f = RegularGridInterpolator((g.zs, ph), rr, bounds_error=False, fill_value=None)
        P = self.mid_P
        phi = np.arctan2(P[:, 1], P[:, 0])
        z = P[:, 2]
        dz, dp = 0.5e-3, math.radians(1.0)
        r = f(np.c_[z, phi])
        r_z = (f(np.c_[z + dz, phi]) - f(np.c_[z - dz, phi])) / (2 * dz)
        r_p = (f(np.c_[z, phi + dp]) - f(np.c_[z, phi - dp])) / (2 * dp)
        c, s = np.cos(phi), np.sin(phi)
        Pz = np.stack([r_z * c, r_z * s, np.ones_like(z)], 1)                 # dP/dz (up the wall)
        Pp = np.stack([r_p * c - r * s, r_p * s + r * c, np.zeros_like(z)], 1)  # dP/dphi
        nw = np.cross(Pp, Pz)
        nw /= np.linalg.norm(nw, axis=1)[:, None]                             # outward wall normal
        d = np.cross(nw, self.seg_P)
        d /= np.linalg.norm(d, axis=1)[:, None]
        d *= np.sign(np.sum(d * Pz, 1))[:, None]                               # out of the opening
        self.dirW_P = d                                                        # exit tangent (body)
        self.wall_n_P = nw
        self.wall_slope = r_z                                                  # dr/dz at the rim point
        self.seglen = np.linalg.norm(self.seg_P, axis=1)                       # 3-D rim length per segment
        i0 = int(np.argmin(np.abs(phi)))
        self.beta_tip_deg = float(np.degrees(np.arctan(r_z[i0])))

    @staticmethod
    def _section(y, b, dw, seglen):
        """Strip discretisation of the rim profile across the main direction at water depth y above the lowest rim
        point: area A, top width B, first moment S1 = int (y - b)^2 / 2, wetted rim length P."""
        e = np.maximum(y - b, 0.0)
        wet = b < y
        return dict(A=float((dw * e).sum()), B=float(dw[wet].sum()), S1=float((dw * e * e).sum() / 2), P=float(seglen[wet].sum()))

    def _wall_exit(self, pose, z_f, v_tip, omega, out):
        """P1-P3 at one step (port of M5 ExitPitcher.weir): None when the lip is dry, else dict(e, U0, kM, Cvis, yc, yb,
        A_b, W_b, P_b, Qc, origin, v_lip, rising, elev_raw, H, n_wet)."""
        law = self.law
        G_ = law.g
        if not (out["Q"] > 0 and out["H"] > 0):
            return None
        R = rotation(pose.yaw, pose.tilt, pose.roll)
        tip = np.asarray(pose.tip, float)
        mid = (self.mid_P - self.geom.tip) @ R.T + tip
        seg = self.seg_P @ R.T
        dW = self.dirW_P @ R.T
        Qi = out["Qi"]
        w = Qi / out["Q"]
        e = (w[:, None] * dW).sum(0)
        e /= np.linalg.norm(e)
        elev_raw = math.degrees(math.asin(max(-1.0, min(1.0, e[2]))))
        rising = e[2] > 0
        if rising and law.clip_rise:
            e = np.array([e[0], e[1], 0.0]); e /= np.linalg.norm(e)
        # section across the horizontal main direction
        eh = np.array([e[0], e[1], 0.0]); nh = np.linalg.norm(eh)
        eh = eh / nh if nh > 1e-9 else out["e_main"]
        sh = seg.copy(); sh[:, 2] = 0.0
        dw = np.linalg.norm(sh - (sh @ eh)[:, None] * eh[None, :], axis=1)
        zlow = float(mid[:, 2].min())
        b = mid[:, 2] - zlow
        H = float(z_f - zlow)
        # critical depth: H = y + A/(2B)
        lo, hi = 0.0, H
        for _ in range(40):
            m = 0.5 * (lo + hi)
            s = self._section(m, b, dw, self.seglen)
            if s["B"] <= 0 or m + s["A"] / (2 * s["B"]) < H:
                lo = m
            else:
                hi = m
        yc = 0.5 * (lo + hi)
        sc = self._section(yc, b, dw, self.seglen)
        if not (sc["A"] > 0 and sc["B"] > 0):
            return None
        Qc = math.sqrt(G_ * sc["A"] ** 3 / sc["B"])
        uj = Qc / sc["A"] + G_ * sc["S1"] / Qc
        kM = uj / math.sqrt(2 * G_ * H)
        yb = law.kappa_b * yc
        sb = self._section(yb, b, dw, self.seglen)
        if sb["A"] <= 0:
            sb = sc; yb = yc
        Ui = kM * math.sqrt(2 * G_ * H)
        # Thwaites, theta^2 = 0.45 nu / U^6 int U^5 dx, U rising linearly from 0 to Ui over L_w = c_L H
        th_m = math.sqrt(0.45 / 6.0) * math.sqrt(law.nu * law.c_L * H / Ui)
        # numerical guard: inactive on every flowing step (Q > 1 mL/s, T4); it is hit on some trickle steps below 1 mL/s
        # at stroke ends (counted in real_program info['inlet'] Cvis_guard_hits; build/DEVIATIONS.md item 3)
        Cvis = max(0.5, 1.0 - th_m * sb["P"] / sb["A"])
        U0 = Cvis * Ui
        wet = (yb - b) > 0
        wa = dw * np.maximum(yb - b, 0.0)
        cen = (wa[:, None] * mid).sum(0) / wa.sum()
        cen[2] = zlow + (yb - sb["S1"] / sb["A"])
        v_lip = np.asarray(v_tip, float) + np.cross(np.asarray(omega, float), cen - tip)
        return dict(e=e, U0=U0, kM=kM, Cvis=Cvis, yc=yc, yb=yb, A_b=sb["A"], W_b=sb["B"], P_b=sb["P"],
                    origin=cen, v_lip=v_lip, elev_raw=elev_raw, rising=bool(rising), H=H, Qc=Qc, n_wet=int(wet.sum()))

    # ---- geometry at a pose
    def world(self, pose, P):
        R = rotation(pose.yaw, pose.tilt, pose.roll)
        return (np.asarray(P, float) - self.geom.tip) @ R.T + np.asarray(pose.tip, float)

    def free_surface(self, pose, V):
        from .v05_pitcher_check import FastPitcher   # same weighted-quantile surface as the old model (n = e_z)
        return FastPitcher.free_surface(self, pose, V)

    def onset_tilt(self, pose, V, lo=0.0, hi=math.radians(150.0), iters=50):
        """Tilt at which the max front head is exactly 0 (the lip just wets) for fill V at this tip / yaw / roll."""
        p = Pose(pose.tip, pose.yaw, lo, pose.roll)

        def H(t):
            p.tilt = t
            z, _ = self.free_surface(p, V)
            return float((z - self.world(p, self.mid_P)[:, 2]).max())
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            if H(m) <= 0:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

    def weir(self, pose, z_f, v_tip=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
        """Steps 2-4 and 7 at one pose and free-surface height: per-segment heads, projected widths, flows and exit
        velocities, and their flux-weighted means."""
        law = self.law
        R = rotation(pose.yaw, pose.tilt, pose.roll)
        tip = np.asarray(pose.tip, float)
        mid = (self.mid_P - self.geom.tip) @ R.T + tip
        seg = self.seg_P @ R.T
        d_o = self.dir_P @ R.T
        h = np.maximum(z_f - mid[:, 2] - law.h_k, 0.0)
        H = float(h.max())
        seg_h = seg.copy(); seg_h[:, 2] = 0.0
        # main outflow direction: flux-weighted horizontal lip normal (arc weights), beak axis when dry
        nh = d_o.copy(); nh[:, 2] = 0.0
        nh /= np.maximum(np.linalg.norm(nh, axis=1), 1e-12)[:, None]
        wa = h ** 1.5 * np.linalg.norm(seg_h, axis=1)
        e = (wa[:, None] * nh).sum(0) if H > 0 else R @ np.array([math.cos(self.beak_phi), math.sin(self.beak_phi), 0.0])
        e[2] = 0.0
        e /= max(np.linalg.norm(e), 1e-12)
        if law.width == "projected":
            dw = np.linalg.norm(seg_h - (seg_h @ e)[:, None] * e[None, :], axis=1)
        elif law.width == "arc":
            dw = np.linalg.norm(seg_h, axis=1)
        else:
            raise ValueError(law.width)
        K = (2.0 / 3.0) * law.C_d * math.sqrt(2 * law.g)
        Qi = K * dw * h ** 1.5
        Q = float(Qi.sum())
        # side overflow (rim outside the front edge under the surface): spill
        Qs = 0.0
        if law.side_spill:
            smid = (self.side_mid_P - self.geom.tip) @ R.T + tip
            hs = np.maximum(z_f - smid[:, 2] - law.h_k, 0.0)
            if hs.max() > 0:
                ss = self.side_seg_P @ R.T
                Qs = float(K * (np.hypot(ss[:, 0], ss[:, 1]) * hs ** 1.5).sum())
        out = dict(Q=Q, H=H, e_main=e, Q_spill=Qs, width=float(dw[h > 0].sum()), z_f=z_f)
        if Q <= 0 or H <= 0:                          # dry lip
            out.update(Q=0.0, U=0.0, origin=mid[np.argmin(mid[:, 2])], v_o=np.zeros(3), d_o=np.zeros(3), A_o=0.0,
                       h_bar=0.0, Qi=Qi * 0.0)
            return out
        if law.U_head == "max":
            Ui = np.full(len(h), law.C_v * math.sqrt(2 * law.g * H))
        elif law.U_head == "local":
            Ui = law.C_v * np.sqrt(2 * law.g * h)
        elif law.U_head == "bar":
            hb = float((Qi * h).sum() / Q)
            Ui = np.full(len(h), law.C_v * math.sqrt(2 * law.g * hb))
        else:
            raise ValueError(law.U_head)
        v_lip = np.asarray(v_tip, float)[None, :] + np.cross(np.asarray(omega, float)[None, :], mid - tip[None, :])
        v_oi = v_lip + Ui[:, None] * d_o
        w = Qi / Q
        dsum = (w[:, None] * d_o).sum(0)
        on = Qi > 0
        out.update(U=float((w * Ui).sum()), origin=(w[:, None] * mid).sum(0), v_o=(w[:, None] * v_oi).sum(0),
                   d_o=dsum / max(np.linalg.norm(dsum), 1e-12), A_o=float((Qi[on] / Ui[on]).sum()),
                   h_bar=float((Qi * h).sum() / Q), Qi=Qi, U_i=Ui)
        if getattr(law, "exit", "lip") == "wall":      # P1-P3 (Q, H, Qi above unchanged)
            out["wall"] = self._wall_exit(pose, z_f, v_tip, omega, out)
        return out

    # ---- pose series
    @staticmethod
    def pose_rates(t, tip, yaw, tilt, roll, smooth_s=0.0):
        """Tip velocity and body angular velocity (world frame) from a uniformly sampled pose series (central
        differences; optional gaussian smoothing of the pose first).  omega from dR/dt R^T."""
        t = np.asarray(t, float)
        dt = float(t[1] - t[0])
        tip = np.asarray(tip, float).copy()
        ang = [np.broadcast_to(np.asarray(a, float), t.shape).copy() for a in (yaw, tilt, roll)]
        if smooth_s > 0:
            from scipy.ndimage import gaussian_filter1d
            s = smooth_s / dt
            tip = gaussian_filter1d(tip, s, axis=0, mode="nearest")
            ang = [gaussian_filter1d(a, s, mode="nearest") for a in ang]
        v = np.gradient(tip, dt, axis=0)
        Rs = np.stack([rotation(a, b, c) for a, b, c in zip(*ang)])
        dR = np.gradient(Rs, dt, axis=0)
        Wm = dR @ np.transpose(Rs, (0, 2, 1))
        W = 0.5 * (Wm - np.transpose(Wm, (0, 2, 1)))
        om = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], -1)
        return v, om

    def run(self, t, tip, yaw, tilt, roll, V0, dt=None):
        """Emission history for a pose series sampled at times t (pose i holds over [t_i - dt/2, t_i + dt/2]).
        Returns a dict of per-step arrays (SI): t, V_P (after the step), Q_o, Q_spill, U, H, h_bar, A_o, width,
        origin (n,3), v_o (n,3), d_o (n,3), v_tip, omega, e_main, and the ballistic t_f, x_hit, U_perp, u_h, t_arr."""
        law = self.law
        t = np.asarray(t, float)
        n = len(t)
        dt = float(t[1] - t[0]) if dt is None else dt
        tip = np.asarray(tip, float)
        yaw, tilt, roll = (np.broadcast_to(np.asarray(a, float), (n,)) for a in (yaw, tilt, roll))
        v_tip, om = self.pose_rates(t, tip, yaw, tilt, roll, law.smooth_s)
        keys1 = ("V_P", "Q_o", "Q_eq", "Q_spill", "U", "H", "h_bar", "A_o", "width", "z_f", "U_state")
        E = {k: np.zeros(n) for k in keys1}
        for k in ("origin", "v_o", "d_o", "e_main"):
            E[k] = np.zeros((n, 3))
        V = float(V0)
        Ust = 0.0
        g = law.g
        wall = getattr(law, "exit", "lip") == "wall"
        X = [None] * n if wall else None               # per-step wall-exit dicts (P1-P3)
        for i in range(n):
            pose = Pose(tuple(tip[i]), float(yaw[i]), float(tilt[i]), float(roll[i]))
            z_f, over = self.free_surface(pose, V)
            Wr = self.weir(pose, z_f, v_tip[i], om[i])
            if wall:
                X[i] = Wr.get("wall")
            Qeq = Wr["Q"]
            Qo = Qeq
            if law.inertia:
                H = Wr["H"]
                # semi-implicit: L_e (U' ) = g H - U|U| / (2 C_v^2)
                a_ = dt / (2 * law.C_v ** 2 * law.L_e)
                b_ = Ust + dt * g * max(H, 0.0) / law.L_e
                Ust = (-1 + math.sqrt(1 + 4 * a_ * b_)) / (2 * a_) if b_ > 0 else b_ / (1 + a_ * abs(Ust))
                Ueq = law.C_v * math.sqrt(2 * g * H) if H > 0 else 0.0
                Qo = Qeq / Ueq * max(Ust, 0.0) if (Qeq > 0 and Ueq > 0) else 0.0
            sp = Wr["Q_spill"] + over / dt
            out = (Qo + sp) * dt
            if out > V:                                # never more than what is left
                f = V / out
                Qo, sp = Qo * f, sp * f
            V -= (Qo + sp) * dt
            E["V_P"][i], E["Q_o"][i], E["Q_eq"][i], E["Q_spill"][i] = V, Qo, Qeq, sp
            E["U_state"][i] = Ust
            for k in ("U", "H", "h_bar", "A_o", "width", "z_f"):
                E[k][i] = Wr[k]
            if Qeq > 0 and Qo != Qeq:
                E["A_o"][i] *= Qo / Qeq
            for k in ("origin", "v_o", "d_o", "e_main"):
                E[k][i] = Wr[k]
        E.update(t=t, dt=dt, v_tip=v_tip, omega=om, V0=float(V0))
        if law.exit_lift or law.anchor:                 # step 7b
            em = E["e_main"].copy()
            em[:, 2] = 0.0
            hh = em / np.maximum(np.linalg.norm(em, axis=1), 1e-12)[:, None]
            d, U = E["d_o"], E["U"]
            v_lip = E["v_o"] - U[:, None] * d
            phi = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])) + law.exit_lift
            d2 = np.c_[np.cos(phi)[:, None] * hh[:, :2], np.sin(phi)]
            on = E["Q_o"] > 0
            E["d_o"] = np.where(on[:, None], d2, d)
            E["v_o"] = np.where(on[:, None], v_lip + U[:, None] * d2, E["v_o"])
            E["origin"] = np.where(on[:, None], E["origin"] + law.anchor * hh, E["origin"])
        # ballistic flight to the coffee surface z = 0 (step 8)
        vz, z0 = E["v_o"][:, 2], E["origin"][:, 2]
        tf = np.where(z0 > 0, (vz + np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))) / g, 0.0)
        E["t_f"] = np.where(E["Q_o"] > 0, tf, np.nan)
        E["x_hit"] = E["origin"][:, :2] + E["v_o"][:, :2] * tf[:, None]
        E["U_perp"] = np.sqrt(vz ** 2 + 2 * g * np.maximum(z0, 0.0))
        E["u_h"] = E["v_o"][:, :2].copy()
        E["t_arr"] = t + tf
        if wall:
            self._wall_run(E, X)
        return E

    def _wall_run(self, E, X):
        """Exit 'wall': steps 7 / 7b / 8 of the flowing steps replaced by P1-P4 (port of M5 exitlaw.run_pour, law
        phys_<capillary>).  Steps with Q_o > 0 but no brink section (never seen; M5 n_miss 0) keep the lip-normal exit
        and its ballistics and are flagged in E['wall_miss'].  Adds the per-step exit quantities (nan where dry)."""
        law = self.law
        n = len(E["t"])
        Q = E["Q_o"]
        on = np.array([x is not None for x in X])
        onq = on & (Q > 0)
        keys3 = ("e", "origin", "v_lip")
        keys1 = ("U0", "kM", "Cvis", "yc", "yb", "A_b", "W_b", "P_b", "elev_raw", "H", "Qc", "n_wet")
        P = {k: np.full((n, 3), np.nan) for k in keys3}
        P.update({k: np.full(n, np.nan) for k in keys1})
        P["rising"] = np.zeros(n, bool)
        for i in np.nonzero(onq)[0]:
            x = X[i]
            for k in keys3 + keys1:
                P[k][i] = x[k]
            P["rising"][i] = x["rising"]
        idx = np.nonzero(onq)[0]
        if len(idx):
            tf_, xh_, vh_, T0_, M0_ = jet_flight(P["origin"][idx], P["v_lip"][idx], P["e"][idx], P["U0"][idx], Q[idx],
                                                 P["A_b"][idx], P["W_b"][idx], P["P_b"][idx], law.rho, law.sigma,
                                                 law.capillary, g=law.g)
            if not np.all(np.isfinite(tf_)):
                raise RuntimeError(f"wall exit: {int((~np.isfinite(tf_)).sum())} parcels did not reach the surface")
            E["origin"][idx] = P["origin"][idx]
            E["v_o"][idx] = P["v_lip"][idx] + P["U0"][idx, None] * P["e"][idx]
            E["d_o"][idx] = P["e"][idx]
            E["U"][idx] = P["U0"][idx]
            E["t_f"][idx] = tf_
            E["x_hit"][idx] = xh_
            E["U_perp"][idx] = -vh_[:, 2]
            E["u_h"][idx] = vh_[:, :2]
            E["t_arr"][idx] = E["t"][idx] + tf_
            T0 = np.full(n, np.nan); M0 = np.full(n, np.nan)
            T0[idx], M0[idx] = T0_, M0_
        else:
            T0 = np.full(n, np.nan); M0 = np.full(n, np.nan)
        E["wall_miss"] = (Q > 0) & ~onq
        E.update({k: P[k] for k in ("kM", "Cvis", "W_b", "P_b", "A_b", "Qc", "yc", "yb", "elev_raw", "rising", "n_wet")})
        E["v_lip"] = P["v_lip"]
        E["H_wall"] = P["H"]
        E["T0"], E["M0"] = T0, M0
        E["beta_tip_deg"] = float(self.beta_tip_deg)

    @staticmethod
    def arrivals(E, t0, n_out, dt_out=None):
        """Bin the emitted parcels onto arrival bins [t0 + j dt, t0 + (j+1) dt).  Parcel i leaves over [t_i - dt/2,
        t_i + dt/2]; its arrival window runs between the midpoints of neighbouring parcels' arrival times (one-sided
        +-dt/2 at the ends of a stream), so a changing flight time stretches / compresses the arrivals.  Returns the
        arrival record (SI): Q (received), x_hit (2), u_h (2), U_perp, A_perp, r1, r2, phi, speed, and the ledger at
        the bin ends (V_P, V_jet, V_cup, V_spill)."""
        dt = E["dt"]
        dto = dt if dt_out is None else dt_out
        Q = E["Q_o"]
        on = Q > 0
        ta = E["t_arr"]
        n = len(Q)
        lo = ta - 0.5 * dt
        hi = ta + 0.5 * dt
        prev_on = np.r_[False, on[:-1]]
        next_on = np.r_[on[1:], False]
        lo = np.where(prev_on, 0.5 * (np.r_[ta[0], ta[:-1]] + ta), lo)
        hi = np.where(next_on, 0.5 * (ta + np.r_[ta[1:], ta[-1]]), hi)
        a, b = np.minimum(lo, hi), np.maximum(lo, hi)
        b = np.maximum(b, a + 1e-9)
        vol = Q * dt
        R = {k: np.zeros(n_out) for k in ("Q", "U_perp", "speed")}
        R["x_hit"] = np.zeros((n_out, 2)); R["u_h"] = np.zeros((n_out, 2))
        acc_v = np.zeros(n_out)
        for i in np.nonzero(on)[0]:
            j0 = int(math.floor((a[i] - t0) / dto))
            j1 = int(math.floor((b[i] - t0) / dto))
            for j in range(max(j0, 0), min(j1, n_out - 1) + 1):
                s0, s1 = t0 + j * dto, t0 + (j + 1) * dto
                ov = min(b[i], s1) - max(a[i], s0)
                if ov <= 0:
                    continue
                v = vol[i] * ov / (b[i] - a[i])
                acc_v[j] += v
                R["x_hit"][j] += v * E["x_hit"][i]
                R["u_h"][j] += v * E["u_h"][i]
                R["U_perp"][j] += v * E["U_perp"][i]
        m = acc_v > 0
        R["Q"] = acc_v / dto
        for k in ("x_hit", "u_h"):
            R[k][m] /= acc_v[m, None]
        R["U_perp"][m] /= acc_v[m]
        R["on"] = m
        sp = np.hypot(np.hypot(R["u_h"][:, 0], R["u_h"][:, 1]), R["U_perp"])
        R["speed"] = sp
        R["v_hit"] = np.c_[R["u_h"], -R["U_perp"]]
        R["A_perp"] = np.where(m, R["Q"] / np.maximum(sp, 1e-12), 0.0)
        r = np.where(m, np.sqrt(R["Q"] / (math.pi * np.maximum(R["U_perp"], 1e-12))), 0.0)
        asp = np.where(m, sp / np.maximum(R["U_perp"], 1e-12), 1.0)
        R["r1"], R["r2"] = r * np.sqrt(asp), r / np.sqrt(asp)
        R["phi"] = np.arctan2(R["u_h"][:, 1], R["u_h"][:, 0])
        # ledger at the bin ends
        te = t0 + (np.arange(n_out) + 1) * dto
        rel_t = E["t"] + 0.5 * dt                                # emission of step i complete at t_i + dt/2
        V_rel = np.interp(te, rel_t, np.cumsum(Q * dt), left=0.0)
        V_sp = np.interp(te, rel_t, np.cumsum(E["Q_spill"] * dt), left=0.0)
        V_cup = np.cumsum(acc_v)
        R.update(t=te, V_cup=V_cup, V_jet=V_rel - V_cup, V_spill=V_sp, V_P=np.interp(te, rel_t, E["V_P"], left=E["V0"]))
        return R
