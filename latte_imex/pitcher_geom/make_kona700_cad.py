"""CAD model (STEP + STL) of the rig pitcher: CAFEDE KONA x 元一 "Allround" milk jug, 700 mL, white.  Replaces the
NX stand-in (pitcher_nx.stp: straight 101 mm cylinder, 0.52 L) with the jug actually used in the recordings.

Dimensions (mm; z up from the outside bottom, spout towards +x, handle towards -x):
  body      outer radius profile of the catalogue drawing (height 122, 128 mm handle-to-spout), rim diameter 84.5
            (tape, across the spout axis), wall 1.0, bottom 1.2
  spout     V beak, tip 53.5 mm from the axis (tape: spout tip to the opposite rim 96.5 mm); the crease starts
            ~60 mm up the body and opens to +-35 deg at the rim; the lip bends DOWN towards the tip: the rim edge
            drops by SPOUT_DIP (5 mm, catalogue drawing) at the tip while the beak keeps flaring outwards
  handle    flat strap 13.5 x 2.0 mm (tape), top arm at the rim out to x = -72, vertical drop, rounded return to the
            body at ~22 mm; thumb tab above the rim to x = -74.5 (128 mm overall)
    python -m latte_imex.pitcher_geom.make_kona700_cad [--out latte_imex/pitcher_geom/pitcher_kona700]
"""
import argparse, math, os
import numpy as np

PROFILE_Z = [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 118, 122]
PROFILE_R = [38.0, 38.6, 39.6, 43.0, 44.0, 44.0, 44.0, 44.0, 43.7, 43.5, 43.3, 42.8, 42.6, 42.4, 42.3, 42.25, 42.25]
H, WALL, BOTTOM = 122.0, 1.0, 1.2
LIP_R, CREASE_Z, HALF_RIM = 55.3, 60.0, 35.0   # LIP_R: beak radius extrapolated to z = H (tip at the dipped lip: ~53.5)
SPOUT_DIP, DIP_HALF = 5.0, 40.0                  # lip drop at the tip [mm], angular half-width of the dip [deg]
NPH = 144


def r_body(z):
    return float(np.interp(z, PROFILE_Z, PROFILE_R))


def radius(z, offset, ph):
    """Wall radius at height z [mm] and azimuths ph [rad] (spout at 0): outer (offset 0) or inner (offset -WALL)."""
    rb = r_body(min(z, H)) + offset
    s = np.clip((min(z, H) - CREASE_Z) / (H - CREASE_Z), 0, 1)
    tip = rb + (LIP_R + offset - rb) * s ** 1.6           # beak tip radius rises towards the rim
    half = math.radians(12 + (HALF_RIM - 12) * s)
    w = np.clip(1 - np.abs(ph) / half, 0, 1)
    return np.maximum(rb, rb + (tip - rb) * w ** 1.3)


def rim_height(ph):
    """Rim edge height [mm] vs azimuth: H, dipping by SPOUT_DIP at the spout tip (downward-bent lip)."""
    w = np.clip(1 - np.abs(np.degrees(ph)) / DIP_HALF, 0, 1)
    return H - SPOUT_DIP * w ** 1.5


def section(z, offset=0.0):
    """Horizontal section at height z as points (NPH, 3)."""
    ph = np.linspace(0, 2 * math.pi, NPH, endpoint=False) - math.pi
    r = radius(z, offset, ph)
    return np.stack([r * np.cos(ph), r * np.sin(ph), np.full_like(r, z)], 1)


def grid_json(nz=121, nphi=72):
    """pitcher.GridPitcherGeometry JSON straight from the analytic shape (z from the inside bottom, spout at phi = 0);
    slicing the STEP (pitcher_cad.py) stops at the lowest point of the rim and would lose the bent lip."""
    phis = np.arange(nphi) / nphi * 2 * math.pi - math.pi
    zs = BOTTOM + (np.arange(nz) + 0.5) / nz * (H - BOTTOM)
    r_in = np.array([radius(z, -WALL, phis) for z in zs])
    r_out = np.array([radius(z, 0.0, phis) for z in zs])
    z_rim = rim_height(phis)
    r_rim = np.array([radius(zr, -WALL, np.array([p]))[0] for zr, p in zip(z_rim, phis)])
    dz = zs[1] - zs[0]; dphi = 2 * math.pi / nphi
    vol = float(sum((0.5 * r_in[i, j] ** 2 * dphi * dz) for i in range(nz) for j in range(nphi) if zs[i] < z_rim[j]))
    return dict(units="mm", name="pitcher_kona700", zs=(zs - BOTTOM).tolist(), phis=phis.tolist(), r_in=r_in.tolist(),
                r_out=r_out.tolist(), z_rim=(z_rim - BOTTOM).tolist(), r_rim=r_rim.tolist(), spout_phi_residual=0.0,
                height=float(H - BOTTOM), cavity_volume_mm3=vol,
                source="latte_imex/pitcher_geom/make_kona700_cad.py (CAD of the CAFEDE KONA Allround 700 mL jug)")


def loft(zs, offset):
    import cadquery as cq
    wires = []
    for z in zs:
        P = section(z, offset)
        e = cq.Edge.makeSpline([cq.Vector(*p) for p in P], periodic=True)
        wires.append(cq.Wire.assembleEdges([e]))
    return cq.Solid.makeLoft(wires, ruled=False)


def build():
    import cadquery as cq
    zo = [0.0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 115, 118, 122]
    outer = loft(zo, 0.0)
    zi = [BOTTOM, 3, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 115, 118, 122, 126]
    inner = loft(zi, -WALL)
    shell = cq.Workplane().add(outer).cut(cq.Workplane().add(inner))
    # downward-bent lip: cut away a wedge above z_rim(phi) = H - SPOUT_DIP (1 - |phi| / DIP_HALF)^1.5 around the spout,
    # lofted through planar radial sections (phi = const), so the rim edge slopes down to the tip
    secs = []
    for d in np.linspace(-DIP_HALF, DIP_HALF, 17):
        a = math.radians(d)
        zb = float(rim_height(np.array([a]))[0]) - (0.05 if abs(d) == DIP_HALF else 0.0)
        c, s_ = math.cos(a), math.sin(a)
        pts = [(r * c, r * s_, z) for r, z in ((20.0, zb), (80.0, zb), (80.0, H + 20), (20.0, H + 20))]
        secs.append(cq.Wire.makePolygon([cq.Vector(*p) for p in pts], close=True))
    cutter = cq.Solid.makeLoft(secs, ruled=False)
    shell = shell.cut(cq.Workplane().add(cutter))
    # handle: strap 13.5 (y) x 2.0, path in the x-z plane on the -x side
    T, Wd = 2.0, 13.5
    path = (cq.Workplane("XZ").moveTo(-r_body(117) + 0.6, 117).lineTo(-66, 117)
            .threePointArc((-70 + 6 * (1 - math.cos(math.radians(45))), 117 - 6 * (1 - math.cos(math.radians(45)))), (-70, 111))
            .lineTo(-70, 52)
            .threePointArc((-63.5, 32), (-r_body(22) + 0.6, 22)))
    prof = cq.Workplane("YZ", origin=(-r_body(117) + 0.6, 0, 117)).rect(Wd, T)
    handle = prof.sweep(path, transition="round")
    tab = (cq.Workplane("XY").box(32, Wd, T).translate((-58.5, 0, 0))     # thumb tab: rim -> x = -74.5, rising outwards
           .rotate((-42.5, 0, 0), (-42.5, 1, 0), 9).translate((0, 0, 120.0)))
    solid = shell.union(handle).union(tab)
    for nm, part in (("shell", shell), ("handle", handle), ("tab", tab)):
        b = part.val().BoundingBox()
        print(f"  {nm:6s} x {b.xmin:6.1f}..{b.xmax:6.1f}  z {b.zmin:6.1f}..{b.zmax:6.1f}")
    return solid, outer, inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitcher_kona700"))
    a = ap.parse_args()
    import cadquery as cq
    solid, outer, inner = build()
    import json
    g = grid_json()
    json.dump(g, open(a.out + ".json", "w"))
    print(a.out + ".json", f"grid cavity {g['cavity_volume_mm3'] / 1e3:.0f} mL, lip r {max(g['r_rim']):.1f} mm at z {min(g['z_rim']):.1f}")
    cq.exporters.export(solid, a.out + ".step")
    cq.exporters.export(solid, a.out + ".stl", tolerance=0.05, angularTolerance=0.1)
    cav = cq.Workplane().add(inner).intersect(cq.Workplane().box(300, 300, H - BOTTOM, centered=(True, True, False))
                                              .translate((0, 0, BOTTOM)))
    bb = solid.val().BoundingBox()
    print(a.out + ".step", f"bbox x {bb.xmin:.1f}..{bb.xmax:.1f} y {bb.ymin:.1f}..{bb.ymax:.1f} z {bb.zmin:.1f}..{bb.zmax:.1f} mm;",
          f"material {solid.val().Volume() / 1e3:.1f} cm3, cavity to the rim {cav.val().Volume() / 1e3:.0f} mL")


if __name__ == "__main__":
    main()
