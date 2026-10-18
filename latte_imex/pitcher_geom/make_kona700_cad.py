"""CAD model (STEP + STL) of the rig pitcher: CAFEDE KONA x 元一 "Allround" milk jug, 700 mL, white.  Replaces the
NX stand-in (pitcher_nx.stp: straight 101 mm cylinder, 0.52 L) with the jug actually used in the recordings.

Dimensions (mm; z up from the outside bottom, spout towards +x, handle towards -x):
  body      straight sides (photos): bevel at the bottom, wall from r 44.0 (z 9) to the rim, rim diameter 84.5 (tape),
            height 122 / 128 mm handle-to-spout (catalogue)
            (tape, across the spout axis), wall 1.0, bottom 1.2
  spout     a pressed channel shows outside as a rounded ridge from ~48 mm up (front photo: rounded lower end
            ~23 mm wide, tapering out to ~44 mm at the rim, ~3 mm proud), flaring in the top ~18 mm into the V beak
            (+-31 deg at the rim); the lip bends DOWN: the rim edge drops by SPOUT_DIP = 7 mm at the tip (side photo) while the beak
            keeps flaring outwards, tip ~49 mm from the axis (tape across the top: spout tip to the opposite rim
            ~90-96 mm; the catalogue drawing gives ~49)
  facets    the lower body is a 14-sided prism (product page '14边对称设计', flats centred on the spout and the handle), blending into the round
            body between FACET_Z0 and FACET_Z1; flats ~1.5 mm inside the round profile
  bottom    inside: raised round boss (diameter 44, 2 mm) as seen from above
  handle    two separate flat straps 13.5 x 2.0 mm (tape; product photos): the grip strap (arm at 110 mm out to
            x = -72, vertical drop, rounded return to the body at ~21 mm) and above it the thumb strap (at 119 mm,
            parallel, curling up beyond the vertical part to x = -77); 128 mm thumb-rest-to-spout overall
    python -m latte_imex.pitcher_geom.make_kona700_cad [--out latte_imex/pitcher_geom/pitcher_kona700]
"""
import argparse, math, os
import numpy as np

# straight sides (photos): bottom bevel 0..9 mm, then a straight, slightly inward-leaning wall up to the rim
PROFILE_Z = [0.0, 9.0, 122.0]
PROFILE_R = [39.5, 44.0, 42.25]
H, WALL, BOTTOM = 122.0, 0.6, 0.8        # sheet thickness: < 1 mm, ~0.5-0.6 mm (user, by hand)
CREASE_Z, BEAK_Z, RIDGE_H0, RIDGE_H1 = 50.0, 104.0, 2.0, 6.5   # ridge start, beak flare start [mm]; ridge height at both
# (product side photo: the spout side of the silhouette leans out from the channel's end to ~6.5 mm at the beak)
RIDGE_HALF0, HALF_RIM = 15.0, 31.0               # channel half-angles [deg]: round lower end, rim
# (front photo: the channel tapers from ~44 mm wide at the rim to ~23 mm at its rounded lower end at ~48 mm)
LIP_R = 49.1                                     # beak radius extrapolated to z = H (the dipped tip ends at ~49)
SPOUT_DIP, DIP_HALF = 7.0, 40.0                  # lip drop at the tip [mm], angular half-width of the dip [deg]
NPH = 144
N_FACETS, FACET_Z0, FACET_Z1 = 14, 30.0, 48.0      # product page: 14-sided symmetric body, flats at spout and handle
BOSS_R, BOSS_H = 22.0, 2.0


def r_body(z):
    return float(np.interp(z, PROFILE_Z, PROFILE_R))


def radius(z, offset, ph):
    """Wall radius at height z [mm] and azimuths ph [rad] (spout at 0): outer (offset 0) or inner (offset -WALL)."""
    rb = r_body(min(z, H)) + offset
    zc = min(z, H)
    ridge = ((RIDGE_H0 + (RIDGE_H1 - RIDGE_H0) * min(1.0, (zc - CREASE_Z) / (BEAK_Z - CREASE_Z)))
             * min(1.0, (zc - CREASE_Z) / 4.0) ** 0.5) if zc > CREASE_Z else 0.0
    b = np.clip((zc - BEAK_Z) / (H - BEAK_Z), 0, 1) ** 1.3                                        # beak flare 0..1
    tip = rb + ridge + (LIP_R + offset - rb - ridge) * b
    # straight channel edges (front photo): the half-width grows linearly from the round end to the rim
    half = math.radians(RIDGE_HALF0 + (HALF_RIM - RIDGE_HALF0) * np.clip((zc - CREASE_Z) / (H - CREASE_Z), 0, 1))
    r_end = (rb * math.radians(RIDGE_HALF0))                        # rounded (semicircular) lower end of the channel
    if zc < CREASE_Z + r_end:
        half *= math.sqrt(max(1e-4, 1 - ((CREASE_Z + r_end - zc) / r_end) ** 2))
    x = np.clip(np.abs(ph) / half, 0, 1)
    prof = (1 - b) * np.sqrt(1 - x ** 2) + b * (1 - x) ** 1.3                                     # round ridge -> V beak
    r = rb + (tip - rb) * prof
    f = np.clip((FACET_Z1 - z) / (FACET_Z1 - FACET_Z0), 0, 1)       # 1: full polygon, 0: round
    if f > 0:
        a = 2 * math.pi / N_FACETS
        d = np.mod(ph + a / 2, a) - a / 2                            # angle from the nearest flat centre (one at the spout)
        poly = math.cos(a / 2) / np.cos(d)                           # vertices on the circle, flats inside
        r = r * (1 - f + f * poly * (1 if offset == 0 else 1))
    return r


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
    return cq.Solid.makeLoft(wires, ruled=True)          # ruled: straight walls between sections, no spline overshoot


def build():
    import cadquery as cq
    zo = [0.0, 2, 5, 9, 20, 30, 36, 42, 48, 50, 51.5, 53, 55, 57, 59.5, 62, 66, 70, 80, 90, 100, 104, 108, 112, 115, 118, 120, 122]
    outer = loft(zo, 0.0)
    zi = [BOTTOM, 3, 6, 9, 20, 30, 36, 42, 48, 50, 51.5, 53, 55, 57, 59.5, 62, 66, 70, 80, 90, 100, 104, 108, 112, 115, 118, 120, 122, 126]
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
    # handle (product photos): TWO separate straps leaving the body near the rim.
    #   grip strap (fingers): arm at Z_ARM out to the vertical part at xh, down, rounded return to the body at 21 mm
    #   thumb strap: above it at Z_THUMB, parallel to the arm, curling up at its outer end beyond the vertical part
    T, Wd = 2.0, 13.5
    xh, Z_ARM, Z_THUMB = -72.0, 110.0, 119.0
    c45 = 1 - math.cos(math.radians(45))
    path = (cq.Workplane("XZ").moveTo(-r_body(Z_ARM) + 0.6, Z_ARM).lineTo(xh + 6, Z_ARM)
            .threePointArc((xh + 6 * c45, Z_ARM - 6 * c45), (xh, Z_ARM - 6))
            .lineTo(xh, 52)
            .threePointArc((xh + 7.5, 31), (-r_body(21) + 0.6, 21)))
    prof = cq.Workplane("YZ", origin=(-r_body(Z_ARM) + 0.6, 0, Z_ARM)).rect(Wd, T)
    handle = prof.sweep(path, transition="round")
    tpath = (cq.Workplane("XZ").moveTo(-r_body(Z_THUMB) + 0.6, Z_THUMB).lineTo(-63.0, Z_THUMB)
             .threePointArc((-70.5, Z_THUMB + 1.6), (-77.0, Z_THUMB + 6.0)))
    tab = cq.Workplane("YZ", origin=(-r_body(Z_THUMB) + 0.6, 0, Z_THUMB)).rect(Wd, T).sweep(tpath, transition="round")
    boss = cq.Workplane("XY").circle(BOSS_R).extrude(BOSS_H + 0.2).translate((0, 0, BOTTOM - 0.2)).edges(">Z").fillet(1.5)
    shell = shell.union(boss)
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
