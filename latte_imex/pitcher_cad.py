"""Pitcher geometry from a CAD model (STEP) -> grid description used by pitcher.GridPitcherGeometry and the web page.

    python -m latte_imex.pitcher_cad path/to/pitcher.stp --out latte_imex/pitcher_geom/pitcher.json

Method: tessellate the solid (cadquery/OCP), slice it with horizontal planes (trimesh), take the cavity as the
holes of the wall ring, and describe it in cylindrical coordinates about the cavity axis:
    r_in(z, phi)   inner wall radius (first crossing of a ray from the axis with the hole boundary)
    r_out(z, phi)  outer wall radius (first crossing with the exterior boundary; the handle is skipped)
    z_rim(phi)     highest z at which the inner wall exists at that azimuth (flat rim -> constant; V notch -> dip)
Body frame of the output: z up from the cavity bottom, x towards the spout (largest rim radius), units metres.
Needs cadquery + trimesh (+ shapely, networkx, rtree) only for the extraction; the JSON has no dependencies.
"""
import argparse, json, math, os
import numpy as np


def load_mesh(path, tol=0.05):
    import cadquery as cq, trimesh
    shape = cq.importers.importStep(path)
    tmp = os.path.splitext(path)[0] + "_tess.stl"
    cq.exporters.export(shape, tmp, tolerance=tol, angularTolerance=0.1)
    m = trimesh.load(tmp)
    return m


def slice_polys(m, z):
    import trimesh, shapely.geometry as sg
    sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return []
    p2, T = sec.to_planar(normal=[0, 0, 1])
    out = []
    for poly in p2.polygons_full:
        ext = trimesh.transform_points(np.c_[np.array(poly.exterior.coords), np.zeros(len(poly.exterior.coords))], T)[:, :2]
        holes = [trimesh.transform_points(np.c_[np.array(h.coords), np.zeros(len(h.coords))], T)[:, :2] for h in poly.interiors]
        out.append(sg.Polygon(ext, holes))
    return out


def ring_radius(ring, center, phi, rmax):
    """Distance from center to the first crossing of ring (LineString) along azimuth phi, or None."""
    import shapely.geometry as sg
    ray = sg.LineString([center, (center[0] + rmax * math.cos(phi), center[1] + rmax * math.sin(phi))])
    x = ring.intersection(ray)
    if x.is_empty:
        return None
    pts = []
    if x.geom_type == "Point":
        pts = [x]
    elif x.geom_type == "MultiPoint":
        pts = list(x.geoms)
    else:
        pts = [g for g in getattr(x, "geoms", [x]) if g.geom_type == "Point"]
        if not pts:
            return None
    return min(math.hypot(p.x - center[0], p.y - center[1]) for p in pts)


def extract(m, dz=1.0, nphi=72, z_fine=3.0, dz_fine=0.25):
    import shapely.geometry as sg
    zmin, zmax = m.bounds[0][2], m.bounds[1][2]
    # 1. cavity axis from hole centroids over the lower 80 % of the height
    cents, zs_ok = [], []
    for z in np.arange(zmin + 2, zmax - 2, 4.0):
        for p in slice_polys(m, z):
            if p.interiors:
                big = max(p.interiors, key=lambda h: sg.Polygon(h).area)
                c = sg.Polygon(big).centroid; cents.append((c.x, c.y)); zs_ok.append(z)
    cents = np.array(cents)
    center = cents[: max(1, int(0.8 * len(cents)))].mean(0)
    # 2. z levels: coarse plus fine near the rim
    zs = list(np.arange(zmin + 0.5, zmax - z_fine, dz)) + list(np.arange(zmax - z_fine, zmax, dz_fine))
    phis = (np.arange(nphi) + 0.5) / nphi * 2 * math.pi - math.pi
    rmax = 2.0 * max(m.extents[0], m.extents[1])
    r_in = np.full((len(zs), nphi), np.nan); r_out = np.full((len(zs), nphi), np.nan)
    z_bottom = None
    for iz, z in enumerate(zs):
        polys = [p for p in slice_polys(m, z) if p.interiors]
        if not polys:
            continue
        body = max(polys, key=lambda p: p.area)
        hole = max(body.interiors, key=lambda h: sg.Polygon(h).area)
        hole_ring = sg.LineString(hole.coords); ext_ring = sg.LineString(body.exterior.coords)
        if not sg.Polygon(hole).contains(sg.Point(center)):
            continue
        if z_bottom is None:
            z_bottom = z
        for ip, phi in enumerate(phis):
            r_in[iz, ip] = ring_radius(hole_ring, center, phi, rmax)
            r_out[iz, ip] = ring_radius(ext_ring, center, phi, rmax)
    zs = np.array(zs)
    valid = ~np.isnan(r_in).all(1)
    zs, r_in, r_out = zs[valid], r_in[valid], r_out[valid]
    # bottom of the cavity: extend half a step below the lowest valid slice
    z0 = zs[0] - 0.5 * dz
    # 3. rim height per azimuth: highest z with an inner wall
    z_rim = np.array([zs[~np.isnan(r_in[:, ip])].max() if (~np.isnan(r_in[:, ip])).any() else z0 for ip in range(nphi)])
    # fill gaps (above the rim) by holding the last valid radius so that bilinear lookups are defined
    for ip in range(nphi):
        col = r_in[:, ip]; last = np.nan
        for iz in range(len(zs)):
            if np.isnan(col[iz]):
                col[iz] = last
            else:
                last = col[iz]
        col = r_out[:, ip]; last = np.nan
        for iz in range(len(zs)):
            if np.isnan(col[iz]):
                col[iz] = last
            else:
                last = col[iz]
    r_in = np.where(np.isnan(r_in), np.nanmin(r_in), r_in); r_out = np.where(np.isnan(r_out), np.nanmin(r_out), r_out)
    # 4. spout azimuth: largest inner radius near the rim -> rotate to +x
    top = r_in[-4:].mean(0)
    ip_spout = int(np.argmax(top)); phi_spout = phis[ip_spout]
    shift = -ip_spout + nphi // 2   # roll so that the spout lands at phi = 0 (index nphi//2 has phi ~ 0)
    # phis grid has phi=0 between indices nphi/2-1 and nphi/2; roll by (nphi//2 - ip_spout) and record the residual
    r_in = np.roll(r_in, shift, axis=1); r_out = np.roll(r_out, shift, axis=1); z_rim = np.roll(z_rim, shift)
    phi_res = phis[nphi // 2] - 0.0   # residual half-step offset of the spout from phi=0 (keep in metadata)
    rim_r = np.array([r_in[np.searchsorted(zs, z_rim[ip]) - 1 if z_rim[ip] > zs[0] else 0, ip] for ip in range(nphi)])
    # cavity volume (polar integration)
    dphi = 2 * math.pi / nphi
    vol = 0.0
    for iz in range(len(zs)):
        dzl = (zs[iz + 1] - zs[iz]) if iz + 1 < len(zs) else dz_fine
        rr = np.where(zs[iz] <= z_rim, r_in[iz], 0.0)
        vol += float((0.5 * rr ** 2 * dphi).sum()) * dzl
    return dict(units="mm", center_xy=center.tolist(), z0=float(z0), zs=(zs - z0).tolist(), phis=phis.tolist(),
                r_in=r_in.tolist(), r_out=r_out.tolist(), z_rim=(z_rim - z0).tolist(), r_rim=rim_r.tolist(),
                spout_phi_original=float(phi_spout), spout_phi_residual=float(phi_res), cavity_volume_mm3=vol,
                height=float(zs[-1] - z0), zmax_mesh=float(zmax - z0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--out", default="latte_imex/pitcher_geom/pitcher.json")
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--nphi", type=int, default=72)
    a = ap.parse_args()
    m = load_mesh(a.step)
    g = extract(m, dz=a.dz, nphi=a.nphi)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(g, f)
    print(f"cavity {g['cavity_volume_mm3']/1000:.0f} mL, height {g['height']:.1f} mm, levels {len(g['zs'])}, "
          f"r_in at mid-height {np.mean(g['r_in'][len(g['zs'])//2]):.1f} mm, spout r {max(g['r_rim']):.1f} mm, "
          f"rim z range {min(g['z_rim']):.1f}..{max(g['z_rim']):.1f} mm -> {a.out}")
