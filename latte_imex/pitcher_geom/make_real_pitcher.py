"""Parametric geometry of the real pitcher (the one with the AprilTags), in the GridPitcherGeometry JSON format
(same as pitcher_nx.json: mm, zs, phis, r_in/r_out [nz, nphi], z_rim/r_rim [nphi], spout at phi = 0).

Current numbers come from the rig data (to be replaced by calliper measurements, see raw_data/pitcher/):
  * outer radius profile r_out(z): merged multi-window depth reconstruction (raw_data/rig/results/
    pitcher_merged3.json), checked against the tag-fit cylinder radius 46.3-46.8 mm at the tag height (~33 mm);
  * height 126 mm; wall 1.0 mm (assumed); flat bottom 1.5 mm (assumed);
  * spout: lip 53 mm from the axis at rim height (tag-frame tip (0, 53, 93) mm, rim 93 mm above the tag centre);
    the spout channel narrows from ~44 mm to ~24 mm wide (pitcher_model width profile) -> half-angle at the rim
    ~ asin(12/53) ~ 13 deg at the lip, ~30 deg where it leaves the body; channel depth below the rim SPOUT_DEPTH.
    python -m latte_imex.pitcher_geom.make_real_pitcher [--out pitcher_real.json] [--H 126] ...
"""
import argparse, json, math, os
import numpy as np

# outer radius [mm] vs height [mm] above the outside bottom (merged reconstruction, smoothed)
PROFILE_Z = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 115, 120, 126]
PROFILE_R = [47.5, 47.2, 47.0, 46.8, 46.5, 46.3, 46.1, 45.8, 45.9, 45.4, 44.9, 43.9, 43.3, 42.4, 41.2, 39.6]
# CAFEDE KONA x 元一 "Allround" milk jug, 700 mL version (the rig pitcher): product drawing (height 122 mm, handle-to-spout
# 128 mm), outer radius measured on the drawing's silhouette at 0.232 mm/px (handle side, the spout side is shaded);
# spout lip ~49 mm from the axis.  Inner cavity with a 1 mm wall: ~0.70 L, the catalogue capacity.
PROFILES = dict(
    rig=dict(z=PROFILE_Z, r=PROFILE_R, H=126.0, lip_r=53.0),
    kona700=dict(z=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 118, 122],
                 r=[38.0, 39.3, 43.5, 44.0, 44.0, 44.0, 44.0, 43.7, 43.5, 43.3, 42.8, 42.6, 42.3, 42.2, 42.1, 42.1],
                 H=122.0, lip_r=49.0),
)


def build(H=126.0, wall=1.0, bottom=1.5, lip_r=53.0, spout_half_deg=30.0, lip_half_deg=13.0, spout_depth=18.0,
          nz=126, nphi=72, scale_r=1.0, profile="rig"):
    PROFILE_Z, PROFILE_R = PROFILES[profile]["z"], PROFILES[profile]["r"]
    zs = (np.arange(nz) + 0.5) / nz * (H - bottom)                    # inner z from the inside bottom
    phis = np.arange(nphi) / nphi * 2 * math.pi - math.pi          # includes phi = 0 (spout)
    r_out_z = np.interp(zs + bottom, PROFILE_Z, PROFILE_R) * scale_r
    r_out = np.repeat(r_out_z[:, None], nphi, 1)
    r_in = r_out - wall
    Hin = H - bottom
    r_top_in = float(np.interp(H, PROFILE_Z, PROFILE_R) * scale_r - wall)
    # spout channel: in the top band (Hin - spout_depth .. Hin) the inner radius bulges towards phi = 0, reaching the
    # lip radius at the rim; angular half-width shrinks from spout_half_deg (bottom of band) to lip_half_deg (rim)
    for iz, z in enumerate(zs):
        s = (z - (Hin - spout_depth)) / spout_depth
        if s <= 0: continue
        s = min(s, 1.0)
        half = math.radians(spout_half_deg + (lip_half_deg - spout_half_deg) * s)
        rmax = r_in[iz, 0] + (lip_r - wall - r_in[iz, 0]) * s ** 1.5
        w = np.clip(1 - np.abs(phis) / half, 0, None)
        prof = r_in[iz] + (rmax - r_in[iz]) * (1 - (1 - w) ** 2) * (w > 0)
        r_in[iz] = np.maximum(r_in[iz], prof)
        r_out[iz] = np.maximum(r_out[iz], r_in[iz] + wall)
    # rim: constant height; rim radius = inner radius at the top row (spout lip at phi = 0)
    z_rim = np.full(nphi, Hin)
    r_rim = r_in[-1].copy()
    # cavity volume
    dz = zs[1] - zs[0]; dphi = 2 * math.pi / nphi
    vol = float((0.5 * r_in ** 2 * dphi).sum() * dz)
    return dict(units="mm", name="pitcher_real", zs=zs.tolist(), phis=phis.tolist(), r_in=r_in.tolist(), r_out=r_out.tolist(),
                z_rim=z_rim.tolist(), r_rim=r_rim.tolist(), spout_phi_residual=0.0, height=float(Hin),
                cavity_volume_mm3=vol, params=dict(H=H, wall=wall, bottom=bottom, lip_r=lip_r, spout_half_deg=spout_half_deg,
                                                    lip_half_deg=lip_half_deg, spout_depth=spout_depth, scale_r=scale_r, profile=profile,
                                                    profile_z=PROFILE_Z, profile_r=PROFILE_R, r_top_inner=r_top_in),
                source="parametric from rig reconstruction (latte_imex/pitcher_geom/make_real_pitcher.py); replace with calliper data")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitcher_real.json"))
    ap.add_argument("--profile", default="rig", choices=list(PROFILES))
    for k, v in dict(H=None, wall=1.0, bottom=1.5, lip_r=None, spout_half_deg=30.0, lip_half_deg=13.0, spout_depth=18.0).items():
        ap.add_argument("--" + k, type=float, default=v)
    a = ap.parse_args()
    P = PROFILES[a.profile]
    d = build(a.H or P["H"], a.wall, a.bottom, a.lip_r or P["lip_r"], a.spout_half_deg, a.lip_half_deg, a.spout_depth,
              profile=a.profile)
    json.dump(d, open(a.out, "w"))
    print(a.out, "cavity mL", round(d["cavity_volume_mm3"] / 1e3, 1), "lip r", round(max(d["r_rim"]), 1), "top inner r", round(d["params"]["r_top_inner"], 1))
