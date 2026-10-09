"""Top-down movie of the physical chain: cup with the milk field, the pitcher (outer wall and rim) and the stream.

    python -m latte_imex.topview_movie --from runs/joint_sweep_r2fix/results.json --idx 9 --pattern tulip --N 256 --out runs/codexmap/r2fix_9_N256

Same chain as joint_sweep.run_chain (pitcher -> ledger -> IMEX); at 24 fps the milk field and the pitcher pose
(tip, tilt, jet, hit point) are captured and drawn from above, like the web playground's top view.
"""
import argparse, json, math, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .joint_sweep import MOVES, D_L
import latte_imex.joint_sweep as js
from .pitcher import Pitcher, GridPitcherGeometry, Coupling, BaristaScript, Pose, Cup
from .solver import Solver, Params, Numerics
from .render import c_to_rgb


def run(name, action, physics, N, snap_dt=1 / 24, V0=250e-6):
    if js._GEOM is None:
        js._GEOM = GridPitcherGeometry.load(nr=6)
    pitcher = Pitcher(geom=js._GEOM, V0=V0)
    cp = Coupling(c_S=physics["kQ"] / D_L ** 3, c_U=1.0 / D_L, c_u=1.0 / D_L, footprint="physical")
    script = BaristaScript(moves=MOVES[name](action), pitcher=pitcher, coupling=cp)
    pd = {k: v for k, v in physics.items() if k != "kQ"}
    pd.update(D=1e-7, kappa_Q=125.0, return_law="v05")
    num = Numerics(N=N); sol = Solver(Params(**pd), num)
    frames, next_snap, t0 = [], 0.0, time.time()
    while sol.t < script.T - 1e-9:
        probe, _ = script.sample(sol.t)
        fdt = num.frame_dt
        if probe.active and probe.S_eff > 0 and probe.scan_speed > 0:
            fdt = min(fdt, max(num.scan_safety * min(sol.g.h, probe.r1, probe.r2) / probe.scan_speed, 1e-4))
        inl, phase = script.sample(sol.t + 0.5 * fdt)
        sol.advance_frame(inl, frame_dt=fdt)
        if sol.t + 1e-9 >= next_snap and script.records:
            rec = script.records[-1]
            pose = Pose(tip=tuple(rec["tip"]), yaw=script.barista.yaw, tilt=rec["tilt"], roll=0.0)
            wall = pitcher.wall_samples(pose, nz=20, nphi=48)
            rim = pitcher.world(pose, pitcher.geom.rim)
            jet = pitcher.jet_polyline(rec["jet"]) if rec["jet"] is not None else np.zeros((0, 3))
            hit = rec["hit"]["x_hit"] if rec.get("hit") else None
            frames.append(dict(t=sol.t, c=sol.concentration().astype(np.float32), wall=wall, rim=rim, jet=jet,
                               hit=None if hit is None else np.asarray(hit), Q=rec["Q"], Q_hit=rec.get("Q_hit", 0.0),
                               tilt=rec["tilt"], z_tip=rec["tip"][2], lift=rec.get("lift", 0.0), phase=phase))
            next_snap += snap_dt
    print(f"{name} N={N}: {len(frames)} frames, {time.time() - t0:.0f}s", flush=True)
    return frames, sol.g.mask, script.barista.cup


def render(path, frames, mask, cup, title, fps=24, hold=24, view=0.085):
    import imageio.v2 as imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    fig = plt.figure(figsize=(8, 8), dpi=96)
    fig.patch.set_facecolor("#1b1b1c")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    seq = list(frames) + [frames[-1]] * hold
    for k, f in enumerate(seq):
        ax.clear(); ax.set_facecolor("#1b1b1c")
        rgba = np.concatenate([c_to_rgb(f["c"], mask), mask[::-1][..., None].astype(float)], -1)   # c_to_rgb is y-up (row 0 = top)
        ax.imshow(rgba, origin="upper", extent=[-D_L / 2, D_L / 2, -D_L / 2, D_L / 2], interpolation="bilinear", zorder=1)
        th = np.linspace(0, 2 * math.pi, 200)
        ax.fill(cup.r_out * np.cos(th), cup.r_out * np.sin(th), color="#d8d2c8", zorder=0)
        ax.plot(cup.r_in * np.cos(th), cup.r_in * np.sin(th), color="#6b665f", lw=1.0, zorder=2)
        w = f["wall"]; z = w[:, 2]
        ax.scatter(w[:, 0], w[:, 1], s=4, c=z, cmap="Greys_r", vmin=-0.02, vmax=0.16, alpha=0.35, linewidths=0, zorder=3)
        r = f["rim"]
        ax.plot(np.append(r[:, 0], r[0, 0]), np.append(r[:, 1], r[0, 1]), color="#9fb4c7", lw=1.6, zorder=4)
        if len(f["jet"]):
            ax.plot(f["jet"][:, 0], f["jet"][:, 1], color="white", lw=2.5, zorder=5)
        if f["hit"] is not None and f["Q_hit"] > 0:
            ax.plot(f["hit"][0], f["hit"][1], "o", color="white", ms=6, mec="#333", zorder=6)
        ax.set_xlim(-view, view); ax.set_ylim(-view, view); ax.set_aspect("equal"); ax.axis("off")
        hold_txt = "   FINAL HOLD" if k >= len(frames) else ""
        ax.text(0.02, 0.97, f"{title}   t = {f['t']:5.2f} s{hold_txt}", color="white", fontsize=12, va="top", transform=ax.transAxes)
        ax.text(0.02, 0.93, f"{f['phase']}   Q {f['Q']*1e6:4.1f} mL/s   tilt {math.degrees(f['tilt']):4.0f}°   "
                            f"spout height {f['z_tip']*100:4.1f} cm" + (f"   lift {f['lift']*100:.1f} cm" if f["lift"] > 0 else ""),
                color="#dddddd", fontsize=10, va="top", transform=ax.transAxes)
        ax.text(0.98, 0.03, "top view, cup Ø 8 cm; pitcher wall shaded by height, rim in blue, stream in white",
                color="#999999", fontsize=8, ha="right", transform=ax.transAxes)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close(); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--pattern", default="tulip")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = json.load(open(a.src))[a.idx]
    os.makedirs(a.out, exist_ok=True)
    frames, mask, cup = run(a.pattern, rec["actions"][a.pattern], rec["physics"], a.N)
    path = os.path.join(a.out, f"{a.pattern}_N{a.N}_topview.mp4")
    render(path, frames, mask, cup, f"{a.pattern} #{a.idx}")
    print("wrote", path)


if __name__ == "__main__":
    main()
