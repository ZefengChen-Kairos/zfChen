"""Diagnostic rendering of the milk fraction c (not an optical renderer)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BROWN = np.array([0.42, 0.25, 0.12])
WHITE = np.array([0.97, 0.95, 0.91])
BG = np.array([0.16, 0.16, 0.17])
RIM = np.array([0.85, 0.85, 0.86])


def c_to_rgb(c, mask, c_sat=0.8):
    """Map c in [0, c_sat] to brown->white (saturating above), outside the cup a neutral background."""
    v = np.clip(np.nan_to_num(c, nan=0.0) / c_sat, 0.0, 1.0)
    v = v ** 0.8
    rgb = BROWN[None, None, :] * (1 - v[..., None]) + WHITE[None, None, :] * v[..., None]
    rgb = np.where(mask[..., None], rgb, BG[None, None, :])
    return rgb[::-1]  # y up


def save_final(path, c, mask, title=None):
    rgb = c_to_rgb(c, mask)
    fig = plt.figure(figsize=(6, 6), dpi=128)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", ha="left", color="w", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def make_video(path, snaps, traj, mask, title, fps=12, hold_frames=12):
    """snaps: list of (t, c); traj: list of (t, x, y, active, phase)."""
    import imageio.v2 as imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axL = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    axR = fig.add_axes([0.55, 0.08, 0.42, 0.84])
    tt = np.array([p[0] for p in traj]); xx = np.array([p[1] for p in traj]); yy = np.array([p[2] for p in traj])
    act = np.array([p[3] for p in traj]); phases = [p[4] for p in traj]
    frames = list(snaps) + [snaps[-1]] * hold_frames
    for fi, (t, c) in enumerate(frames):
        axL.clear(); axR.clear()
        axL.imshow(c_to_rgb(c, mask), interpolation="bilinear"); axL.axis("off")
        label = f"{title}   t = {t:5.2f} s" + ("   FINAL HOLD" if fi >= len(snaps) else "")
        axL.text(0.02, 0.98, label, transform=axL.transAxes, va="top", color="w", fontsize=12)
        axR.add_patch(plt.Circle((0, 0), 0.49, fill=False, color="0.4", lw=1.5))
        sel = tt <= t + 1e-9
        if sel.any():
            on = act & sel; off = (~act) & sel
            axR.plot(xx[off], yy[off], ".", color="0.6", ms=2)
            axR.plot(xx[on], yy[on], "-", color="tab:blue", lw=1.5)
            k = int(np.searchsorted(tt, t, side="right")) - 1
            k = max(0, min(k, len(tt) - 1))
            if act[k]:
                axR.plot(xx[k], yy[k], "o", color="tab:red", ms=9)
            axR.set_title(f"inlet path  |  phase: {phases[k]}", fontsize=11)
        axR.set_xlim(-0.55, 0.55); axR.set_ylim(-0.55, 0.55); axR.set_aspect("equal")
        axR.set_xlabel("x [cup diameters]"); axR.set_ylabel("y")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close()
    plt.close(fig)


def overview(path, items, ncols=3):
    """items: list of (name, c, mask)."""
    n = len(items); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=110)
    for ax, (name, c, mask) in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(c_to_rgb(c, mask), interpolation="bilinear"); ax.set_title(name); ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
