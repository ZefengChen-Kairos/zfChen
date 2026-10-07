"""Diagnostic rendering of the milk fraction c (not an optical renderer)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BROWN = np.array([0.42, 0.25, 0.12])
WHITE = np.array([0.97, 0.95, 0.91])
BG = np.array([0.16, 0.16, 0.17])


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


def make_video(path, snaps, traj, mask, title, fps=24, hold_frames=24, cn=""):
    """1536x768 movie: left = native milk-fraction field, right = inlet path, phase, clock and flow curves.
    snaps: list of (t, c) at the movie frame rate;  traj: list of dicts per control frame with keys
    t, x, y, active, phase, S, chi.  The last `hold_frames` frames repeat the final state (FINAL HOLD)."""
    import imageio.v2 as imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    fig = plt.figure(figsize=(16, 8), dpi=96)
    fig.patch.set_facecolor("#1b1b1c")
    axL = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    axT = fig.add_axes([0.57, 0.40, 0.40, 0.55])
    axF = fig.add_axes([0.57, 0.08, 0.40, 0.24])
    tt = np.array([p["t"] for p in traj]); xx = np.array([p["x"] for p in traj]); yy = np.array([p["y"] for p in traj])
    act = np.array([p["active"] for p in traj]); phases = [p["phase"] for p in traj]
    SS = np.array([p["S"] for p in traj]); chi = np.array([p["chi"] for p in traj])
    dep = SS * chi
    T_end = snaps[-1][0]
    frames = list(snaps) + [snaps[-1]] * hold_frames
    for fi, (t, c) in enumerate(frames):
        hold = fi >= len(snaps)
        axL.clear(); axT.clear(); axF.clear()
        for ax in (axT, axF):
            ax.set_facecolor("#242426")
            for sp in ax.spines.values():
                sp.set_color("0.5")
            ax.tick_params(colors="0.8", labelsize=9)
        axL.imshow(c_to_rgb(c, mask), interpolation="bilinear"); axL.axis("off")
        k = int(np.searchsorted(tt, t + 1e-9, side="right")) - 1
        k = max(0, min(k, len(tt) - 1))
        ph = "settle" if hold else phases[k]
        axL.text(0.02, 0.98, f"{title}  {cn}", transform=axL.transAxes, va="top", color="w", fontsize=15)
        axL.text(0.02, 0.93, f"t = {t:5.2f} s   phase: {ph}", transform=axL.transAxes, va="top", color="0.85", fontsize=12)
        axL.text(0.02, 0.03, "native milk fraction c = m/l,  display saturates at c = 0.8", transform=axL.transAxes,
                 color="0.6", fontsize=9)
        if hold:
            axL.text(0.98, 0.98, "FINAL HOLD", transform=axL.transAxes, va="top", ha="right", color="#ffd166", fontsize=15)
        # --- trajectory panel
        axT.add_patch(plt.Circle((0, 0), 0.49, fill=False, color="0.55", lw=1.5))
        n = int((tt <= t + 1e-9).sum())
        i0 = 0
        while i0 < n:
            i1 = i0
            while i1 + 1 < n and act[i1 + 1] == act[i0]:
                i1 += 1
            stop = i1 + 2 if i1 + 1 < n else i1 + 1
            if act[i0]:
                axT.plot(xx[i0:stop], yy[i0:stop], "-", color="#4c9be8", lw=1.8)
            else:
                axT.plot(xx[i0:stop], yy[i0:stop], "--", color="0.55", lw=1.0)
            i0 = i1 + 1
        if not hold and n > 0:
            if act[k]:
                axT.plot(xx[k], yy[k], "o", color="#e8504c", ms=10, mec="w")
            else:
                axT.plot(xx[k], yy[k], "o", color="0.7", ms=7, mec="w")
        axT.set_xlim(-0.55, 0.55); axT.set_ylim(-0.55, 0.55); axT.set_aspect("equal")
        axT.set_title("inlet hit point in the cup frame  (blue: pouring, grey dashed: no flow)", color="0.9", fontsize=10)
        axT.set_xlabel("x [cup diameters]", color="0.8"); axT.set_ylabel("y  (barista at y<0)", color="0.8")
        # --- flow panel
        smax = max(float(SS.max()), 1e-6)
        axF.plot(tt, SS, color="#4c9be8", lw=1.2, label="effective inlet rate S = κ_Q Q")
        axF.plot(tt, dep, color="#f2f2f2", lw=1.2, label="deposition rate χS")
        axF.plot(tt, chi * smax, color="#ffd166", lw=0.9, ls=":", label="χ (scaled)")
        axF.axvline(min(t, T_end), color="#e8504c", lw=1.2)
        axF.set_xlim(0, T_end); axF.set_ylim(0, 1.1 * smax)
        axF.set_xlabel("model time [s]", color="0.8")
        axF.legend(loc="upper right", fontsize=8, facecolor="#242426", labelcolor="0.9", edgecolor="0.4")
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
