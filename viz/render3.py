"""viz29 render3: the three-view videos of one pour from a forward_fields npz (Warp ray tracer, pure Python).

    python render3.py --pour tulip --fields s1kq_params_tulip.npz --out DIR [--fps 30] [--size 1280x720]
        [--views pour,guest,top,combo] [--model v05n] [--model-desc "v05n 基线"] [--which start|best|extra]
        [--label text] [--device cuda|cpu] [--t0 s] [--t1 s] [--ss 2] [--cup dark|white] [--root TREE]
        [--font wqy-zenhei.ttc] [--no-sheet]
Caption: model (description) · which closure (起点参数（未重新拟合，<file>） / 拟合后参数（best_closure.json）) · inlet ·
N (of the fields, once) · second line: 杯子为示意 and where everything comes from.
-> DIR/<p>_pour.mp4    倒奶视角: side view, translucent CAD pitcher so the milk level shows, the stream; HUD: time,
                       Q at the lip [mL/s], poured [mL], left in the pitcher [mL], in the cup [mL], head, axis tilt,
                       arrival speed, Q(t) chart with a cursor
   DIR/<p>_guest.mp4   客人视角: from across the cup (customer side, 40 deg elevation), latte surface textured with the
                       solver field, pitcher + stream above
   DIR/<p>_top.mp4     俯视二维: the solver field C top-down (render512 orientation, C[::-1]), solver landing marker +
                       the footprint ellipse the solver used, projected spout tip
   DIR/<p>_combo.mp4   2 x 2: pour | guest / top (sim) | top (real rectified video, video_obs), one clock, caption
   DIR/<p>_sheet.jpg   6 times x (pour, guest, top sim, top real)
   DIR/<p>_viz.json    what was drawn (cameras, frames, snap statistics, timings)
Everything that moves comes from the simulation chain (viz/scene.py): measured pitcher pose v2 + the CAD pitcher, the
pour model's ledger (milk left, free surface), its emitted parcels (the stream; foot snapped to the solver landing),
the solver's white fraction C.  The cup beyond the liquid disc is NOT measured: a stand-in mug (caption says so).
Videos: imageio-ffmpeg, libx264, yuv420p, frames piped (no PNG sequences)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raytrace as RT  # noqa: E402
import scene as S  # noqa: E402

EXT = 1.05                                       # top views: disc radii shown (render512.EXT)
CUP_TOP = np.array([28, 24, 26], float)          # render512 CUP (outside the disc in the top views)
NAMES = dict(tulip="郁金香 tulip", leaf="树叶 leaf", heart="爱心 heart", push_heart="推推乐 push_heart", swan="天鹅 swan")
VIEW_ZH = dict(pour="倒奶视角", guest="客人视角", top="俯视二维（模拟）", real="俯视（真实视频，校正后）")
VIEW_EN = dict(pour="pour view", guest="guest view", top="top 2-D (simulation)", real="top (real video, rectified)")
INLET_ZH = dict(replay="实测入口 MI-R（回放）", law="MI-L 出口定律", record="旧入口 R1（记录 + E0）")
INLET_EN = dict(replay="measured inlet MI-R (replay)", law="MI-L exit law", record="record inlet R1 (+E0)")
# which closure of a params run: the start closure as it is, or the fitted best_closure.json (autodl_viz21.sh)
WHICH_ZH = dict(start="起点参数（未重新拟合）", best="拟合后参数", extra="指定闭包")
WHICH_EN = dict(start="start closure (not refitted)", best="fitted closure", extra="given closure")
WHICH_SHORT_ZH = dict(start="起点参数", best="拟合后参数", extra="指定闭包")
WHICH_SHORT_EN = dict(start="start", best="fitted", extra="given")


def strip_grid(label):
    """A label without its own grid size ('..., N=96'): the caption shows the N of the fields once."""
    import re
    s = re.sub(r"[\s,，;；·、]*(?<![A-Za-z0-9_])N\s*=\s*\d+(?![0-9])", "", str(label or ""))
    return re.sub(r"[\s,，;；·、]+$", "", s).strip()


# ------------------------------------------------------------------------------------------------ fonts / text
def find_font(cli=None, root=None):
    cands = [cli, os.environ.get("VIZ_FONT")]
    for r in (root, os.path.join(HERE, ".."), HERE, os.path.join(HERE, "..", "..", "mi27", "pkg", "tree"),
              os.path.join(HERE, "..", "..", "diag20", "tree")):
        if r:
            cands += [os.path.join(r, "fonts", "wqy-zenhei.ttc"), os.path.join(r, "assets", "wqy-zenhei.ttc")]
    cands += ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


class Fonts:
    def __init__(self, path, scale=1.0):
        from PIL import ImageFont
        self.cjk = path is not None
        self.cache = {}
        self.path = path
        self.scale = scale
        self._IF = ImageFont

    def __call__(self, px):
        px = max(8, int(round(px * self.scale)))
        if px not in self.cache:
            if self.path:
                self.cache[px] = self._IF.truetype(self.path, px)
            else:
                try:
                    self.cache[px] = self._IF.load_default(size=px)
                except TypeError:
                    self.cache[px] = self._IF.load_default()
        return self.cache[px]


def panel_box(im, xy, wh, alpha=150):
    """Darken a rectangle of a PIL RGB image in place (HUD backdrop)."""
    x, y = int(xy[0]), int(xy[1])
    w, h = int(wh[0]), int(wh[1])
    a = np.asarray(im)
    sub = a[max(y, 0):y + h, max(x, 0):x + w].astype(np.float32)
    sub = sub * (1 - alpha / 255.0) + np.array([12, 12, 16], np.float32) * (alpha / 255.0)
    from PIL import Image
    im.paste(Image.fromarray(sub.astype(np.uint8)), (max(x, 0), max(y, 0)))


# ------------------------------------------------------------------------------------------------ scene per frame
class Viz:
    """Scene + renderer + cameras of one pour."""

    def __init__(self, fields, device="cpu", ss=2, cup="dark", W=1280, H=720, stl=None):
        self.sc = S.Scene(fields, stl=stl)
        sc = self.sc
        self.W, self.H = W, H
        V, F = sc.stl()
        self.V_stl = V
        c0 = sc.cup(0.0)
        self.rise0 = c0["rise_left"]
        cv, cf, cn, self.z_foot = RT.cup_mesh(sc.R_disc, self.rise0)
        self.R = RT.Renderer(device, pitcher=(V, F), cup=(cv, cf, cn), cup_white=(cup == "white"), ss=ss)
        self.cams = self._cameras()
        self.snap = self._snap_stats()
        N = int(sc.meta.get("N", sc.N))
        self.N_solver = N

    # -------------------------------------------------------------- cameras (fixed per pour, from the data)
    def _cameras(self):
        sc = self.sc
        T = sc.t_end()
        ts = np.linspace(0.0, T, 16)
        Vs = self.V_stl[::150]
        P, tips = [], []
        for t in ts:
            p = sc.pose(t)
            P.append(Vs @ p["R"].T + p["t"]); tips.append(p["tip"])
        P = np.concatenate(P); tips = np.array(tips)
        ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        Ro = sc.R_disc + 0.004
        ring = np.c_[np.cos(ang) * Ro, np.sin(ang) * Ro]
        cup_top = np.c_[ring, np.full(len(ring), 0.003 + self.rise0)]
        cup_bot = np.c_[ring * 0.8, np.full(len(ring), self.z_foot + self.rise0)]
        cup_pts = np.concatenate([cup_top, cup_bot])
        asp = self.W / self.H
        # pour view: the whole pitcher over the pour + the cup, above the caption band (HUD panels are translucent)
        pour = RT.fit_camera(np.concatenate([P, cup_pts]), 180.0, 14.0, fov_y_deg=30.0, aspect=asp, margin=0.04,
                             box=(-0.97, 0.97, -0.84, 0.97))
        # guest view: the cup, the stream and the lower part of the pitcher (its top may leave the frame)
        ztop = float(tips[:, 2].max()) + 0.03
        Pg = P[P[:, 2] <= ztop]
        guest = RT.fit_camera(np.concatenate([Pg, cup_pts, tips]), -90.0, 40.0, fov_y_deg=30.0, aspect=asp, margin=0.04,
                              box=(-0.97, 0.97, -0.85, 0.99))
        return dict(pour=pour, guest=guest)

    def _snap_stats(self):
        sc = self.sc
        sn = []
        for t in sc.clock():
            s = sc.stream(t)
            if s["foot"] is not None and sc.solver_foot(t) is not None:
                sn.append(float(np.hypot(*s["snap"])) * 1e3)
        sn = np.array(sn)
        if not len(sn):
            return dict(p50=None, p90=None, max=None, n=0)
        return dict(p50=float(np.percentile(sn, 50)), p90=float(np.percentile(sn, 90)), max=float(sn.max()), n=int(len(sn)))

    # -------------------------------------------------------------- one frame
    def frame(self, t):
        sc = self.sc
        pose = sc.pose(t)
        fl = sc.flow(t)
        st = sc.stream(t)
        cu = sc.cup(t)
        fd = sc.field(t)
        A = sc.filled(fd["C"]).astype(np.float32)
        Vm, Fm, cap = (None, None, None)
        if np.isfinite(fl["z_f"]) and fl["V_P"] > 1e-7:
            Vm, Fm, cap = sc.milk_mesh(t, pose, fl["z_f"], inset=2e-4)
        sm = RT.stream_mesh(st["segments"])
        return dict(t=float(t), pose=pose, flow=fl, stream=st, cup=cu, A=A, milk=(Vm, Fm), stream_mesh=sm,
                    foot=sc.solver_foot(t))

    def load(self, fr):
        parts = []
        if fr["milk"][0] is not None:
            parts.append((fr["milk"][0], fr["milk"][1], None, RT.MAT_MILK))
        if fr["stream_mesh"] is not None:
            parts.append((fr["stream_mesh"][0], fr["stream_mesh"][1], fr["stream_mesh"][2], RT.MAT_STREAM))
        self.R.set_dyn(parts)
        self.R.set_field(fr["A"], self.sc.L, self.sc.R_disc)

    def render(self, fr, view, W=None, H=None, ss=None):
        W, H = W or self.W, H or self.H
        dz = fr["cup"]["rise_left"]
        p = fr["pose"]
        if view == "pour":
            return self.R.render(self.cams["pour"], W, H, pitcher_pose=(p["R"], p["t"]), pitcher_alpha=0.22,
                                 pitcher_shadow=0.55, cup_dz=dz, table_z=self.z_foot + dz, ss=ss)
        return self.R.render(self.cams["guest"], W, H, pitcher_pose=(p["R"], p["t"]), pitcher_alpha=1.0,
                             pitcher_shadow=0.0, cup_dz=dz, table_z=self.z_foot + dz, ss=ss)

    # -------------------------------------------------------------- top views
    def top_sim(self, fr, Hs, marks=True):
        """(Hs, Hs, 3) uint8: the field top-down (C[::-1] orientation, disc +-EXT), mug outside the disc; markers:
        solver landing (+), the footprint ellipse the solver used, the projected spout tip."""
        from PIL import Image, ImageDraw
        sc = self.sc
        s = np.linspace(-EXT, EXT, Hs)
        U, V = np.meshgrid(s, s)
        c = sc.sample(fr["A"], U * sc.R_disc, -V * sc.R_disc, filled=False)
        img = S.latte_rgb(c)
        img[U ** 2 + V ** 2 > 1.0] = CUP_TOP
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if not marks:
            return np.asarray(im)
        d = ImageDraw.Draw(im)

        def px(X, Y):
            return ((X / sc.R_disc + EXT) / (2 * EXT) * Hs - 0.5, (-Y / sc.R_disc + EXT) / (2 * EXT) * Hs - 0.5)
        lw = max(1, Hs // 240)
        tip = fr["pose"]["tip"]
        if tip[2] > 0:
            u, v = px(tip[0], tip[1]); r = max(3, Hs // 110)
            d.polygon([(u, v - r), (u + r, v), (u, v + r), (u - r, v)], outline=(90, 200, 255), width=lw)
        f = self._sched(fr["t"])
        if f is not None:
            x0, y0 = sc.S["x"][f], sc.S["y"][f]
            hmin = 1.5 / self.N_solver
            r1, r2, ph = max(sc.S["r1"][f], hmin), max(sc.S["r2"][f], hmin), sc.S["phi"][f]
            th = np.linspace(0, 2 * math.pi, 64)
            ex = x0 + r1 * math.cos(ph) * np.cos(th) - r2 * math.sin(ph) * np.sin(th)
            ey = y0 + r1 * math.sin(ph) * np.cos(th) + r2 * math.cos(ph) * np.sin(th)
            pts = [px(a * sc.L, b * sc.L) for a, b in zip(ex, ey)]
            d.line(pts + [pts[0]], fill=(255, 70, 70), width=lw)
            u, v = px(x0 * sc.L, y0 * sc.L); r = max(4, Hs // 70)
            d.line([(u - r, v), (u + r, v)], fill=(255, 40, 40), width=lw + 1)
            d.line([(u, v - r), (u, v + r)], fill=(255, 40, 40), width=lw + 1)
        return np.asarray(im)

    def _sched(self, t):
        sc = self.sc
        if not sc.S:
            return None
        f = int(np.clip(np.searchsorted(sc.S_t1, t, side="right"), 0, len(sc.S_t1) - 1))
        return f if bool(sc.S["active"][f]) else None


# ------------------------------------------------------------------------------------------------ real top view
class RealTop:
    def __init__(self, p, root=None, meta=None, path=None):
        cands = [path]
        for r in (root, os.path.join(os.path.dirname(str((meta or {}).get("package") or "")), ""), os.getcwd(),
                  os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "mi27", "pkg", "tree")):
            if r:
                cands.append(os.path.join(r, "raw_data", "rig", "video_obs", f"{p}.npz"))
        self.fn = next((c for c in cands if c and os.path.exists(c)), None)
        if self.fn:
            Z = np.load(self.fn)
            self.t, self.img = np.asarray(Z["t"], float), Z["img"]
        self._cache = {}

    def __call__(self, t, Hs):
        from PIL import Image
        if self.fn is None:
            a = np.full((Hs, Hs, 3), 40, np.uint8)
            return a, None
        j = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))
        key = (j, Hs)
        if key not in self._cache:
            self._cache = {}
            s = np.linspace(-EXT, EXT, Hs)
            U, V = np.meshgrid(s, s)
            a = np.asarray(Image.fromarray(self.img[j]).resize((Hs, Hs), Image.BICUBIC), float)
            a[U ** 2 + V ** 2 > 1.0] = CUP_TOP
            self._cache[key] = np.clip(a, 0, 255).astype(np.uint8)
        return self._cache[key], float(self.t[j])


# ------------------------------------------------------------------------------------------------ HUD
class Hud:
    def __init__(self, viz, fonts, label, p, W, H, model=None, desc=None, which=None):
        """label: free text (legacy --label; its own 'N=..' is dropped); model / desc: model id and description
        ('v05n', 'v05n 基线'); which: start | best | extra (or any text)."""
        self.v, self.F, self.p, self.W, self.H = viz, fonts, p, W, H
        sc = viz.sc
        m = sc.meta
        cjk = fonts.cjk
        self.cjk = cjk
        inl = str(m.get("inlet") or "")
        held = bool(m.get("heldout"))
        il = (INLET_ZH if cjk else INLET_EN).get(inl, inl)
        if held:
            # e0 as the forward run used it (forward_fields meta: str(e0) = 'False' | 'jet' | 'world'): an MI closure's
            # held-out pour runs MI-L with e0 off; a record closure's runs record + its own e0 (the control fit's test)
            e0 = m.get("e0")
            e0_off = (inl == "law") if e0 is None else str(e0).strip().lower() in ("false", "0", "0.0", "none", "off", "")
            if e0_off:
                il += "（留出，e0 关，仅报告）" if cjk else " (held out, e0 off, report only)"
            else:
                il += "（留出，仅报告）" if cjk else " (held out, report only)"
        N = m.get("N", sc.N)
        name = NAMES.get(p, p) if cjk else p
        self.pattern = name + ("（留出）" if held and cjk else "（标定）" if cjk else "")
        stem = os.path.basename(str(m.get("closure") or ""))
        label = strip_grid(label)
        model = strip_grid(model) or label or os.path.splitext(stem)[0] or "closure"
        desc = strip_grid(desc) if desc and desc != "-" else ""
        mod = f"{model}（{desc}）" if (desc and cjk) else (f"{model} ({desc})" if desc else model)
        if label and label != model:
            mod += f" · {label}"
        wz = (WHICH_ZH if cjk else WHICH_EN).get(which, which) if which else ""
        if wz and stem:
            wz = (wz[:-1] + f"，{stem}）" if wz.endswith("）") else wz + f"（{stem}）") if cjk else f"{wz} ({stem})"
        ws = (WHICH_SHORT_ZH if cjk else WHICH_SHORT_EN).get(which, which) if which else ""
        self.model, self.which = model, which
        self.title = " · ".join(x for x in (model, ws, self.pattern) if x)
        snap = viz.snap
        sn = "" if snap["p50"] is None else (f"奶柱下端对到求解器落点（校正中位 {snap['p50']:.1f} mm）" if cjk else
                                             f"stream foot snapped to the solver landing (median {snap['p50']:.1f} mm)")
        if snap["p50"] is not None and snap["p50"] > 5.0:
            sn += ("：旧入口的落点修正，奶柱因此倾斜" if cjk else ": the old inlet's landing correction, hence the lean")
        orc = ""
        if m.get("oracle"):
            orc = "含诊断用入口修正（oracle，仅诊断）" if cjk else "with diagnostic oracle inlet corrections"
        self.line1 = " · ".join(x for x in (mod, wz, il, f"N={N}", orc) if x)
        self.line2 = ("杯子为示意（只测了液面圆盘）· 壶 = CAD + 实测位姿 · 奶量/奶柱 = 倒奶模型 · 拉花 = 求解器白度场 · "
                      + sn) if cjk else ("cup is a stand-in (only the liquid disc is measured) · pitcher = CAD + measured "
                                         "pose · milk / stream = pour model · pattern = solver white fraction · " + sn)
        self.chart = None
        self._cache = {}

    def _width(self, s, px, stroke=True):
        """Width [px] of a string as _text draws it."""
        return self._bitmap(s, px, (255, 255, 255), stroke)[0].size[0]

    def _text(self, im, xy, s, px, fill=(255, 255, 255), stroke=True):
        """Draw text on the PIL image `im` (RGB) at xy."""
        img, (ox, oy) = self._bitmap(s, px, fill, stroke)
        im.paste(img, (int(xy[0]) + ox, int(xy[1]) + oy), img)

    def _bitmap(self, s, px, fill, stroke):
        """Rendered strings are cached as RGBA bitmaps (the captions and labels are the same in every frame; font
        rendering is most of the HUD's cost)."""
        from PIL import Image, ImageDraw
        key = (s, int(round(px * self.F.scale)), tuple(fill), bool(stroke))
        b = self._cache.get(key)
        if b is None:
            f = self.F(px)
            sw = 1 if stroke else 0
            x0, y0, x1, y1 = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), s, font=f, stroke_width=sw)
            b = Image.new("RGBA", (max(1, x1 - min(x0, 0) + 2), max(1, y1 - min(y0, 0) + 2)), (0, 0, 0, 0))
            ImageDraw.Draw(b).text((-min(x0, 0), -min(y0, 0)), s, font=f, fill=tuple(fill) + (255,), stroke_width=sw,
                                   stroke_fill=(0, 0, 0, 255))
            b = (b, (min(x0, 0), min(y0, 0)))
            if len(self._cache) > 3000:
                self._cache.clear()
            self._cache[key] = b
        return b

    def caption(self, im, y=None, px=15):
        from PIL import ImageDraw
        W, H = im.size
        s = W / 1280.0
        y = H - int(46 * s) if y is None else y
        panel_box(im, (0, y), (W, H - y), 160)
        d = ImageDraw.Draw(im)
        self._text(im, (int(12 * s), y + int(5 * s)), self.line1, px * s, (255, 236, 170))
        self._text(im, (int(12 * s), y + int(25 * s)), self.line2, (px - 3) * s, (215, 215, 215))

    def q_chart(self, cw, ch):
        """Static Q(t) chart (lip flow, mL/s) as an RGB image + (t0, t1, q1, origin) for the cursor."""
        from PIL import Image, ImageDraw
        sc = self.v.sc
        E = sc.E
        t = E["t"]; q = E["Q_o"] * 1e6
        nb = max(1, int(round(0.02 / max(float(np.median(np.diff(t))), 1e-6))))
        n = len(t) // nb * nb
        tb, qb = t[:n].reshape(-1, nb).mean(1), q[:n].reshape(-1, nb).mean(1)
        T1 = sc.t_end()
        q1 = max(5.0, float(np.ceil(qb.max() / 10.0) * 10.0))
        im = Image.new("RGB", (cw, ch), (14, 14, 18))
        d = ImageDraw.Draw(im)
        s = ch / 120.0
        L, R, Tp, B = int(40 * s), cw - int(8 * s), int(22 * s), ch - int(20 * s)
        d.rectangle([L, Tp, R, B], outline=(90, 90, 100))
        for k in range(1, 4):
            y = B - (B - Tp) * k / 4
            d.line([(L, y), (R, y)], fill=(40, 40, 48))
        xs = L + (tb / T1) * (R - L)
        ys = B - np.clip(qb / q1, 0, 1) * (B - Tp)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=(120, 200, 255), width=max(1, int(2 * s)))
        f = self.F(12 * s)
        d.text((L, 3 * s), "壶口流量 Q(t) [mL/s]" if self.cjk else "lip flow Q(t) [mL/s]", font=f, fill=(200, 200, 210))
        d.text((3 * s, Tp - 6 * s), f"{q1:.0f}", font=f, fill=(170, 170, 180))
        d.text((3 * s, B - 7 * s), "0", font=f, fill=(170, 170, 180))
        d.text((L, B + 3 * s), "0 s", font=f, fill=(170, 170, 180))
        d.text((R - 34 * s, B + 3 * s), f"{T1:.0f} s", font=f, fill=(170, 170, 180))
        self.chart = (np.asarray(im), (L, R, Tp, B, T1, q1))
        return self.chart

    def pour(self, img, fr):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        W, H = im.size
        s = W / 1280.0
        fl = fr["flow"]
        p = fr["pose"]
        cjk = self.cjk
        rows = [(f"{VIEW_ZH['pour'] if cjk else VIEW_EN['pour']}   t = {fr['t']:5.2f} s", 22, (255, 255, 255)),
                (self.title, 15, (255, 236, 170))]
        Q = fl["Q_lip"] * 1e6
        vals = [("壶口流量 Q" if cjk else "lip flow Q", f"{Q:5.1f} mL/s"),
                ("已倒出" if cjk else "poured", f"{fl['V_out'] * 1e6:5.1f} mL"),
                ("壶内剩余" if cjk else "left in pitcher", f"{fl['V_P'] * 1e6:5.1f} mL"),
                ("已落入杯中" if cjk else "arrived in cup", f"{fl.get('V_cup', float('nan')) * 1e6:5.1f} mL"),
                ("液头 H" if cjk else "head H", f"{fl['H'] * 1e3:4.1f} mm"),
                ("壶轴倾角" if cjk else "axis tilt", f"{p['axis_tilt_deg']:4.0f}°"),
                ("落点速度 U⊥" if cjk else "arrival U_perp", f"{fl.get('U_perp', 0.0):4.2f} m/s")]
        bw, bh = max(int(330 * s), self._width(self.title, 15 * s) + int(22 * s)), int((64 + 22 * len(vals)) * s)
        panel_box(im, (int(10 * s), int(10 * s)), (bw, bh), 150)
        d = ImageDraw.Draw(im)
        y = int(16 * s)
        for txt, px, col in rows:
            self._text(im, (int(20 * s), y), txt, px * s, col); y += int((px + 8) * s)
        for k, v in vals:
            self._text(im, (int(20 * s), y), k, 15 * s, (210, 210, 220))
            self._text(im, (int(190 * s), y), v, 15 * s, (255, 255, 255)); y += int(22 * s)
        # Q(t) chart with the cursor
        cw, ch = int(330 * s), int(120 * s)
        if self.chart is None or self.chart[0].shape[1] != cw:
            self.q_chart(cw, ch)
        ci, (L, R, Tp, B, T1, q1) = self.chart
        x0, y0 = int(10 * s), H - ch - int(56 * s)          # bottom left: the table side, away from the spout
        im.paste(Image.fromarray(ci), (x0, y0))
        d = ImageDraw.Draw(im)
        xc = x0 + L + (fr["t"] / T1) * (R - L)
        d.line([(xc, y0 + Tp), (xc, y0 + B)], fill=(255, 210, 90), width=max(1, int(2 * s)))
        yc = y0 + B - min(Q / q1, 1.0) * (B - Tp)
        r = 4 * s
        d.ellipse([xc - r, yc - r, xc + r, yc + r], fill=(255, 210, 90))
        self.caption(im)
        return np.asarray(im)

    def guest(self, img, fr):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        s = im.size[0] / 1280.0
        panel_box(im, (int(10 * s), int(10 * s)), (max(int(330 * s), self._width(self.title, 15 * s) + int(22 * s)),
                                                 int(62 * s)), 150)
        d = ImageDraw.Draw(im)
        self._text(im, (int(20 * s), int(16 * s)), f"{VIEW_ZH['guest'] if self.cjk else VIEW_EN['guest']}   t = {fr['t']:5.2f} s",
                   22 * s)
        self._text(im, (int(20 * s), int(46 * s)), self.title, 15 * s, (255, 236, 170))
        self.caption(im)
        return np.asarray(im)

    def top(self, sq, fr, W, H):
        """Standalone top view: the square field on the left, legend on the right."""
        from PIL import Image, ImageDraw
        s = W / 1280.0
        im = Image.new("RGB", (W, H), (18, 18, 22))
        capH = int(46 * s)
        Hs = sq.shape[0]
        x0, y0 = int(24 * s), (H - capH - Hs) // 2
        im.paste(Image.fromarray(sq), (x0, max(y0, 0)))
        d = ImageDraw.Draw(im)
        xt = x0 + Hs + int(30 * s)
        cjk = self.cjk
        self._text(im, (xt, int(24 * s)), f"{VIEW_ZH['top'] if cjk else VIEW_EN['top']}", 24 * s)
        self._text(im, (xt, int(60 * s)), f"t = {fr['t']:5.2f} s", 22 * s)
        self._text(im, (xt, int(92 * s)), self.title, 16 * s, (255, 236, 170))
        y = int(140 * s)
        leg = ([("+", (255, 40, 40), "求解器落点（奶柱落点）"), ("○", (255, 70, 70), "落点足迹椭圆（求解器所用）"),
                ("◇", (90, 200, 255), "壶嘴尖的投影")] if cjk else
               [("+", (255, 40, 40), "solver landing (stream foot)"), ("o", (255, 70, 70), "footprint ellipse (solver)"),
                ("<>", (90, 200, 255), "spout tip, projected")])
        for sym, col, txt in leg:
            self._text(im, (xt, y), sym, 18 * s, col, stroke=False)
            self._text(im, (xt + int(30 * s), y), txt, 16 * s, (220, 220, 225)); y += int(30 * s)
        # palette strip
        y += int(14 * s)
        bw, bh = int(300 * s), int(16 * s)
        strip = S.latte_rgb(np.linspace(0, 1, bw))[None].repeat(bh, 0)
        im.paste(Image.fromarray(strip.astype(np.uint8)), (xt, y))
        d = ImageDraw.Draw(im)
        self._text(im, (xt, y + bh + int(4 * s)), "白度 C = 0（咖啡）" if cjk else "white fraction C = 0 (crema)", 13 * s,
                   (200, 200, 205))
        self._text(im, (xt + bw - int(110 * s), y + bh + int(4 * s)), "1（奶泡）" if cjk else "1 (foam)", 13 * s, (200, 200, 205))
        y += bh + int(40 * s)
        note = (["方位与真实俯视视频一致：", "x 向右，y 向上（拉花师在上方，", "客人在下方，壶嘴朝下）"] if cjk else
                ["orientation = the rectified real top view:", "x right, y up (barista at the top,", "customer at the bottom)"])
        for k, n in enumerate(note):
            self._text(im, (xt, y + k * int(22 * s)), n, 14 * s, (180, 180, 190))
        self.caption(im)
        return np.asarray(im)

    def combo(self, pour, guest, topsq, realsq, t_real, fr, W, H):
        """2 x 2: pour | guest / top sim | top real (each W/2 x H/2) under a caption band."""
        from PIL import Image, ImageDraw
        s = W / 1280.0
        band = int(64 * s)
        band += (-(H + band)) % 16
        im = Image.new("RGB", (W, H + band), (18, 18, 22))
        w2, h2 = W // 2, H // 2
        def half(a):
            im_ = Image.fromarray(a)
            return im_.reduce(2) if (a.shape[1] == 2 * w2 and a.shape[0] == 2 * h2) else im_.resize((w2, h2), Image.LANCZOS)
        pa, ga = half(pour), half(guest)
        im.paste(pa, (0, band)); im.paste(ga, (w2, band))
        for k, sq in enumerate((topsq, realsq)):
            x = k * w2 + (w2 - sq.shape[1]) // 2
            im.paste(Image.fromarray(sq), (x, band + h2 + (h2 - sq.shape[0]) // 2))
        d = ImageDraw.Draw(im)
        d.line([(w2, band), (w2, band + H)], fill=(60, 60, 70), width=2)
        d.line([(0, band + h2), (W, band + h2)], fill=(60, 60, 70), width=2)
        cjk = self.cjk
        names = VIEW_ZH if cjk else VIEW_EN
        labels = [(names["pour"], 0, band), (names["guest"], w2, band), (names["top"], 0, band + h2),
                  (names["real"] + (f"  t = {t_real:.2f} s" if t_real is not None else ""), w2, band + h2)]
        for txt, x, y in labels:
            self._text(im, (x + int(8 * s), y + int(6 * s)), txt, 16 * s, (255, 236, 170))
        fl = fr["flow"]
        self._text(im, (int(10 * s), int(5 * s)),
                   f"{self.title}   t = {fr['t']:5.2f} s   Q = {fl['Q_lip'] * 1e6:4.1f} mL/s   "
                   + (f"已倒出 {fl['V_out'] * 1e6:5.1f} mL   壶内 {fl['V_P'] * 1e6:5.1f} mL" if cjk else
                      f"poured {fl['V_out'] * 1e6:5.1f} mL   left {fl['V_P'] * 1e6:5.1f} mL"), 19 * s)
        self._text(im, (int(10 * s), int(30 * s)), self.line1, 13 * s, (255, 236, 170))
        self._text(im, (int(10 * s), int(47 * s)), self.line2, 11 * s, (210, 210, 210))
        return np.asarray(im)


# ------------------------------------------------------------------------------------------------ video writer
class Writer:
    def __init__(self, fn, fps):
        import imageio
        self.fn = fn
        self.tmp = fn[:-4] + ".part.mp4"
        self.w = imageio.get_writer(self.tmp, fps=fps, codec="libx264", quality=8, macro_block_size=2,
                                    pixelformat="yuv420p", ffmpeg_log_level="error")
        self.n = 0

    def add(self, img):
        a = np.asarray(img)
        h, w = a.shape[:2]
        if h % 2 or w % 2:
            a = np.pad(a, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
        self.w.append_data(a)
        self.n += 1

    def close(self, ok=True):
        self.w.close()
        if ok and self.n:
            os.replace(self.tmp, self.fn)
        elif os.path.exists(self.tmp):
            os.remove(self.tmp)


# ------------------------------------------------------------------------------------------------ main
def sheet(cells, times, fn, fonts, title):
    """cells: list over times of [pour, guest, topsq, realsq] (uint8)."""
    from PIL import Image, ImageDraw
    hh = 230
    cols = []
    for k in range(4):
        a = cells[0][k]
        cols.append(int(round(a.shape[1] * hh / a.shape[0])))
    pad, left, top = 8, 70, 60
    Wd = left + sum(cols) + pad * 5
    Hd = top + len(cells) * (hh + pad) + pad
    im = Image.new("RGB", (Wd, Hd), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((pad, 8), title, font=fonts(20), fill=(255, 255, 255))
    names = VIEW_ZH if fonts.cjk else VIEW_EN
    x = left
    for k, nm in enumerate((names["pour"], names["guest"], names["top"], names["real"])):
        d.text((x + 4, 36), nm, font=fonts(15), fill=(255, 236, 170)); x += cols[k] + pad
    for r, (row, t) in enumerate(zip(cells, times)):
        y = top + r * (hh + pad)
        d.text((8, y + hh // 2 - 10), f"{t:5.2f} s", font=fonts(16), fill=(255, 255, 255))
        x = left
        for k, a in enumerate(row):
            im.paste(Image.fromarray(a).resize((cols[k], hh), Image.LANCZOS), (x, y)); x += cols[k] + pad
    im.save(fn, quality=90)


def run(a):
    t_all = time.time()
    W, H = (int(v) for v in a.size.lower().split("x"))
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    os.makedirs(a.out, exist_ok=True)
    p = a.pour
    viz = Viz(a.fields, device=a.device, ss=a.ss, cup=a.cup, W=W, H=H, stl=a.stl)
    sc = viz.sc
    if sc.meta.get("pour") and sc.meta["pour"] != p:
        raise SystemExit(f"--pour {p} but the fields are of {sc.meta['pour']}")
    fonts = Fonts(find_font(a.font, a.root))
    label = a.label
    if not (label or a.model):          # legacy default: results_diag21/<model>/params/best_closure.json -> <model>
        label = os.path.basename(os.path.dirname(os.path.dirname(str(sc.meta.get("closure", "closure")))))
    hud = Hud(viz, fonts, label, p, W, H, model=a.model, desc=a.model_desc, which=a.which)
    real = RealTop(p, a.root, sc.meta, a.video_obs)
    t1 = sc.t_end() if a.t1 is None else min(a.t1, sc.t_end())
    ts = np.arange(max(a.t0, 0.0), t1 + 1e-9, 1.0 / a.fps)
    if not len(ts):
        raise SystemExit(f"{p}: no frames in [{a.t0}, {t1:.2f}] s (the fields end at {sc.t_end():.2f} s)")
    need_pour = "pour" in views or "combo" in views or a.sheet
    need_guest = "guest" in views or "combo" in views or a.sheet
    st_times = np.linspace(ts[0] + 0.04 * (ts[-1] - ts[0]), ts[-1] - 0.04 * (ts[-1] - ts[0]), 6) if len(ts) > 1 else ts
    st_idx = sorted({int(np.argmin(np.abs(ts - x))) for x in st_times})
    cells, cell_t = [], []
    Hs_top = H - 2 * int(24 * W / 1280.0) - int(46 * W / 1280.0)
    Hs_top -= Hs_top % 2
    h2 = H // 2
    ws = {v: Writer(os.path.join(a.out, f"{p}_{v}.mp4"), a.fps) for v in views}
    tim = dict(scene=0.0, trace=0.0, top=0.0, hud=0.0, encode=0.0)
    ok = False
    try:
        for i, t in enumerate(ts):
            t0 = time.time()
            fr = viz.frame(t)
            viz.load(fr)
            t1_ = time.time(); tim["scene"] += t1_ - t0
            ip = viz.render(fr, "pour") if need_pour else None
            ig = viz.render(fr, "guest") if need_guest else None
            t2_ = time.time(); tim["trace"] += t2_ - t1_
            sq_big = viz.top_sim(fr, Hs_top) if "top" in views else None
            sq = viz.top_sim(fr, h2) if ("combo" in views or a.sheet) else None
            rs, tr = real(t, h2) if ("combo" in views or a.sheet) else (None, None)
            t3_ = time.time(); tim["top"] += t3_ - t2_
            out = {}
            if "pour" in views:
                out["pour"] = hud.pour(ip, fr)
            if "guest" in views:
                out["guest"] = hud.guest(ig, fr)
            if "top" in views:
                out["top"] = hud.top(sq_big, fr, W, H)
            if "combo" in views:
                out["combo"] = hud.combo(ip, ig, sq, rs, tr, fr, W, H)
            t4_ = time.time(); tim["hud"] += t4_ - t3_
            for v in views:
                ws[v].add(out[v])
            tim["encode"] += time.time() - t4_
            if a.sheet and i in st_idx:
                cells.append([ip, ig, sq, rs]); cell_t.append(float(t))
            if a.verbose and (i % max(1, int(a.fps)) == 0):
                print(f"{p}: frame {i + 1}/{len(ts)} t {t:.2f} s ({time.time() - t_all:.0f} s)", flush=True)
        ok = True
    finally:
        for w in ws.values():
            w.close(ok)
    if a.sheet and cells:
        sheet(cells, cell_t, os.path.join(a.out, f"{p}_sheet.jpg"), fonts, f"{hud.pattern} · {hud.line1}")
    info = dict(pour=p, fields=os.path.abspath(a.fields), views=views, size=[W, H], fps=a.fps, frames=int(len(ts)),
                t0=float(ts[0]), t1=float(ts[-1]), device=a.device, ss=a.ss, cup=a.cup + " (stand-in, not measured)",
                caption=[hud.line1, hud.line2], title=hud.title, model=hud.model, which=hud.which,
                snap_mm=viz.snap, font=fonts.path, video_obs=real.fn,
                cams={k: {kk: (np.asarray(vv).tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in c.items()}
                      for k, c in viz.cams.items()},
                meta={k: sc.meta.get(k) for k in ("closure", "closure_sha256", "inlet", "cal_inlet", "heldout", "e0", "oracle",
                                                  "N", "L_unit_m", "R_disc_m", "V0_mL", "poured_mL")},
                timing_s=dict(tim, total=time.time() - t_all))
    json.dump(info, open(os.path.join(a.out, f"{p}_viz.json"), "w"), indent=1, default=str)
    print(f"{p}: {len(ts)} frames x {len(views)} views ({W}x{H}, ss {a.ss}, {a.device}) -> {a.out} in "
          f"{time.time() - t_all:.0f} s (scene {tim['scene']:.0f}, trace {tim['trace']:.0f}, top {tim['top']:.0f}, "
          f"hud {tim['hud']:.0f}, encode {tim['encode']:.0f} s)", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pour", required=True)
    ap.add_argument("--fields", required=True, help="forward_fields npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--views", default="pour,guest,top,combo")
    ap.add_argument("--label", default=None, help="free label for the captions (an 'N=..' in it is dropped: the caption "
                                                  "shows the fields' N once)")
    ap.add_argument("--model", default=None, help="model id for the captions, e.g. v05n")
    ap.add_argument("--model-desc", default=None, help="model description, e.g. 'v05n 基线'")
    ap.add_argument("--which", default=None, help="start (起点参数，未重新拟合) | best (拟合后参数) | extra | free text")
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: cuda:0 if available)")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--ss", type=int, default=2, help="supersampling per axis")
    ap.add_argument("--cup", default="dark", choices=("dark", "white"), help="stand-in cup colour (the real one is black)")
    ap.add_argument("--root", default=None, help="package tree (fonts/, raw_data/rig/video_obs)")
    ap.add_argument("--font", default=None)
    ap.add_argument("--video-obs", default=None)
    ap.add_argument("--stl", default=None)
    ap.add_argument("--no-sheet", dest="sheet", action="store_false")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.device is None:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        a.device = "cuda:0" if wp.is_cuda_available() else "cpu"
    elif a.device == "cuda":
        a.device = "cuda:0"
    return run(a)


if __name__ == "__main__":
    main()
