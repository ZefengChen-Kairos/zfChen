"""Subpixel tag corners: lines fitted to the black-square edges (max of the directional luminance gradient along
the edge normal, parabolic subpixel peak), corners = intersections of adjacent edge lines."""
import numpy as np, cv2


def refine(g, q, n_s=24, rng=2.5, trim=0.18, sigma=0.8):
    """g: gray float32 image (blurred), q: (4,2) initial corners (aruco order TL,TR,BR,BL, black square boundary).
    Returns refined (4,2) or None."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    cen = q.mean(0)
    lines = []
    for e in range(4):
        a, b = q[e], q[(e + 1) % 4]
        L = np.linalg.norm(b - a)
        if L < 8:
            return None
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]])
        if np.dot(nrm, (a + b) / 2 - cen) < 0:
            nrm = -nrm                                   # outward normal (dark inside -> bright outside)
        s = np.linspace(trim, 1 - trim, n_s)
        P = a[None] + s[:, None] * (b - a)[None]
        offs = np.arange(-rng, rng + 1e-6, 0.25)
        Q = P[:, None, :] + offs[None, :, None] * nrm[None, None, :]
        mx = cv2.remap(gx, Q[..., 0].astype(np.float32), Q[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        my = cv2.remap(gy, Q[..., 0].astype(np.float32), Q[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        gn = mx * nrm[0] + my * nrm[1]                     # derivative along the outward normal (dark -> bright: > 0)
        j = np.argmax(gn, 1)
        ok = (j > 0) & (j < len(offs) - 1) & (gn[np.arange(n_s), j] > 4.0)
        jj = np.clip(j, 1, len(offs) - 2)
        y0, y1, y2 = gn[np.arange(n_s), jj - 1], gn[np.arange(n_s), jj], gn[np.arange(n_s), jj + 1]
        den = y0 - 2 * y1 + y2
        sub = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / den, 0.0)
        off = offs[jj] + np.clip(sub, -0.5, 0.5) * 0.25
        pts = P + off[:, None] * nrm[None]
        pts = pts[ok]
        if len(pts) < 6:
            return None
        # robust line fit (two passes)
        for _ in range(2):
            m = pts.mean(0)
            U, S, Vt = np.linalg.svd(pts - m)
            dirn = Vt[0]; nn = np.array([-dirn[1], dirn[0]])
            r = (pts - m) @ nn
            keep = np.abs(r) < max(3 * np.median(np.abs(r)) + 1e-3, 0.3)
            if keep.sum() < 6:
                break
            pts = pts[keep]
        m = pts.mean(0); U, S, Vt = np.linalg.svd(pts - m); dirn = Vt[0]; nn = np.array([-dirn[1], dirn[0]])
        lines.append((nn, nn @ m))
    out = []
    for e in range(4):
        (n1, c1), (n2, c2) = lines[(e - 1) % 4], lines[e]   # corner e is the end of edge e-1 and the start of edge e
        A = np.stack([n1, n2]); 
        if abs(np.linalg.det(A)) < 1e-3:
            return None
        out.append(np.linalg.solve(A, [c1, c2]))
    out = np.array(out)
    if np.max(np.linalg.norm(out - q, axis=1)) > 3.0:
        return None
    return out
