/* latte2d.js — JavaScript port of latte_imex/solver.py (V1 surface-layer model, IMEX + Rusanov).
   State l, qx, qy, m on an N x N grid over the unit cup diameter; cup = disc of radius cup_radius.
   One control frame: inlet fields frozen, sub-steps by the material CFL; each sub-step
     A explicit Rusanov momentum advection -> B implicit mass/pressure/drag/traction (PCG) ->
     C milk with the same face flux (van Leer upwind) -> D explicit viscosity and mixing.
   Flat Float64Array storage, index j*N+i (j = y row, i = x column). Faces: fx (N x N-1) index j*(N-1)+i,
   fy (N-1 x N) index j*N+i.  Exposed as global `Latte2D` (browser) or module.exports (Node). */
(function (global) {
  'use strict';

  const DEFAULT_PARAMS = { cp: 0.3, beta: 3.0, nu: 1e-3, D: 1e-7, kappa_Q: 125.0, kappa_c: 1.0, kappa_t: 0.7, kappa_r: 0.3,
    B_dep: 2.5, p_dep: 2.0, return_law: 'v05',
    // "push first, whiten later" delay closure (port of solver.Params): tau_d = 0 reproduces the instantaneous closure
    tau_d: 0, sub_advect: 0.3, sub_pressure: 0.5,
    // closure 'twolayer' (port of solver.Params): the layer is the floating foam/crema film, dragged (beta) by a coffee layer
    // driven by the poured volume; the foam that survives the impact, chi(Fr), surfaces after the fountain rise time 2U/g'
    closure: 'chi', g_red: 75.0, Fr_c2: 23.0, phi_foam: 0.5, ent_coef: 0.74, cell_frac: 1.0,
    bulk_tol: 1e-8, bulk_rtol: 0.0 };   // coffee-layer solve: PCG tolerance; reuse U_b while the source changes < bulk_rtol (0 = every frame)
  const DEFAULT_NUM = { cup_radius: 0.49, cfl: 0.5, frame_dt: 1 / 120, max_substeps: 64, cg_tol: 1e-10, cg_maxiter: 500,
    kernel_quadrature: 3, scan_safety: 0.5, explicit_mixing_limit: 0.05, explicit_visc_limit: 0.2, visc_tol: 1e-8, l_floor: 1e-3 };

  class Latte2D {
    constructor(N, params, num) {
      this.N = N; this.h = 1 / N; this.area = this.h * this.h;
      this.P = Object.assign({}, DEFAULT_PARAMS, params || {});
      this.num = Object.assign({}, DEFAULT_NUM, num || {});
      const n = N * N, nfx = N * (N - 1);
      this.mask = new Uint8Array(n); this.x = new Float64Array(n); this.y = new Float64Array(n);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const x = (i + 0.5) * this.h - 0.5, y = (j + 0.5) * this.h - 0.5, k = j * N + i;
        this.x[k] = x; this.y[k] = y; this.mask[k] = Math.hypot(x, y) <= this.num.cup_radius ? 1 : 0;
      }
      this.fx = new Uint8Array(nfx); this.fy = new Uint8Array(nfx);
      this.nfx = new Float64Array(n); this.nfy = new Float64Array(n);
      for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) {
        const f = this.mask[j * N + i] & this.mask[j * N + i + 1]; this.fx[j * (N - 1) + i] = f;
        this.nfx[j * N + i] += f; this.nfx[j * N + i + 1] += f;
      }
      for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) {
        const f = this.mask[j * N + i] & this.mask[(j + 1) * N + i]; this.fy[j * N + i] = f;
        this.nfy[j * N + i] += f; this.nfy[(j + 1) * N + i] += f;
      }
      this.l = new Float64Array(n).fill(1); this.qx = new Float64Array(n); this.qy = new Float64Array(n); this.m = new Float64Array(n);
      this.t = 0; this.deposited = 0; this.stats = { steps: 0, frames: 0, cg_mass: 0, max_u: 0, retries: 0, max_substeps: 0, clip_mass: 0 };
      // work arrays
      const A = () => new Float64Array(n), F = () => new Float64Array(nfx);
      this.w = { lsafe: A(), ux: A(), uy: A(), c: A(), Fxqx: F(), Fxqy: F(), Fyqx: F(), Fyqy: F(), qxs: A(), qys: A(),
        ax: A(), ay: A(), coef: A(), Gex: F(), Gey: F(), kx: F(), ky: F(), diag: A(), rhs: A(), lnew: A(), Gx: F(), Gy: F(),
        qxn: A(), qyn: A(), cfx: F(), cfy: F(), mnew: A(), tmp: A(), r: A(), p: A(), Ap: A(), z: A(), lfx: F(), lfy: F(),
        s: A(), Lam: A(), vsx: A(), vsy: A(), sl: A(), sqx: A(), sqy: A(), sm: A(), msubn: A(), ssurf: A(), smsub: A(),
        f1: A(), t1: A(), sf: A(), sft: A(), tvx: A(), tvy: A(), Kn: A(), fsrc: A(), brhs: A(), bdiag: A() };
      this.msub = A(); this.tauCur = null;      // sub-surface milk waiting to surface (delay closure)
      // two-layer closure: submerged foam f and its rise-time moment ft, coffee-layer potential and velocity
      this.f = A(); this.ft = A(); this.phib = A(); this.ubx = A(); this.uby = A(); this.hasUb = false; this.ubKey = null; this.tl = { active: false };
      this.fxf = new Float64Array(nfx); this.fyf = new Float64Array(nfx);
      for (let k = 0; k < nfx; k++) { this.fxf[k] = this.fx[k]; this.fyf[k] = this.fy[k]; }
    }

    // ---------------------------------------------------------------- inlet fields
    inletFields(inlet) {
      const N = this.N, h = this.h, P = this.P, w = this.w;
      w.s.fill(0); w.Lam.fill(0); w.vsx.fill(0); w.vsy.fill(0); this.tl.active = false;
      if (!inlet || !inlet.active || inlet.S_eff <= 0) return 0;
      const r1 = Math.max(inlet.r1, 1.5 * h), r2 = Math.max(inlet.r2, 1.5 * h), phi = inlet.phi || 0;
      const x0 = inlet.x_hit[0], y0 = inlet.x_hit[1];
      let q = Math.max(this.num.kernel_quadrature, Math.ceil(4 * h / Math.min(r1, r2))); q = Math.min(q, 12);
      const R = Math.max(r1, r2);
      const i0 = Math.max(0, Math.floor((x0 - R + 0.5) / h) - 1), i1 = Math.min(N, Math.floor((x0 + R + 0.5) / h) + 2);
      const j0 = Math.max(0, Math.floor((y0 - R + 0.5) / h) - 1), j1 = Math.min(N, Math.floor((y0 + R + 0.5) / h) + 2);
      if (i1 <= i0 || j1 <= j0) throw new Error('inlet footprint outside the grid');
      const cph = Math.cos(phi), sph = Math.sin(phi);
      const K = w.tmp; K.fill(0); let Z = 0;
      for (let j = j0; j < j1; j++) for (let i = i0; i < i1; i++) {
        const k = j * N + i; if (!this.mask[k]) continue;
        let acc = 0;
        for (let a = 0; a < q; a++) for (let b = 0; b < q; b++) {
          const dx = this.x[k] + ((a + 0.5) / q - 0.5) * h - x0, dy = this.y[k] + ((b + 0.5) / q - 0.5) * h - y0;
          const u = (cph * dx + sph * dy) / r1, v = (-sph * dx + cph * dy) / r2;
          const g = 1 - u * u - v * v; if (g > 0) acc += g * g;
        }
        K[k] = acc; Z += acc;
      }
      Z *= this.area; if (Z <= 0) throw new Error('inlet footprint does not intersect the cup');
      const d = Math.max(inlet.d_jet, 1e-6);
      if (P.closure === 'twolayer') {
        const Fr2 = inlet.U_perp * inlet.U_perp / (P.g_red * d), chi = 1 / (1 + Math.pow(Fr2 / P.Fr_c2, P.p_dep)), S = inlet.S_eff;
        const tl = this.tl; tl.active = true; tl.S = S; tl.SE = P.ent_coef * Math.sqrt(Fr2) * S; tl.x0 = x0; tl.y0 = y0;
        tl.Rc = Math.min(P.cell_frac * 2.32 * Math.sqrt(Fr2) * d, 0.45); tl.tauNew = Math.max(2 * inlet.U_perp / P.g_red, 1e-3);
        tl.jx = P.kappa_t * inlet.u_in[0]; tl.jy = P.kappa_t * inlet.u_in[1];
        for (let k = 0; k < N * N; k++) {
          const Kk = K[k] / Z; w.Kn[k] = Kk; w.fsrc[k] = chi * P.phi_foam * S * Kk;
          w.Lam[k] = P.kappa_c * S * Kk; w.vsx[k] = tl.jx; w.vsy[k] = tl.jy;   // jet impact traction on the film
        }
        return chi;
      }
      const zc = inlet.U_perp * inlet.U_perp / (P.B_dep * d);
      const chi = 1 / (1 + Math.pow(zc, P.p_dep));
      if (P.tau_d > 0) this.tauCur = P.tau_d;
      const Rdep = Math.sqrt(Math.max(inlet.r1 * inlet.r2, 1e-12));
      let rad;
      if (P.return_law === 'v05') {
        let VR = inlet.U_perp * Math.pow(chi, 1 / (2 * P.p_dep)); VR = Math.min(VR, inlet.U_perp, Math.sqrt(P.B_dep * d)); rad = P.kappa_r * VR;
      } else rad = chi * P.kappa_r * inlet.U_perp;
      const ktx = P.kappa_t * inlet.u_in[0], kty = P.kappa_t * inlet.u_in[1];
      for (let k = 0; k < N * N; k++) {
        const Kk = K[k] / Z;
        w.s[k] = chi * inlet.S_eff * Kk; w.Lam[k] = P.kappa_c * (1 - chi) * inlet.S_eff * Kk;
        const dx = this.x[k] - x0, dy = this.y[k] - y0, den = Math.sqrt(Rdep * Rdep + dx * dx + dy * dy);
        w.vsx[k] = ktx + rad * dx / den; w.vsy[k] = kty + rad * dy / den;
      }
      return chi;
    }

    // ---------------------------------------------------------------- helpers
    div(Fx, Fy, out) {
      const N = this.N, ih = 1 / this.h;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        let v = 0; const k = j * N + i;
        if (i < N - 1) v += Fx[j * (N - 1) + i]; if (i > 0) v -= Fx[j * (N - 1) + i - 1];
        if (j < N - 1) v += Fy[k]; if (j > 0) v -= Fy[k - N];
        out[k] = v * ih;
      }
    }
    // out = a*phi - b*div(k grad phi)   (a scalar or array)
    applyDiffusion(phi, kx, ky, aArr, aScal, b, out) {
      const N = this.N, ih2 = 1 / (this.h * this.h), fx = this.fx, fy = this.fy;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i, p = phi[k]; let v = 0;
        if (i < N - 1 && fx[j * (N - 1) + i]) v += kx[j * (N - 1) + i] * (phi[k + 1] - p);
        if (i > 0 && fx[j * (N - 1) + i - 1]) v += kx[j * (N - 1) + i - 1] * (phi[k - 1] - p);
        if (j < N - 1 && fy[k]) v += ky[k] * (phi[k + N] - p);
        if (j > 0 && fy[k - N]) v += ky[k - N] * (phi[k - N] - p);
        out[k] = (aArr ? aArr[k] : aScal) * p - b * v * ih2;
      }
    }
    faceAvg(a, outx, outy) {
      const N = this.N;
      for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) outx[j * (N - 1) + i] = this.fx[j * (N - 1) + i] ? 0.5 * (a[j * N + i] + a[j * N + i + 1]) : 0;
      for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) outy[j * N + i] = this.fy[j * N + i] ? 0.5 * (a[j * N + i] + a[(j + 1) * N + i]) : 0;
    }
    static vl(r) { return (r + Math.abs(r)) / (1 + Math.abs(r)); }
    limitedFaces(c, Gx, Gy, outx, outy) {
      const N = this.N, fx = this.fx, fy = this.fy, eps = 1e-14;
      for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) {
        const f = j * (N - 1) + i; if (!fx[f]) { outx[f] = 0; continue; }
        const cL = c[j * N + i], cR = c[j * N + i + 1], d = cR - cL; let v;
        if (Math.abs(d) <= eps) v = Gx[f] >= 0 ? cL : cR;
        else if (Gx[f] >= 0) { const dl = (i > 0 && fx[f - 1]) ? cL - c[j * N + i - 1] : 0; v = cL + 0.5 * Latte2D.vl(dl / d) * d; }
        else { const dr = (i + 1 < N - 1 && fx[f + 1]) ? c[j * N + i + 2] - cR : 0; v = cR - 0.5 * Latte2D.vl(dr / d) * d; }
        outx[f] = Math.min(Math.max(v, Math.min(cL, cR)), Math.max(cL, cR));
      }
      for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) {
        const f = j * N + i; if (!fy[f]) { outy[f] = 0; continue; }
        const cL = c[f], cR = c[f + N], d = cR - cL; let v;
        if (Math.abs(d) <= eps) v = Gy[f] >= 0 ? cL : cR;
        else if (Gy[f] >= 0) { const dl = (j > 0 && fy[f - N]) ? cL - c[f - N] : 0; v = cL + 0.5 * Latte2D.vl(dl / d) * d; }
        else { const dr = (j + 1 < N - 1 && fy[f + N]) ? c[f + 2 * N] - cR : 0; v = cR - 0.5 * Latte2D.vl(dr / d) * d; }
        outy[f] = Math.min(Math.max(v, Math.min(cL, cR)), Math.max(cL, cR));
      }
    }
    pcgMass(kx, ky, dt, b, diag, x) { return this.pcg(kx, ky, null, 1, dt, b, diag, x, this.num.cg_tol); }
    // PCG for  (aArr|aScal)*phi - bmul*div(k grad phi) = b,  Jacobi preconditioner
    pcg(kx, ky, aArr, aScal, bmul, b, diag, x, tol) {
      const w = this.w, n = this.N * this.N, r = w.r, p = w.p, Ap = w.Ap, z = w.z;
      this.applyDiffusion(x, kx, ky, aArr, aScal, bmul, Ap);
      let rz = 0, bn = 0;
      for (let k = 0; k < n; k++) { r[k] = b[k] - Ap[k]; p[k] = r[k] / diag[k]; rz += r[k] * p[k]; bn += b[k] * b[k]; }
      bn = Math.sqrt(bn) + 1e-300; let it = 0;
      for (it = 1; it <= this.num.cg_maxiter; it++) {
        this.applyDiffusion(p, kx, ky, aArr, aScal, bmul, Ap);
        let pAp = 0; for (let k = 0; k < n; k++) pAp += p[k] * Ap[k];
        const alpha = rz / (pAp + 1e-300); let rr = 0, rzn = 0;
        for (let k = 0; k < n; k++) { x[k] += alpha * p[k]; r[k] -= alpha * Ap[k]; rr += r[k] * r[k]; z[k] = r[k] / diag[k]; rzn += r[k] * z[k]; }
        if (Math.sqrt(rr) <= tol * bn) break;
        const beta = rzn / rz; for (let k = 0; k < n; k++) p[k] = z[k] + beta * p[k]; rz = rzn;
      }
      return it;
    }
    // diag of  a*phi - dt*div(k grad phi)
    diagOf(aArr, kx, ky, dt, out) {
      const N = this.N, ih2 = 1 / (this.h * this.h);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i; let v = 0;
        if (i < N - 1) v += kx[j * (N - 1) + i]; if (i > 0) v += kx[j * (N - 1) + i - 1];
        if (j < N - 1) v += ky[k]; if (j > 0) v += ky[k - N];
        out[k] = aArr[k] + dt * ih2 * v;
      }
    }

    // first-order upwind transport of a with the coffee-layer velocity (ubx, uby): out = div(F)
    upwindDiv(a, out) {
      const N = this.N, w = this.w, fx = this.fx, fy = this.fy, ux = this.ubx, uy = this.uby;
      for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) { const f = j * (N - 1) + i, k = j * N + i;
        const u = fx[f] ? 0.5 * (ux[k] + ux[k + 1]) : 0; w.Fxqy[f] = u * (u > 0 ? a[k] : a[k + 1]); }
      for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) { const f = j * N + i;
        const v = fy[f] ? 0.5 * (uy[f] + uy[f + N]) : 0; w.Fyqy[f] = v * (v > 0 ? a[f] : a[f + N]); }
      this.div(w.Fxqy, w.Fyqy, out);
    }
    // coffee layer under the film (quasi-steady potential flow): div U_b = S K + S_E (K - K_cell) - <.>, U_b = grad phi_b
    bulkFlow() {
      const N = this.N, n = N * N, h = this.h, w = this.w, tl = this.tl, mask = this.mask;
      if (!tl.active || !(tl.S > 0)) { this.hasUb = false; this.ubKey = null; return; }
      // quasi-steady: keep U_b while the source has not moved by half a cell and its strengths changed < bulk_rtol
      const key = this.ubKey, rt = this.P.bulk_rtol;
      if (this.hasUb && key && rt > 0 && Math.hypot(tl.x0 - key.x0, tl.y0 - key.y0) < 0.5 * h && Math.abs(tl.S - key.S) <= rt * key.S
          && Math.abs(tl.SE - key.SE) <= rt * Math.max(key.SE, 1e-12) && Math.abs(tl.Rc - key.Rc) < 0.5 * h) return;
      this.ubKey = { x0: tl.x0, y0: tl.y0, S: tl.S, SE: tl.SE, Rc: tl.Rc };
      const R = Math.max(tl.Rc, 2 * h); let nc = 0;
      for (let k = 0; k < n; k++) if (mask[k] && Math.hypot(this.x[k] - tl.x0, this.y[k] - tl.y0) < R) nc++;
      const kc = 1 / Math.max(nc * this.area, 1e-12); let mean = 0, nm = 0;
      for (let k = 0; k < n; k++) {
        const cell = mask[k] && Math.hypot(this.x[k] - tl.x0, this.y[k] - tl.y0) < R ? kc : 0;
        w.brhs[k] = tl.S * w.Kn[k] + tl.SE * (w.Kn[k] - cell); if (mask[k]) { mean += w.brhs[k]; nm++; }
      }
      mean /= Math.max(nm, 1);
      const eps = 1e-6, ih2 = 1 / (h * h), fxf = this.fxf, fyf = this.fyf;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i; w.brhs[k] = mask[k] ? -(w.brhs[k] - mean) : 0; let v = 0;
        if (i < N - 1) v += fxf[j * (N - 1) + i]; if (i > 0) v += fxf[j * (N - 1) + i - 1];
        if (j < N - 1) v += fyf[k]; if (j > 0) v += fyf[k - N];
        w.bdiag[k] = eps + v * ih2;
      }
      this.stats.cg_bulk = (this.stats.cg_bulk || 0) + this.pcg(fxf, fyf, null, eps, 1, w.brhs, w.bdiag, this.phib, this.P.bulk_tol);
      const ih = 1 / h, fx = this.fx, fy = this.fy, p = this.phib;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i; let gx = 0, gy = 0;
        if (i < N - 1 && fx[j * (N - 1) + i]) gx += (p[k + 1] - p[k]) * ih; if (i > 0 && fx[j * (N - 1) + i - 1]) gx += (p[k] - p[k - 1]) * ih;
        if (j < N - 1 && fy[k]) gy += (p[k + N] - p[k]) * ih; if (j > 0 && fy[k - N]) gy += (p[k] - p[k - N]) * ih;
        this.ubx[k] = mask[k] ? gx / Math.max(this.nfx[k], 1) : 0; this.uby[k] = mask[k] ? gy / Math.max(this.nfy[k], 1) : 0;
      }
      this.hasUb = true;
    }

    // ---------------------------------------------------------------- one sub-step
    substep(dt) {
      const N = this.N, n = N * N, h = this.h, P = this.P, w = this.w, mask = this.mask, fx = this.fx, fy = this.fy;
      const l = this.l, qx = this.qx, qy = this.qy, m = this.m, s = w.s, Lam = w.Lam, vsx = w.vsx, vsy = w.vsy;
      for (let k = 0; k < n; k++) { const ls = mask[k] ? l[k] : 1; w.lsafe[k] = ls; w.ux[k] = mask[k] ? qx[k] / ls : 0; w.uy[k] = mask[k] ? qy[k] / ls : 0; w.c[k] = mask[k] ? m[k] / ls : 0; }
      // delay closure: the deposited milk enters a sub-surface reservoir (impact momentum acts now), surfaces after tau,
      // is carried by a fraction of the surface velocity and pushes the layer through the pressure as a submerged mound
      const twol = P.closure === 'twolayer';
      let sm = s; const smom = s; const delay = !twol && P.tau_d > 0 && this.tauCur !== null; const mn = w.msubn;
      if (delay) {
        const tau = this.tauCur, msub = this.msub, ss = w.ssurf;
        for (let k = 0; k < n; k++) { const tot = msub[k] + dt * s[k]; mn[k] = tot / (1 + dt / tau); ss[k] = (tot - mn[k]) / dt; }
        if (P.sub_advect > 0) {
          const a = P.sub_advect;
          for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) { const f = j * (N - 1) + i, k = j * N + i;
            const u = fx[f] ? a * 0.5 * (w.ux[k] + w.ux[k + 1]) : 0; w.Fxqx[f] = u * (u > 0 ? mn[k] : mn[k + 1]); }
          for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) { const f = j * N + i;
            const v = fy[f] ? a * 0.5 * (w.uy[f] + w.uy[f + N]) : 0; w.Fyqx[f] = v * (v > 0 ? mn[f] : mn[f + N]); }
          this.div(w.Fxqx, w.Fyqx, w.tmp);
          for (let k = 0; k < n; k++) mn[k] = mask[k] ? Math.max(mn[k] - dt * w.tmp[k], 0) : 0;
        }
        sm = ss;
      }
      let tvx = vsx, tvy = vsy;
      if (twol) {
        // submerged foam moves with the coffee layer (upwind), surfaces after its rise time and feeds the film
        const f1 = w.f1, t1 = w.t1, tl = this.tl, ss = w.ssurf;
        f1.set(this.f); t1.set(this.ft);
        if (this.hasUb) {
          this.upwindDiv(this.f, w.tmp); for (let k = 0; k < n; k++) f1[k] -= dt * w.tmp[k];
          this.upwindDiv(this.ft, w.tmp); for (let k = 0; k < n; k++) t1[k] -= dt * w.tmp[k];
        }
        for (let k = 0; k < n; k++) {
          if (tl.active) { f1[k] += dt * w.fsrc[k]; t1[k] += dt * w.fsrc[k] * tl.tauNew; }
          f1[k] = mask[k] ? Math.max(f1[k], 0) : 0; t1[k] = mask[k] ? Math.max(t1[k], 0) : 0;
          const tau = Math.max(f1[k] > 1e-14 ? t1[k] / Math.max(f1[k], 1e-300) : 1, 1e-3);
          const surf = f1[k] * (1 - Math.exp(-dt / tau));
          f1[k] -= surf; t1[k] = tau * f1[k]; ss[k] = surf / dt;
          // explicit momentum: surfacing foam arrives with the coffee velocity, traction pulls towards the jet velocity
          const ub = this.hasUb ? this.ubx[k] : 0, vb = this.hasUb ? this.uby[k] : 0, L = Lam[k], jx = tl.active ? tl.jx : 0, jy = tl.active ? tl.jy : 0;
          const wsum = ss[k] + L; w.tvx[k] = wsum > 0 ? (ss[k] * ub + L * jx) / wsum : 0; w.tvy[k] = wsum > 0 ? (ss[k] * vb + L * jy) / wsum : 0;
        }
        sm = ss; tvx = w.tvx; tvy = w.tvy;
      }
      // A: Rusanov momentum advection
      for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) {
        const f = j * (N - 1) + i, k = j * N + i;
        if (!fx[f]) { w.Fxqx[f] = 0; w.Fxqy[f] = 0; continue; }
        const uL = w.ux[k], uR = w.ux[k + 1], a = 2 * Math.max(Math.abs(uL), Math.abs(uR));
        w.Fxqx[f] = 0.5 * (qx[k] * uL + qx[k + 1] * uR) - 0.5 * a * (qx[k + 1] - qx[k]);
        w.Fxqy[f] = 0.5 * (qy[k] * uL + qy[k + 1] * uR) - 0.5 * a * (qy[k + 1] - qy[k]);
      }
      for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) {
        const f = j * N + i;
        if (!fy[f]) { w.Fyqx[f] = 0; w.Fyqy[f] = 0; continue; }
        const vL = w.uy[f], vR = w.uy[f + N], a = 2 * Math.max(Math.abs(vL), Math.abs(vR));
        w.Fyqx[f] = 0.5 * (qx[f] * vL + qx[f + N] * vR) - 0.5 * a * (qx[f + N] - qx[f]);
        w.Fyqy[f] = 0.5 * (qy[f] * vL + qy[f + N] * vR) - 0.5 * a * (qy[f + N] - qy[f]);
      }
      this.div(w.Fxqx, w.Fyqx, w.qxs); this.div(w.Fxqy, w.Fyqy, w.qys);
      for (let k = 0; k < n; k++) { w.qxs[k] = qx[k] - dt * w.qxs[k]; w.qys[k] = qy[k] - dt * w.qys[k]; }
      if (twol && this.hasUb) {   // drag towards the moving coffee layer: -beta (q - l U_b); the -beta q part is implicit in B
        for (let k = 0; k < n; k++) { w.qxs[k] += dt * P.beta * w.lsafe[k] * this.ubx[k]; w.qys[k] += dt * P.beta * w.lsafe[k] * this.uby[k]; }
      }
      // B: implicit mass + pressure + drag/traction
      const cp2 = P.cp * P.cp;
      for (let k = 0; k < n; k++) {
        const ls = w.lsafe[k], theta = 1 + dt * (P.beta + Lam[k] / ls), f = dt * (smom[k] + Lam[k]);
        w.ax[k] = (w.qxs[k] + f * tvx[k]) / theta; w.ay[k] = (w.qys[k] + f * tvy[k]) / theta; w.coef[k] = cp2 * ls / theta;
      }
      if (delay && P.sub_pressure > 0) {   // submerged mound pushes the layer: a -= dt (cp^2 l/theta) w grad(m_sub)
        const ihh = 1 / h, wsp = P.sub_pressure;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const k = j * N + i; let gx = 0, gy = 0;
          if (i < N - 1 && fx[j * (N - 1) + i]) gx += (mn[k + 1] - mn[k]) * ihh; if (i > 0 && fx[j * (N - 1) + i - 1]) gx += (mn[k] - mn[k - 1]) * ihh;
          if (j < N - 1 && fy[k]) gy += (mn[k + N] - mn[k]) * ihh; if (j > 0 && fy[k - N]) gy += (mn[k] - mn[k - N]) * ihh;
          gx = this.nfx[k] > 0 ? gx / this.nfx[k] : 0; gy = this.nfy[k] > 0 ? gy / this.nfy[k] : 0;
          w.ax[k] -= dt * w.coef[k] * wsp * gx; w.ay[k] -= dt * w.coef[k] * wsp * gy; }
      }
      for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) {
        const f = j * (N - 1) + i, k = j * N + i;
        if (!fx[f]) { w.Gex[f] = 0; w.kx[f] = 0; continue; }
        const ww = Math.max(Math.abs(w.ax[k] / w.lsafe[k]), Math.abs(w.ax[k + 1] / w.lsafe[k + 1]));
        w.Gex[f] = 0.5 * (w.ax[k] + w.ax[k + 1]) - 0.5 * ww * (l[k + 1] - l[k]);
        w.kx[f] = dt * 0.5 * (w.coef[k] + w.coef[k + 1]);
      }
      for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) {
        const f = j * N + i;
        if (!fy[f]) { w.Gey[f] = 0; w.ky[f] = 0; continue; }
        const ww = Math.max(Math.abs(w.ay[f] / w.lsafe[f]), Math.abs(w.ay[f + N] / w.lsafe[f + N]));
        w.Gey[f] = 0.5 * (w.ay[f] + w.ay[f + N]) - 0.5 * ww * (l[f + N] - l[f]);
        w.ky[f] = dt * 0.5 * (w.coef[f] + w.coef[f + N]);
      }
      const ih2 = 1 / (h * h);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i; let v = 0;
        if (i < N - 1) v += w.kx[j * (N - 1) + i]; if (i > 0) v += w.kx[j * (N - 1) + i - 1];
        if (j < N - 1) v += w.ky[k]; if (j > 0) v += w.ky[k - N];
        w.diag[k] = 1 + dt * ih2 * v;
      }
      this.div(w.Gex, w.Gey, w.rhs);
      for (let k = 0; k < n; k++) { w.rhs[k] = l[k] + dt * sm[k] - dt * w.rhs[k]; w.lnew[k] = l[k]; }
      this.stats.cg_mass += this.pcgMass(w.kx, w.ky, dt, w.rhs, w.diag, w.lnew);
      const ih = 1 / h;
      for (let j = 0; j < N; j++) for (let i = 0; i < N - 1; i++) { const f = j * (N - 1) + i, k = j * N + i; w.Gx[f] = fx[f] ? w.Gex[f] - w.kx[f] * (w.lnew[k + 1] - w.lnew[k]) * ih : 0; }
      for (let j = 0; j < N - 1; j++) for (let i = 0; i < N; i++) { const f = j * N + i; w.Gy[f] = fy[f] ? w.Gey[f] - w.ky[f] * (w.lnew[f + N] - w.lnew[f]) * ih : 0; }
      this.div(w.Gx, w.Gy, w.tmp);
      for (let k = 0; k < n; k++) w.lnew[k] = l[k] + dt * sm[k] - dt * w.tmp[k];
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = j * N + i; let gx = 0, gy = 0;
        if (i < N - 1 && fx[j * (N - 1) + i]) gx += (w.lnew[k + 1] - w.lnew[k]) * ih;
        if (i > 0 && fx[j * (N - 1) + i - 1]) gx += (w.lnew[k] - w.lnew[k - 1]) * ih;
        if (j < N - 1 && fy[k]) gy += (w.lnew[k + N] - w.lnew[k]) * ih;
        if (j > 0 && fy[k - N]) gy += (w.lnew[k] - w.lnew[k - N]) * ih;
        gx = this.nfx[k] > 0 ? gx / this.nfx[k] : 0; gy = this.nfy[k] > 0 ? gy / this.nfy[k] : 0;
        w.qxn[k] = w.ax[k] - dt * w.coef[k] * gx; w.qyn[k] = w.ay[k] - dt * w.coef[k] * gy;
        if (delay) { w.qxn[k] += dt * sm[k] * w.ux[k]; w.qyn[k] += dt * sm[k] * w.uy[k]; }   // surfacing milk moves with the surface
      }
      // C: milk with the same flux
      this.limitedFaces(w.c, w.Gx, w.Gy, w.cfx, w.cfy);
      for (let f = 0; f < w.cfx.length; f++) { w.cfx[f] *= w.Gx[f]; w.cfy[f] *= w.Gy[f]; }
      this.div(w.cfx, w.cfy, w.mnew);
      let clip = 0;
      for (let k = 0; k < n; k++) {
        let mv = m[k] + dt * sm[k] - dt * w.mnew[k];
        if (mask[k]) { clip += Math.max(mv - w.lnew[k], 0) + Math.max(-mv, 0); mv = Math.min(Math.max(mv, 0), w.lnew[k]); }
        else { mv = 0; w.lnew[k] = 1; w.qxn[k] = 0; w.qyn[k] = 0; }
        w.mnew[k] = mv;
      }
      this.stats.clip_mass += clip * this.area;
      // D: explicit viscosity and mixing (l frozen at l^{n+1}); the implicit branch is not ported (nu dt/h^2 is small here)
      this.faceAvg(w.lnew, w.lfx, w.lfy);
      if (P.nu > 0) {
        for (let f = 0; f < w.kx.length; f++) { w.kx[f] = P.nu * w.lfx[f]; w.ky[f] = P.nu * w.lfy[f]; }
        for (let k = 0; k < n; k++) { w.ux[k] = w.qxn[k] / w.lnew[k]; w.uy[k] = w.qyn[k] / w.lnew[k]; }
        if (P.nu * dt / (h * h) < this.num.explicit_visc_limit) {
          this.applyDiffusion(w.ux, w.kx, w.ky, null, 0, -1, w.tmp); for (let k = 0; k < n; k++) if (mask[k]) w.qxn[k] += dt * w.tmp[k];
          this.applyDiffusion(w.uy, w.kx, w.ky, null, 0, -1, w.tmp); for (let k = 0; k < n; k++) if (mask[k]) w.qyn[k] += dt * w.tmp[k];
        } else {
          // implicit: l (u+ - u-)/dt = div(nu l grad u+), l frozen at l^{n+1}; one PCG per component
          this.diagOf(w.lnew, w.kx, w.ky, dt, w.diag);
          for (let k = 0; k < n; k++) w.rhs[k] = w.lnew[k] * w.ux[k];
          this.stats.cg_visc = (this.stats.cg_visc || 0) + this.pcg(w.kx, w.ky, w.lnew, 0, dt, w.rhs, w.diag, w.ux, this.num.visc_tol);
          for (let k = 0; k < n; k++) w.rhs[k] = w.lnew[k] * w.uy[k];
          this.stats.cg_visc += this.pcg(w.kx, w.ky, w.lnew, 0, dt, w.rhs, w.diag, w.uy, this.num.visc_tol);
          for (let k = 0; k < n; k++) { w.qxn[k] = mask[k] ? w.lnew[k] * w.ux[k] : 0; w.qyn[k] = mask[k] ? w.lnew[k] * w.uy[k] : 0; }
        }
      }
      if (P.D > 0) {
        for (let f = 0; f < w.kx.length; f++) { w.kx[f] = P.D * w.lfx[f]; w.ky[f] = P.D * w.lfy[f]; }
        for (let k = 0; k < n; k++) w.c[k] = mask[k] ? w.mnew[k] / w.lnew[k] : 0;
        if (P.D * dt / (h * h) < this.num.explicit_mixing_limit) {
          this.applyDiffusion(w.c, w.kx, w.ky, null, 0, -1, w.tmp);
          for (let k = 0; k < n; k++) w.mnew[k] = mask[k] ? Math.min(Math.max(w.mnew[k] + dt * w.tmp[k], 0), w.lnew[k]) : 0;
        } else {
          this.diagOf(w.lnew, w.kx, w.ky, dt, w.diag);
          for (let k = 0; k < n; k++) w.rhs[k] = w.lnew[k] * w.c[k];
          this.pcg(w.kx, w.ky, w.lnew, 0, dt, w.rhs, w.diag, w.c, this.num.cg_tol);
          for (let k = 0; k < n; k++) w.mnew[k] = mask[k] ? w.lnew[k] * Math.min(Math.max(w.c[k], 0), 1) : 0;
        }
      }
      // accept
      let ok = true, lmin = Infinity, ssum = 0;
      for (let k = 0; k < n; k++) {
        if (!isFinite(w.lnew[k]) || !isFinite(w.qxn[k]) || !isFinite(w.qyn[k]) || !isFinite(w.mnew[k])) { ok = false; break; }
        if (mask[k]) { if (w.lnew[k] < lmin) lmin = w.lnew[k]; ssum += sm[k]; }
      }
      if (!ok || lmin <= this.num.l_floor) return false;
      this.l.set(w.lnew); this.qx.set(w.qxn); this.qy.set(w.qyn); this.m.set(w.mnew); if (delay) this.msub.set(mn);
      if (twol) { this.f.set(w.f1); this.ft.set(w.t1); }
      this.deposited += ssum * this.area * dt; this.stats.steps++;
      return true;
    }

    // ---------------------------------------------------------------- one control frame
    advanceFrame(inlet, frameDt) {
      frameDt = frameDt || this.num.frame_dt;
      if (inlet && inlet.active && inlet.S_eff > 0 && inlet.scan_speed > 0) {
        const lim = this.num.scan_safety * Math.min(this.h, inlet.r1, inlet.r2) / inlet.scan_speed;
        frameDt = Math.min(frameDt, Math.max(lim, 1e-4));
      }
      const chi = this.inletFields(inlet);
      if (this.P.closure === 'twolayer') this.bulkFlow();
      const n = this.N * this.N, w = this.w; let umax = 0;
      for (let k = 0; k < n; k++) {
        const ls = this.mask[k] ? this.l[k] : 1; const u = Math.hypot(this.qx[k], this.qy[k]) / ls; if (u > umax) umax = u;
        if (w.s[k] + w.Lam[k] > 0) { const v = Math.hypot(w.vsx[k], w.vsy[k]); if (v > umax) umax = v; }
        if (this.hasUb) { const v = Math.hypot(this.ubx[k], this.uby[k]); if (v > umax) umax = v; }
      }
      this.stats.max_u = Math.max(this.stats.max_u, umax);
      let nsub = Math.ceil(frameDt / (this.num.cfl * this.h / (2 * Math.max(umax, 1e-9))));
      nsub = Math.max(1, Math.min(nsub, this.num.max_substeps));
      w.sl.set(this.l); w.sqx.set(this.qx); w.sqy.set(this.qy); w.sm.set(this.m); w.smsub.set(this.msub); w.sf.set(this.f); w.sft.set(this.ft); const dep0 = this.deposited;
      let done = false;
      for (let attempt = 0; attempt < 8 && !done; attempt++) {
        const dt = frameDt / nsub; let ok = true;
        for (let k = 0; k < nsub; k++) if (!this.substep(dt)) { ok = false; break; }
        if (ok) done = true;
        else { this.l.set(w.sl); this.qx.set(w.sqx); this.qy.set(w.sqy); this.m.set(w.sm); this.msub.set(w.smsub); this.f.set(w.sf); this.ft.set(w.sft); this.deposited = dep0; this.stats.retries++; nsub *= 2; }
      }
      if (!done) throw new Error('frame failed after retries');
      this.stats.max_substeps = Math.max(this.stats.max_substeps, nsub);
      this.t += frameDt; this.stats.frames++;
      return { chi, nsub, frameDt };
    }

    concentration(out) {
      const n = this.N * this.N; out = out || new Float32Array(n);
      for (let k = 0; k < n; k++) out[k] = this.mask[k] ? this.m[k] / this.l[k] : NaN;
      return out;
    }
    reset() { this.l.fill(1); this.qx.fill(0); this.qy.fill(0); this.m.fill(0); this.msub.fill(0); this.tauCur = null; this.t = 0; this.deposited = 0;
      this.f.fill(0); this.ft.fill(0); this.phib.fill(0); this.ubx.fill(0); this.uby.fill(0); this.hasUb = false; this.ubKey = null; this.tl = { active: false };
      this.stats = { steps: 0, frames: 0, cg_mass: 0, max_u: 0, retries: 0, max_substeps: 0, clip_mass: 0 }; }
  }

  /* SI impact record -> inlet (cup units), port of pitcher.Coupling */
  const COUPLING = { D_L: 0.08, c_S: 0.05 / 15e-6, c_U: 0.263 / 0.70, c_u: 0.5 / 0.15, kappa_Q: 125.0, max_jet_angle: Math.PI / 3, footprint: 'physical' };
  function coupleInlet(rec, scanSpeed, cp) {
    cp = Object.assign({}, COUPLING, cp || {});
    // what arrives on the coffee this frame (flight delay) when the record carries it, else what leaves the spout
    const J = rec.hit !== undefined ? rec.hit : rec.jet; const Q = rec.hit !== undefined ? (J ? J.Q : 0) : rec.Q;
    if (!J || Q <= 0) return { active: false };
    const x = J.x_hit[0] / cp.D_L, y = J.x_hit[1] / cp.D_L, S = cp.c_S * Q, U = cp.c_U * J.U_perp;
    let ux = cp.c_u * J.v0[0], uy = cp.c_u * J.v0[1]; let vt = Math.hypot(ux, uy);
    if (vt > 0) { const sc = Math.min(1, U * Math.tan(cp.max_jet_angle) / vt); ux *= sc; uy *= sc; vt *= sc; }
    const Qrec = S / cp.kappa_Q, speed = Math.hypot(U, vt);
    let r1, r2, phi, d;
    if (cp.footprint !== 'v05' && J.r1 !== undefined) { r1 = J.r1 / cp.D_L; r2 = J.r2 / cp.D_L; phi = J.phi; d = 2 * Math.sqrt(r1 * r2); }
    else { const aspect = speed / U, r = Math.sqrt(Qrec / (Math.PI * U)); r1 = r * Math.sqrt(aspect); r2 = r / Math.sqrt(aspect); phi = Math.atan2(uy, ux); d = 2 * Math.sqrt(Qrec / speed / Math.PI); }
    return { active: true, x_hit: [x, y], S_eff: S, u_in: [ux, uy], U_perp: U, d_jet: d, r1, r2, phi, scan_speed: scanSpeed || 0 };
  }

  const api = { Latte2D, coupleInlet, DEFAULT_PARAMS, DEFAULT_NUM, COUPLING };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else global.Latte2D = api;
})(typeof window !== 'undefined' ? window : globalThis);
