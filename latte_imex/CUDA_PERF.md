# CUDA back-end: what was changed for throughput, and what to expect

Scope: `cuda_solver.py` (surface-layer solver), `cuda_pitcher.py` (forward pitcher). The CPU solver (`solver.py`,
`kernels.py`) and `pitcher.ForwardPitcher` are the references and are unchanged. Numbers marked *measured* come from
the CUDA simulator (counts are exact there; times are not); timings on a real GPU are estimates until
`python -m latte_imex.bench_cuda --quick` has been run on the AutoDL machine (`scripts/autodl_setup.sh` does it).

## 1. Launches and host synchronisations per sub-step

`i_m`, `i_v` = PCG iterations of the mass (step B) and viscous (step D, `visc="symmetric"`) solves.

| | before | now |
|---|---|---|
| kernel launches | 19 + 5 i_m + 6 i_v | 5 + 2 (i_m + i_v) |
| small host->device copies (zeroing reduction slots) | 6 + 2 (i_m + i_v) | 0 |
| device->host reads (each one a full sync) | 6 + 2 (i_m + i_v) | ~2 (one poll per PCG solve when the iteration count is predicted right) |
| per control frame, extra | inlet: 2 launches + 1 read, CFL: 2 launches + 1 read, 4 frame-save copies | 2 launches (+1 restore per retry), 1 upload of the inlet scalars, 1 read (Z, CFL maxima), 1 read (check flags) |

Example (typical late-pour sub-step, i_m = 12, i_v = 8): 127 launches + 40 copies + 46 syncs before, 45 launches +
~2 syncs now.  *Measured* (simulator, heart, N = 20, 1 sub-step per frame, i_m + i_v = 7): 23.0 launches per sub-step
including the frame kernels; the host polls 1-2 times per solve after the first frames (the first solve has no
prediction and polls every iteration).

## 2. What was done

1. **Fusion.** Momentum (step A) + step-B prepare in one kernel; explicit layer flux + rhs + Jacobi diagonal + x0 +
   initial residual + the two initial dot products in one kernel; the concentration c = m/l is formed on the fly in
   the final-flux/milk kernel (no c array pass); viscous weights + X0 + b + diagonal + initial residual (corner shear
   of X0 on the fly) in one kernel; q = l X, explicit mixing and the finite / positive-layer check in one kernel.
2. **PCG with device-resident scalars, 2 launches per iteration.** k1: p = r/diag + beta p is computed on the fly at the
   stencil points (double-buffered p, no separate update kernel), then A p and the block-reduced p.Ap; k2: x, r update
   and r.r, r.z.  All scalars live in a per-member slot array [B, it, 3]; k1 evaluates the CPU stopping rule
   `sqrt(r.r) <= tol |b|` on the device and turns the remaining launches into no-ops, so the iteration count and the
   iterates are those of `solver.pcg` (checked: same `cg_mass` / `cg_visc` counts as the CPU).  The host polls a
   [B, 2] status word only once the previous solve's iteration count is reached (`poll_every`, default 1).
   Memory passes per mass-PCG iteration: 20 array passes before, 13 now.
3. **Retry guard once per frame.** The check flags are accumulated on the device and read at the end of the frame; a
   failure restarts the frame with 2x the sub-steps, which is also what the CPU solver does (accepted states equal).
4. **Inlet on the device.** Per frame only the inlet scalars are uploaded (24 doubles per member); the footprint
   kernel is evaluated on its unclipped bounding box only, and the in-cup / total mass, i.e. the off-cup fraction Z of
   `solver.kernel_fraction`, is reduced on the device (no host `kernel_fraction` call); s, Lam, v*, the CFL maxima and
   the frame save are one kernel.
5. **Launch overhead.** A launcher calls the compiled kernel object directly after the first call (skips the
   dispatcher's per-call argument typing; `LATTE_CUDA_FASTLAUNCH=0` turns it off; any error falls back).
6. **Batched mode.** `BatchCudaSolver`: state [B, N, N], grid z = member; each member has its own inlet, frame length,
   sub-step count, retries, ledger and stats (members idle in a round are copied through, so a member reproduces its
   B = 1 run: *measured* max difference 0 in the simulator, B = 3 with one member idle for a frame).  At N = 256 one
   member is only 256 blocks; B = 8-32 fills an RTX 4090 / A100 and amortises the per-launch host cost over B runs.
   `batch_run.run_batch` is `run.run` for many programs (same outputs).
7. **Snapshots** go through one pinned host buffer (`cuda.pinned_array`); one concentration launch serves all members.
8. **FP32 storage behind a flag** (`dtype=np.float32`; arithmetic and reductions FP64, PCG tolerances raised to
   1e-6).  *Measured* 1e-4 relative difference to FP64 after 2 frames; experimental, not used by the pipelines.

Evaluated and not done:
- **CUDA graphs**: numba-cuda has no supported stream-capture API; with the device-side PCG control the launch sequence
  of a sub-step is fixed except for its length, so it could be captured once such an API is available.
- **Shared-memory tiles for the stencils**: each value is used by at most 5 (9 for the corner shear) threads of a
  block and is served by L1/L2; the kernels are bandwidth/launch bound, a tiled version would add halo logic for a
  small gain.  Layout: [B, N, N], 32 x 8 blocks with i fastest, i.e. 256-byte coalesced rows per warp.
- **Further warm starts / fewer iterations** (extrapolated initial guess, looser tolerance): would change the
  iteration counts and the results beyond rounding; the CPU solver's x0 = l^n and X0 = u^- are kept.

## 3. Expected effect (estimates; replace with `bench_cuda --quick` numbers)

- Single run, N = 768: before ~0.5 ms per sub-step, launch/sync bound.  Now ~45 launches and ~2 syncs per sub-step:
  ~0.15-0.25 ms per sub-step if the launch cost is ~3-5 us; a 10 000-step pattern ~2-3 s (FP64, RTX 4090).
- Single run, N = 256: still launch bound (~10-20 us of host time per launch); the batched mode is the lever there:
  B = 32 members cost roughly 2-4x one member's wall time, i.e. ~8-15x more member-sub-steps per second.
- Robustness grid (60 runs at 256^2): one batch; expected 10-25 min on one GPU vs ~60 CPU-hours-equivalent of the
  serial CPU runs.

## 4. GPU forward pitcher (`cuda_pitcher.py`)

One block (128 threads) per pose sequence walks the sequence (the pitcher volume couples the steps).  Per step: three
passes over the 47 088 cavity points (range, 256-bin histogram of the volumes with shared-memory atomics, gather of the
bin that holds V), ranking inside that bin in shared memory (same weighted quantile and interpolation as
`FastPitcher.free_surface`), front-edge weir with projected widths and flux-weighted exit quantities (incl. omega x r),
side overflow, volume update; ~25 block reductions per step.  Pose rates (np.gradient of the tip and of R, omega from
dR R^T) are a separate per-step kernel.  Ballistic flight and arrival binning stay on the host (the CPU code).  A
histogram bin with more than 768 points (none on the S1 poses: max 372) flags the sequence, which is then recomputed
on the CPU.  Expected: ~30-60 us per step per block, all sequences in parallel: a 10 000-step sequence in < 1 s, the
whole robustness calibration (30 sequences x ~30 calls) in 1-3 min (the CPU needs 10-16 s per sequence and call).

## 5. Only verifiable on the real GPU

Timings; the fast-launch path (falls back by itself on error; the CPU-vs-GPU self-test runs with it on);
float64 atomics (global max/add, shared-memory add) and the shared-memory budget of the pitcher kernel (~35 KB);
FMA contraction makes the GPU results differ from the CPU in the last bits (the tests compare with tolerances).
