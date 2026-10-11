import sys, json, numpy as np
R = "/tmp/claude-0/-home-user/699ad593-ca9a-58b9-a0ac-50a2002cb19c/scratchpad/review/"
def load(tag):
    return np.load(R + f"skeptic_A16_heart_{tag}.npz")
def l1(a, b, mask):
    return float(np.mean(np.abs(a[mask] - b[mask])))
base = load("cfl0.5_f120"); mask = base["mask"]
g = (np.arange(96) + 0.5) / 96 - 0.5
X = g[None, :].repeat(96, 0); Y = g[:, None].repeat(96, 1)
roi = mask & (np.abs(X) < 0.08) & (Y > -0.46) & (Y < 0.34)   # cut path of the heart script
pairs = [("cfl0.5_f120", "cfl0.25_f120"), ("cfl0.25_f120", "cfl0.125_f120"), ("cfl0.5_f120", "cfl0.5_f240"), ("cfl0.5_f240", "cfl0.5_f480")]
for a, b in pairs:
    try:
        A, B = load(a), load(b)
    except FileNotFoundError:
        print("missing", a, b); continue
    ca, cb = A["c"], B["c"]
    print(f"{a} vs {b}: L1(cup) {l1(ca, cb, mask):.3e}  L1(cut ROI) {l1(ca, cb, roi):.3e}  max|dc| {np.nanmax(np.abs(ca - cb)[mask]):.3e}"
          f"  L1(l) {l1(A['l'], B['l'], mask):.3e}  white>0.4 {np.nansum(ca[mask] > 0.4)} vs {np.nansum(cb[mask] > 0.4)}")
for tag in ["cfl0.5_f120", "cfl0.25_f120", "cfl0.125_f120", "cfl0.5_f240", "cfl0.5_f480"]:
    try:
        A = load(tag)
    except FileNotFoundError:
        continue
    ru = A["ru"]; act = A["active"]; t = A["t"]
    i = int(np.argmax(ru))
    print(f"{tag}: frames {len(ru)} retries {int(A['retries'].sum())} ratio(u_sub_max/umax0) max {ru.max():.2f} at t={t[i]:.2f} "
          f"(umax0 {A['umax0'][i]:.3f} -> u_sub {A['u_sub'][i]:.3f}), p99 {np.percentile(ru, 99):.2f}, "
          f"#>1 {(ru > 1).sum()}, #>1.41 {(ru > 1.414).sum()}, #>2 {(ru > 2).sum()}, nsub max {A['nsub'].max()} mean {A['nsub'].mean():.1f}, "
          f"fdt min {A['fdt'].min():.4f}")
    # in which phases do the >1 frames occur
    bad = np.where(ru > 1.0)[0]
    if len(bad):
        print("   t of ratio>1 frames: ", np.round(t[bad][:15], 2), "... total", len(bad), "active:", int(act[bad].sum()))
