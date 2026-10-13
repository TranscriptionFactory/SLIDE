"""Summarize floor_sweep.py: R tie rate vs pair distance, and per-draw W zero/sign agreement and Z20/Z21 selection
for the current Rust fitter (tol 1e-4, 1e-7), candidate tie rules applied to it, and the entry-order fitter.
usage: sweep_summary.py <scratch_dir> <out_dir>   (env: PYTHONPATH=src)"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from loveslide.knockoff.filter import knockoff_threshold

SCR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
d = pd.read_csv(SCR / "floor_sweep.csv")
d["delta_bin"] = pd.cut(d.delta, [0, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2, 1])
nz = d[d.near_zero & (d.r_entry > 0)]  # near-duplicate pairs R entered; W = 0 there is a tie
tie = lambda v: (v == 0).mean()
curve = nz.groupby("delta_bin", observed=True).agg(n=("w_r", "size"), r_tie=("w_r", tie), rust_1e4_tie=("w_rust", tie),
                                                   rust_1e7_tie=("w_rust7", tie), entry_order_tie=("w_entry", tie))
curve.round(3).to_csv(OUT / "sweep_tie_curve.csv")

rules = {"R": d.w_r, "rust_1e-4": d.w_rust, "rust_1e-7": d.w_rust7, "entry_order_1e-7": d.w_entry}
for tau in (3e-4, 1e-3, 2e-3):
    rules[f"rust_1e-4+delta<{tau:g}"] = d.w_rust.where(d.delta >= tau, 0.0)
    rules[f"rust_1e-7+delta<{tau:g}"] = d.w_rust7.where(d.delta >= tau, 0.0)
for k in (1, 3):
    rules[f"rust_1e-4+gap<={k}"] = d.w_rust.where(~(d.rust_gap_steps <= k), 0.0)
w = pd.DataFrame(rules)
key = d[["floor", "draw"]]
# Selection: W_j >= knockoff threshold of that draw's full 22-vector (FDR 0.1, offset 0).
thr = w.groupby([key.floor, key.draw]).transform(lambda v: knockoff_threshold(v.to_numpy(), fdr=0.1, offset=0))
sel = w >= thr
agree = np.sign(w).eq(np.sign(d.w_r), axis=0)
m = d.near_zero & (d.r_entry > 0)
focus = d.factor.isin(["Z20", "Z21"])
rows = []
for f, g in d.groupby("floor"):
    i = g.index
    for name in rules:
        rows.append(dict(floor=f, delta_med_nz=g.delta[g.near_zero].median(), rule=name,
                         agree_nz=agree.loc[i[m[i]], name].mean(), agree_Z20_Z21=agree.loc[i[focus[i]], name].mean(),
                         agree_all=agree.loc[i, name].mean(),
                         sel_Z20=sel.loc[i[d.factor[i] == "Z20"], name].mean(),
                         sel_Z21=sel.loc[i[d.factor[i] == "Z21"], name].mean()))
res = pd.DataFrame(rows)
res.to_csv(OUT / "sweep_rule_agreement.csv", index=False)
pd.set_option("display.width", 250)
print(curve.round(3).to_string())
for col in ("agree_nz", "agree_Z20_Z21", "agree_all", "sel_Z20", "sel_Z21"):
    print(f"\n{col}\n", res.pivot(index="rule", columns="floor", values=col).round(3).to_string())
