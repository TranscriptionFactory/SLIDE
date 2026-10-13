"""Summarize experiment.py outputs: W sign/zero fractions, selection rates and per-draw sign agreement
with R glmnet; Python-glmnet-vs-R agreement from the s-zero draws.csv is the target rate.
usage: summarize.py <scratch_dir> <out_dir>"""
import sys, numpy as np, pandas as pd
from pathlib import Path

SCR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ROOT = Path(__file__).resolve().parents[2]
d = pd.concat([pd.read_csv(f) for f in sorted(SCR.glob("w_*.csv"))])
sgn = d.assign(sign=np.sign(d.w)).pivot_table(index=["arm", "draw", "factor"], columns="stat", values="sign")
focus = d[d.factor.isin(["Z20", "Z21"])]
g = focus.groupby(["arm", "factor", "stat"])
summ = pd.DataFrame({"frac_w_pos": g.w.apply(lambda v: (v > 0).mean()),
                     "frac_w_zero": g.w.apply(lambda v: (v == 0).mean()),
                     "frac_w_neg": g.w.apply(lambda v: (v < 0).mean()),
                     "frac_selected": g.selected.mean()})
agree = sgn.drop(columns="r_glmnet").eq(sgn["r_glmnet"], axis=0)
summ["sign_agree_r"] = agree.loc[agree.index.get_level_values("factor").isin(["Z20", "Z21"])] \
    .groupby(level=["arm", "factor"]).mean().stack().rename_axis(["arm", "factor", "stat"])
# Target: Python vendored glmnet vs R glmnet on the same draws (from the s-zero check).
prev = pd.read_csv(ROOT / ".tmpfiles/2026-10-13_s-zero/draws.csv")
pv = prev.pivot_table(index=["arm", "draw", "factor"], columns="stat", values="w")
target = (np.sign(pv.python) == np.sign(pv.r_glmnet)).groupby(level=["arm", "factor"]).mean()
summ = summ.reset_index()
summ["target_python_vs_r"] = [target.get((a, f), np.nan) for a, f in zip(summ.arm, summ.factor)]
summ.round(4).to_csv(OUT / "focus_summary.csv", index=False)
allf = agree.groupby(level="arm").mean().round(4)
allf.to_csv(OUT / "all_factor_sign_agreement.csv")
pd.set_option("display.width", 200)
print(summ.round(3).to_string(index=False))
print("sign agreement with R over all 22 factors x draws:\n", allf.to_string())
