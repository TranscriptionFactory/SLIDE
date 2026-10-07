"""Readback of the native-draw replay: final sets, union all-or-nothing rate, repeat check, figure.

Usage: readback.py <run> <repeat run> <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "benchmarks")
from analyze_paired_xtilde import aggregate_selection  # noqa: E402

run, repeat, out = map(Path, sys.argv[1:4])
FIXTURES = Path(".tmpfiles/2026-09-14_paired-convergence/results")
summaries = json.loads((run / "sampler_summary.json").read_text())
rows, union_votes = [], []
for summary in summaries:
    design, draws = summary["design"], summary["draws"]
    meta = json.loads((FIXTURES / design / "inputs/metadata.json").read_text())
    names = meta["lf_names"]
    differing = 0
    for arm in summary["arms"]:
        sets, sizes = [], []
        for i in range(draws):
            fit = json.loads((run / design / arm / f"iter_{i:03d}.json").read_text())
            again = json.loads((repeat / design / arm / f"iter_{i:03d}.json").read_text())
            differing += any(fit[k] != again[k] for k in fit if k != "wall_ms")
            sets.append(set(fit["selected"]))
            sizes.append(len(fit["selected"]))
        agg = aggregate_selection(sets, len(names), meta["spec"])
        sizes = np.array(sizes)
        rows.append(dict(design=design, arm=arm, final=" ".join(names[j] for j in agg["findOptIter"]["selected"]),
                         n_final=len(agg["findOptIter"]["selected"]), n_frequent=len(agg["frequent"]),
                         empty_draws=int((sizes == 0).sum()), near_all_draws=int((sizes >= len(names) - 2).sum()),
                         minimum_votes=agg["minimum_votes"]))
        if design == "rust_stage_two_union":
            union_votes += [dict(arm=arm, feature=f, votes=v) for f, v in zip(names, agg["votes"])]
    rows[-1]["repeat_differing_fits"] = rows[-2]["repeat_differing_fits"] = differing
table = pd.DataFrame(rows)
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "final_sets.csv", index=False)
print(table.to_string(index=False))

union = table[table.design == "rust_stage_two_union"].set_index("arm")
a, b = (int(union.loc[arm, "near_all_draws"]) for arm in ("rust_native", "r_native"))
n = summaries[0]["draws"]
print(f"union draws selecting >= p-2 features: rust {a}/{n}, R {b}/{n}; "
      f"Fisher p = {stats.fisher_exact([[a, n - a], [b, n - b]]).pvalue:.3f}; cutoff {union.minimum_votes.iloc[0]} votes")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
votes = pd.DataFrame(union_votes)
votes["rate"] = votes.votes / n
z = 1.959963984540054
votes["half"] = z * np.sqrt(votes.rate * (1 - votes.rate) / n)
sns.set_theme(style="whitegrid", font_scale=0.9)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for offset, (arm, colour) in zip((-0.15, 0.15), (("rust_native", "#c0392b"), ("r_native", "#2c6fbb"))):
    g = votes[votes.arm == arm].reset_index(drop=True)
    ax.errorbar(np.arange(len(g)) + offset, 100 * g.rate, yerr=100 * g.half, fmt="o", color=colour,
                capsize=2, label=f"{arm} ({int(union.loc[arm, 'n_final'])} in final set)")
ax.axhline(100 * union.minimum_votes.iloc[0] / n, color="black", linestyle="--", linewidth=1, label="spec cutoff (10%)")
ax.set_xticks(range(votes.feature.nunique()), votes.feature.unique())
ax.set_ylabel("Draws selecting feature (%)")
ax.set_xlabel("Rust stage-two union feature")
ax.set_title("Stage-two union: 1,000 native draws per side, common swaps and fitter")
ax.legend(fontsize=8, loc="upper right")
fig.text(0.01, 0.01, "Error bars: pointwise 95% normal intervals per arm. Independent innovation streams; not FDR inference.", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(out / "union_votes.png", dpi=160)
