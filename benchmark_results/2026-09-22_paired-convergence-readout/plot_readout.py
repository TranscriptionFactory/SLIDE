"""Plot saved paired-draw disagreement; arguments: result root, output directory."""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

root, out = map(Path, sys.argv[1:])
rows = []
pairs = [("python", "Python"), ("rust_500", "Rust production"),
         ("rust_500_tie_zero", "Rust tie-zero counterfactual")]
for directory in sorted(root.iterdir()):
    frame = pd.read_csv(directory / "pairwise_draw_jaccard.csv")
    for backend, label in pairs:
        paired = frame[((frame.left == "r") & (frame.right == backend)) |
                       ((frame.right == "r") & (frame.left == backend))]
        assert len(paired) == 1000 and paired.iteration.nunique() == 1000
        count = int((paired.jaccard < 1).sum())
        n, p, z = len(paired), count / len(paired), 1.959963984540054
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        rows.append(dict(design=directory.name, comparison=label, draws=n,
                         disagreements=count, percent=100*p,
                         low=100*(center-half), high=100*(center+half)))
data = pd.DataFrame(rows)
out.mkdir(parents=True, exist_ok=True)
data.to_csv(out / "disagreement.csv", index=False)
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(11, 5.5))
labels = [label for _, label in pairs]
sns.barplot(data=data, x="design", y="percent", hue="comparison", hue_order=labels,
            palette=["#789494", "#b84d43", "#3676a3"], errorbar=None, ax=ax)
for container, label in zip(ax.containers, labels):
    subset = data[data.comparison == label]
    centers = [bar.get_x() + bar.get_width()/2 for bar in container]
    ax.errorbar(centers, subset.percent, yerr=[subset.percent-subset.low, subset.high-subset.percent],
                fmt="none", color="#333333", capsize=3, linewidth=1)
    ax.bar_label(container, fmt="%.1f", padding=8, fontsize=8)
ax.set(title="Tied-entry handling explains much, but not all, of Rust's selection disagreement",
       xlabel="Fixed design", ylabel="Draws selecting a different set than R (%)", ylim=(0, 70))
ax.set_xticks(range(5), ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Saved Rust union"])
ax.legend(title="", loc="upper right", fontsize=9)
fig.text(.02, .025, "1,000 paired draws per design; bars show pointwise 95% Wilson intervals.\n"
         "Counterfactual changes W ties only, not fitted paths. Fixed-design results are not full-pipeline parity.", fontsize=9)
fig.tight_layout(rect=(0, .10, 1, 1))
fig.savefig(out / "paired_disagreement.png", dpi=160)
plt.close(fig)
