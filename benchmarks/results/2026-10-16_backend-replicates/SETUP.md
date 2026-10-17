# Five-arm backend statistical repeats

Status: four included arms completed 40 runs / 160 cells. The ten
Python-with-R-knockoffs tasks hit their 20-minute limits; 17 of their planned
40 cells finished before timeout. The user excluded that arm from statistical
analysis. Final four-arm analysis completed in job 11958124; tables and
figures verified. See
[final results](README.md).
Source: c4726e3 (Gaussian arithmetic and interaction-residual fixes).

## Design

Reuses the maintained `benchmarks/scripts/benchmark_backends.py` harness and the October 13 `run.sbatch`,
with the arm list extended and array range changed to 0–49 (10 concurrent).
No model or analysis code changes.

- Arms: rust, python (DSDP), pure_r, rust_r_knockoffs, r_knockoffs. In the
  harness, `rust_r_knockoffs` is the Rust CLI with R-generated knockoffs (R SDP and Gaussian draws);
  `r_knockoffs` is `slide-py` with Python LOVE and R knockoffs via `rpy2`.
  See [`benchmarks/scripts/benchmark_backends.py`](../../scripts/benchmark_backends.py).
- Ten runs each: seed labels 42, 1000, 2000, 3000, 4000, 5000, 6000,
  7000, 8000, 9000.
- Delta 0.01/0.1 × lambda 0.1/1.0: 50 tasks, 200 cells.
- Parity settings, 500 iterations, specificity 0.1, SLIDE selection,
  four workers/threads; BLAS threads 1.
- HTC Turin, four CPUs, 4 GB, 20 minutes per task.
- Frozen X/Y, R SLIDE package and Python 3.11 DSDP/glmnet extensions
  reused from the October 13 run. Fresh Rust build from frozen source.
- Fresh pure_r runs are the primary reference; no historical pooling.

Staging: .tmpfiles/2026-10-16_backend-replicates-all-arms/
under the repository root. This preserves all previous runs.

## Jobs and checks

- Build attempts 11957806 and 11957808 failed because Homebrew binutils
  require GLIBC_2.38, unavailable on the compute node.
- Build 11957812 uses PATH=$HOME/.cargo/bin:/usr/bin:/bin, system cc/c++/ar/ranlib,
  a new Cargo target directory, and the existing with-rust-blas-env.sh wrapper.
- Benchmark array: 11957807, dependent on the successful build.
- Original analysis: 11957810, cancelled after its dependency became unsatisfiable.
- Final four-arm analysis: 11958124, completed successfully.

Python DSDP, compiled glmnet and rpy2 import checks passed. R imports passed:
SLIDE 0.1.1, glmnet 5.0, knockoff 0.3.6, Rdsdp 1.0.6.
Each task checks the frozen input/extension manifest, all four cell statuses,
and the presence of four selection outputs.

## Python with R knockoffs: partial evidence only

The `r_knockoffs` arm uses `slide-py` with Python LOVE and R knockoffs via
`rpy2`. Each of its ten tasks timed out at 20 minutes, but successful cell
shards record 17 completed cells. The remaining 23 cells are incomplete or
were not started; do not treat them as completed 1200-second observations or
as zero selections. The user excluded this arm from the four-arm statistical
analysis.

| Delta | Lambda | Completed | Partial | Not started |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.1 | 10/10 | 0 | 0 |
| 0.01 | 1.0 | 7/10 | 3 | 0 |
| 0.1 | 0.1 | 0/10 | 7 | 3 |
| 0.1 | 1.0 | 0/10 | 0 | 10 |

Among successful cells only, `delta=0.01, lambda=0.1` selected Z18, Z27,
Z45 and Z136 in 10/10 runs each, and Z74 in 8/10; median wall time was
581.0 seconds (25th–75th percentile: 577.1–585.7). At `delta=0.01, lambda=1.0`, Z27 and Z45 appeared in 7/7,
Z18 and Z136 in 6/7, Z73 in 3/7, and Z57 in 2/7; median wall time was
521.7 seconds (25th–75th percentile: 519.8–612.8). These are completion-conditioned summaries, especially the
7-run cell, and do not support statistical comparison. There are no completed
high-delta frequency or runtime estimates.

The original staging directory is
`../omni-SLIDE/.tmpfiles/2026-10-16_backend-replicates-all-arms/` relative to
this repository root. The compact evidence was copied from there on 2026-10-17;
no benchmark jobs were rerun. Seeds 42, 2000 and 6000 stopped during the
0.01/1.0 cell. The other seven stopped during 0.1/0.1. No task started 0.1/1.0.

Retained compact evidence under [`results/r_knockoffs/`](results/r_knockoffs/) includes timing shards
and both selection files for successful cells only. The ten task logs are in
[`logs/11957807_40.out`](logs/11957807_40.out) through
[`logs/11957807_49.out`](logs/11957807_49.out). The partial run belongs to
the c4726e3 campaign; it is not pooled with the separate 4b32111 pilot or the
older September run. `results/rust_r_knockoffs/seed_*/benchmark_summary.json`
retains the ten matching Rust-hybrid summaries, copied byte-for-byte from the
staging results; these summaries contain no patient inputs. For the 17
successful Python-hybrid cells, Z-matrix row labels and factor-column order
match the same-cell pure-R seed-1000 reference; maximum absolute difference
was 1.12e-11. This check used source matrices only; no patient-level matrices
were copied. A selection file without a successful timing shard does not count
as a completed cell.

Recompute the successful-only Python-hybrid counts and medians from this
repository root. The retained files omit incomplete cells; use the source
staging directory and task logs for the partial/not-started classification.

```python
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import json

root = Path("benchmarks/results/2026-10-16_backend-replicates/results/r_knockoffs")
cells = defaultdict(list)
for path in root.glob("seed_*/r_knockoffs/timing_shards/*.json"):
    timing = json.loads(path.read_text())
    assert timing["status"] == "success"
    cell = timing["delta"], timing["lam"]
    selected = path.parent.parent / f"{cell[0]}_{cell[1]}_out/sig_LFs.txt"
    cells[cell].append((timing["wall_seconds"], set(selected.read_text().split())))
assert {cell: len(runs) for cell, runs in cells.items()} == {(0.01, 0.1): 10, (0.01, 1.0): 7}
for cell, runs in sorted(cells.items()):
    counts = Counter(factor for _, factors in runs for factor in factors)
    print(cell, len(runs), median(seconds for seconds, _ in runs), dict(sorted(counts.items())))
```

The original plan was for analyze_seed_replicates.py to compare all four non-reference
arms with fresh pure_r, plus rust:python, rust:rust_r_knockoffs and
python:r_knockoffs. It requires ten runs per arm/cell and K/Z agreement
with R within 1e-8. It writes count tests, factor frequencies and Wilson
intervals, Fisher BH q-values, Jaccard/permutation comparisons, bootstrap
runtime intervals, and three figures here. It uses 10,000 permutations
and 10,000 bootstrap samples. Inspect these outputs after completion.

## Reproduction

Set RUN to the absolute staging directory. Environment:
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library

Submit the prepared configuration:
sbatch --output="$RUN/logs/%A_%a.out" "$RUN/run.sbatch" "$RUN" "$ENV" "$R_DEPS"

The source snapshot, REVISION, inputs, run.sbatch, compiled extensions,
build logs and job IDs are in staging. SLURM retains the submitted analysis
command in job 11957810.

## Limits

These are independent backend draws, not seed-paired comparisons; stock R
PSOCK workers are not fully controlled by seed labels. Ten repeats on one
dataset cannot establish equivalence or FDR control. Runtime includes node
variation and backend-specific timing boundaries. The analysis does not
extract knockoff-stage timing for rust_r_knockoffs; total wall timing remains
available. Permutation p-values are exploratory and not multiplicity-adjusted.
