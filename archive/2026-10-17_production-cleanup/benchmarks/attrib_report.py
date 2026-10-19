#!/usr/bin/env python
"""CD4 over-selection attribution report (decomposes the rust-vs-native-R gap).

Reads the bundled emit_knockoff_attribution + attrib_xinject outputs from <attr_dir> and
splits the per-LF over-selection into a SAMPLING term (H1) and a LASSO term (H2):

  (a)  rust production, WITH swap        rust_counts.csv
  (a') rust lasso, NO swap, rust X-tilde rust_counts_noswap.csv   (auxiliary)
  (c)  R glmnet (swaps), rust's byte-identical X-tilde   r_on_rust_counts.csv
  (b)  R glmnet (swaps), R's own create.gaussian X-tilde r_on_r_counts.csv

  knockoff::stat.glmnet_lambdasmax swaps internally, as does rust production, so:
  H2 (lasso)    = (a) - (c)   same X-tilde, rust CD vs R glmnet (both swap)
  H1 (sampling) = (c) - (b)   same R glmnet, rust X-tilde vs R X-tilde
  total gap     = (a) - (b)   = H1 + H2
  (a') (no-swap) is shown only to expose the raw original-over-knockoff bias the swap masks.

plus realized mean |corr(Z_j, X~_j)|: rust (rust_realized_corr.csv) vs R (r_realized_corr.csv).

Usage: python attrib_report.py <attr_dir> <ndraws> [target_lfs_csv]
       e.g. python attrib_report.py .tmpfiles/2026-06-25_cd4_attribution 100 7,17,26
"""
import csv
import sys
from pathlib import Path


def read_counts(p: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            out[int(row["lf_index"])] = int(float(row["count"]))
    return out


def read_corr(p: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            out[int(row["lf_index"])] = float(row["mean_abs_corr"])
    return out


def main() -> int:
    d = Path(sys.argv[1])
    ndraws = int(sys.argv[2])
    targets = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [7, 17, 26]

    a = read_counts(d / "rust_counts.csv")
    ap = read_counts(d / "rust_counts_noswap.csv")
    c = read_counts(d / "r_on_rust_counts.csv")
    b = read_counts(d / "r_on_r_counts.csv")
    rc_rust = read_corr(d / "rust_realized_corr.csv")
    rc_r = read_corr(d / "r_realized_corr.csv")

    def rate(cnt: dict[int, int], i: int) -> float:
        return cnt.get(i, 0) / ndraws

    print(f"# CD4 d=0.1 over-selection attribution  (ndraws={ndraws}, threshold>=10% of draws)\n")
    hdr = (f"{'LF':>5} {'a':>4} {'aps':>4} {'c':>4} {'b':>4} "
           f"{'H2_lasso':>9} {'H1_samp':>8} {'swap':>5} "
           f"{'corr_rust':>9} {'corr_R':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = sorted(set(a) | set(ap) | set(c) | set(b))
    h1_tot = h2_tot = 0.0
    for i in rows:
        if max(a.get(i, 0), ap.get(i, 0), c.get(i, 0), b.get(i, 0)) == 0:
            continue
        h2 = a.get(i, 0) - c.get(i, 0)
        h1 = c.get(i, 0) - b.get(i, 0)
        sw = a.get(i, 0) - ap.get(i, 0)
        mark = "  <==" if i in targets else ""
        # accumulate only over LFs rust production over-selects vs R's own draws
        if a.get(i, 0) > b.get(i, 0):
            h2_tot += h2
            h1_tot += h1
        print(f"Z{i:<4} {a.get(i,0):>4} {ap.get(i,0):>4} {c.get(i,0):>4} {b.get(i,0):>4} "
              f"{h2:>9} {h1:>8} {sw:>5} "
              f"{rc_rust.get(i,0):>9.4f} {rc_r.get(i,0):>7.4f}{mark}")

    print("\n## targets (rates over draws)")
    for i in targets:
        print(f"  Z{i}: a={rate(a,i):.3f}  a'={rate(ap,i):.3f}  c={rate(c,i):.3f}  b={rate(b,i):.3f}"
              f"   |  H2(lasso)={rate(a,i)-rate(c,i):+.3f}  H1(samp)={rate(c,i)-rate(b,i):+.3f}"
              f"  |  corr rust={rc_rust.get(i,0):.4f} R={rc_r.get(i,0):.4f}")

    tot = h1_tot + h2_tot
    print(f"\n## verdict (summed over LFs where rust production over-selects vs R-own-draws)")
    print(f"  H2 lasso term  (a'-c): {h2_tot:+.0f} votes")
    print(f"  H1 sampling    (c-b) : {h1_tot:+.0f} votes")
    if tot != 0:
        print(f"  -> lasso explains {100*h2_tot/tot:.0f}%, sampling explains {100*h1_tot/tot:.0f}% "
              f"of the over-selection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
