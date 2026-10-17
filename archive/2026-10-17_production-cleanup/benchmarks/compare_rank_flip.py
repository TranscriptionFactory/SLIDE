#!/usr/bin/env python
"""Head-to-head: R (instrumented selectShortFreq) vs rust per-LF knockoff counts at d=0.1.

Loads the R dump dir (stage_one_union.txt, stage_two_tab.csv, chunk_*_tab.csv,
final_marginal.txt) and the rust checkpoint, and tabulates, for every LF in either
stage-one union, R's vs rust's stage-two selection_count + union membership + final pick.
Localizes the rust-vs-pure_r ranking flip: is it the stage-one union (Z62 in/out) or the
stage-two per-LF counts (R counts Z25/Z33/Z53 high, rust ~25)?

Usage: python compare_rank_flip.py <r_dump_dir> <rust_checkpoint.json>
"""
import csv
import json
import sys
from pathlib import Path


def read_tab(p: Path) -> dict[str, int]:
    out = {}
    if not p.exists():
        return out
    with p.open() as f:
        for row in csv.DictReader(f):
            out[row["variable"]] = int(row["count"])
    return out


def read_csv_line(p: Path) -> set[str]:
    if not p.exists():
        return set()
    txt = p.read_text().strip()
    return {t for t in txt.replace("\n", ",").split(",") if t}


def main() -> int:
    rdir = Path(sys.argv[1])
    rust_ck = Path(sys.argv[2])

    r_union = read_csv_line(rdir / "stage_one_union.txt")
    r_stage2 = read_tab(rdir / "stage_two_tab.csv")
    r_final = read_csv_line(rdir / "final_marginal.txt")
    # per-chunk counts merged (stage-one screening view)
    r_chunk = {}
    for cf in sorted(rdir.glob("chunk_*_tab.csv")):
        r_chunk.update(read_tab(cf))

    ck = json.load(rust_ck.open())
    ev = [e for e in ck["evaluations"]
          if e and abs(e["delta"] - 0.1) < 1e-9 and abs(e["lambda"] - 0.1) < 1e-9][0]
    v = ev["marginal"]["voting"]
    rust_counts = {f"Z{i}": c for i, c in enumerate(v["selection_counts"])}
    rust_union = {f"Z{i}" for i in (v.get("stage_one_union") or [])}
    rust_final = {f"Z{i}" for i in ev["marginal"]["sig_lfs"]}
    min_sel = v["min_selections"]

    pure_r_ref = {"Z25", "Z33", "Z53", "Z61", "Z62"}

    print(f"# Rank-flip head-to-head @ d=0.1/l=0.1 (niter={v['niter']}, cliff={min_sel})\n")
    print(f"R   final (this standalone): {sorted(r_final, key=lambda z: int(z[1:]))}")
    print(f"    pure_r reported set      : {sorted(pure_r_ref, key=lambda z: int(z[1:]))}")
    print(f"rust final                   : {sorted(rust_final, key=lambda z: int(z[1:]))}\n")
    print(f"R    stage-one union ({len(r_union)}): {sorted(r_union, key=lambda z: int(z[1:]))}")
    print(f"rust stage-one union ({len(rust_union)}): {sorted(rust_union, key=lambda z: int(z[1:]))}")
    print(f"  in R-union NOT rust-union: {sorted(r_union - rust_union, key=lambda z: int(z[1:]))}")
    print(f"  in rust-union NOT R-union: {sorted(rust_union - r_union, key=lambda z: int(z[1:]))}\n")

    lfs = sorted(set(r_stage2) | set(rust_counts.keys()) & (r_union | rust_union)
                 | r_union | rust_union | pure_r_ref, key=lambda z: int(z[1:]))
    print(f"{'LF':>5} {'R_s2':>5} {'rust':>5} {'Rwin':>5} {'rustwin':>7} "
          f"{'Runion':>7} {'rstUni':>7} {'pure_r':>7}")
    for lf in lfs:
        rc = r_stage2.get(lf, 0)
        uc = rust_counts.get(lf, 0)
        if rc == 0 and uc == 0 and lf not in (r_union | rust_union | pure_r_ref):
            continue
        print(f"{lf:>5} {rc:>5} {uc:>5} {'Y' if rc>=min_sel else '.':>5} "
              f"{'Y' if uc>=min_sel else '.':>7} "
              f"{'Y' if lf in r_union else '.':>7} {'Y' if lf in rust_union else '.':>7} "
              f"{'Y' if lf in pure_r_ref else '.':>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
