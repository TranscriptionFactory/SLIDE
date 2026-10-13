#!/usr/bin/env python3
"""Prepare saved marginal designs for paired solver replay, without refitting LOVE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from loveslide.tools import calc_default_fsize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header, *rows = csv.reader(handle)
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError(f"Empty or inconsistent CSV: {path}")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate row labels: {path}")
    return header, rows


def verify_matrix(rows: list[list[str]], saved: dict, label: str) -> float:
    if saved["dim"] != [len(rows), len(rows[0]) - 1]:
        raise ValueError(f"{label} shape differs from checkpoint")
    values = [float(value) for row in rows for value in row[1:]]
    if len(values) != len(saved["data"]):
        raise ValueError(f"{label} checkpoint data length differs")
    if any(not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10)
           for a, b in zip(values, saved["data"])):
        raise ValueError(f"{label} values/order differ from checkpoint")
    return max(abs(a - b) for a, b in zip(values, saved["data"]))


def prepare(cell_dir: Path, checkpoint: Path, y_csv: Path, out: Path, delta: float = 0.1,
            lam: float = 1.0, shape: tuple[int, int] = (24, 88), union_size: int = 11) -> dict:
    sources = {"z": cell_dir / "z_matrix.csv", "a": cell_dir / "A.csv",
               "y": y_csv, "checkpoint": checkpoint}
    saved = json.loads(checkpoint.read_text())
    if len(saved["evaluations"]) != 1:
        raise ValueError("Expected a single-cell checkpoint")
    cell = saved["evaluations"][0]
    if cell["delta"] != delta or cell["lambda"] != lam:
        raise ValueError(f"Expected the delta={delta}, lambda={lam} cell")
    zh, z = read_csv(sources["z"])
    ah, a = read_csv(sources["a"])
    yh, y = read_csv(y_csv)
    n, p = len(z), len(zh) - 1
    if (n, p) != tuple(shape):
        raise ValueError(f"Expected the saved {shape[0]} x {shape[1]} design")
    if zh[1:] != [f"Z{i}" for i in range(p)] or ah[1:] != zh[1:]:
        raise ValueError("A/Z column names or order differ")
    if len(y) != n or len(yh) != 2 or sha256(y_csv) != saved["y_ref"]["sha256"]:
        raise ValueError("Y shape/hash differs from the checkpoint response")
    z_error = verify_matrix(z, cell["z"], "Z")
    a_error = verify_matrix(a, cell["love"]["a"], "A")
    original_samples = [row[0] for row in z]
    samples = [row[0] for row in y]
    if original_samples not in (samples, [f"V{i + 1}" for i in range(n)]):
        raise ValueError("Z sample order is neither Y labels nor verified positional labels")
    # As in SLIDE's select_short_freq: calc_default_fsize(24, 88) = 24 gives four
    # balanced chunks of 22; calc_default_fsize(24, 172) = 24 gives eight (seven of 22, one of 18).
    f_size = saved["config"]["knockoff"]["f_size"] or calc_default_fsize(n, p)
    splits = math.ceil(p / f_size)
    width = math.ceil(p / splits)
    designs = [(f"marginal_chunk_{i + 1}", list(range(i * width, min((i + 1) * width, p))))
               for i in range(splits)]
    union = cell["marginal"]["voting"]["stage_one_union"]
    if len(union) != union_size or union != sorted(set(union)) or any(i < 0 or i >= p for i in union):
        raise ValueError(f"Expected the saved ordered {union_size}-column second-stage union")
    designs.append(("rust_stage_two_union", union))
    metadata = {
        "delta": cell["delta"], "lambda": cell["lambda"],
        "sources": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                    for name, path in sources.items()},
        "original_z_sample_order": original_samples, "sample_order": samples,
        "alignment": "Y hash and rowwise Z/A checkpoint comparison verify positional alignment; output Z uses Y sample labels.",
        "checkpoint_max_abs_difference": {"z": z_error, "a": a_error},
        "checkpoint_comparison_tolerance": {"atol": 1e-10, "rtol": 1e-10},
        "f_size": f_size, "n_splits": splits, "feature_split": width,
        "rationale": "Replay actual balanced marginal chunks and one fixed saved Rust stage-two union; no LOVE regeneration or interaction replay.",
        "designs": [],
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, columns in designs:
        directory = out / name
        directory.mkdir()
        names = [zh[i + 1] for i in columns]
        for filename, header, rows in (
            ("z_matrix.csv", [zh[0], *names],
             [[sample, *[row[i + 1] for i in columns]] for sample, row in zip(samples, z)]),
            ("A.csv", [ah[0], *names],
             [[row[0], *[row[i + 1] for i in columns]] for row in a]),
            ("Y.csv", yh, y),
        ):
            with (directory / filename).open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
        metadata["designs"].append({
            "name": name, "column_indices": columns, "lf_names": names,
            "shape": {"z": [n, len(columns)], "a": [len(a), len(columns)], "y": [n, 1]},
            "hashes": {filename: sha256(directory / filename)
                       for filename in ("z_matrix.csv", "A.csv", "Y.csv")},
        })
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("cell-dir", "checkpoint", "y-csv", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--shape", type=int, nargs=2, default=[24, 88], metavar=("N", "P"))
    parser.add_argument("--union-size", type=int, default=11)
    args = parser.parse_args()
    prepare(args.cell_dir, args.checkpoint, args.y_csv, args.out, args.delta, args.lam,
            args.shape, args.union_size)


if __name__ == "__main__":
    main()
