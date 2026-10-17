#!/usr/bin/env python3
"""R-S vs R-S control and focus-factor vote table for the delta=0.01 S-only replay.

Control: every fit in <run>/replay must equal the matching fit in <run>/replay_repeat
except elapsed time, and the per-design votes/draws tables must be byte-identical.
Table: per design, Rust-S and R-S votes for Z20/Z21/Z73/Z74 with Wilson 95% intervals
and the paired difference with its Monte Carlo SE (not FDR inference).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = ["Z20", "Z21", "Z73", "Z74"]
ARMS = ("rust_s", "r_s")


def wilson(votes, draws, z=1.959963984540054):
    phat = votes / draws
    centre = (phat + z**2 / (2 * draws)) / (1 + z**2 / draws)
    half = z * np.sqrt(phat * (1 - phat) / draws + z**2 / (4 * draws**2)) / (1 + z**2 / draws)
    return centre - half, centre + half


def check_control(run, design):
    first, repeat = run / "replay" / design, run / "replay_repeat" / design
    for name in ("votes.csv", "draws.csv"):
        if (first / name).read_bytes() != (repeat / name).read_bytes():
            raise AssertionError(f"{design}: {name} differs between replay and repeat")
    fits = 0
    for arm in ARMS:
        for path in sorted((first / arm).glob("iter_*.json")):
            left, right = (json.loads(p.read_text()) for p in (path, repeat / arm / path.name))
            left.pop("wall_ms"), right.pop("wall_ms")
            if left != right:
                raise AssertionError(f"{design}/{arm}/{path.name}: repeat fit differs")
            fits += 1
    return fits


def main():
    run, designs = Path(sys.argv[1]), sys.argv[2:]
    rows = []
    for design in designs:
        fits = check_control(run, design)
        logging.info("%s: %d repeat fits identical apart from elapsed time", design, fits)
        votes = pd.read_csv(run / "replay" / design / "votes.csv")
        draws = json.loads((run / "replay" / design / "sampler_summary.json").read_text())["draws"]
        for row in votes[votes.feature.isin(FOCUS)].itertuples():
            record = {"design": design, "feature": row.feature, "draws": draws}
            for arm in ARMS:
                count = getattr(row, f"{arm}_votes")
                low, high = wilson(count, draws)
                record |= {f"{arm}_votes": count, f"{arm}_wilson_low": low, f"{arm}_wilson_high": high}
            record |= {"frequency_delta": row.frequency_delta, "paired_mcse": row.paired_mcse,
                       "rust_s_only": row.rust_s_only, "r_s_only": row.r_s_only}
            rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(run / "focus_factors.csv", index=False)
    logging.info("\n%s", table.to_string(index=False, float_format="%.4g"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
