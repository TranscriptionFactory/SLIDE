"""Cost summaries retain the planned denominator and do not pair seed labels."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("compare_timings", Path(__file__).parents[2] / "benchmarks/compare_timings.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_manifest_failures_pending_and_units(tmp_path):
    manifest = tmp_path / "first.json"
    tasks = [dict(dataset="scale_n50_p12000", backend="python", seed=i, out=str(i), cells=[[.1, .1], [.1, 1]]) for i in range(3)]
    manifest.write_text(json.dumps(tasks))
    done = [{**tasks[0], "status": "failed", "timings": [dict(delta=.1, lam=.1, status="timeout", wall_seconds=300, peak_rss_bytes=2**30)]},
            {**tasks[1], "status": "not_attempted_after_limit"}]
    manifest.with_suffix(".status.jsonl").write_text("\n".join(map(json.dumps, done)))
    records = report.manifest_records([manifest])
    assert len(records) == 6
    assert records.status.tolist() == ["timeout", "failed", "not_attempted_after_limit", "not_attempted_after_limit", "pending", "pending"]
    assert records.iloc[0].peak_rss_gib == 1
    assert (records.p == 12000).all() and (records.n == 50).all()
    assert (records.K_truth == 20).all() and records.K_measured.isna().all()
    summary, speedup = report.summarize_manifest(records)
    assert summary.planned.sum() == 6 and summary.attempts.sum() == 2
    assert summary.timeouts.sum() == 1 and summary.failures.sum() == 1
    assert summary.pending.sum() == 2 and summary.successes.sum() == 0
    assert speedup.empty
    with pytest.raises(ValueError, match="Duplicate"):
        report.manifest_records([manifest, manifest])


def test_speedup_independent_bootstrap_and_incomplete_coverage():
    records = pd.DataFrame([dict(dataset="ssc", backend=backend, delta=.1, lam=.1, status=status,
                                wall_seconds=seconds, peak_rss_gib=1)
                            for backend, status, seconds in [("python", "success", 10), ("python", "success", 20),
                                                            ("python", "timeout", 300), ("pure_r", "success", 20),
                                                            ("pure_r", "success", 40)]])
    summary, speedup = report.summarize_manifest(records, bootstrap=200)
    result = speedup.iloc[0]
    assert result.speedup == 2
    assert result.ci95_lo <= 2 <= result.ci95_hi
    assert result.coverage_warning
    assert summary.loc[summary.backend == "python", "wall_seconds_max"].item() == 20
    _, again = report.summarize_manifest(records, bootstrap=200)
    pd.testing.assert_frame_equal(speedup, again)
    _, single = report.summarize_manifest(records.iloc[[0, 3]])
    assert np.isnan(single.iloc[0].ci95_lo)


def test_old_parser_unchanged(tmp_path):
    path = tmp_path / "sample.out"
    path.write_text("INFO rust delta=0.1 lam=1.0: 12.5s (success)\n")
    assert report.parse_timings([path])[0]["seconds"] == 12.5
