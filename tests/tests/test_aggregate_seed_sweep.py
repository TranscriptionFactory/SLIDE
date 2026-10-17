import importlib.util
import json
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "aggregate_seed_sweep.py"
SPEC = importlib.util.spec_from_file_location("aggregate_seed_sweep", MODULE_PATH)
aggregate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = aggregate
SPEC.loader.exec_module(aggregate)


def _write_seed(parent, seed, results):
    seed_dir = parent / f"seed_{seed}"
    seed_dir.mkdir()
    (seed_dir / "benchmark_summary.json").write_text(json.dumps({"results": results}))


def test_stable_core_and_cross_backend_comparison(tmp_path):
    _write_seed(tmp_path, 1, {
        "pure_r_0.1_1.0": {"sig_lfs": ["Z1", "Z2"], "sig_interacts": ["Z1.Z2"]},
        "rust_0.1_1.0": {"sig_lfs": ["Z1", "Z3"], "sig_interacts": ["Z2:Z1"]},
    })
    _write_seed(tmp_path, 2, {
        "pure_r_0.1_1.0": {"sig_lfs": ["Z1"], "sig_interacts": ["Z1.Z2"]},
        "rust_0.1_1.0": {"sig_lfs": ["Z1"], "sig_interacts": []},
    })
    _write_seed(tmp_path, 3, {
        "pure_r_0.1_1.0": {"sig_lfs": ["Z1", "Z2"], "sig_interacts": []},
        "rust_0.1_1.0": {"sig_lfs": ["Z1", "Z2"], "sig_interacts": ["Z1.Z2"]},
    })

    summary = aggregate.summarize(tmp_path)

    assert summary["groups"]["pure_r_0.1_1.0"]["marginal"]["stable_core"] == ["Z1", "Z2"]
    comparison = summary["comparisons"][0]
    assert comparison["marginal_jaccard"] == 0.5
    assert comparison["marginal_recall"] == 0.5
    assert comparison["interaction_jaccard"] == 1.0


def test_writer_emits_machine_and_human_readable_outputs(tmp_path):
    summary = {"seeds": [], "stable_threshold": 0.5, "groups": {}, "comparisons": []}

    aggregate.write_outputs(summary, tmp_path)

    assert (tmp_path / "seed_sweep_summary.json").exists()
    assert "Stable core" in (tmp_path / "seed_sweep_report.md").read_text()
