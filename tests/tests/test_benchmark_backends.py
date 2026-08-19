import csv
import importlib.util
import json
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "benchmark_backends.py"
SPEC = importlib.util.spec_from_file_location("benchmark_backends", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_timing_shards_replace_cells_and_merge_in_grid_order(tmp_path):
    benchmark._save_timing(
        benchmark.TimingRecord("python", 0.1, 1.0, 5.0, "success"), tmp_path
    )
    benchmark._save_timing(
        benchmark.TimingRecord("python", 0.01, 0.1, 7.0, "success"), tmp_path
    )
    benchmark._save_timing(
        benchmark.TimingRecord("python", 0.1, 1.0, 3.0, "success"), tmp_path
    )

    records = benchmark._load_saved_timings(tmp_path)

    assert [(r.delta, r.lam, r.wall_seconds) for r in records] == [
        (0.01, 0.1, 7.0),
        (0.1, 1.0, 3.0),
    ]
    assert len(list((tmp_path / "python" / "timing_shards").glob("*.json"))) == 2


def test_legacy_timing_file_is_still_loaded(tmp_path):
    backend_dir = tmp_path / "rust"
    backend_dir.mkdir()
    (backend_dir / "timings.json").write_text(json.dumps([{
        "backend": "rust", "delta": 0.1, "lam": 1.0,
        "wall_seconds": 2.0, "status": "success", "error": "",
    }]))

    records = benchmark._load_saved_timings(tmp_path)

    assert records == [benchmark.TimingRecord("rust", 0.1, 1.0, 2.0, "success", "")]


def test_cell_summaries_are_aggregated_without_shared_input(tmp_path):
    backend_dir = tmp_path / "pure_r"
    for cell, row in [
        ("0.1_1.0_out", ["0.1", "1.0", "88", "2", "4"]),
        ("0.01_0.1_out", ["0.01", "0.1", "172", "5", "14"]),
    ]:
        cell_dir = backend_dir / cell
        cell_dir.mkdir(parents=True)
        with (cell_dir / "benchmark_summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "delta", "lambda", "Num_of_LFs", "Num_of_Sig_LFs",
                "Num_of_Interactors",
            ])
            writer.writerow(row)

    benchmark._merge_cell_summaries(tmp_path)

    with (backend_dir / "benchmark_cell_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["delta"], row["lambda"]) for row in rows] == [
        ("0.01", "0.1"), ("0.1", "1.0"),
    ]
