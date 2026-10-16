import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "benchmark_backends.py"
SPEC = importlib.util.spec_from_file_location("benchmark_backends", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_parity_settings_override_inherited_rust_shortcuts(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIDE_KNOCKOFF_TOL", "1e-4")
    monkeypatch.setenv("SLIDE_KNOCKOFF_MAX_ITER", "1000")
    monkeypatch.setenv("SLIDE_KNOCKOFF_MATCH_R_COV", "0")
    monkeypatch.setenv("SLIDE_KNOCKOFF_MATCH_R_LAMBDA", "0")
    args = benchmark.parse_args(["--configuration", "parity", "--source-revision", "frozen-sha"])

    assert args.knockoff_nlambda == 500
    assert args.knockoff_base_seed == 42
    assert args.threads == 4
    env = benchmark._rust_environment(args)
    assert {key: env[key] for key in benchmark._PARITY_RUST_ENV} == benchmark._PARITY_RUST_ENV
    assert env["RAYON_NUM_THREADS"] == "4"
    benchmark._write_environment_metadata(args, tmp_path)
    metadata = json.loads((tmp_path / "benchmark_environment.json").read_text())
    assert metadata["configuration"] == "parity"
    assert metadata["rust_solver_environment"] == benchmark._PARITY_RUST_ENV
    assert metadata["inherited_environment"]["SLIDE_KNOCKOFF_TOL"] == "1e-4"
    assert metadata["source_revision"] == "frozen-sha"
    assert metadata["enclosing_git_sha"] != "frozen-sha"
    assert metadata["workers_by_backend"]["python_serial"] == 1
    assert metadata["workers_by_backend"]["pure_r"] == 4
    assert metadata["paired_random_draws"] is False


def test_native_configuration_preserves_rust_environment(monkeypatch):
    monkeypatch.setenv("SLIDE_KNOCKOFF_TOL", "2e-5")
    args = benchmark.parse_args([])
    assert args.configuration == "native"
    assert args.knockoff_nlambda is None
    assert args.threads == 10
    assert benchmark._rust_environment(args)["SLIDE_KNOCKOFF_TOL"] == "2e-5"


@pytest.mark.parametrize("flags", [
    ["--y-factor"], ["--knockoff-nlambda", "250"],
    ["--knockoff-fdr", "0.2"], ["--threads", "0"],
    ["--slurm", "--threads", "5", "--slurm-cpus", "4"],
])
def test_parity_rejects_unmatched_or_oversubscribed_configuration(flags):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--configuration", "parity", *flags])


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


def test_python_runner_passes_base_seed_into_pipeline(monkeypatch, tmp_path):
    captured = {}

    class FakeOptimizeSLIDE:
        def __init__(self, input_params):
            captured.update(input_params)

        def run_pipeline(self, verbose):
            assert verbose is True

    monkeypatch.setitem(
        sys.modules, "loveslide",
        types.SimpleNamespace(OptimizeSLIDE=FakeOptimizeSLIDE),
    )
    monkeypatch.setattr(benchmark, "np", __import__("numpy"))
    args = types.SimpleNamespace(
        profile=False, knockoff_base_seed=123, threads=2,
        x_path="x.csv", y_path="y.csv", y_factor=False,
        niter=5, spec=0.1, slide_selection=False,
    )

    result = benchmark.run_python_backend(args, "python", 0.1, 1.0, tmp_path)

    assert result.status == "success"
    assert captured["knockoff_base_seed"] == 123
    assert captured["slide_selection"] is False


def test_measured_mode_requires_prebuilt_rust():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--measure-process", "--backends", "rust"])
    for value in ("0", "nan", "inf"):
        with pytest.raises(SystemExit):
            benchmark.parse_args(["--cell-timeout", value])


def test_process_monitor_records_nonzero_exit_and_memory(tmp_path):
    elapsed, peak, code, timed_out = benchmark._monitor_process(
        [sys.executable, "-c", "import time; time.sleep(0.2); raise SystemExit(7)"],
        tmp_path / "child.log", 5,
    )
    assert elapsed >= 0.2
    assert peak > 0
    assert code == 7
    assert not timed_out


def test_process_monitor_timeout_kills_worker(tmp_path):
    import time

    pid_path = tmp_path / "worker.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    elapsed, peak, code, timed_out = benchmark._monitor_process(
        [sys.executable, "-c", code], tmp_path / "timeout.log", 0.5,
    )
    assert timed_out and code != 0 and elapsed < 5 and peak > 0
    pid = int(pid_path.read_text())
    for _ in range(50):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"worker {pid} survived process-group cleanup")


def test_measurement_child_failure_is_retained(tmp_path):
    args = benchmark.parse_args(["--measure-process", "--backends", "pure_r",
                                 "--rscript", str(tmp_path / "missing-Rscript"),
                                 "--out-dir", str(tmp_path), "--cell-timeout", "10"])
    record = benchmark.run_measured_cell(args, "pure_r", 0.1, 1.0)
    assert record.status == "failed"
    assert "missing-Rscript" in record.error
    assert record.wall_seconds >= record.backend_wall_seconds
    assert record.peak_rss_bytes > 0
    assert "not cgroup peak" in record.memory_method
    benchmark._save_timing(record, tmp_path)
    assert benchmark._load_saved_timings(tmp_path) == [record]


def test_measured_python_rejects_fallback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "loveslide.knockoff.stats.glmnet",
                        types.SimpleNamespace(HAS_GLMNET=False))
    args = benchmark.parse_args(["--backends", "python", "--out-dir", str(tmp_path)])
    config = tmp_path / "child.json"
    result = tmp_path / "result.json"
    config.write_text(json.dumps({"args": vars(args), "backend": "python", "delta": 0.1,
                                  "lam": 1.0, "result_path": str(result)}, default=str))
    assert benchmark._measurement_child(config) == 1
    record = json.loads(result.read_text())
    assert record["python_glmnet_available"] is False
    assert "fallback is forbidden" in record["error"]


@pytest.mark.parametrize("configuration", ["parity", "native"])
def test_measured_parity_checks_dsdp_before_pipeline(monkeypatch, tmp_path, configuration):
    calls = []

    def missing_dsdp(solver):
        calls.append(solver)
        raise ImportError("DSDP unavailable")

    monkeypatch.setitem(sys.modules, "loveslide.knockoff.stats.glmnet",
                        types.SimpleNamespace(HAS_GLMNET=True))
    monkeypatch.setitem(sys.modules, "loveslide.knockoff.solve",
                        types.SimpleNamespace(_get_sdp_solver=missing_dsdp))
    monkeypatch.setattr(benchmark, "_run_cell", lambda args, backend, delta, lam:
                        benchmark.TimingRecord(backend, delta, lam, 0.01, "success"))
    args = benchmark.parse_args(["--configuration", configuration, "--backends", "python",
                                 "--out-dir", str(tmp_path)])
    config = tmp_path / "child.json"
    result = tmp_path / "result.json"
    config.write_text(json.dumps({"args": vars(args), "backend": "python", "delta": 0.1,
                                  "lam": 1.0, "result_path": str(result)}, default=str))
    exit_code = benchmark._measurement_child(config)
    record = json.loads(result.read_text())
    assert record["python_glmnet_available"] is True
    if configuration == "parity":
        assert calls == ["dsdp"]
        assert exit_code == 1 and record["status"] == "failed"
        assert record["error"] == "DSDP unavailable"
    else:
        assert calls == []
        assert exit_code == 0 and record["status"] == "success"
