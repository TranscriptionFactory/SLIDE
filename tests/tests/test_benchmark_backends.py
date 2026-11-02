import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "scripts" / "benchmark_backends.py"
SPEC = importlib.util.spec_from_file_location("benchmark_backends", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_benchmark_project_root_and_r_runner():
    assert benchmark._PROJECT_ROOT == Path(__file__).resolve().parents[2]
    assert (benchmark._PROJECT_ROOT / "benchmarks/scripts/run_slide_r.R").is_file()


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
            cell = Path(captured["out_path"]) / "0.1_1.0_out"
            (cell / "sampleCV_model.json").write_text(json.dumps({"status": "no_marginals", "metric": None}))

    monkeypatch.setitem(
        sys.modules, "slide_py",
        types.SimpleNamespace(OptimizeSLIDE=FakeOptimizeSLIDE),
    )
    monkeypatch.setattr(benchmark, "np", __import__("numpy"))
    args = benchmark.parse_args(["--knockoff-base-seed", "123", "--threads", "2",
                                 "--x-path", "x.csv", "--y-path", "y.csv",
                                 "--niter", "5", "--no-slide-selection"])

    result = benchmark.run_python_backend(args, "python", 0.1, 1.0, tmp_path)

    assert result.status == "success"
    assert captured["knockoff_base_seed"] == 123
    assert captured["slide_selection"] is False
    assert captured["auto_select"] is False
    assert captured["sample_cv_iterations"] == 500
    assert captured["sample_cv_k"] == 4
    assert captured["eval_type"] == "corr"


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
    monkeypatch.setitem(sys.modules, "slide_py.knockoff.stats.glmnet",
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

    monkeypatch.setitem(sys.modules, "slide_py.knockoff.stats.glmnet",
                        types.SimpleNamespace(HAS_GLMNET=True))
    monkeypatch.setitem(sys.modules, "slide_py.knockoff.solve",
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


@pytest.mark.parametrize("flags", [
    ["--workflow", "cv", "--backends", "python"],
    ["--workflow", "cv", "--z-path", "z.csv", "--backends", "r_knockoffs"],
    ["--sample-cv-iterations", "0"], ["--sample-cv-k", "1"],
    ["--cv-repeats", "0"], ["--cv-folds", "1"],
])
def test_workflow_settings_reject_invalid_values(flags):
    with pytest.raises(SystemExit):
        benchmark.parse_args(flags)


@pytest.mark.parametrize("workflow", ["full-fit", "cv"])
def test_rust_commands_export_settings_and_native_cv(monkeypatch, tmp_path, workflow):
    import yaml

    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda cmd, **kwargs:
                        calls.append(cmd) or types.SimpleNamespace(returncode=0))
    args = benchmark.parse_args(["--backends", "rust", "--workflow", workflow,
        "--z-path", "saved.csv", "--rust-bin", "/prebuilt/slide", "--threads", "4",
        "--knockoff-base-seed", "71", "--sample-cv-iterations", "37", "--sample-cv-k", "3"])
    record = benchmark.run_rust_backend(args, "rust", 0.1, 1.0, tmp_path)
    config = yaml.safe_load((tmp_path / "rust_config.yaml").read_text())
    assert record.status == "success"
    assert config["out_path"] == str(tmp_path.resolve())
    assert config["sample_cv_iterations"] == 37 and config["sample_cv_k"] == 3
    assert config["auto_select"] is False and config["eval_type"] == "corr"
    assert calls[0][:2] == ["/prebuilt/slide", "cv" if workflow == "cv" else "run"]
    if workflow == "cv":
        assert config["delta"] == 0.1 and config["lambda"] == 1.0
        for flag, value in (("--z", str(Path("saved.csv").resolve())),
                            ("--nrep", "15"), ("--k", "5"), ("--seed", "71"), ("--threads", "4")):
            assert calls[0][calls[0].index(flag) + 1] == value
        assert "--checkpoint" not in calls[0]
    else:
        assert config["delta"] == [0.1] and config["lambda"] == [1.0]
        assert "--checkpoint" in calls[0]


def test_python_cv_wraps_saved_factors_without_love(monkeypatch, tmp_path):
    import pandas as pd

    captured = {}
    class FakeCV:
        def __init__(self, saved, **kwargs):
            captured.update(params=saved.input_params, z=saved.latent_factors,
                            y=saved.data.Y, settings=kwargs)
        def run(self, **kwargs):
            captured.update(run=kwargs)

    monkeypatch.setitem(sys.modules, "slide_py", types.SimpleNamespace(SLIDEcv=FakeCV))
    z = tmp_path / "z.csv"
    y = tmp_path / "y.csv"
    pd.DataFrame({"Z0": [1., 2., 3.]}, index=["a", "b", "c"]).to_csv(z)
    pd.DataFrame({"y": [2., 5., 1.]}, index=["a", "b", "c"]).to_csv(y)
    args = benchmark.parse_args(["--workflow", "cv", "--backends", "python", "--z-path", str(z),
                                 "--y-path", str(y), "--knockoff-base-seed", "19"])
    assert benchmark.run_python_cv_backend(args, .1, 1., tmp_path).status == "success"
    assert captured["settings"] == dict(nrep=15, k=5, eval_type="corr")
    assert captured["run"]["seed"] == 19
    assert captured["params"]["niter"] == 500
    assert captured["z"].index.equals(captured["y"].index)
    pd.DataFrame({"y": [2., 5., 1.]}, index=["c", "b", "a"]).to_csv(y)
    failed = benchmark.run_python_cv_backend(args, .1, 1., tmp_path)
    assert failed.status == "failed" and "identically ordered" in failed.error


def test_artifacts_keep_missing_and_zero_samplecv_distinct(tmp_path):
    record = benchmark.TimingRecord("python", .1, 1., .2, "success")
    missing = benchmark._cell_artifacts(record, tmp_path)
    assert missing["sampleCV"]["metric"] is None
    assert missing["sampleCV"]["status"] == "unavailable"
    cell = tmp_path / "python/0.1_1.0_out"
    cell.mkdir(parents=True)
    (cell / "sampleCV_model.json").write_text(json.dumps({"status": "ok", "metric": 0.0,
        "settings": {"score_provenance": "pooled_holdout"}}))
    (cell / "sampleCV_predictions.csv").write_text("iteration,sample_id,y_true,y_pred\n0,a,1,1\n")
    (cell / "slidecv_results.csv").write_text("replicate,method,metric_value\n0,SLIDE,0.2\n0,SLIDE_y,0.0\n")
    (cell / "slidecv_predictions.csv").write_text("replicate,method,sample_id,y_true,y_pred\n")
    result = benchmark._cell_artifacts(record, tmp_path)
    assert result["sampleCV"]["metric"] == 0.0
    assert result["sampleCV"]["provenance"] == "pooled_holdout"
    assert result["sampleCV"]["predictions"].endswith("sampleCV_predictions.csv")
    assert result["finalCV"]["status"] == "exported"
    assert len(result["finalCV"]["repeat_scores"]) == 2
    assert result["stage_timings"] is None


def test_cv_defaults_use_native_backends_and_six_hour_cap():
    args = benchmark.parse_args(["--workflow", "cv", "--z-path", "saved.csv"])
    assert args.backends == ["python", "rust", "pure_r"]
    assert args.cell_timeout == 21600
    assert benchmark.parse_args([]).cell_timeout == 600


def test_stage_timer_retains_failed_call_without_inferred_memory(tmp_path):
    def fail():
        raise ValueError("selection failure")
    with pytest.raises(ValueError, match="selection failure"):
        benchmark._timed_stage(fail, "selection", tmp_path)()
    with (tmp_path / "benchmark_stage_timings.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert float(rows[0]["seconds"]) >= 0
    assert "rss" not in rows[0]
