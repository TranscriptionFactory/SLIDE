#!/usr/bin/env python3
"""Run the shared Rust artifact fixture (stdlib only); optionally test a real Unix PTY."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import shutil
import statistics
import sys
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def validate(checkpoint, out_path=None):
    data = json.loads(checkpoint.read_text())
    final_cv = data.get("final_cv")
    if not final_cv or not final_cv.get("folds"):
        raise RuntimeError(f"{checkpoint} does not contain final_cv folds")
    evaluations = data.get("evaluations", [])
    factors = max((run.get("love", {}).get("k", 0) for run in evaluations), default=0)
    if factors < 2:
        raise RuntimeError(f"{checkpoint} only produced {factors} factors")
    if out_path is not None:
        plots = list(out_path.rglob("ControlPerformancePlot.pdf"))
        if not plots:
            raise RuntimeError("No Cairo control plot was exported")
        for path in plots:
            pdf = path.read_bytes()
            if not pdf.startswith(b"%PDF-") or b"/FontFile" not in pdf:
                raise RuntimeError(f"PDF has no embedded font: {path}")
    logging.info("checkpoint=%s factors=%s folds=%s", checkpoint, factors, len(final_cv["folds"]))


def tui_pty(binary, checkpoint, config):
    # TestBackend/help cannot establish real terminal startup; use a controlling PTY.
    if os.name == "nt":
        raise RuntimeError("Windows TUI startup requires a native ConPTY harness; Unix PTY is unavailable")
    import pty
    import select
    import fcntl
    import struct
    import termios

    pid, fd = pty.fork()
    if pid == 0:
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
        os.environ["TERM"] = "xterm-256color"
        os.execv(str(binary), [str(binary), "view", "--checkpoint", str(checkpoint),
                              "--config", str(config), "--no-follow"])
    output = bytearray()
    deadline = time.monotonic() + 30
    sent_quit = False
    status = None
    try:
        while time.monotonic() < deadline:
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    chunk = b""
                output.extend(chunk)
                # Wait for an actual alternate-screen render before quitting.
                if not sent_quit and b"\x1b[?1049h" in output and b"Overview" in output:
                    os.write(fd, b"q")
                    sent_quit = True
            child, status = os.waitpid(pid, os.WNOHANG)
            if child:
                break
            status = None
        if status is None:
            child, status = os.waitpid(pid, os.WNOHANG)
            if not child:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                raise RuntimeError("TUI did not exit within the PTY startup deadline")
        if not sent_quit or os.waitstatus_to_exitcode(status) != 0:
            raise RuntimeError(f"TUI PTY startup failed: {bytes(output[-2000:])!r}")
    finally:
        os.close(fd)


def smoke(binary, workdir, out_path=None, reuse=False, check_tui=False, full_fit=False, rscript=None):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "README.md").write_text(
        "# Rust artifact smoke\n\n40-feature HIV fixture, final CV, optional PDF/font checks and Unix PTY startup.\n"
        f"Command: python scripts/slide_artifact_smoke.py --binary {binary} --workdir {workdir}\n"
        f"PDF output: {out_path}; reuse: {reuse}; PTY: {check_tui}.\n")
    checkpoint = workdir / "tui-smoke-checkpoint.json"
    config = workdir / "tui-smoke.yaml"
    if not reuse:
        for previous in workdir.glob("tui-smoke-checkpoint.*"):
            archive = workdir / "archive" / (time.strftime("%Y-%m-%d_checkpoint-") + str(time.time_ns()))
            archive.mkdir(parents=True)
            shutil.move(str(previous), archive / previous.name)
    x_path, y_path = workdir / "tui-smoke-hiv-x.csv", workdir / "tui-smoke-hiv-y.csv"
    with (ROOT / "examples/HIV+cytokines_X.csv").open(newline="") as src, x_path.open("w", newline="") as dst:
        writer = csv.writer(dst)
        for row in csv.reader(src):
            writer.writerow(row[:41])
    shutil.copyfile(ROOT / "examples/HIV+cytokines_y.csv", y_path)
    if full_fit:
        return full_fit_smoke(binary, workdir, x_path, y_path, rscript)
    # JSON strings are valid YAML scalars, including Windows backslashes/spaces.
    config.write_text(
        f"x_path: {json.dumps(str(x_path))}\ny_path: {json.dumps(str(y_path))}\n"
        "delta: [0.1]\nlambda: [0.5]\nspec: 0.1\nfdr: 0.1\nSLIDE_iter: 8\n"
        "thresh_fdr: 0.1\ny_factor: true\ndo_interacts: false\npure_homo: true\n"
        "cv:\n  enabled: true\n  nrep: 1\n  k: 3\n  eval_type: auc\n"
        + (f"out_path: {json.dumps(str(out_path))}\n" if out_path else "")
    )
    subprocess.run([str(binary), "run", "--config", str(config), "--checkpoint", str(checkpoint)], check=True)
    validate(checkpoint, out_path)
    if check_tui:
        tui_pty(binary, checkpoint, config)


def full_fit_smoke(binary, workdir, x_path, y_path, rscript):
    """Exercise fixed fitting, factor-only CV and ordered-weight validation."""
    with y_path.open() as handle:
        rows = list(csv.reader(handle))
    levels = sorted({float(row[1]) for row in rows[1:]})
    assert len(levels) == 2
    for row in rows[1:]:
        row[1] = str(int(float(row[1]) == levels[1]))
    with y_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    weights = workdir / "weights.csv"
    with weights.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "weight"])
        writer.writerows((row[0], 1 + i % 3) for i, row in enumerate(rows[1:]))
    out = workdir / "full-fit"
    config = workdir / "full-fit.yaml"
    config.write_text(f"x_path: {json.dumps(str(x_path))}\ny_path: {json.dumps(str(y_path))}\n"
                      f"out_path: {json.dumps(str(out))}\n"
                      "delta: 0.1\nlambda: 0.5\nthresh_fdr: 0.1\nspec: 0.1\nfdr: 0.1\n"
                      "SLIDE_iter: 8\ndo_interacts: yes\neval_type: auc\ny_levels: [0, 1]\n")
    subprocess.run([str(binary), "fit", "--config", str(config), "--weights", str(weights),
                    "--out", str(out), "--seed", "37", "--threads", "2"], check=True)
    status = json.loads((out / "status.json").read_text())
    assert status["factors"] == "complete"
    with (out / "z_matrix.csv").open() as handle:
        z = list(csv.reader(handle))
    assert [r[0] for r in z[1:]] == [r[0] for r in rows[1:]]
    effective = json.loads((out / "effective-config.json").read_text())
    assert effective["knockoff"]["f_size"] == min(20, len(z[0]) - 1)
    assert effective["knockoff"]["r_compatible"] and effective["knockoff"]["nlambda"] == 500
    bad = workdir / "misaligned-weights.csv"
    with weights.open() as handle:
        w = list(csv.reader(handle))
    w[1], w[2] = w[2], w[1]
    with bad.open("w", newline="") as handle:
        csv.writer(handle).writerows(w)
    rejected = workdir / "must-not-exist"
    run = subprocess.run([str(binary), "fit", "--config", str(config), "--weights", str(bad),
                          "--out", str(rejected)], capture_output=True, text=True)
    assert run.returncode != 0 and "sample IDs" in run.stderr and not rejected.exists()
    run = subprocess.run([str(binary), "fit", "--config", str(config), "--weights", str(weights),
                          "--out", str(out)], capture_output=True, text=True)
    assert run.returncode != 0 and "already exists" in run.stderr
    subprocess.run([str(binary), "cv", "--config", str(config), "--z", str(out / "z_matrix.csv"),
                    "--out", str(workdir / "full-fit-cv"), "--nrep", "2", "--k", "3",
                    "--seed", "38", "--threads", "2"], check=True)
    if rscript:
        code = r"""
            a <- commandArgs(TRUE); suppressPackageStartupMessages(library(SLIDE))
            x <- as.matrix(read.csv(a[1],row.names=1)); y <- read.csv(a[2],row.names=1)[,1]
            w <- read.csv(a[3],row.names=1)[,1]; out <- a[4]
            lf <- getLatentFactors(y,x,scale(x),std_y=TRUE,delta=.1,lambda=.5,thresh_fdr=.1)
            write.csv(lf$A,file.path(out,'r-A.csv')); write.csv(predZ(scale(x),lf),file.path(out,'r-Z.csv'))
            d <- read.csv(file.path(out,'design.csv'),row.names=1,check.names=FALSE)
            if(ncol(d)>0) {
                d$.response <- y
                fit <- glm(.response~.,data=d,family=binomial(),weights=w)
                status <- fit$converged && !fit$boundary && all(is.finite(coef(fit))) &&
                    all(fitted(fit)>10*.Machine$double.eps & fitted(fit)<1-10*.Machine$double.eps)
                writeLines(if(status) 'complete' else 'failed',file.path(out,'r-predictor-status.txt'))
                if(status) write.csv(data.frame(pred=predict(fit,type='response')),file.path(out,'r-predictions.csv'))
            }
        """
        subprocess.run([str(rscript), "--vanilla", "-e", code, str(x_path), str(y_path), str(weights), str(out)], check=True)
        def matrix(path):
            with path.open() as handle:
                return [[float(v) for v in row[1:]] for row in list(csv.reader(handle))[1:]]
        rz, zz = matrix(out / "r-Z.csv"), matrix(out / "z_matrix.csv")
        assert len(rz) == len(zz) and len(rz[0]) == len(zz[0])
        # Factor numbering and signs need not agree across implementations.
        matches = []
        for j in range(len(zz[0])):
            candidates = [(max(abs(u[j] - sign * v[k]) for u, v in zip(zz, rz)), k, sign)
                          for k in range(len(rz[0])) for sign in [-1, 1]]
            error, k, sign = min(candidates)
            assert error < 1e-6, (j, k, error)
            matches.append((k, sign))
        assert len({k for k, _ in matches}) == len(matches)
        ra, aa = matrix(out / "r-A.csv"), matrix(out / "A.csv")
        assert max(abs(u[j] - sign*v[k]) for u,v in zip(aa,ra) for j,(k,sign) in enumerate(matches)) < 1e-6
        if (out / "r-predictor-status.txt").exists():
            assert (out / "r-predictor-status.txt").read_text().strip() == status["predictor"]
        if status["predictor"] == "complete":
            assert max(abs(a[0]-b[0]) for a,b in zip(matrix(out / "predictions.csv"),matrix(out / "r-predictions.csv"))) < 1e-8
    logging.info("Full-fit smoke passed: %s", status)


def timed_cv(command, log_path, env, timeout, profile=False):
    """Measure each child separately; wait4 supplies peak RSS and final CPU usage."""
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            if profile:
                if not hasattr(os, "wait4"):
                    raise RuntimeError("Per-process profiling requires POSIX wait4")
                while True:
                    pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(status)
                        break
                    if time.monotonic() - started > timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    time.sleep(0.01)
            else:
                process.wait(timeout=timeout)
        finally:
            if process.returncode is None:
                process.kill()
                process.wait()
    elapsed = time.monotonic() - started
    if process.returncode:
        raise RuntimeError(f"CV failed ({process.returncode}); see {log_path}")
    metrics = {"elapsed_s": elapsed}
    if profile:
        metrics.update(cpu_s=usage.ru_utime + usage.ru_stime,
                       peak_bytes=usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                       dsdp_wait_s=0.0, dsdp_solve_s=0.0, dsdp_calls=0)
        for line in log_path.read_text().splitlines():
            if line.startswith('{"event":"dsdp"'):
                event = json.loads(line)
                metrics["dsdp_wait_s"] += event["wait_seconds"]
                metrics["dsdp_solve_s"] += event["solve_seconds"]
                metrics["dsdp_calls"] += 1
    return metrics


def cv_concurrency_smoke(binary, workdir, args):
    """One acceptance entrypoint; no fitting, production jobs or historical sweep."""
    workdir.mkdir(parents=True, exist_ok=False)
    (workdir / "README.md").write_text(
        "# Resource-aware CV acceptance\n\n"
        "Reuses saved factors; no LOVE fitting or production jobs. This run tests automatic resource "
        "ceilings, a bounded queue and optional paired inner-voting profiles.\n\n"
        f"Command: {sys.argv!r}\n")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key in list(env):
        if key.startswith("SLIDE_KNOCKOFF_"):
            env.pop(key)
    # The existing CLI test fixture can be supplied locally; CI uses committed HIV factors.
    if args.cv_fixture:
        z = args.cv_fixture.resolve() / "z_matrix.csv"
        y = args.cv_fixture.resolve() / "y.csv"
        spec = 0.25
    else:
        z = ROOT / "examples/out/2026-10-17_rust-example/0.1_0.1_out/z_matrix.csv"
        y = workdir / "y.csv"
        with (ROOT / "examples/HIV+cytokines_y.csv").open(newline="") as handle:
            rows = list(csv.reader(handle))
        levels = sorted({float(row[1]) for row in rows[1:]})
        assert len(levels) == 2
        for row in rows[1:]:
            row[1] = str(int(float(row[1]) == levels[1]))
        with y.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        spec = 0.1
    config = workdir / "input.yaml"
    def write_config(iterations):
        config.write_text(f"y_path: {json.dumps(str(y))}\nout_path: {json.dumps(str(z.parent))}\n"
                          f"eval_type: auc\nSLIDE_iter: {iterations}\nspec: {spec}\ndo_interacts: no\n")
    write_config(101)
    base = [str(binary), "cv", "--config", str(config), "--z", str(z),
            "--nrep", "1", "--k", "3", "--seed", "17", "--threads", "4", "--memory-budget-mib", "2048"]
    filenames = ["cv-result.json", "predictions.csv", "SLIDECV_boxplot_data.csv"]
    def same(left, right):
        for name in filenames:
            assert (left / name).read_bytes() == (right / name).read_bytes(), (name, left, right)
    def run(name, flags=(), run_env=None, profile=False, command=None):
        out = workdir / name
        metrics = timed_cv((command or base) + ["--out", str(out), *flags],
                           workdir / f"{name}.log", run_env or env, args.cv_timeout, profile)
        record = json.loads((out / "requested-config.json").read_text())
        folds = record["config"]["max_active_folds"]
        voting = record["config"]["knockoff"]["max_voting_workers"]
        cpu = record["resources"]["cpu_budget"]
        assert 1 <= folds * voting <= cpu <= 4
        headroom = record["resources"]["headroom"]
        if headroom is not None:
            estimate = (record["estimated_retained_bytes"] + folds * record["estimated_fold_bytes"]
                        + folds * (voting - 1) * record["estimated_additional_voting_worker_bytes"])
            assert estimate <= headroom // 4 * 3
        metrics.update(cpu_budget=cpu, active_folds=folds, voting_workers=voting)
        if profile:
            metrics["budget_utilization"] = metrics["cpu_s"] / metrics["elapsed_s"] / cpu
            assert metrics["peak_bytes"] <= 2048 * 1024**2
            assert metrics["budget_utilization"] <= 1.05
        return out, metrics
    # Existing Rust suites cover policy edge cases, queue admission/failure and exact numerics.
    for executable in args.test_binary:
        command = [str(executable.resolve()), "--test-threads=1"]
        if executable.name.startswith("slide-"):
            command.append("run::tests::")
        with (workdir / f"{executable.name}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True, timeout=args.cv_timeout)
    serial, _ = run("serial", ["--max-active-folds", "1", "--voting-workers", "1"])
    automatic, decision = run("automatic")
    ceiling, _ = run("ceiling", ["--max-active-folds", "999"])
    same(serial, automatic)
    same(serial, ceiling)
    spare, spare_decision = run("spare-capacity", ["--max-active-folds", "1"])
    same(serial, spare)
    record = json.loads((spare / "requested-config.json").read_text())
    headroom = record["resources"]["headroom"]
    capacity = ((headroom // 4 * 3 - record["estimated_retained_bytes"]) // record["estimated_fold_bytes"]
                if headroom is not None else 1)
    assert spare_decision["voting_workers"] == min(spare_decision["cpu_budget"], capacity)
    if args.cv_reference:
        same(serial, args.cv_reference)
    text = (workdir / "automatic.log").read_text()
    for expected in ["CV resources:", "CV completed fold pairs=3/3", "Knockoff voting completed draws=100/101",
                     "Knockoff voting completed draws=101/101"]:
        assert expected in text, expected
    data = json.loads((automatic / "cv-result.json").read_text())
    with y.open(newline="") as handle:
        n = len(list(csv.reader(handle))) - 1
    assert len(data["predictions"]) == len(data["permuted_predictions"]) == n
    assert len(data["folds"]) == len(data["permuted_folds"]) == 3
    assert len(data["repeat_scores"]) == 2
    limited_env = dict(env, SLURM_CPUS_PER_TASK="1", SLURM_MEM_PER_NODE="1024")
    inherited, limited = run("scheduler", run_env=limited_env)
    assert limited["cpu_budget"] == limited["active_folds"] == limited["voting_workers"] == 1
    same(serial, inherited)
    for flag, value in [("--threads", "0"), ("--max-active-folds", "0"), ("--memory-budget-mib", "0"),
                        ("--memory-budget-mib", "1"), ("--memory-budget-mib", str(2**64-1)), ("--voting-workers", "0")]:
        out = workdir / f"rejected-{flag[2:]}-{value}"
        # Replace rather than duplicate clap arguments.
        command = base[:]
        if flag in command:
            index = command.index(flag)
            command[index + 1] = value
        else:
            command += [flag, value]
        if flag == "--memory-budget-mib" and value == "1":
            command += ["--max-active-folds", "999"]
        rejected = subprocess.run(command + ["--out", str(out)], env=env, capture_output=True,
                                  text=True, timeout=args.cv_timeout)
        assert rejected.returncode and not out.exists(), (flag, value)
        assert "CV started:" not in rejected.stderr
    if sys.platform.startswith("linux"):
        import resource
        def limit_address_space():
            resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        out = workdir / "inherited-process-limit"
        with (workdir / "inherited-process-limit.log").open("w") as log:
            subprocess.run(base + ["--out", str(out)], env=env, stdout=log, stderr=log,
                           preexec_fn=limit_address_space, check=True, timeout=args.cv_timeout)
        record = json.loads((out / "requested-config.json").read_text())
        assert any(s.startswith("Max address space:") for s in record["resources"]["sources"])
        assert record["resources"]["headroom"] < 1024**3
        same(serial, out)
    if args.saved_config:
        command = [str(binary), "cv", "--config", str(args.saved_config.resolve()), "--z", str(args.saved_z.resolve()),
                   "--nrep", "1", "--k", "5", "--seed", "20261602", "--threads", "4", "--memory-budget-mib", "2048"]
        saved, _ = run("saved-factor-automatic", command=command)
        same(saved, args.saved_reference.resolve())
        saved_data = json.loads((saved / "cv-result.json").read_text())
        assert len(saved_data["predictions"]) == len(saved_data["permuted_predictions"]) == 778
    profile_result = {"gate_passed": None, "reason": "not requested", "pairs": []}
    if args.profile_voting:
        write_config(1000)
        profile_env = dict(env, SLIDE_KNOCKOFF_PROFILE="1")
        profile_result.update(gate_passed=False, reason="gate failed")
        try:
            for pair in range(3):
                results = {}
                outputs = {}
                # Alternate order to reduce warm-cache/order bias across pairs.
                for label in (["serial", "parallel"] if pair % 2 == 0 else ["parallel", "serial"]):
                    outputs[label], results[label] = run(f"profile-{pair + 1}-{label}",
                        ["--max-active-folds", "1", "--voting-workers", "1" if label == "serial" else "4"],
                        run_env=profile_env, profile=True)
                same(outputs["serial"], outputs["parallel"])
                profile_result["pairs"].append(results)
            pairs = profile_result["pairs"]
            ratios = [p["parallel"]["elapsed_s"] / p["serial"]["elapsed_s"] for p in pairs]
            passed = (all(p["serial"]["budget_utilization"] < 0.8 for p in pairs)
                      and all(p["parallel"]["voting_workers"] > 1 for p in pairs)
                      and all(r < 1 for r in ratios) and statistics.median(ratios) <= 0.9)
            profile_result.update(gate_passed=passed, reason="gate passed" if passed else "gate failed",
                                  median_speedup_fraction=1-statistics.median(ratios))
        except subprocess.TimeoutExpired:
            profile_result["reason"] = "profile timed out; gate does not authorize enabling inner voting"
        except RuntimeError as error:
            profile_result["reason"] = f"profile unavailable or failed: {error}"
    summary = {"resource_queue_numerics": "passed", "automatic": decision,
               "saved_factor": "passed" if args.saved_config else "not supplied",
               "inner_voting": profile_result, "platform": sys.platform}
    (workdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("Combined CV acceptance: %s", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Executable; on Windows use bin/slide.exe and set FONTCONFIG_FILE")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--out-path", type=Path, help="Export and check embedded-font PDFs (not visual validation)")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--tui-pty", action="store_true", help="Require real Unix terminal startup/render/quit")
    parser.add_argument("--full-fit", action="store_true", help="Test the fixed R-compatible fit and CV handoff")
    parser.add_argument("--rscript", type=Path, help="Check full-fit factors and weighted predictions against R")
    parser.add_argument("--cv-concurrency", action="store_true", help="Run combined saved-factor concurrency acceptance instead of fitting")
    parser.add_argument("--cv-fixture", type=Path, help="Existing small CLI fixture directory containing z_matrix.csv and y.csv")
    parser.add_argument("--cv-reference", type=Path, help="Accepted small-fixture outputs (101 draws, seed 17, three folds)")
    parser.add_argument("--saved-config", type=Path, help="Existing 778-cell diagnostic YAML (two draws, interactions)")
    parser.add_argument("--saved-z", type=Path)
    parser.add_argument("--saved-reference", type=Path)
    parser.add_argument("--profile-voting", action="store_true", help="Three bounded serial/parallel pairs at 1000 draws; POSIX only")
    parser.add_argument("--cv-timeout", type=float, default=600)
    parser.add_argument("--test-binary", type=Path, action="append", default=[], help="Prebuilt Rust test executable; repeat for selected suites")
    args = parser.parse_args()
    if any([args.saved_config, args.saved_z, args.saved_reference]) and not all([args.saved_config, args.saved_z, args.saved_reference]):
        parser.error("--saved-config, --saved-z and --saved-reference must be provided together")
    logging.basicConfig(level=logging.INFO)
    if args.cv_concurrency:
        cv_concurrency_smoke(args.binary.resolve(), args.workdir.resolve(), args)
    else:
        smoke(args.binary.resolve(), args.workdir.resolve(), args.out_path.resolve() if args.out_path else None,
              args.reuse, args.tui_pty, args.full_fit, args.rscript)
