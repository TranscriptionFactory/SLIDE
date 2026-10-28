#!/usr/bin/env python3
"""Run the shared Rust artifact fixture (stdlib only); optionally test a real Unix PTY."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import shutil
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Executable; on Windows use bin/slide.exe and set FONTCONFIG_FILE")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--out-path", type=Path, help="Export and check embedded-font PDFs (not visual validation)")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--tui-pty", action="store_true", help="Require real Unix terminal startup/render/quit")
    parser.add_argument("--full-fit", action="store_true", help="Test the fixed R-compatible fit and CV handoff")
    parser.add_argument("--rscript", type=Path, help="Check full-fit factors and weighted predictions against R")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    smoke(args.binary.resolve(), args.workdir.resolve(), args.out_path.resolve() if args.out_path else None,
          args.reuse, args.tui_pty, args.full_fit, args.rscript)
