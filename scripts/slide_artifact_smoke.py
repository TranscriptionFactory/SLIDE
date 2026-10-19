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
        os.execv(str(binary), [str(binary), "tui", "--checkpoint", str(checkpoint),
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


def smoke(binary, workdir, out_path=None, reuse=False, check_tui=False):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Executable; on Windows use bin/slide.exe and set FONTCONFIG_FILE")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--out-path", type=Path, help="Export and check embedded-font PDFs (not visual validation)")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--tui-pty", action="store_true", help="Require real Unix terminal startup/render/quit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    smoke(args.binary.resolve(), args.workdir.resolve(), args.out_path.resolve() if args.out_path else None,
          args.reuse, args.tui_pty)
