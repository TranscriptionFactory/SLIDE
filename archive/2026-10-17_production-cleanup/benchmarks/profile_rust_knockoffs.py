#!/usr/bin/env python3
"""Run the release voting profiler on manifest-listed saved subproblems.

Manifest: [{"label": "...", "x": "X.raw", "y": "y.raw", "n": 24, "p": 22}].
Raw files are little-endian float64, row-major. Paths are relative to the manifest.
R-generated setup includes startup, generation, text transfer and parsing.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, SLIDE_KNOCKOFF_PROFILE="1", TMPDIR=str(args.out.resolve()))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[key] = "1"
    provenance = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "inputs": [],
        "hostname": platform.node(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {key: env.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "R_LIBS")},
        "config": {"method": "asdp", "max_block_size": 500, "nlambda": 250, "seed": 42, "fdr": 0.1, "spec": 0.2, "parallel": False},
    }
    rows = []
    for case in json.loads(args.manifest.read_text()):
        paths = [(args.manifest.parent / case[key]).resolve() for key in ("x", "y")]
        provenance["inputs"].append({"case": case, "sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]})
        for backend in ("native", "r_generated"):
            for repeat in range(args.repeats):
                prefix = args.out / f"{case['label']}_{backend}_{repeat + 1}"
                command = [str(args.binary.resolve()), *map(str, paths), str(case["n"]), str(case["p"]), str(args.niter)]
                if backend == "r_generated":
                    command.append(str(args.rscript.resolve()))
                with prefix.with_suffix(".jsonl").open("w") as stdout, prefix.with_suffix(".stages.jsonl").open("w") as stderr:
                    try:
                        subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout, check=True)
                    except subprocess.TimeoutExpired:
                        rows.append({"label": case["label"], "backend": backend, "repeat": repeat + 1, "stage": "timeout", "seconds": args.timeout})
                        break
                events = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
                stages = {}
                for line in prefix.with_suffix(".stages.jsonl").read_text().splitlines():
                    if line.startswith('{"event":"slide_knockoff_profile"'):
                        event = json.loads(line)
                        stages[event["stage"]] = stages.get(event["stage"], 0) + event["seconds"]
                draw_seconds = sum(stages.values())
                for event in events:
                    if event["event"] == "voting_total":
                        stages["unprofiled_setup_and_dispatch"] = event["seconds"] - draw_seconds
                    stages[event["event"]] = event["seconds"]
                rows.extend({"label": case["label"], "backend": backend, "repeat": repeat + 1, "stage": stage, "seconds": seconds} for stage, seconds in stages.items())
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.out / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "backend", "repeat", "stage", "seconds"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
