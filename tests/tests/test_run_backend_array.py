import json
from pathlib import Path
import subprocess


def test_manifest_records_failure_and_stops_ladder(tmp_path):
    script = Path(__file__).parents[2] / "benchmarks/scripts/run_backend_array.sh"
    (tmp_path / "benchmarks/scripts").mkdir(parents=True)
    (tmp_path / "benchmarks/scripts/benchmark_backends.py").write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--out-dir')+1]) / 'rust'\n"
        "out.mkdir()\n"
        "(out/'timings.json').write_text(json.dumps([{'status':'timeout'}]))\n"
    )
    tasks = [dict(backend="rust", out=str(tmp_path / str(i)), ladder="rust",
                  args=["--measure-process", "--out-dir", str(tmp_path / str(i))]) for i in range(2)]
    manifest = tmp_path / "cells.json"
    manifest.write_text(json.dumps(tasks))
    subprocess.run(["bash", str(script), "--manifest", str(manifest), "--validate-only"],
                   cwd=tmp_path, check=True)
    subprocess.run(["bash", str(script), "--manifest", str(manifest)], cwd=tmp_path, check=True)
    rows = [json.loads(line) for line in manifest.with_suffix(".status.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["failed", "not_attempted_after_limit"]
    assert rows[0]["timings"][0]["status"] == "timeout"
    assert not (tmp_path / "1").exists()
    # A rerun cannot truncate the retained outcomes.
    repeat = subprocess.run(["bash", str(script), "--manifest", str(manifest)], cwd=tmp_path)
    assert repeat.returncode != 0
