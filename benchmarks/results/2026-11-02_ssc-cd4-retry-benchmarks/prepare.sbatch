#!/usr/bin/env bash
set -euo pipefail
RUN=${1:?campaign directory required}
source "$RUN/environment.sh"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$RUN/source"
sha256sum --check --quiet "$RUN/source-files.sha256"
sha256sum --check --quiet "$RUN/input-files.sha256"
export CC="$ENV/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV/bin/x86_64-conda-linux-gnu-c++"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CC"
export LIBCLANG_PATH="$ENV/lib" CARGO_BUILD_JOBS=1
export PKG_CONFIG_PATH="$RUN/build/pkgconfig"
export CARGO_TARGET_DIR="$REPO/target"
# Direct binaries avoid rustup settings locks on the shared home filesystem.
toolchain="$HOME/.rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin"
export RUSTC="$toolchain/rustc" RUSTDOC="$toolchain/rustdoc"
export PATH="$toolchain:$PATH"
"$toolchain/cargo" build --release --locked -p slide-rs --bin slide
cp "$CARGO_TARGET_DIR/release/slide" "$RUN/bin/slide"
sha256sum "$RUN/bin/slide" > "$RUN/build-artifacts.sha256"
{
    date -Is
    hostname
    taskset -pc $$
    lscpu
    "$RUN/bin/slide" --version
} > "$RUN/logs/preparation-hardware.txt"
bash benchmarks/scripts/run_backend_array.sh --manifest "$RUN/manifests/pilot-full-fit.json"

# The runner records failed tasks but can still exit zero; inspect every cell.
python - <<'PY'
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
run = Path(os.environ['RUN'])
tasks = json.loads((run / 'manifests/pilot-full-fit.json').read_text())
statuses = [json.loads(line) for line in
            (run / 'manifests/pilot-full-fit.status.jsonl').read_text().splitlines()]
assert len(tasks) == len(statuses) == 6
assert {row['out'] for row in statuses} == {task['out'] for task in tasks}
for row in statuses:
    assert row['status'] == 'success', (row['out'], row['status'])
    assert len(row['timings']) == 4, row['out']
    assert all(cell['status'] == 'success' for cell in row['timings']), row['out']

hashes = []
for dataset in ('ssc', 'cd4'):
    x = pd.read_csv(run / 'inputs' / dataset / 'X.csv', index_col=0)
    y = pd.read_csv(run / 'inputs' / dataset / 'Y.csv', index_col=0)
    assert x.index.equals(y.index) and y.shape[1] == 1, dataset
    assert y.index.is_unique and not y.index.hasnans, dataset
    for delta in (0.01, 0.1):
        for lam in (0.1, 1.0):
            cell = f'{delta}_{lam}_out'
            root = run / 'pilot/full-fit' / dataset
            reference_path = root / 'pure_r/seed_42/pure_r' / cell / 'z_matrix.csv'
            reference = pd.read_csv(reference_path, index_col=0)
            assert reference.index.equals(y.index), reference_path
            assert reference.columns.is_unique and np.isfinite(reference.to_numpy()).all()
            for backend in ('python', 'rust'):
                path = root / backend / 'seed_42' / backend / cell / 'z_matrix.csv'
                factors = pd.read_csv(path, index_col=0)
                assert factors.index.equals(reference.index), path
                assert factors.columns.equals(reference.columns), path
                np.testing.assert_allclose(factors.to_numpy(), reference.to_numpy(), atol=1e-8, rtol=0,
                                           err_msg=str(path))
            hashes.append(f'{hashlib.sha256(reference_path.read_bytes()).hexdigest()}  {reference_path}\n')
with (run / 'saved-factors.sha256').open('x') as handle:
    handle.writelines(hashes)
logging.info('Pilot gate passed: 24 full-fit cells, eight aligned R factor matrices.')
PY
