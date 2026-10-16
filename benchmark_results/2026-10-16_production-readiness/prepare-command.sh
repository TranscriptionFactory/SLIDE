#!/usr/bin/env bash
set -euo pipefail
: "${RUN:?}" "${ENV:?}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$ENV/bin/python" - "$RUN" <<'PY'
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
run = Path(sys.argv[1])
sys.path.insert(0, str(run / 'source/benchmarks'))
from generate_factor_inputs import generate, write_inputs
inventory = {}
for dataset in ('ssc', 'cd4'):
    directory = run / 'inputs' / dataset
    x, y = (pd.read_csv(directory / name, index_col=0) for name in ('X.csv', 'Y.csv'))
    if dataset == 'ssc':
        assert list(x.index) == [f'V{i}' for i in range(1,25)] and len(y) == 24
    else:
        assert x.index.equals(y.index), f'{dataset}: sample order mismatch'
    assert y.shape[1] == 1 and x.index.is_unique
    assert np.isfinite(x.to_numpy(dtype=float)).all() and np.isfinite(y.to_numpy(dtype=float)).all()
    inventory[dataset] = {'n': len(x), 'p': x.shape[1], 'response_unique': int(y.iloc[:, 0].nunique()),
                          'sample_ids_match': x.index.equals(y.index),
                          'pairing': 'unchanged historical row order; biological pairing unverified' if dataset == 'ssc' else 'identical sample IDs and row order'}
tasks, scaling = [], []
rng = random.Random(20261016)
seeds = [42,1000,2000,3000,4000,5000,6000,7000,8000,9000]
def add(target, dataset, directory, seed, deltas, lambdas, ladder=None):
    arms = ['python', 'rust', 'pure_r']
    rng.shuffle(arms)
    for backend in arms:
        out = run / 'results' / dataset / backend / f'seed_{seed}'
        args = ['--backends', backend, '--configuration', 'parity', '--source-revision',
                (run / 'source/REVISION').read_text().strip(), '--x-path', str(directory / 'X.csv'),
                '--y-path', str(directory / 'Y.csv'), '--delta', *map(str, deltas),
                '--lam', *map(str, lambdas), '--niter', '500', '--spec', '.1',
                '--knockoff-base-seed', str(seed), '--slide-selection', '--threads', '4',
                '--rust-bin', str(run / 'bin/slide'), '--rscript', str(Path(sys.executable).with_name('Rscript')),
                '--out-dir', str(out), '--measure-process', '--cell-timeout',
                '600' if target is scaling else '300', '--profile']
        target.append({'dataset': dataset, 'backend': backend, 'seed': seed, 'out': str(out),
                       'cells': [[d,l] for d in deltas for l in lambdas], 'args': args,
                       'ladder': f'{ladder}:{backend}' if ladder else None})
for seed in seeds:
    add(tasks, 'ssc', run / 'inputs/ssc', seed, [.01,.1], [.1,1.])
for seed in seeds:
    add(tasks, 'cd4', run / 'inputs/cd4', seed, [.1], [.1])
for near in (False, True):
    for null in (True, False):
        regime = ('near' if near else 'ordinary') + ('_null' if null else '_sparse')
        for seed in range(10000,10020):
            directory = run / 'inputs' / regime / f'seed_{seed}'
            write_inputs(generate(seed=seed, near_degenerate=near, null=null), directory)
            add(tasks, regime, directory, seed, [.1], [.1])
for n,p in [(50,2000),(50,4000),(50,8000),(50,12000),(200,2000)]:
    dataset = f'scale_n{n}_p{p}'
    directory = run / 'inputs' / dataset
    write_inputs(generate(n=n,p=p,k=20,seed=20000), directory)
    add(scaling, dataset, directory, 20000, [.1], [.1], 'n50' if n==50 else None)
assert sum(len(t['cells']) for t in tasks) == 390 and len(scaling) == 15
for name, rows in [('first',tasks[:3]),('remaining',tasks[3:]),('scaling',scaling)]:
    with (run / f'{name}.json').open('x') as stream:
        json.dump(rows, stream, indent=2)
(run / 'input_inventory.json').write_text(json.dumps(inventory, indent=2))
PY
