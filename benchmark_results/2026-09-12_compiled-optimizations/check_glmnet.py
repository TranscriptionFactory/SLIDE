import sys
import time
from pathlib import Path
import numpy as np
from loveslide.knockoff.stats.glmnet import HAS_GLMNET, stat_glmnet_lambdasmax
from loveslide.knockoff._vendor.glmnet import _glmnet
assert HAS_GLMNET
outputs = {}
times = []
for seed in range(4):
    rng = np.random.default_rng(seed)
    n, p = [(24, 87), (80, 30)][seed % 2]
    x = rng.normal(size=(n, p))
    xt = rng.normal(size=(n, p))
    y = x[:, 0] + rng.normal(size=n)
    started = time.perf_counter()
    w, diagnostics = stat_glmnet_lambdasmax(x, xt, y, swap=np.zeros(p, dtype=bool), return_diagnostics=True)
    times.append(time.perf_counter() - started)
    outputs[f'w_{seed}'] = w
    outputs[f'coef_{seed}'] = diagnostics['coefficients']
    outputs[f'lambda_{seed}'] = diagnostics['lambda']
path = Path(sys.argv[1])
np.savez(path, **outputs)
print('extension:', _glmnet.__file__)
print('seconds:', times)
if len(sys.argv) > 2:
    baseline = np.load(sys.argv[2])
    for name, actual in outputs.items():
        np.testing.assert_allclose(actual, baseline[name], rtol=1e-10, atol=1e-12)
    print('PASS:', len(outputs), 'paired arrays; max absolute difference:', max(np.max(np.abs(outputs[k]-baseline[k])) for k in outputs))
