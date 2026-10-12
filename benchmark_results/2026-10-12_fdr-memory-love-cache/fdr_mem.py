"""Peak RSS and time of one Python FDR threshold call (old or new) on synthetic p x p correlation."""
import sys, time, resource, importlib.util
import numpy as np
which, p = sys.argv[1], int(sys.argv[2])
if which == "old":
    spec = importlib.util.spec_from_file_location("loveslide.love_python.love.love_old", sys.argv[3])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fn = mod._apply_fdr_threshold
else:
    from loveslide.love_python.love.love import _apply_fdr_threshold as fn
rng = np.random.default_rng(0)
n = 50
X = rng.standard_normal((n, 20)) @ rng.standard_normal((20, p)) + rng.standard_normal((n, p))
R = np.corrcoef(X, rowvar=False)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
t = time.perf_counter(); out = fn(R, n, 0.2); dt = time.perf_counter() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"{which} p={p} time={dt:.1f}s peak_rss={peak:.2f}GB before_call={base:.2f}GB kept={(out != 0).mean():.4f} checksum={np.abs(out).sum():.6f}")
