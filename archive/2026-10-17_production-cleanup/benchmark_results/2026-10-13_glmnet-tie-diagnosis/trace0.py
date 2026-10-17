import sys, numpy as np
from cdsim import *
dr = int(sys.argv[1]); j = int(sys.argv[2])
z, y, xt, sw = load("rust_s")
xs, yc = prep(augmented(z, xt[dr], sw[dr]), y)
lam = grid(xs, yc)
n = N
pr = (j, P + j)
print("swap", sw[dr, j], "rho", xs[:, j] @ xs[:, P + j] / (n - 1), "c_left-c_right", (xs[:, j] - xs[:, P + j]) @ yc / n)
for name, fn in (("rust", lambda: rust_cd(xs, yc, lam, float(sys.argv[3]), 100000, trace=pr)), ("glmnet", lambda: glmnet_naive(xs, yc, lam, trace=pr))):
    path, log = fn()
    e = entry(path, lam)
    print(name, "entry idx", [int(np.argmax(path[k] != 0)) for k in pr], "lam", lam[:6].round(5))
    for li, ph, k, g, old, new in log:
        if 19 <= li <= 21:
            print(f"  {name} li={li} {ph:6s} col={k:2d} grad={g:.10f} lam={lam[li]:.10f} beta {old:.3e}->{new:.3e}")
    print("  beta at entry+0..3:", path[list(pr)][:, 19:24].round(8).tolist())
