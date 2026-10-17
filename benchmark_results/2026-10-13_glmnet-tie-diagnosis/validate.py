"""Check the Python mirrors against recorded W (draws.csv) for Z20/Z21, rust_s arm, first K draws."""
import sys, numpy as np, pandas as pd
from cdsim import *
K = int(sys.argv[1])
z, y, xt, sw = load("rust_s")
d = pd.read_csv(ROOT / ".tmpfiles/2026-10-13_s-zero/draws.csv")
d = d[d.arm == "rust_s"].pivot_table(index=["draw", "factor"], columns="stat", values="w")
def W(e, j):
    o, k = e[j], e[P + j]
    return 0.0 if o == k else max(o, k) * np.sign(o - k)
agree = {"rust_default": 0, "rust_replay": 0, "python": 0}
for dr in range(K):
    xs, yc = prep(augmented(z, xt[dr], sw[dr]), y)
    lam = grid(xs, yc)
    er4 = entry(rust_cd(xs, yc, lam, 1e-4, 1000)[0], lam)
    er7 = entry(rust_cd(xs, yc, lam, 1e-7, 100000)[0], lam)
    eg = entry(glmnet_naive(xs, yc, lam)[0], lam) * N
    for f, j in (("Z20", 20), ("Z21", 21)):
        s = -1 if sw[dr, j] else 1
        mine = {"rust_default": s * W(er4, j), "rust_replay": s * W(er7, j), "python": s * W(eg, j)}
        for k in agree:
            ok = np.isclose(mine[k], d.loc[(dr, f), k], rtol=1e-9, atol=0)
            agree[k] += ok
            if not ok: print("mismatch", dr, f, k, mine[k], d.loc[(dr, f), k])
print({k: f"{v}/{2*K}" for k, v in agree.items()})
