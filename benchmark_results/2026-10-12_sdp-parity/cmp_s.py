"""Compare Python create_solve_asdp against saved R create.solve_asdp output (hex)."""
import sys
import numpy as np, pandas as pd
from loveslide.knockoff.solve import create_solve_asdp
D, Z = sys.argv[1], sys.argv[2]
z = pd.read_csv(Z, index_col=0).values
hx = float.fromhex
for i in range(8):
    S = np.array([[hx(t) for t in l.split()] for l in open(f"{D}/rSig_{i}.txt")])
    r = np.array([hx(t) for t in open(f"{D}/rsh_{i}.txt").read().split()])
    X = z[:, i*22:min(i*22+22, z.shape[1])]
    p = np.asarray(create_solve_asdp(S)).ravel()
    pp = np.asarray(create_solve_asdp(np.cov(X, rowvar=False))).ravel()
    print(i, "R-Sigma: max|d|", f"{np.abs(p-r).max():.1e}", "bitwise", np.array_equal(p, r),
          "| py-Sigma: max|d|", f"{np.abs(pp-r).max():.1e}", "bitwise", np.array_equal(pp, r))
