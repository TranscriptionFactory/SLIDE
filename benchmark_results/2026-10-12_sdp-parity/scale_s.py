"""Ratio of ASDP s/var for raw vs standardized Z per chunk (scale equivariance check)."""
import sys
import numpy as np, pandas as pd
from loveslide.knockoff.solve import create_solve_asdp
z = pd.read_csv(sys.argv[1], index_col=0).values
for i in range(8):
    X = z[:, i*22:min(i*22+22, z.shape[1])]; Xs = (X - X.mean(0)) / X.std(0)
    a = np.asarray(create_solve_asdp(np.cov(X, rowvar=False))).ravel() / X.var(0, ddof=1)
    b = np.asarray(create_solve_asdp(np.cov(Xs, rowvar=False))).ravel() / Xs.var(0, ddof=1)
    tiny = a < 1e-6; q = b[tiny] / a[tiny]
    print(i, f"tiny scaled/raw ratio: {q.min():.2f}-{q.max():.2f}")
