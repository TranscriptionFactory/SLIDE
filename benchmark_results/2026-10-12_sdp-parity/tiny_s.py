"""Relative agreement of near-zero s entries (R vs Python), with optional R posdef rule."""
import sys
import numpy as np, pandas as pd
from scipy import linalg
import loveslide.knockoff.utils as u, loveslide.knockoff.solve as s, loveslide.knockoff.filter as f
if sys.argv[3] == "rtol":
    def is_posdef(A, tol=1e-9, **k):
        return np.min(linalg.eigvalsh(np.asarray(A))) > tol * 10
    for m in (u, s, f):
        m.is_posdef = is_posdef
D, Z = sys.argv[1], sys.argv[2]
z = pd.read_csv(Z, index_col=0).values
for i in range(8):
    X = z[:, i*22:min(i*22+22, z.shape[1])]; S = np.cov(X, rowvar=False); v = np.diag(S)
    r = np.array([float.fromhex(t) for t in open(f"{D}/rsh_{i}.txt").read().split()]) / v
    p = np.asarray(s.create_solve_asdp(S)).ravel() / v
    tiny = r < 1e-6
    q = p[tiny] / r[tiny]
    print(i, f"n_tiny={tiny.sum()} py/R ratio on tiny s: min {q.min():.2f} median {np.median(q):.2f} max {q.max():.2f}")
