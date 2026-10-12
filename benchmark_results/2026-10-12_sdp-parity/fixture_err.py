"""Relative error of Python ASDP diag_s against the saved high-delta R fixture."""
import numpy as np
from loveslide.knockoff.filter import _cov_shrink_corpcor
from loveslide.knockoff.solve import create_solve_asdp
f = np.load("tests/fixtures/knockoff_corpcor/high_delta_n24_p88.npz")
d = np.asarray(create_solve_asdp(_cov_shrink_corpcor(f["z"]))).ravel(); r = f["r_diag_s"]
v = np.diag(f["r_sigma"]); tiny = r / v < 1e-6
print("rel err", np.linalg.norm(d - r) / np.linalg.norm(r), "n_tiny", int(tiny.sum()),
      "tiny ratio range", (d[tiny] / r[tiny]).min() if tiny.any() else None, (d[tiny] / r[tiny]).max() if tiny.any() else None)
