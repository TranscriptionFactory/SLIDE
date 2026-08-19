import numpy as np
import pandas as pd

from loveslide.knockoffs import Knockoffs
from loveslide.slide import OptimizeSLIDE


def test_interaction_subproblems_use_disjoint_seed_ranges(monkeypatch):
    seeds = []

    def fake_select(**kwargs):
        seeds.append(kwargs["base_seed"])
        return np.array([], dtype=int)

    monkeypatch.setattr(Knockoffs, "select_short_freq", staticmethod(fake_select))
    slide = OptimizeSLIDE.__new__(OptimizeSLIDE)
    slide.latent_factors = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))
    slide.marginal_idxs = np.array([0, 1])
    slide.data = type("Data", (), {"Y": pd.Series([0.0, 1.0, 2.0, 3.0])})()

    slide.find_interaction_LFs(
        None, spec=0.1, fdr=0.1, niter=2, f_size=2,
        knockoff_backend="python", knockoff_base_seed=10,
    )

    assert seeds == [18, 26]
