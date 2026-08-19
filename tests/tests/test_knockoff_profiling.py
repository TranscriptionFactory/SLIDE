import json

import numpy as np
import pytest

from loveslide.knockoff.filter import (
    VotingResult,
    _partition_iteration_seeds,
    knockoff_filter_voting,
)
from loveslide.knockoff.stats.glmnet import stat_glmnet_lambdasmax


def _sampler(x, _cache, seed):
    return x[:, ::-1] + np.random.RandomState(seed).normal(0, 0.01, x.shape)


def _run(monkeypatch, n_jobs):
    x = np.random.RandomState(3).normal(size=(24, 4))
    y = x[:, 0] - x[:, 2] + np.random.RandomState(4).normal(size=24)
    monkeypatch.setattr(
        "loveslide.knockoff.filter._prepare_knockoff_cache", lambda *args, **kwargs: {}
    )
    return knockoff_filter_voting(
        x, y, statistic=stat_glmnet_lambdasmax, niter=2, n_jobs=n_jobs,
        base_seed=40, knockoff_sampler=_sampler, use_cache=True,
    )


def test_profiling_disabled_preserves_return_type(monkeypatch, tmp_path):
    monkeypatch.delenv("SLIDE_KNOCKOFF_PROFILE", raising=False)
    result = _run(monkeypatch, 1)
    assert isinstance(result, VotingResult)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_profiling_emits_all_iteration_stages(monkeypatch, tmp_path, n_jobs):
    profile = tmp_path / "profile.jsonl"
    monkeypatch.setenv("SLIDE_KNOCKOFF_PROFILE", str(profile))
    result = _run(monkeypatch, n_jobs)
    records = [json.loads(line) for line in profile.read_text().splitlines()]
    assert isinstance(result, VotingResult)
    assert {record["stage"] for record in records} == {
        "sample", "preparation", "lasso_path", "threshold"
    }
    assert all(record["backend"] == "python" for record in records)


def test_serial_and_parallel_voting_match_for_fixed_seeds(monkeypatch):
    serial = _run(monkeypatch, 1)
    parallel = _run(monkeypatch, 2)
    assert np.array_equal(serial.selection_counts, parallel.selection_counts)
    assert np.array_equal(serial.selected, parallel.selected)


def test_iteration_batches_cover_ordered_seeds_once():
    batches = _partition_iteration_seeds(niter=23, n_jobs=2, base_seed=100)
    flattened = [item for batch in batches for item in batch]

    assert len(batches) == 8
    assert flattened == [(i, 100 + i) for i in range(23)]


def test_unseeded_iteration_batches_keep_iteration_identity():
    batches = _partition_iteration_seeds(niter=3, n_jobs=10, base_seed=None)

    assert [item for batch in batches for item in batch] == [
        (0, None), (1, None), (2, None),
    ]
