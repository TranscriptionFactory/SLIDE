"""Boundary cases for caret-style folds and pooled CV scores."""

import numpy as np
import pytest

from test_cv import make_cv


def test_sample_sd_global_factors_and_training_response_scaling():
    cv = make_cv()
    np.testing.assert_allclose(cv.z_matrix.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(cv.z_matrix.std(axis=0, ddof=1), 1)
    train, valid = np.array([2., 4., 6.]), np.array([8., 10.])
    a, b = cv._standardize_fold(train, valid)
    np.testing.assert_allclose(a, [-1, 0, 1])
    np.testing.assert_allclose(b, [2, 3])


def test_continuous_quantile_strata_and_binary_rare_class():
    cv = make_cv(y=np.arange(32))
    folds = cv._make_folds(np.random.default_rng(8), 10)
    all_valid = np.concatenate([valid for _, valid in folds])
    np.testing.assert_array_equal(np.sort(all_valid), np.arange(32))
    # Four quantile bins; each contributes two rows to every fold.
    bins = np.searchsorted(np.quantile(cv.y, [.25, .5, .75]), cv.y, side='left')
    for _, valid in folds:
        np.testing.assert_array_equal(np.bincount(bins[valid], minlength=4), [2]*4)
    rare = make_cv(y=[0]*30 + [1]*2, eval_type='auc')
    assert len(rare._make_folds(np.random.default_rng(8), 10)) == 4
    impossible = make_cv(y=[0]*31 + [1], eval_type='auc')
    with pytest.raises(ValueError, match='nondegenerate'):
        impossible._make_folds(np.random.default_rng(8), 3)


def test_leave_one_out_and_binary_corr_branch():
    cv = make_cv(y=[0, 1]*16)
    assert not cv.is_classifier
    cv.k = cv.n_samples
    assert len(cv._make_folds(np.random.default_rng(2), 1)) == cv.n_samples


@pytest.mark.parametrize('pred, expected, raw', [([1, 1, 0, 0], 1, 0),
                                               ([0, 0, 1, 1], 1, 1),
                                               ([1, 1, 1, 1], .5, .5),
                                               ([0, np.nan, 1, 1], .75, .75)])
def test_auc_imputes_median_then_flips_direction(pred, expected, raw):
    cv = make_cv(y=[0]*16 + [1]*16, eval_type='auc')
    actual, actual_raw, mse = cv._score(np.array([0, 0, 1, 1]), pred)
    assert actual == expected
    assert actual_raw == raw
    assert np.isnan(mse)
    assert np.isnan(cv._compute_metric(np.array([0, 0, 1, 1]), np.full(4, np.nan)))


def test_continuous_spearman_mse_and_undefined_metrics():
    cv = make_cv()
    assert cv._compute_metric(np.array([1., 2., 3.]), np.array([9., 5., 1.])) == -1
    metric, raw, mse = cv._score(np.array([1., 2., 3.]), [1., 1., 1.])
    assert np.isnan(metric) and np.isnan(raw)
    assert mse == pytest.approx(5/3)
    assert np.isnan(cv._compute_metric(np.array([1., 2., 3.]), [1., np.nan, 3.]))


def test_response_uses_train_stats_and_saved_z_is_not_rescaled(monkeypatch):
    cv = make_cv()
    folds = cv._make_folds(np.random.default_rng(10), 10)
    seen = []
    def predict(train, y, valid, **kwargs):
        seen.append((train.copy(), y.copy(), valid.copy()))
        return np.zeros(len(valid))
    monkeypatch.setattr(cv, '_run_slide_fold', predict)
    result = cv.run(seed=10)
    for fold, (train, valid) in enumerate(folds):
        observed = seen[2*fold]
        np.testing.assert_array_equal(observed[0], cv.z_matrix[train])
        np.testing.assert_allclose(observed[1].mean(), 0, atol=1e-15)
        np.testing.assert_allclose(observed[1].std(ddof=1), 1)
        actual = cv.predictions_.query('replicate == 0 and fold == @fold and method == "SLIDE"')
        np.testing.assert_allclose(actual.y_true, (cv.y[valid]-cv.y[train].mean())/cv.y[train].std(ddof=1))
    assert result.status.eq('undefined_metric').all()


def test_incomplete_prediction_shape_and_coverage_fail(monkeypatch):
    cv = make_cv()
    monkeypatch.setattr(cv, '_run_slide_fold', lambda *a, **k: np.array([]))
    with pytest.raises(ValueError, match='incomplete predictions'):
        cv.run()
    monkeypatch.setattr(cv, '_run_slide_fold', lambda a, b, c, **k: np.zeros(len(c)))
    monkeypatch.setattr(cv, '_make_folds', lambda *a: [(np.arange(24), np.arange(24, 32))])
    with pytest.raises(ValueError, match='sample coverage'):
        cv.run()


def test_marginal_and_interaction_size_loops_are_bounded(monkeypatch):
    cv = make_cv(z=np.random.default_rng(3).normal(size=(5, 8)), y=np.arange(5))
    specs = []
    def oversized(z, y, spec, f_size, rng):
        specs.append(spec)
        return np.arange(z.shape[1])
    monkeypatch.setattr(cv, '_select', oversized)
    with pytest.raises(ValueError, match='oversized'):
        cv._run_slide_fold(cv.z_matrix, cv.y, cv.z_matrix[:1])
    assert specs[0] == .2 and specs[-1] == pytest.approx(1)
    specs.clear()
    with pytest.raises(ValueError, match='oversized'):
        cv._find_interactions_fold(cv.z_matrix, cv.y, [0, 1], cv._f_size)
    assert specs[-1] == pytest.approx(1)


@pytest.mark.parametrize('change', ['ids', 'constant_z', 'constant_y', 'nan', 'multicolumn_y'])
def test_reject_invalid_data(change):
    cv = make_cv()
    slide = cv.slide_obj
    if change == 'ids':
        slide.data.Y = slide.data.Y.iloc[::-1]
    elif change == 'constant_z':
        slide.latent_factors.iloc[:, 0] = 1
    elif change == 'constant_y':
        slide.data.Y.iloc[:, 0] = 1
    elif change == 'nan':
        slide.latent_factors.iloc[0, 0] = np.nan
    else:
        slide.data.Y['extra'] = 1
    with pytest.raises(ValueError, match='SLIDEcv'):
        type(cv)(slide, k=4)


@pytest.mark.parametrize('parameters', [dict(k=1), dict(k=33), dict(nrep=0), dict(nrep=True),
                                        dict(eval_type='mse'), dict(f_size=0)])
def test_reject_invalid_settings(parameters):
    cv = make_cv()
    with pytest.raises(ValueError, match='SLIDEcv'):
        type(cv)(cv.slide_obj, **parameters)


def test_auc_is_pooled_across_folds_and_counts_imputation(monkeypatch):
    z = np.column_stack((np.arange(8), np.arange(8)**2))
    cv = make_cv(z=z, y=[0]*4 + [1]*4, eval_type='auc')
    folds = [(np.setdiff1d(np.arange(8), [i, i+1]), np.array([i, i+1])) for i in range(0, 8, 2)]
    monkeypatch.setattr(cv, '_make_folds', lambda *args: folds)
    monkeypatch.setattr(cv, '_run_slide_fold', lambda train, y, valid, **kwargs: valid[:, 0])
    pooled = cv.run()
    assert pooled.metric_value.eq(1).all()
    assert pooled.imputed_predictions.eq(0).all()
    def missing(train, y, valid, **kwargs):
        result = valid[:, 0].copy()
        result[0] = np.nan
        return result
    monkeypatch.setattr(cv, '_run_slide_fold', missing)
    scored = cv.run()
    assert scored.imputed_predictions.eq(4).all()
    assert scored.status.eq('ok').all()
    assert cv.predictions_.y_pred.isna().sum() == 16


@pytest.mark.parametrize('name,value', [('knockoff_backend', 'r_knockoffs'), ('knockoff_method', 'equi'),
                                       ('knockoff_shrink', True), ('knockoff_offset', 1), ('fstat', 'glmnet_coefdiff')])
def test_reject_conflicting_method4_knobs(name, value):
    cv = make_cv()
    cv.slide_obj.input_params[name] = value
    with pytest.raises(ValueError, match=name):
        type(cv)(cv.slide_obj)
