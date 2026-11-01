"""Small fixed-design screening and export checks; no LOVE or knockoff jobs."""
import json

import numpy as np
import pandas as pd
import pytest

from slide_py import SLIDE, OptimizeSLIDE, sample_cv
from slide_py.tools import init_data


def test_explicit_ols_products_and_last_model():
    z = np.random.default_rng(3).normal(size=(12, 2))
    y = 2 + z[:, 0] + 3 * z[:, 0] * z[:, 1]
    train = [list(range(9)), list(range(3, 12))]
    original = z.copy()
    result = sample_cv(z, y, [0], [[0], [1]], iterations=2, train_splits=train)
    assert result['metric'] == pytest.approx(1)
    assert result['last_coefficients'] == pytest.approx([2, 1, 3])
    assert result['last_train_indices'] == train[-1]
    assert result['term_order'] == ['(Intercept)', 'Z1', 'Z1.Z2']
    assert [p['row_index'] for p in result['predictions']] == [9, 10, 11, 0, 1, 2]
    np.testing.assert_array_equal(z, original)


def test_ordered_aliases_not_minimum_norm():
    z = np.arange(8., dtype=float)
    z = np.column_stack((z, z))
    z[-1, 1] = 100
    result = sample_cv(z, 1 + 2*z[:, 0], [0, 1], iterations=1,
                       train_splits=[list(range(6))])
    assert result['last_coefficients'][:2] == pytest.approx([1, 2])
    assert result['last_coefficients'][2] is None
    assert result['predictions'][-1]['y_pred'] == pytest.approx(15)


def test_endpoint_guard_not_design_column_guard():
    z = np.random.default_rng(0).normal(size=(4, 4))
    result = sample_cv(z, np.arange(4.), [0], [[0, 0, 0], [1, 2, 3]], iterations=1)
    assert result['guard'] == 'model_too_large'
    assert result['metric'] is None and result['last_coefficients'] is None
    result = sample_cv(z, np.arange(4.), [0], [[0, 0, 0], [1, 1, 1]], iterations=1)
    assert result['guard'] == 'ok'


@pytest.mark.parametrize('y,condition', [(np.ones(8), 'corr')])
def test_missing_metrics(y, condition):
    result = sample_cv(np.arange(8.)[:, None], y, [0], condition=condition,
                       iterations=1, train_splits=[list(range(6))])
    assert result['metric'] is None
    assert result['status'] == 'missing_metric'


@pytest.mark.parametrize('y,condition', [
    ([0, 1, 2, 3, 4, 5, 6, np.nan], 'corr'),
    ([0, 1, 2, 0, 1, 2, 0, 1], 'auc'),
    (np.ones(8), None),
    (np.ones(8), 'auc'),
])
def test_invalid_sample_cv_responses(y, condition):
    with pytest.raises(ValueError, match='finite|exactly two'):
        sample_cv(np.arange(8.)[:, None], y, [0], condition=condition)


def test_reproducible_local_rng_and_no_marginals():
    z = np.arange(24.).reshape(12, 2)
    state = np.random.get_state()
    a = sample_cv(z, z[:, 0], [0], seed=9, iterations=3)
    b = sample_cv(z, z[:, 0], [0], seed=9, iterations=3)
    assert a == b
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert sample_cv(z, z[:, 0], [], iterations=1)['guard'] == 'no_marginals'


@pytest.mark.parametrize('kwargs', [dict(k=1), dict(k=9), dict(iterations=0),
                                     dict(condition='pearson'), dict(train_splits=[[0]*6], iterations=1)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        sample_cv(np.arange(8.)[:, None], np.arange(8.), [0], **kwargs)


def test_tiny_n_and_optimizer_settings():
    result = sample_cv([[0], [1]], [0, 1], [0], k=2, iterations=2, seed=1)
    assert len(result['predictions']) == 2
    _, params = init_data({'sampleCV_K': 2, 'sampleCV_iter': 3, 'SLIDE_iter': 5},
                          x=np.zeros((8, 2)), y=np.arange(8.))
    assert params['sample_cv_k'] == 8
    assert params['sample_cv_iterations'] == 3
    assert params['niter'] == 5
    assert params['auto_select'] is False


def test_export_and_summary_preserve_candidate_order(tmp_path):
    slider = SLIDE({'y_factor': False}, x=np.zeros((8, 2)), y=np.arange(8.))
    slider.latent_factors = pd.DataFrame(np.arange(8.), columns=['Z0'])
    slider.sig_LFs, slider.sig_interacts = ['Z0'], []
    screening = sample_cv(slider.latent_factors, np.arange(8.), [0], iterations=1,
                          train_splits=[list(range(6))], sample_ids=[f's{i}' for i in range(8)])
    dirs = [tmp_path / '0.2_0.5_out', tmp_path / '0.1_0.5_out']
    for directory, control in zip(dirs, [.9, .1]):
        directory.mkdir()
        slider.save_params(directory, {'real': control, 'full_random': [.5]}, screening)
    slider.create_summary_table(tmp_path, dirs)
    summary = pd.read_csv(tmp_path / 'summary_table.csv')
    assert summary.delta.tolist() == [.2, .1]
    assert 'Control_Performance' in summary
    assert summary.sampleCV_Performance.tolist() == pytest.approx([1, 1])
    model = json.loads((dirs[0] / 'sampleCV_model.json').read_text())
    assert model['last_train_ids'] == [f's{i}' for i in range(6)]
    assert model['settings']['model'] == 'last_split_ols'
    assert pd.read_csv(dirs[0] / 'sampleCV_predictions.csv').sample_id.tolist() == ['s6', 's7']


@pytest.mark.parametrize('auto_select', [False, True])
@pytest.mark.parametrize('has_marginals', [False, True])
def test_optimizer_grid_screening_without_love(tmp_path, monkeypatch, auto_select, has_marginals):
    from slide_py.plotting import Plotter
    from slide_py.score import SLIDE_Estimator

    z = pd.DataFrame({'Z0': np.arange(8.)}, index=[f's{i}' for i in range(8)])
    slider = OptimizeSLIDE(dict(y_factor=False, delta=[.2, .1], **{'lambda': [.5]},
                                out_path=str(tmp_path), do_interacts=False,
                                sampleCV_iter=12, auto_select=auto_select,
                                knockoff_base_seed=5), x=z, y=np.arange(8.))

    def fake_love(**kwargs):
        slider.love_result = {}
        slider.latent_factors = z
        slider.A = pd.DataFrame([[1.]], index=z.columns, columns=z.columns)

    def fake_slide(**kwargs):
        slider.marginal_idxs = [0] if has_marginals else []
        slider.sig_LFs = ['Z0'] if has_marginals else []
        slider.sig_interacts = []
        slider.interaction_pairs = np.empty((2, 0), dtype=int)

    monkeypatch.setattr(slider, 'get_latent_factors', fake_love)
    monkeypatch.setattr(slider, 'run_SLIDE', fake_slide)
    monkeypatch.setattr(SLIDE, 'get_LF_genes', lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(SLIDE_Estimator, 'score_performance',
                        lambda **kwargs: {'real': .1, 'full_random': [.5]})
    stale_selection = tmp_path / 'selected_candidate.json'
    stale_selection.write_text('{"status":"selected","candidate":{"delta":9}}')
    for name in ('plot_latent_factors', 'plot_controlplot', 'plot_corr_network'):
        monkeypatch.setattr(Plotter, name, lambda *args, **kwargs: None)
    slider.run_pipeline(verbose=False)
    summary = pd.read_csv(tmp_path / 'summary_table.csv')
    assert summary.delta.tolist() == [.2, .1]
    if has_marginals:
        assert summary.sampleCV_Performance.between(.9, 1.).all()
        assert summary.sampleCV_Performance.iloc[0] == summary.sampleCV_Performance.iloc[1]
    else:
        assert summary.sampleCV_Performance.isna().all()
        assert summary.sampleCV_Status.tolist() == ['no_marginals', 'no_marginals']
    if auto_select:
        selected = json.loads((tmp_path / 'selected_candidate.json').read_text())
        if has_marginals:
            assert selected['status'] == 'selected'
            assert selected['candidate']['delta'] == .2
        else:
            assert selected['status'] == 'no_scorable_candidate'
            assert selected['candidate'] is None
    else:
        assert slider.selected_candidate is None
        selected = json.loads(stale_selection.read_text())
        assert selected == {'status': 'manual', 'candidate': None}
    model = json.loads((tmp_path / '0.2_0.5_out/sampleCV_model.json').read_text())
    assert model['settings']['k'] == 8
    assert model['settings']['iterations'] == 12
    assert model['settings']['seed'] != 5


def test_load_state_restores_exact_interaction_pairs(tmp_path):
    slider = SLIDE({'y_factor': False}, x=np.zeros((8, 2)), y=np.arange(8.))
    z = pd.DataFrame(np.arange(16.).reshape(8, 2), columns=['Z0', 'Z1'])
    z.to_csv(tmp_path / 'z_matrix.csv')
    pd.DataFrame(np.eye(2), columns=z.columns).to_csv(tmp_path / 'A.csv')
    (tmp_path / 'sig_LFs.txt').write_text('Z0\n')
    (tmp_path / 'sig_interacts.txt').write_text('Z0.Z1\n')
    slider.load_state(tmp_path)
    np.testing.assert_array_equal(slider.interaction_pairs, [[0], [1]])
    assert slider.sig_interacts == ['Z1']
    assert slider.marginal_idxs.tolist() == [0]


def test_rerun_recomputes_screening_from_cached_factors(tmp_path, monkeypatch):
    from slide_py.plotting import Plotter
    from slide_py.score import SLIDE_Estimator

    x = pd.DataFrame({'g0': np.arange(8.), 'g1': np.arange(8.)**2},
                     index=[f's{i}' for i in range(8)])
    slider = OptimizeSLIDE(dict(y_factor=False, delta=[.1], **{'lambda': [.5]},
                                out_path=str(tmp_path), do_interacts=False,
                                sampleCV_iter=12, knockoff_base_seed=5),
                           x=x, y=np.arange(8.))
    cell = tmp_path / '0.1_0.5_out'
    cell.mkdir()
    pd.DataFrame({'Z0': np.arange(8.)}, index=x.index).to_csv(cell / 'z_matrix.csv')
    pd.DataFrame({'Z0': [1., 1.]}, index=x.columns).to_csv(cell / 'A.csv')
    (cell / 'sig_LFs.txt').write_text('Z0\n')
    (cell / 'scores.txt').write_text('True Scores: 0.999\n')
    monkeypatch.setattr(SLIDE, 'get_LF_genes', lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(SLIDE_Estimator, 'score_performance',
                        lambda **kwargs: {'real': .1, 'full_random': [.5]})
    for name in ('plot_latent_factors', 'plot_controlplot', 'plot_corr_network'):
        monkeypatch.setattr(Plotter, name, lambda *args, **kwargs: None)
    slider.run_pipeline(verbose=False, rerun=True)
    assert (cell / 'sampleCV_model.json').exists()
    summary = pd.read_csv(tmp_path / 'summary_table.csv')
    assert summary.Control_Performance.iloc[0] == .1
    assert summary.sampleCV_Performance.iloc[0] > .9


def test_failed_selection_is_reported_and_not_auto_selected(tmp_path, monkeypatch, caplog):
    from slide_py.plotting import Plotter
    from slide_py.score import SLIDE_Estimator

    z = pd.DataFrame({'Z0': np.arange(8.)}, index=[f's{i}' for i in range(8)])
    slider = OptimizeSLIDE(dict(y_factor=False, delta=[.2, .1], **{'lambda': [.5]},
                                out_path=str(tmp_path), do_interacts=False,
                                auto_select=True, sampleCV_iter=12,
                                knockoff_base_seed=5), x=z, y=np.arange(8.))
    calls = iter((False, True))

    def fake_slide(**kwargs):
        if not next(calls):
            raise RuntimeError('controlled selection failure')
        slider.marginal_idxs = [0]
        slider.sig_LFs, slider.sig_interacts = ['Z0'], []
        slider.interaction_pairs = np.empty((2, 0), dtype=int)

    def fake_love(**kwargs):
        slider.love_result = {}
        slider.latent_factors = z
        slider.A = pd.DataFrame([[1.]], index=z.columns, columns=z.columns)

    monkeypatch.setattr(slider, 'get_latent_factors', fake_love)
    monkeypatch.setattr(slider, 'run_SLIDE', fake_slide)
    monkeypatch.setattr(SLIDE, 'get_LF_genes', lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(SLIDE_Estimator, 'score_performance',
                        lambda **kwargs: {'real': .1, 'full_random': [.5]})
    for name in ('plot_latent_factors', 'plot_controlplot', 'plot_corr_network'):
        monkeypatch.setattr(Plotter, name, lambda *args, **kwargs: None)
    slider.run_pipeline(verbose=False)

    summary = pd.read_csv(tmp_path / 'summary_table.csv')
    assert summary.delta.tolist() == [.2, .1]
    assert summary.sampleCV_Status.tolist() == ['selection_failed', 'ok']
    assert pd.isna(summary.sampleCV_Performance.iloc[0])
    assert pd.isna(summary.num_of_Sig_LFs.iloc[0])
    selected = json.loads((tmp_path / 'selected_candidate.json').read_text())
    assert selected['candidate']['delta'] == .1
    assert 'controlled selection failure' in caplog.text
