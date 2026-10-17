"""
Test coverage for slide_py.score module - Estimator and SLIDE_Estimator classes.

Major gaps:
- Estimator model initialization and auto-selection
- Feature scaling edge cases
- Cross-validation scoring accuracy
- SLIDE_Estimator performance scoring
- Error handling for invalid inputs
- Memory management with large datasets
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from unittest.mock import Mock, patch

from slide_py.score import Estimator, SLIDE_Estimator, _mean_valid_scores


def test_mean_valid_scores_ignores_undefined_resamples():
    assert _mean_valid_scores(np.array([0.8, None, 0.6], dtype=object)) == pytest.approx(0.7)


def test_mean_valid_scores_returns_nan_when_no_split_is_evaluable():
    assert np.isnan(_mean_valid_scores(np.array([None, np.nan], dtype=object)))


class TestEstimatorInit:
    """Test Estimator initialization and model setup."""

    def test_estimator_auto_model_classification(self):
        """Test auto model selection for classification data."""
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

        estimator = Estimator(model='auto')
        estimator.fit(X, y)

        # Should automatically select classification model
        assert hasattr(estimator.model, 'predict_proba')

    def test_estimator_auto_model_regression(self):
        """Test auto model selection for regression data."""
        X, y = make_regression(n_samples=100, n_features=20, random_state=42)

        estimator = Estimator(model='auto')
        estimator.fit(X, y)

        # Should automatically select regression model
        assert hasattr(estimator.model, 'predict')


class TestEstimatorScaling:
    """Test feature scaling functionality."""

    def test_estimator_standard_scaling(self):
        """Test standard scaling of features."""
        X = np.random.randn(100, 20) * 100 + 50  # Non-standard scale
        y = np.random.randint(0, 2, 100)

        estimator = Estimator(model='logistic', scaler='standard')
        estimator.fit(X, y)

        # Features should be scaled during fitting
        # Test internal scaling behavior

    def test_estimator_minmax_scaling(self):
        """Test min-max scaling of features."""
        X = np.random.randn(100, 20) * 100 + 50
        y = np.random.randint(0, 2, 100)

        estimator = Estimator(model='logistic', scaler='minmax')
        estimator.fit(X, y)

        # Test that scaling is applied correctly

    def test_estimator_no_scaling(self):
        """Test without feature scaling."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        estimator = Estimator(model='logistic', scaler=None)
        estimator.fit(X, y)

        # Should work without scaling

    def test_estimator_scale_features_static(self):
        """Test static scale_features method."""
        X = np.random.randn(50, 10) * 100 + 50

        # Test different scalers
        X_standard = Estimator.scale_features(X, 'standard')
        X_minmax = Estimator.scale_features(X, 'minmax')

        # Standard scaling should have ~0 mean, ~1 std
        assert np.allclose(np.mean(X_standard, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_standard, axis=0), 1, atol=1e-10)

        # Min-max scaling maps to feature_range, (-1, 1) by default
        assert np.allclose(X_minmax.min(axis=0), -1)
        assert np.allclose(X_minmax.max(axis=0), 1)


    def test_estimator_scale_features_edge_cases(self):
        """Test feature scaling with edge cases."""
        # Constant features
        X_constant = np.ones((50, 3))
        X_scaled = Estimator.scale_features(X_constant, 'standard')
        # Should handle without error (might be NaN or 0)

        # Single sample
        X_single = np.random.randn(1, 5)
        X_scaled_single = Estimator.scale_features(X_single, 'standard')
        # Should handle gracefully


class TestEstimatorPrediction:
    """Test prediction functionality."""

    @pytest.fixture
    def fitted_classifier(self):
        """Create a fitted classifier for testing."""
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        estimator = Estimator(model='random_forest')
        estimator.fit(X, y)
        return estimator, X, y

    @pytest.fixture
    def fitted_regressor(self):
        """Create a fitted regressor for testing."""
        X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        estimator = Estimator(model='random_forest')
        estimator.fit(X, y)
        return estimator, X, y


    def test_estimator_predict_unfitted(self):
        """Test prediction without fitting first."""
        X = np.random.randn(50, 10)

        estimator = Estimator(model='random_forest')

        with pytest.raises((AttributeError, ValueError)):
            estimator.predict(X)


class TestEstimatorEvaluation:
    """Test evaluation functionality."""

    def test_estimator_train_test_split(self):
        """Test train-test split functionality."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        estimator = Estimator()
        (X_train, X_test, y_train, y_test) = estimator.train_test_split(
            X, y, test_size=0.2, seed=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_estimator_score_classification(self):
        """Test scoring for classification."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        estimator = Estimator()
        score = estimator.score(y_pred, y_true)

        # Should return accuracy or similar metric
        assert 0 <= score <= 1

    def test_estimator_score_regression(self):
        """Test scoring for regression."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        estimator = Estimator()
        score = estimator.score(y_pred, y_true)

        # Should return R² or similar regression metric
        assert isinstance(score, (int, float))


    def test_estimator_get_aucs(self):
        """Test AUC computation functionality."""
        X, y = make_classification(n_samples=200, n_features=30, random_state=42)

        estimator = Estimator(model='random_forest')
        aucs = estimator.get_aucs(
            X, y,
            n_iters=3,
            test_size=0.3,
            base_seed=42
        )

        assert len(aucs) == 3
        assert all(0 <= auc <= 1 for auc in aucs)


class TestSLIDEEstimator:
    """Test SLIDE_Estimator extended functionality."""

    @pytest.fixture
    def slide_estimator_setup(self):
        """Create setup for SLIDE_Estimator testing."""
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        estimator = SLIDE_Estimator(model='random_forest')
        return estimator, X, y

    def test_slide_estimator_inheritance(self):
        """Test SLIDE_Estimator inherits from Estimator correctly."""
        slide_est = SLIDE_Estimator()
        assert isinstance(slide_est, Estimator)

        # TODO: Verify specific keys and value ranges


class TestScoreModuleEdgeCases:
    """Test edge cases and error handling."""

    def test_estimator_empty_data(self):
        """Test behavior with empty datasets."""
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])

        estimator = Estimator()

        with pytest.raises((ValueError, IndexError)):
            estimator.fit(X_empty, y_empty)

    def test_estimator_mismatched_dimensions(self):
        """Test error handling for mismatched X, y dimensions."""
        X = np.random.randn(100, 20)
        y = np.random.randn(90)  # Wrong length

        estimator = Estimator()

        with pytest.raises(ValueError):
            estimator.fit(X, y)

    def test_estimator_single_class_data(self):
        """Test behavior with single-class classification data."""
        X = np.random.randn(100, 20)
        y = np.zeros(100)  # All same class

        estimator = Estimator(model='random_forest')

        # Should either handle gracefully or raise informative error
        # TODO: Determine expected behavior


    def test_estimator_nan_data_handling(self):
        """Test handling of NaN values in data."""
        X = np.random.randn(100, 20)
        X[10, 5] = np.nan  # Introduce NaN
        y = np.random.randint(0, 2, 100)

        estimator = Estimator()

        # Should either handle gracefully or raise informative error
        with pytest.raises((ValueError, RuntimeError)):
            estimator.fit(X, y)

    def test_estimator_infinite_data_handling(self):
        """Test handling of infinite values in data."""
        X = np.random.randn(100, 20)
        X[10, 5] = np.inf  # Introduce infinity
        y = np.random.randint(0, 2, 100)

        estimator = Estimator()

        with pytest.raises((ValueError, RuntimeError)):
            estimator.fit(X, y)


class TestScoreModuleIntegration:
    """Integration tests for score module."""

    def test_complete_scoring_workflow(self):
        """Test complete scoring workflow from data to performance metrics."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=200, n_features=50,
            n_informative=20, n_redundant=5,
            random_state=42
        )

        # Initialize estimator
        estimator = Estimator(model='auto', scaler='standard')

        # Fit and predict
        estimator.fit(X, y)
        predictions = estimator.predict(X)

        # Evaluate performance
        scores = estimator.evaluate(X, y, n_iters=3)
        aucs = estimator.get_aucs(X, y, n_iters=3)

        # Verify workflow completed successfully
        assert len(predictions) == len(y)
        assert len(scores) == 3
        assert len(aucs) == 3

        # TODO: Verify specific performance metrics


@pytest.fixture
def control_reference():
    from pathlib import Path
    root = Path(__file__).parents[1] / 'fixtures' / 'control_plot'
    return pd.read_csv(root / 'input.csv'), pd.read_csv(root / 'expected.csv').iloc[0]


def test_control_matches_r_exact_pairs_and_binary_orientation(control_reference):
    from slide_py.score import _control_score
    data, expected = control_reference
    z = data[['Z0', 'Z1', 'Z2']]
    scores = SLIDE_Estimator.score_performance(
        ['Z0', 'Z1'], ['Z2', 'Z0'], data.y, z,
        interaction_pairs=np.array([[0, 1], [2, 0]]), n_iters=3, seed=4)
    assert scores['real'] == pytest.approx(expected['corr'], abs=1e-12)
    auc = _control_score(data[['binary_x']].values, data.binary_y.values, 'auc')
    assert auc == pytest.approx(expected['auc'])
    assert auc < 0.5  # pROC auto direction is not max(AUC, 1-AUC).


def test_control_sampling_preserves_term_counts_and_full_pool(control_reference):
    import slide_py.score as scoring
    data, _ = control_reference
    z = data[['Z0', 'Z1', 'Z2']]
    designs = []
    def capture(x, y, condition):
        designs.append(x.copy())
        return 0.5
    with patch.object(scoring, '_control_score', side_effect=capture), \
         patch.object(Estimator, 'get_aucs', side_effect=AssertionError('No CV')):
        result = SLIDE_Estimator.score_performance(
            ['Z0', 'Z1', 'Z2'], ['Z2', 'Z0'], data.y, z,
            interaction_pairs=np.array([[0, 1], [2, 0]]), n_iters=4, seed=4)
    assert len(designs) == 9
    assert all(x.shape == (12, 5) for x in designs)
    assert len(result['full_random']) == len(result['partial_random']) == 4
    for x in designs[5:]:
        np.testing.assert_equal(x[:, :3], z.values)
    pool = (z.values[:, :, None] * z.values[:, None, :]).reshape(12, -1)
    for x in designs[1:]:
        for column in x[:, 3:].T:
            assert any(np.array_equal(column, candidate) for candidate in pool.T)


def test_control_defaults_and_reproducibility(control_reference):
    data, _ = control_reference
    z = data[['Z0', 'Z1', 'Z2']]
    with patch('slide_py.score._control_score', return_value=0.5):
        scores = SLIDE_Estimator.score_performance(['Z0'], [], data.y, z)
    assert len(scores['full_random']) == 1000
    assert 'partial_random' not in scores
    a = SLIDE_Estimator.score_performance(['Z0'], [], data.y, z, seed=9, n_iters=5)
    b = SLIDE_Estimator.score_performance(['Z0'], [], data.y, z, seed=9, n_iters=5)
    np.testing.assert_equal(a['full_random'], b['full_random'])
    with pytest.raises(ValueError, match='Exact interaction_pairs'):
        SLIDE_Estimator.score_performance(['Z0'], ['Z1'], data.y, z)
    with pytest.raises(ValueError, match='selected marginal'):
        SLIDE_Estimator.score_performance([], [], data.y, z)
    with pytest.raises(ValueError, match='nonconstant'):
        SLIDE_Estimator.score_performance(['Z0'], [], np.ones(len(z)), z)


def test_control_rejects_overflowed_features_predictions_and_scores():
    from slide_py.score import _control_score
    z = pd.DataFrame({'Z0': [1e200, 2e200, 3e200]})
    with pytest.raises(ValueError, match='features must be finite'):
        SLIDE_Estimator.score_performance(
            ['Z0'], ['Z0'], np.arange(3.), z,
            interaction_pairs=np.array([[0], [0]]), n_iters=1)
    with patch('numpy.linalg.lstsq', return_value=(np.array([np.inf, 1.]),)):
        with pytest.raises(ValueError, match='predictions must be finite'):
            _control_score(np.arange(3.)[:, None], np.arange(3.), 'corr')
    with pytest.raises(ValueError, match='score must be finite'):
        _control_score(np.arange(3.)[:, None], np.arange(3.) * 1e200, 'corr')


@pytest.mark.parametrize('selected,do_interacts', [([], True), ([0], False), ([0], True)])
def test_selection_exports_empty_lists_and_clears_stale_pairs(tmp_path, selected, do_interacts):
    from slide_py.slide import OptimizeSLIDE
    slider = OptimizeSLIDE.__new__(OptimizeSLIDE)
    slider.marginal_idxs = selected
    slider.sig_interacts = ['Z1']
    slider.interaction_pairs = np.array([[0], [1]])
    slider.interaction_terms = np.ones((3, 1))
    slider.calc_default_fsize = lambda count: 2
    slider.find_standalone_LFs = Mock(return_value=None)
    slider.find_interaction_LFs = Mock(return_value=None)
    z = pd.DataFrame({'Z0': [1, 2, 3], 'Z1': [3, 2, 1]})
    slider.run_SLIDE(z, niter=1, spec=.1, fdr=.1, outpath=tmp_path, do_interacts=do_interacts)
    assert (tmp_path / 'sig_LFs.txt').read_text() == ('Z0\n' if selected else '')
    assert (tmp_path / 'sig_interacts.txt').read_text() == ''
    assert slider.sig_interacts == []
    assert slider.interaction_pairs.shape == (2, 0)
    assert slider.interaction_terms.shape == (3, 0)
