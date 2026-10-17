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
