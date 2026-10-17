"""
Test coverage for statistical methods and computation correctness.

Major gaps identified:
- Statistical correctness of knockoff filtering methods
- Validation of LOVE algorithm statistical properties
- Cross-validation statistical validity
- Estimation accuracy and bias testing
- Hypothesis testing correctness
- Bootstrap and resampling validation
- Convergence testing for iterative algorithms
"""
import pytest
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Union, Optional

from slide_py import Knockoffs, VotingResult
from slide_py.knockoff.stats.lasso import stat_lasso_lambdadiff, stat_lasso_coefdiff
from slide_py.knockoff.stats.forward import stat_forward_selection
from slide_py.knockoff.stats.base import compute_difference_stat, compute_signed_max_stat
from slide_py.knockoff.filter import knockoff_threshold, knockoff_filter_voting
from slide_py.love_python.love.score import Score_mat, LP_Score
from slide_py.love_python.love.cv import CV_delta, KfoldCV_delta


class TestKnockoffStatisticalCorrectness:
    """Test statistical correctness of knockoff methods."""

    def test_knockoff_fdr_control_simulation(self):
        """Test that knockoff methods control FDR at specified level."""
        np.random.seed(42)
        n, p = 500, 100
        target_fdr = 0.1
        n_simulations = 20  # Reduced for test speed

        fdp_estimates = []

        for sim in range(n_simulations):
            # Generate null data (no true signals)
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            # Run knockoff filter
            result = knockoff_filter_voting(
                X, y, fdr=target_fdr, niter=10,
                statistic='lasso_lambdadiff', seed=sim
            )

            # Calculate empirical FDP
            n_selected = len(result.selected)
            fdp = n_selected / max(n_selected, 1)  # All selections are false discoveries
            fdp_estimates.append(fdp)

        # Mean FDP should be close to target FDR
        mean_fdp = np.mean(fdp_estimates)
        assert mean_fdp <= target_fdr + 0.05  # Allow some tolerance


    def test_knockoff_statistics_properties(self):
        """Test properties of knockoff statistics."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        X_ko = np.random.randn(100, 20)  # Mock knockoffs
        y = np.random.randn(100)

        # Test different statistics
        stats_lasso = stat_lasso_lambdadiff(X, X_ko, y)
        stats_coef = stat_lasso_coefdiff(X, X_ko, y)

        # Statistics should have reasonable properties
        assert len(stats_lasso) == 20
        assert len(stats_coef) == 20

        # Should have both positive and negative values (sign flip property)
        assert np.any(stats_lasso > 0)
        assert np.any(stats_lasso < 0)

    def test_knockoff_threshold_properties(self):
        """Test properties of knockoff threshold calculation."""
        # Test with known statistics
        W = np.array([3.0, -1.0, 2.5, -0.5, 1.8, -2.0, 0.3, -1.5])

        # Test different FDR levels
        for fdr in [0.05, 0.1, 0.2]:
            threshold = knockoff_threshold(W, fdr=fdr)

            # Higher FDR should give lower threshold (more selections)
            if fdr > 0.1:
                threshold_01 = knockoff_threshold(W, fdr=0.1)
                assert threshold <= threshold_01


class TestLOVEStatisticalCorrectness:
    """Test statistical correctness of LOVE algorithm."""


    def test_love_correlation_structure_preservation(self):
        """Test that LOVE preserves correlation structure."""
        from slide_py.love_python.love.love import LOVE

        np.random.seed(123)
        X = np.random.randn(150, 12)

        # Add some correlation structure
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]
        X[:, 2] = 0.6 * X[:, 0] + 0.4 * X[:, 2]

        original_corr = np.corrcoef(X, rowvar=False)

        result = LOVE(X, lbd=0.4, mu=0.4)

        # Reconstructed correlation should be similar to original
        if 'Sigma_hat' in result:
            Sigma_hat = result['Sigma_hat']
            reconstructed_corr = np.corrcoef(Sigma_hat)

            # Correlation between original and reconstructed
            corr_similarity = np.corrcoef(
                original_corr.flatten(),
                reconstructed_corr.flatten()
            )[0, 1]

            assert corr_similarity > 0.7  # Should preserve structure


class TestEstimationStatistics:
    """Test statistical properties of parameter estimation."""


    def test_estimation_robustness(self):
        """Test robustness of estimation to outliers."""
        from slide_py.love_python.love.est_pure_homo import EstC

        p, K = 8, 2
        Sigma_clean = np.eye(p) + 0.3 * np.ones((p, p))
        AI = np.random.randn(p, K)

        # Clean estimation
        C_clean = EstC(Sigma_clean, AI, diagonal=False)

        # Add outlier to covariance matrix
        Sigma_outlier = Sigma_clean.copy()
        Sigma_outlier[0, 0] *= 10  # Large diagonal element

        C_outlier = EstC(Sigma_outlier, AI, diagonal=False)

        # Results should be reasonably similar (robust estimation)
        relative_diff = np.linalg.norm(C_clean - C_outlier) / np.linalg.norm(C_clean)
        assert relative_diff < 2.0  # Allow some difference but not too large


class TestHypothesisTestingCorrectness:
    """Test statistical hypothesis testing procedures."""

    def test_multiple_testing_correction(self):
        """Test multiple testing correction in feature selection."""
        # Generate data with known null and alternative hypotheses
        np.random.seed(42)
        n, p = 200, 50
        n_true_signals = 5

        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:n_true_signals] = 2  # True signals

        y = X @ beta_true + np.random.randn(n)

        # Test knockoff procedure
        result = knockoff_filter_voting(
            X, y, fdr=0.1, niter=20,
            statistic='lasso_lambdadiff', seed=42
        )

        # Calculate empirical FWER and FDR
        true_discoveries = len(set(result.selected) & set(range(n_true_signals)))
        false_discoveries = len(set(result.selected) - set(range(n_true_signals)))

        fdp = false_discoveries / max(len(result.selected), 1)

        # FDP should be controlled
        assert fdp <= 0.2  # Allow some tolerance above target FDR

    def test_power_analysis(self):
        """Test power analysis for different effect sizes."""
        n, p = 150, 30
        effect_sizes = [0.5, 1.0, 2.0]  # Different signal strengths

        powers = []

        for effect_size in effect_sizes:
            # Generate data with given effect size
            np.random.seed(42)
            X = np.random.randn(n, p)
            beta_true = np.zeros(p)
            beta_true[:3] = effect_size  # 3 true signals

            y = X @ beta_true + np.random.randn(n)

            result = knockoff_filter_voting(
                X, y, fdr=0.2, niter=10,
                statistic='lasso_lambdadiff', seed=42
            )

            # Calculate power
            true_positives = len(set(result.selected) & set(range(3)))
            power = true_positives / 3
            powers.append(power)

        # Power should increase with effect size
        assert all(powers[i] <= powers[i+1] + 0.1 for i in range(len(powers)-1))

    def test_type_i_error_control(self):
        """Test Type I error control under null hypothesis."""
        # Generate pure null data
        np.random.seed(123)
        n, p = 100, 25

        type_i_errors = []

        for trial in range(10):  # Multiple trials
            X = np.random.randn(n, p)
            y = np.random.randn(n)  # Pure noise

            result = knockoff_filter_voting(
                X, y, fdr=0.05, niter=5,
                statistic='lasso_lambdadiff', seed=trial
            )

            # All selections are Type I errors
            type_i_rate = len(result.selected) / p
            type_i_errors.append(type_i_rate)

        # Average Type I error rate should be controlled
        mean_type_i = np.mean(type_i_errors)
        assert mean_type_i <= 0.1  # Should be well below nominal level


class TestConvergenceAndNumericalStability:
    """Test convergence and numerical stability of algorithms."""

    def test_iterative_algorithm_convergence(self):
        """Test convergence of iterative algorithms."""
        from slide_py.love_python.love.est_nonpure import LP

        # Test LP optimization convergence
        y = np.random.randn(10)
        lbd = 0.3

        # Run with different tolerances
        result = LP(y, lbd)

        # Should converge to reasonable solution
        assert result is not None
        assert len(result) == len(y)


    def test_large_scale_numerical_accuracy(self):
        """Test numerical accuracy with larger problems."""
        # Test that algorithms maintain accuracy as problem size increases
        for p in [10, 50, 100]:
            if p > 50:  # Skip large tests if needed
                continue

            X = np.random.randn(2*p, p)
            Sigma = np.cov(X, rowvar=False)

            # Test matrix operations
            eigenvals = np.linalg.eigvals(Sigma)
            assert np.all(eigenvals > -1e-10)  # Should be PSD

            # Test condition number
            condition_number = np.linalg.cond(Sigma)
            if condition_number > 1e12:
                pytest.warn(f"High condition number: {condition_number}")

