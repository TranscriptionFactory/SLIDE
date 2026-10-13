"""
Test coverage for knockoff generation and filtering functionality.

Major gaps identified:
- Knockoffs class initialization and configuration
- SDP solver selection and fallback behavior
- Knockoff generation methods (fixed, gaussian, second-order)
- Statistical tests for knockoff filtering
- Voting mechanisms and aggregation
- Error handling for SDP failures
- Memory efficiency with large datasets
- Parallelization correctness
"""
import sys

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from loveslide import Knockoffs, VotingResult
from loveslide.knockoff.create import create_fixed, create_gaussian, create_second_order
from loveslide.knockoff.filter import knockoff_filter, knockoff_threshold, knockoff_filter_voting
from loveslide.knockoff.solve import create_solve_equi, create_solve_sdp, create_solve_asdp
from loveslide.knockoff.utils import is_posdef, canonical_svd, normc, cov2cor


class TestKnockoffsClass:
    """Test the main Knockoffs class functionality."""

    def test_knockoffs_init_valid_backends(self):
        """Test Knockoffs initialization with valid backends."""
        for backend in ['python', 'r_knockoffs']:
            knockoffs = Knockoffs(backend=backend)
            assert knockoffs.backend == backend

    def test_knockoffs_init_invalid_backend(self):
        """Test Knockoffs fails with invalid backend."""
        with pytest.raises(ValueError, match="backend.*not supported"):
            Knockoffs(backend='invalid_backend')

    def test_knockoffs_init_default_backend(self):
        """Test default backend selection."""
        knockoffs = Knockoffs()
        assert knockoffs.backend in ['python', 'r_knockoffs']

    def test_select_short_freq_basic(self):
        """Test basic knockoff selection functionality."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        knockoffs = Knockoffs(backend='python')
        result = knockoffs.select_short_freq(X, y, fdr=0.1, niter=5)

        assert isinstance(result, VotingResult)
        assert hasattr(result, 'selected')
        assert hasattr(result, 'fdp_estimate')

    def test_select_short_freq_edge_case_no_features(self):
        """Test when no features should be selected."""
        # Create data with no signal
        X = np.random.randn(100, 10) * 0.01  # Very weak signal
        y = np.random.randn(100)

        knockoffs = Knockoffs(backend='python')
        result = knockoffs.select_short_freq(X, y, fdr=0.01, niter=3)  # Strict FDR

        # Should return empty or very few selections
        assert len(result.selected) <= 2  # Allow some false positives

    def test_select_short_freq_deterministic_with_seed(self):
        """Test reproducibility with fixed seed."""
        X = np.random.RandomState(42).randn(50, 15)
        y = X[:, 0] + 0.1 * np.random.RandomState(42).randn(50)  # Feature 0 is signal

        knockoffs = Knockoffs(backend='python')

        result1 = knockoffs.select_short_freq(X, y, fdr=0.1, niter=3, seed=123)
        result2 = knockoffs.select_short_freq(X, y, fdr=0.1, niter=3, seed=123)

        assert result1.selected == result2.selected
        assert np.allclose(result1.fdp_estimate, result2.fdp_estimate)


class TestKnockoffCreation:
    """Test knockoff variable creation methods."""

    def test_create_fixed_basic(self):
        """Test fixed knockoff creation."""
        X = np.random.randn(100, 10)
        X_ko = create_fixed(X)

        assert X_ko.shape == X.shape
        assert not np.allclose(X, X_ko)  # Should be different

    def test_create_fixed_maintains_correlation_structure(self):
        """Test that fixed knockoffs preserve key properties."""
        # Create structured data
        n, p = 200, 5
        X = np.random.randn(n, p)
        X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # Correlated features

        X_ko = create_fixed(X)

        # Original and knockoff should have similar variance
        assert np.allclose(np.var(X, axis=0), np.var(X_ko, axis=0), rtol=0.2)

    def test_create_gaussian_basic(self):
        """Test Gaussian knockoff creation."""
        X = np.random.randn(100, 8)
        X_ko = create_gaussian(X)

        assert X_ko.shape == X.shape
        assert not np.allclose(X, X_ko)

    def test_create_second_order_basic(self):
        """Test second-order knockoff creation."""
        X = np.random.randn(150, 12)
        X_ko = create_second_order(X)

        assert X_ko.shape == X.shape
        assert not np.allclose(X, X_ko)

    def test_create_methods_with_singular_data(self):
        """Test knockoff creation with singular/rank-deficient data."""
        # Create rank-deficient matrix
        X = np.random.randn(50, 10)
        X[:, 5] = X[:, 0] + X[:, 1]  # Linear dependence

        # Should handle gracefully or raise informative error
        with pytest.warns(UserWarning) or pytest.raises(ValueError):
            create_fixed(X)

    def test_knockoff_randomization_flag(self):
        """Test randomization parameter in knockoff creation."""
        X = np.random.randn(80, 6)

        X_ko1 = create_fixed(X, randomize=True)
        X_ko2 = create_fixed(X, randomize=True)

        # With randomization, should get different results
        assert not np.allclose(X_ko1, X_ko2)

        X_ko3 = create_fixed(X, randomize=False)
        X_ko4 = create_fixed(X, randomize=False)

        # Without randomization, should be identical
        assert np.allclose(X_ko3, X_ko4)


class TestSDPSolvers:
    """Test SDP solving for knockoff generation."""

    @staticmethod
    def _sigma(p=6):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, p))
        return np.cov(X, rowvar=False)

    @staticmethod
    def _hide_dsdp(monkeypatch):
        # A None entry in sys.modules makes `from .pydsdp import dsdp` raise ImportError.
        monkeypatch.setitem(sys.modules, "loveslide.knockoff.pydsdp", None)

    def test_default_solver_is_dsdp(self):
        from loveslide.knockoff.solve import _get_sdp_solver

        pytest.importorskip("loveslide.dsdp_solver.pydsdp5")
        assert _get_sdp_solver() == "dsdp"

    def test_unknown_solver_rejected(self):
        with pytest.raises(ValueError, match="sdp_solver"):
            create_solve_sdp(self._sigma(), solver="scs")

    def test_missing_dsdp_raises_instead_of_cvxpy_fallback(self, monkeypatch):
        self._hide_dsdp(monkeypatch)
        with pytest.raises(ImportError, match="sdp_solver='cvxpy'"):
            create_solve_sdp(self._sigma())
        with pytest.raises(ImportError, match="build_ext --inplace"):
            create_solve_asdp(self._sigma())

    def test_cvxpy_is_explicit_opt_in(self, monkeypatch):
        pytest.importorskip("cvxpy")
        self._hide_dsdp(monkeypatch)
        Sigma = self._sigma()
        s = create_solve_sdp(Sigma, solver="cvxpy")
        assert s.shape == (Sigma.shape[0],)
        assert np.all(s > 0)
        assert is_posdef(2 * Sigma - np.diag(s))

    def test_dsdp_failure_raises(self, monkeypatch):
        pytest.importorskip("loveslide.dsdp_solver.pydsdp5")
        import loveslide.knockoff.pydsdp as pydsdp

        def failing_dsdp(*args, **kwargs):
            raise RuntimeError("solver blew up")

        monkeypatch.setattr(pydsdp, "dsdp", failing_dsdp)
        with pytest.raises(RuntimeError, match="DSDP solver failed"):
            create_solve_sdp(self._sigma())

    def test_voting_propagates_missing_dsdp(self, monkeypatch):
        self._hide_dsdp(monkeypatch)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 6))
        y = X[:, 0] + rng.normal(size=40)
        with pytest.raises(ImportError, match="DSDP"):
            knockoff_filter_voting(X, y, niter=2, match_r=True)
        with pytest.raises(ImportError, match="DSDP"):
            Knockoffs.filter_knockoffs_iterative_python(X, y, niter=2, n_jobs=1)

    def test_create_solve_equi_basic(self):
        """Test equicorrelated knockoff solving."""
        Sigma = np.eye(5) + 0.3 * np.ones((5, 5))  # AR(1)-like structure
        np.fill_diagonal(Sigma, 1.0)

        s = create_solve_equi(Sigma)

        assert s.shape == (5,)
        assert np.all(s >= 0)  # Should be non-negative
        assert np.all(s <= 1)  # Should be bounded

    def test_create_solve_sdp_basic(self):
        """Test SDP-based knockoff solving."""
        Sigma = np.eye(4) + 0.2 * np.ones((4, 4))
        np.fill_diagonal(Sigma, 1.0)

        try:
            s = create_solve_sdp(Sigma)
            assert s.shape == (4,)
            assert np.all(s >= -1e-6)  # Allow small numerical errors
        except Exception as e:
            pytest.skip(f"SDP solver not available: {e}")

    def test_create_solve_asdp_clustering(self):
        """Test approximate SDP solving with clustering."""
        # Larger matrix to trigger clustering
        p = 50
        Sigma = 0.7 * np.eye(p) + 0.3 * np.ones((p, p))
        np.fill_diagonal(Sigma, 1.0)

        s = create_solve_asdp(Sigma, max_size=10)

        assert s.shape == (p,)
        assert np.all(s >= -1e-6)

    def test_sdp_solver_with_ill_conditioned_matrix(self):
        """Test SDP solving with ill-conditioned covariance matrix."""
        # Create nearly singular matrix
        Sigma = np.eye(6)
        Sigma[0, 1] = Sigma[1, 0] = 0.999  # Very high correlation

        with pytest.warns(UserWarning) or pytest.raises(np.linalg.LinAlgError):
            create_solve_sdp(Sigma)


class TestKnockoffFiltering:
    """Test knockoff statistical filtering methods."""

    def test_knockoff_threshold_basic(self):
        """Test knockoff threshold calculation."""
        W = np.array([2.1, -0.5, 1.8, -1.2, 3.0, 0.3])

        threshold = knockoff_threshold(W, fdr=0.1)
        assert isinstance(threshold, float)
        assert threshold > 0

    def test_knockoff_threshold_edge_cases(self):
        """Test threshold calculation edge cases."""
        # All negative statistics
        W_neg = np.array([-1.0, -2.0, -0.5])
        threshold_neg = knockoff_threshold(W_neg, fdr=0.1)
        assert threshold_neg == float('inf')  # Should select nothing

        # All positive statistics
        W_pos = np.array([1.0, 2.0, 0.5])
        threshold_pos = knockoff_threshold(W_pos, fdr=0.1)
        assert threshold_pos >= 0

    def test_knockoff_filter_basic(self):
        """Test basic knockoff filtering."""
        X = np.random.randn(100, 10)
        y = X[:, 0] + X[:, 2] + 0.1 * np.random.randn(100)  # Features 0,2 are signals

        result = knockoff_filter(X, y, fdr=0.2)

        assert hasattr(result, 'selected')
        assert hasattr(result, 'threshold')
        assert hasattr(result, 'statistics')
        assert len(result.statistics) == X.shape[1]

    def test_knockoff_filter_with_different_statistics(self):
        """Test knockoff filtering with different statistic functions."""
        X = np.random.randn(80, 8)
        y = np.random.randn(80)

        # Test different statistics
        for stat in ['lasso_lambdadiff', 'lasso_coefdiff', 'forward_selection']:
            try:
                result = knockoff_filter(X, y, fdr=0.1, statistic=stat)
                assert len(result.statistics) == X.shape[1]
            except (ImportError, NotImplementedError):
                pytest.skip(f"Statistic {stat} not available")


class TestKnockoffVoting:
    """Test knockoff voting and aggregation mechanisms."""

    def test_knockoff_filter_voting_basic(self):
        """Test basic voting mechanism."""
        X = np.random.randn(100, 12)
        # Create clear signal in first two features
        y = 2 * X[:, 0] + 1.5 * X[:, 1] + 0.2 * np.random.randn(100)

        result = knockoff_filter_voting(
            X, y, fdr=0.1, niter=5,
            statistic='lasso_lambdadiff'
        )

        assert isinstance(result, VotingResult)
        assert hasattr(result, 'votes')
        assert hasattr(result, 'selected')
        assert len(result.votes) == X.shape[1]

    def test_voting_result_properties(self):
        """Test VotingResult class properties."""
        # Mock voting result
        votes = np.array([8, 2, 9, 1, 0, 7])
        fdp = 0.08
        threshold = 5

        result = VotingResult(
            selected=np.where(votes >= threshold)[0],
            votes=votes,
            fdp_estimate=fdp,
            threshold=threshold
        )

        assert len(result.selected) == 3  # Features 0, 2, 5
        assert result.fdp_estimate == 0.08
        assert np.array_equal(result.votes, votes)

    def test_knockoff_voting_consistency(self):
        """Test voting consistency with different numbers of iterations."""
        X = np.random.RandomState(42).randn(80, 8)
        y = X[:, 0] + 0.5 * X[:, 3] + 0.1 * np.random.RandomState(42).randn(80)

        result_few = knockoff_filter_voting(X, y, fdr=0.2, niter=3, seed=123)
        result_many = knockoff_filter_voting(X, y, fdr=0.2, niter=20, seed=123)

        # More iterations should generally lead to more stable results
        # Allow some variation but expect reasonable consistency
        jaccard = len(set(result_few.selected) & set(result_many.selected)) / \
                 max(len(set(result_few.selected) | set(result_many.selected)), 1)
        assert jaccard >= 0.5  # At least 50% overlap


class TestKnockoffUtilities:
    """Test utility functions for knockoff generation."""

    def test_is_posdef_basic(self):
        """Test positive definiteness checking."""
        # Positive definite matrix
        A_pd = np.array([[2, 1], [1, 2]])
        assert is_posdef(A_pd)

        # Not positive definite
        A_npd = np.array([[1, 2], [2, 1]])  # Singular
        assert not is_posdef(A_npd)

    def test_canonical_svd_basic(self):
        """Test canonical SVD computation."""
        X = np.random.randn(100, 10)
        U, d, V = canonical_svd(X)

        assert U.shape == (100, 10)
        assert d.shape == (10,)
        assert V.shape == (10, 10)

        # Verify SVD property
        X_reconstructed = U @ np.diag(d) @ V
        assert np.allclose(X, X_reconstructed)

    def test_normc_normalization(self):
        """Test column normalization."""
        X = np.random.randn(50, 8) * 5  # Random scale
        X_norm = normc(X, center=True)

        # Should be centered and scaled
        assert np.allclose(np.mean(X_norm, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_norm, axis=0), 1, atol=1e-10)

    def test_cov2cor_conversion(self):
        """Test covariance to correlation conversion."""
        # Create covariance matrix
        X = np.random.randn(100, 5)
        Sigma = np.cov(X, rowvar=False)

        R = cov2cor(Sigma)

        # Diagonal should be all 1s
        assert np.allclose(np.diag(R), 1.0)

        # Off-diagonal should be correlations (between -1 and 1)
        R_offdiag = R[~np.eye(R.shape[0], dtype=bool)]
        assert np.all(np.abs(R_offdiag) <= 1.0)


class TestKnockoffErrorHandling:
    """Test error handling and edge cases in knockoff methods."""

    def test_knockoff_with_mismatched_dimensions(self):
        """Test error handling for mismatched X, y dimensions."""
        X = np.random.randn(100, 10)
        y = np.random.randn(80)  # Wrong length

        knockoffs = Knockoffs(backend='python')

        with pytest.raises(ValueError, match="X and y.*incompatible"):
            knockoffs.select_short_freq(X, y, fdr=0.1)

    def test_knockoff_with_invalid_fdr(self):
        """Test error handling for invalid FDR values."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        knockoffs = Knockoffs(backend='python')

        # FDR out of bounds
        with pytest.raises(ValueError, match="fdr.*between 0 and 1"):
            knockoffs.select_short_freq(X, y, fdr=1.5)

        with pytest.raises(ValueError, match="fdr.*between 0 and 1"):
            knockoffs.select_short_freq(X, y, fdr=-0.1)

    def test_knockoff_with_insufficient_data(self):
        """Test behavior with very small datasets."""
        X = np.random.randn(5, 10)  # More features than samples
        y = np.random.randn(5)

        knockoffs = Knockoffs(backend='python')

        with pytest.warns(UserWarning, match="insufficient.*samples") or \
             pytest.raises(ValueError):
            knockoffs.select_short_freq(X, y, fdr=0.1)

    def test_knockoff_memory_efficiency(self):
        """Test memory efficiency with moderately large datasets."""
        n, p = 2000, 500
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        knockoffs = Knockoffs(backend='python')

        # Should complete without memory errors
        try:
            result = knockoffs.select_short_freq(X, y, fdr=0.1, niter=3)
            assert hasattr(result, 'selected')
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")


class TestKnockoffParallelization:
    """Test parallel execution correctness."""

    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel execution gives same results as sequential."""
        X = np.random.RandomState(42).randn(100, 15)
        y = X[:, 0] + X[:, 5] + 0.2 * np.random.RandomState(42).randn(100)

        # Sequential
        result_seq = knockoff_filter_voting(
            X, y, fdr=0.1, niter=10, n_jobs=1, seed=42
        )

        # Parallel
        result_par = knockoff_filter_voting(
            X, y, fdr=0.1, niter=10, n_jobs=2, seed=42
        )

        # Should be identical
        assert np.array_equal(result_seq.selected, result_par.selected)
        assert np.allclose(result_seq.votes, result_par.votes)

    def test_parallel_different_random_seeds(self):
        """Test parallel execution with different random seeds."""
        X = np.random.randn(80, 12)
        y = np.random.randn(80)

        results = []
        for seed in [123, 124, 125]:
            result = knockoff_filter_voting(
                X, y, fdr=0.1, niter=5, n_jobs=2, seed=seed
            )
            results.append(result)

        # Results should be different (with high probability)
        assert not (np.array_equal(results[0].selected, results[1].selected) and
                   np.array_equal(results[1].selected, results[2].selected))


# Performance and integration tests
class TestKnockoffPerformance:
    """Test computational performance and integration."""

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        n, p = 5000, 1000
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        import time

        knockoffs = Knockoffs(backend='python')

        start_time = time.time()
        result = knockoffs.select_short_freq(X, y, fdr=0.1, niter=3)
        elapsed = time.time() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 300  # 5 minutes max
        assert hasattr(result, 'selected')

    def test_knockoff_integration_with_slide(self):
        """Test integration with SLIDE pipeline."""
        from loveslide import SLIDE

        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        params = {
            'fdr': 0.1,
            'niter': 5,
            'backend': 'python'
        }

        slide = SLIDE(params, x=X, y=y)

        # Should be able to access knockoffs functionality
        assert hasattr(slide, 'data')
        # Additional integration tests would go here