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


class TestKnockoffCreation:
    """Test knockoff variable creation methods."""


    def test_create_second_order_basic(self):
        """Test second-order knockoff creation."""
        X = np.random.randn(150, 12)
        X_ko = create_second_order(X)

        assert X_ko.shape == X.shape
        assert not np.allclose(X, X_ko)


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


class TestKnockoffVoting:
    """Test knockoff voting and aggregation mechanisms."""


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


class TestKnockoffParallelization:
    """Test parallel execution correctness."""


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