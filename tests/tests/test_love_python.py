"""
Test coverage for LOVE Python implementation.

Major gaps identified:
- Core LOVE algorithm (love.py)
- Estimation functions (est_pure_homo.py, est_pure_hetero.py, est_nonpure.py)
- Cross-validation functionality (cv.py)
- Utility functions (utilities.py)
- Preprocessing and screening (prescreen.py)
- Scoring matrices (score.py)
- Omega estimation (est_omega.py)
"""
import pytest
import numpy as np
from typing import Dict, List, Union, Optional

from slide_py.love_python.love.love import LOVE
from slide_py.love_python.love.prescreen import Screen_X
from slide_py.love_python.love.score import Score_mat, LP_Score
from slide_py.love_python.love.utilities import (
    recoverGroup, singleton, threshA, offSum, partition, extract
)
from slide_py.love_python.love.cv import (
    CV_delta, CalFittedSigma, KfoldCV_delta, CV_lbd
)
from slide_py.love_python.love.est_pure_homo import (
    EstAI, EstC, FindRowMax, FindPureNode, TestPure, FindSignPureNode
)
from slide_py.love_python.love.est_pure_hetero import (
    Est_Pure, Est_BI_C, Re_Est_Pure, Post_Est_Pure, Est_K
)
from slide_py.love_python.love.est_nonpure import (
    EstY, EstAJInv, LP, EstAJDant, Dantzig
)
from slide_py.love_python.love.est_omega import estOmega, solve_row


class TestLOVECore:
    """Test the core LOVE algorithm."""


    def test_love_edge_case_small_dataset(self):
        """Test LOVE with very small dataset."""
        X = np.random.randn(10, 5)  # Very small

        with pytest.warns(UserWarning) or pytest.raises(ValueError):
            LOVE(X, lbd=0.5, mu=0.5)


class TestLOVEPreprocessing:
    """Test LOVE preprocessing and screening functionality."""


    def test_screen_x_with_default_grid(self):
        """Test screening with default threshold grid."""
        X = np.random.randn(80, 15)

        result = Screen_X(X)  # Use default grid
        assert isinstance(result, dict)

    def test_screen_x_edge_cases(self):
        """Test screening edge cases."""
        # All identical features
        X = np.ones((50, 10))

        with pytest.warns(UserWarning) or pytest.raises(ValueError):
            Screen_X(X)

    def test_screen_x_with_structured_data(self):
        """Test screening with data containing clear structure."""
        n = 200

        # Create data with clear factor structure
        Z1 = np.random.randn(n)
        Z2 = np.random.randn(n)

        X = np.column_stack([
            Z1 + 0.1 * np.random.randn(n),      # Factor 1
            -Z1 + 0.1 * np.random.randn(n),     # Factor 1 (negative loading)
            Z2 + 0.1 * np.random.randn(n),      # Factor 2
            Z2 + 0.1 * np.random.randn(n),      # Factor 2
            np.random.randn(n),                 # Noise
            np.random.randn(n)                  # Noise
        ])

        result = Screen_X(X, thresh_grid=np.array([0.1, 0.5]))
        assert isinstance(result, dict)


    @pytest.mark.parametrize("n,p", [(8, 30), (24, 60), (60, 45)])
    def test_fdr_threshold_matches_statsmodels_bh_bitwise(self, n, p):
        """Cutoff-based FDR filter keeps exactly what full BH adjustment keeps."""
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
        from slide_py.love_python.love.love import _apply_fdr_threshold

        rng = np.random.default_rng(7)
        factors = rng.uniform(-1, 1, (n, 3))
        cols = np.arange(p - 1)
        X = factors[:, cols % 3] * (cols % 4) + rng.uniform(-1, 1, (n, p - 1))
        X = np.column_stack([X, np.ones(n)])  # constant column gives exact zeros
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.nan_to_num(np.corrcoef(X, rowvar=False))

        for thresh in [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]:
            r_clipped = np.clip(R, -0.9999, 0.9999)
            t_stats = r_clipped * np.sqrt(n - 2) / np.sqrt(1 - r_clipped**2)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - 2))
            _, p_adjusted, _, _ = multipletests(p_values.flatten(), alpha=thresh, method='fdr_bh')
            expected = R * (p_adjusted.reshape(p, p) <= thresh)

            actual = _apply_fdr_threshold(R, n, thresh)
            assert np.array_equal(expected.view(np.uint64), actual.view(np.uint64)), thresh


class TestLOVEScoring:
    """Test LOVE scoring and statistics."""

    def test_score_mat_basic(self):
        """Test basic score matrix computation."""
        # Create correlation matrix
        p = 6
        R = np.random.randn(p, p)
        R = R @ R.T  # Make positive semidefinite
        R = R / np.sqrt(np.diag(R)[:, None] @ np.diag(R)[None, :])  # Normalize

        result = Score_mat(R, q=2)

        assert isinstance(result, dict)
        assert 'score' in result

    def test_score_mat_different_q_values(self):
        """Test score matrix with different q parameters."""
        R = np.corrcoef(np.random.randn(50, 8), rowvar=False)

        for q in [1, 2, 3]:
            result = Score_mat(R, q=q, exact=False)
            assert isinstance(result, dict)

    def test_score_mat_exact_vs_approximate(self):
        """Test exact vs approximate score computation."""
        R = np.corrcoef(np.random.randn(30, 5), rowvar=False)

        result_exact = Score_mat(R, q=2, exact=True)
        result_approx = Score_mat(R, q=2, exact=False)

        assert isinstance(result_exact, dict)
        assert isinstance(result_approx, dict)


class TestLOVEUtilities:
    """Test LOVE utility functions."""

    def test_recover_group_basic(self):
        """Test group recovery from loading matrix."""
        # Simple 2-factor loading matrix
        A = np.array([
            [1, 0],
            [0.8, 0],
            [0, 1],
            [0, 0.9],
            [0.5, 0.3]  # Mixed loading
        ])

        groups = recoverGroup(A)

        assert isinstance(groups, list)
        assert len(groups) > 0
        for group in groups:
            assert isinstance(group, dict)

    def test_singleton_detection(self):
        """Test singleton pure node detection."""
        # List with singleton
        pure_indices_singleton = [[0], [1, 2], [3]]
        assert singleton(pure_indices_singleton)

        # List without singleton
        pure_indices_no_singleton = [[0, 1], [2, 3]]
        assert not singleton(pure_indices_no_singleton)

    def test_thresh_a_thresholding(self):
        """Test matrix thresholding function."""
        A = np.array([
            [0.9, 0.1],
            [0.05, 0.8],
            [0.6, 0.3]
        ])

        A_thresh = threshA(A, mu=0.5, scale=False)

        assert A_thresh.shape == A.shape
        assert np.all(A_thresh >= 0)  # Should be non-negative after thresholding

    def test_thresh_a_with_scaling(self):
        """Test thresholding with scaling."""
        A = np.random.rand(5, 3)

        A_thresh_scaled = threshA(A, mu=0.3, scale=True)
        A_thresh_unscaled = threshA(A, mu=0.3, scale=False)

        assert A_thresh_scaled.shape == A.shape
        assert A_thresh_unscaled.shape == A.shape
        # Results should be different when scaling is applied
        assert not np.allclose(A_thresh_scaled, A_thresh_unscaled)

    def test_off_sum_computation(self):
        """Test off-diagonal sum computation."""
        M = np.array([
            [1, 0.2, 0.3],
            [0.2, 1, 0.4],
            [0.3, 0.4, 1]
        ])

        # Uniform weights
        off_sum_uniform = offSum(M, weights=1.0)
        assert isinstance(off_sum_uniform, float)

        # Different weights
        weights = np.array([1, 2, 1])
        off_sum_weighted = offSum(M, weights=weights)
        assert isinstance(off_sum_weighted, float)

    def test_partition_function(self):
        """Test number partitioning utility."""
        partitions = partition(totalNumb=10, numbGroup=3)

        assert isinstance(partitions, list)
        assert len(partitions) == 3
        assert sum(partitions) == 10
        assert all(p > 0 for p in partitions)

    def test_extract_function(self):
        """Test extraction utility function."""
        preVec = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        indices = [0, 2, 5]

        extracted = extract(preVec, indices)

        assert isinstance(extracted, list)
        assert len(extracted) == len(indices)


class TestLOVEEstimation:
    """Test LOVE parameter estimation functions."""


    def test_heteroscedastic_estimation(self):
        """Test heteroscedastic pure estimation."""
        # Mock score matrix
        score_mat = np.random.rand(10, 10)
        delta = 0.4

        result = Est_Pure(score_mat, delta)

        assert isinstance(result, dict)


    def test_solve_row_optimization(self):
        """Test single row solving in Omega estimation."""
        p = 6
        C_hat = np.random.randn(p, p)
        C_hat = C_hat @ C_hat.T + 0.1 * np.eye(p)

        col_ind = 2
        lbd = 0.2

        omega_row = solve_row(col_ind, C_hat, lbd)

        assert omega_row.shape == (p,)


class TestLOVENonPureEstimation:
    """Test non-pure node estimation functionality."""


    def test_lp_optimization(self):
        """Test LP optimization for single row."""
        y = np.random.randn(5)
        lbd = 0.3

        result = LP(y, lbd)

        assert result.shape == y.shape

    def test_dantzig_selector(self):
        """Test Dantzig selector implementation."""
        p = 8
        C_hat = np.random.randn(p, p)
        C_hat = C_hat @ C_hat.T + 0.1 * np.eye(p)
        y = np.random.randn(p)
        lbd = 0.3

        result = Dantzig(C_hat, y, lbd)

        if result is not None:  # Might return None if optimization fails
            assert result.shape == (p,)


class TestLOVEIntegration:
    """Test integration and end-to-end functionality."""


    def test_love_with_preprocessing(self):
        """Test LOVE with preprocessing steps."""
        X = np.random.randn(150, 20)

        # First screen features
        screen_result = Screen_X(X, thresh_grid=np.array([0.1, 0.3, 0.5]))

        # Then run LOVE (this should work regardless of screening result)
        love_result = LOVE(X, lbd=0.4, mu=0.4)

        assert isinstance(screen_result, dict)
        assert isinstance(love_result, dict)


class TestLOVEErrorHandling:
    """Test error handling and edge cases."""


    def test_love_memory_efficiency(self):
        """Test LOVE memory usage with moderately large data."""
        n, p = 1000, 100
        X = np.random.randn(n, p)

        try:
            result = LOVE(X, lbd=0.5, mu=0.5, verbose=False)
            assert isinstance(result, dict)
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

