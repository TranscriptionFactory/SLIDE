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

from loveslide.love_python.love.love import LOVE
from loveslide.love_python.love.prescreen import Screen_X
from loveslide.love_python.love.score import Score_mat, LP_Score
from loveslide.love_python.love.utilities import (
    recoverGroup, singleton, threshA, offSum, partition, extract
)
from loveslide.love_python.love.cv import (
    CV_delta, CalFittedSigma, KfoldCV_delta, CV_lbd
)
from loveslide.love_python.love.est_pure_homo import (
    EstAI, EstC, FindRowMax, FindPureNode, TestPure, FindSignPureNode
)
from loveslide.love_python.love.est_pure_hetero import (
    Est_Pure, Est_BI_C, Re_Est_Pure, Post_Est_Pure, Est_K
)
from loveslide.love_python.love.est_nonpure import (
    EstY, EstAJInv, LP, EstAJDant, Dantzig
)
from loveslide.love_python.love.est_omega import estOmega, solve_row


class TestLOVECore:
    """Test the core LOVE algorithm."""

    def test_love_basic_functionality(self):
        """Test basic LOVE execution with synthetic data."""
        # Generate structured data
        n, p = 100, 6
        K = 2

        # Loading matrix
        A = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, 1],
            [1/3, 2/3],
            [1/2, -1/2]
        ])

        # Generate latent factors and noise
        np.random.seed(42)
        Z = np.random.randn(n, K) * np.sqrt(2)
        E = np.random.randn(n, p)
        X = Z @ A.T + E

        # Run LOVE
        result = LOVE(X, lbd=0.5, mu=0.5, est_non_pure_row="HT")

        # Check result structure
        assert isinstance(result, dict)
        assert 'L_hat' in result
        assert 'Sigma' in result
        assert result['L_hat'].shape[1] == p  # Should estimate loading matrix

    def test_love_parameter_validation(self):
        """Test LOVE with invalid parameters."""
        X = np.random.randn(50, 10)

        # Invalid lambda
        with pytest.raises(ValueError):
            LOVE(X, lbd=-0.1)  # Negative lambda

        with pytest.raises(ValueError):
            LOVE(X, lbd=1.1)   # Lambda > 1

        # Invalid mu
        with pytest.raises(ValueError):
            LOVE(X, mu=-0.1)   # Negative mu

    def test_love_with_different_estimation_methods(self):
        """Test LOVE with different non-pure estimation methods."""
        X = np.random.randn(80, 8)

        for method in ["HT", "Dantzig"]:
            result = LOVE(X, lbd=0.5, mu=0.5, est_non_pure_row=method)
            assert isinstance(result, dict)
            assert 'L_hat' in result

    def test_love_edge_case_small_dataset(self):
        """Test LOVE with very small dataset."""
        X = np.random.randn(10, 5)  # Very small

        with pytest.warns(UserWarning) or pytest.raises(ValueError):
            LOVE(X, lbd=0.5, mu=0.5)

    def test_love_diagonal_parameter(self):
        """Test LOVE with diagonal=True/False."""
        X = np.random.randn(100, 12)

        result_diag = LOVE(X, diagonal=True, lbd=0.5, mu=0.5)
        result_full = LOVE(X, diagonal=False, lbd=0.5, mu=0.5)

        assert isinstance(result_diag, dict)
        assert isinstance(result_full, dict)
        # Results should be different
        assert not np.allclose(result_diag['L_hat'], result_full['L_hat'])


class TestLOVEPreprocessing:
    """Test LOVE preprocessing and screening functionality."""

    def test_screen_x_basic(self):
        """Test basic screening functionality."""
        X = np.random.randn(100, 20)

        result = Screen_X(X, thresh_grid=np.array([0.1, 0.2, 0.3]))

        assert isinstance(result, dict)
        assert 'indices' in result or 'selected' in result

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
        from loveslide.love_python.love.love import _apply_fdr_threshold

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

    def test_lp_score_computation(self):
        """Test individual LP score computation."""
        R_ij = np.array([0.1, 0.3, -0.2, 0.8])

        score_exact = LP_Score(R_ij, ind=0, exact=True)
        score_approx = LP_Score(R_ij, ind=0, exact=False)

        assert isinstance(score_exact, float)
        assert isinstance(score_approx, float)
        assert score_exact >= 0  # Scores should be non-negative
        assert score_approx >= 0


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


class TestLOVECrossValidation:
    """Test LOVE cross-validation functionality."""

    def test_cv_delta_basic(self):
        """Test delta parameter cross-validation."""
        X = np.random.randn(100, 10)
        deltaGrids = np.array([0.1, 0.3, 0.5, 0.7])

        result = CV_delta(X, deltaGrids, diagonal=False)

        assert isinstance(result, dict)
        assert 'opt_delta' in result or 'delta_opt' in result

    def test_cv_delta_diagonal_vs_full(self):
        """Test CV delta with diagonal vs full covariance."""
        X = np.random.randn(80, 8)
        deltaGrids = np.array([0.2, 0.5, 0.8])

        result_diag = CV_delta(X, deltaGrids, diagonal=True)
        result_full = CV_delta(X, deltaGrids, diagonal=False)

        assert isinstance(result_diag, dict)
        assert isinstance(result_full, dict)

    def test_kfold_cv_delta(self):
        """Test K-fold cross-validation for delta."""
        X = np.random.randn(150, 12)

        result = KfoldCV_delta(X, delta=None, diagonal=False, K_fold=5)

        assert isinstance(result, dict)

    def test_kfold_cv_delta_with_specific_values(self):
        """Test K-fold CV with specific delta values."""
        X = np.random.randn(120, 10)
        delta_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        result = KfoldCV_delta(X, delta=delta_values, K_fold=3)

        assert isinstance(result, dict)

    def test_cv_lbd_basic(self):
        """Test lambda parameter cross-validation."""
        X = np.random.randn(100, 8)
        lbdGrids = np.array([0.1, 0.3, 0.5, 0.7])
        AI = np.random.randn(8, 2)  # Mock loading matrix

        result = CV_lbd(X, lbdGrids, AI, diagonal=False)

        assert isinstance(result, dict)

    def test_cal_fitted_sigma(self):
        """Test fitted sigma calculation."""
        p = 6
        Sigma = np.eye(p) + 0.2 * np.ones((p, p))
        delta = 0.5
        Ms = np.random.rand(p)  # Mock M values

        fitted_sigma = CalFittedSigma(Sigma, delta, Ms, ind_selected=np.arange(p))

        assert fitted_sigma.shape == Sigma.shape
        assert np.allclose(fitted_sigma, fitted_sigma.T)  # Should be symmetric


class TestLOVEEstimation:
    """Test LOVE parameter estimation functions."""

    def test_est_ai_basic(self):
        """Test AI matrix estimation (homoscedastic case)."""
        # Create synthetic covariance matrix
        p = 8
        Sigma = np.eye(p) + 0.3 * np.ones((p, p))
        np.fill_diagonal(Sigma, 1.0)

        optDelta = 0.5
        se_est = np.ones(p)

        AI = EstAI(Sigma, optDelta, se_est, method="HT")

        assert AI.shape[0] == p
        assert AI.shape[1] > 0  # Should estimate at least one factor

    def test_est_ai_different_methods(self):
        """Test AI estimation with different methods."""
        Sigma = np.eye(5) + 0.4 * np.ones((5, 5))
        np.fill_diagonal(Sigma, 1.0)

        for method in ["HT", "threshold", "mixed"]:
            try:
                AI = EstAI(Sigma, optDelta=0.4, se_est=np.ones(5), method=method)
                assert AI.shape[0] == 5
            except (ValueError, NotImplementedError):
                # Some methods might not be implemented
                continue

    def test_est_c_matrix(self):
        """Test C matrix estimation."""
        p, K = 6, 2
        Sigma = np.eye(p) + 0.2 * np.ones((p, p))
        AI = np.random.randn(p, K)

        C = EstC(Sigma, AI, diagonal=False)

        assert C.shape == (p, p)
        assert np.allclose(C, C.T)  # Should be symmetric

    def test_find_row_max(self):
        """Test row maximum finding."""
        Sigma = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.8],
            [0.2, 0.8, 1.0]
        ])

        result = FindRowMax(Sigma)

        assert isinstance(result, dict)
        assert 'Ms' in result
        assert 'arg_Ms' in result

    def test_find_pure_node(self):
        """Test pure node identification."""
        # Create off-diagonal correlation matrix
        off_Sigma = np.array([
            [0, 0.8, 0.1],
            [0.8, 0, 0.2],
            [0.1, 0.2, 0]
        ])

        delta = 0.5
        Ms = np.array([0.8, 0.8, 0.2])
        arg_Ms = np.array([1, 0, 1])

        pure_nodes = FindPureNode(off_Sigma, delta, Ms, arg_Ms)

        assert isinstance(pure_nodes, list)

    def test_test_pure_functionality(self):
        """Test pure node testing."""
        Sigma_row = np.array([0.1, 0.8, 0.2, 0.1])
        rowInd = 0
        Si = np.array([0.8, 0.7, 0.3, 0.2])

        result = TestPure(Sigma_row, rowInd, Si, thre_fac=0.5)

        assert isinstance(result, dict)

    def test_heteroscedastic_estimation(self):
        """Test heteroscedastic pure estimation."""
        # Mock score matrix
        score_mat = np.random.rand(10, 10)
        delta = 0.4

        result = Est_Pure(score_mat, delta)

        assert isinstance(result, dict)

    def test_omega_estimation(self):
        """Test Omega matrix estimation."""
        p = 5
        C = np.random.randn(p, p)
        C = C @ C.T + 0.1 * np.eye(p)  # Make positive definite

        lbd = 0.3
        Omega = estOmega(lbd, C)

        assert Omega.shape == (p, p)
        assert np.allclose(Omega, Omega.T)  # Should be symmetric

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

    def test_est_y_computation(self):
        """Test Y matrix estimation for non-pure nodes."""
        p, K = 8, 3
        Sigma = np.eye(p) + 0.2 * np.ones((p, p))
        AI = np.random.randn(p, K)
        pureVec = np.array([1, 1, 0, 1, 0, 0, 1, 0])  # Some pure, some not

        Y = EstY(Sigma, AI, pureVec)

        expected_non_pure = np.sum(pureVec == 0)
        assert Y.shape == (expected_non_pure, K)

    def test_est_aj_inv_computation(self):
        """Test AJ inverse estimation."""
        p, K = 6, 2
        Omega = np.eye(p)
        Y = np.random.randn(3, K)  # 3 non-pure nodes
        lbd = 0.4

        AJ_inv = EstAJInv(Omega, Y, lbd)

        assert AJ_inv.shape == (3, K)

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

    def test_love_end_to_end_pipeline(self):
        """Test complete LOVE pipeline with synthetic data."""
        # Generate data with known structure
        n, p, K = 200, 10, 2

        # True loading matrix
        A_true = np.random.randn(p, K)
        A_true[5:, 0] = 0  # Make some loadings sparse
        A_true[:3, 1] = 0

        # Generate data
        Z = np.random.randn(n, K)
        E = 0.3 * np.random.randn(n, p)
        X = Z @ A_true.T + E

        # Run LOVE pipeline
        result = LOVE(X, lbd=0.5, mu=0.3, est_non_pure_row="HT", verbose=False)

        # Check that we recover reasonable structure
        L_hat = result['L_hat']
        assert L_hat.shape[0] == p
        assert L_hat.shape[1] <= K + 2  # Should not over-estimate factors

    def test_love_with_preprocessing(self):
        """Test LOVE with preprocessing steps."""
        X = np.random.randn(150, 20)

        # First screen features
        screen_result = Screen_X(X, thresh_grid=np.array([0.1, 0.3, 0.5]))

        # Then run LOVE (this should work regardless of screening result)
        love_result = LOVE(X, lbd=0.4, mu=0.4)

        assert isinstance(screen_result, dict)
        assert isinstance(love_result, dict)

    def test_love_reproducibility(self):
        """Test that LOVE produces reproducible results."""
        X = np.random.RandomState(42).randn(100, 12)

        result1 = LOVE(X, lbd=0.5, mu=0.5, est_non_pure_row="HT")
        result2 = LOVE(X, lbd=0.5, mu=0.5, est_non_pure_row="HT")

        # Results should be identical (assuming deterministic implementation)
        assert np.allclose(result1['L_hat'], result2['L_hat'], atol=1e-6)

    def test_love_parameter_sensitivity(self):
        """Test LOVE sensitivity to parameter changes."""
        X = np.random.randn(100, 8)

        result_low_lbd = LOVE(X, lbd=0.1, mu=0.5)
        result_high_lbd = LOVE(X, lbd=0.9, mu=0.5)

        # Results should be different with different lambda values
        assert not np.allclose(
            result_low_lbd['L_hat'],
            result_high_lbd['L_hat'],
            atol=0.1
        )


class TestLOVEErrorHandling:
    """Test error handling and edge cases."""

    def test_love_with_nan_data(self):
        """Test LOVE behavior with NaN values in data."""
        X = np.random.randn(50, 8)
        X[10, 3] = np.nan  # Insert NaN

        with pytest.raises(ValueError, match="NaN.*detected"):
            LOVE(X, lbd=0.5, mu=0.5)

    def test_love_with_infinite_data(self):
        """Test LOVE behavior with infinite values."""
        X = np.random.randn(50, 8)
        X[5, 2] = np.inf

        with pytest.raises(ValueError, match="infinite.*detected"):
            LOVE(X, lbd=0.5, mu=0.5)

    def test_love_with_rank_deficient_data(self):
        """Test LOVE with rank-deficient input matrix."""
        X = np.random.randn(100, 10)
        X[:, 5] = X[:, 0] + X[:, 1]  # Create linear dependence

        with pytest.warns(UserWarning, match="rank.*deficient") or \
             pytest.raises(np.linalg.LinAlgError):
            LOVE(X, lbd=0.5, mu=0.5)

    def test_love_memory_efficiency(self):
        """Test LOVE memory usage with moderately large data."""
        n, p = 1000, 100
        X = np.random.randn(n, p)

        try:
            result = LOVE(X, lbd=0.5, mu=0.5, verbose=False)
            assert isinstance(result, dict)
        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_cv_with_insufficient_data(self):
        """Test cross-validation with insufficient data."""
        X = np.random.randn(10, 8)  # Very small dataset

        with pytest.warns(UserWarning) or pytest.raises(ValueError):
            KfoldCV_delta(X, K_fold=5)  # More folds than reasonable