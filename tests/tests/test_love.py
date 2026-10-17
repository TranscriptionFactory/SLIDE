"""
Basic tests for the LOVE package.
Reproduces the examples from the R package README.
"""

import numpy as np
import pytest

from slide_py.love_python.love import LOVE, Screen_X


def generate_synthetic_data(n=100, seed=42):
    """
    Generate synthetic data matching the R example.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X, A, Z, E) where X is the data matrix.
    """
    np.random.seed(seed)

    p = 6
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

    # Latent factors
    Z = np.random.randn(n, K) * np.sqrt(2)

    # Noise
    E = np.random.randn(n, p)

    # Observed data
    X = Z @ A.T + E

    return X, A, Z, E


class TestLOVE:
    """Test cases for the main LOVE function."""

    def test_love_hetero(self):
        """Test LOVE with heterogeneous pure loadings (pure_homo=False)."""
        X, A_true, _, _ = generate_synthetic_data(n=100)

        # Run LOVE with heterogeneous approach
        result = LOVE(X, pure_homo=False, delta=None)

        # Basic checks
        assert 'K' in result
        assert 'pureVec' in result
        assert 'A' in result
        assert 'C' in result
        assert 'group' in result

        # Check dimensions
        assert result['A'].shape[0] == X.shape[1]  # p rows
        assert result['K'] > 0

        print(f"Estimated K: {result['K']}")
        print(f"Pure variables: {result['pureVec']}")
        print(f"Estimated A shape: {result['A'].shape}")

    def test_love_homo(self):
        """Test LOVE with homogeneous pure loadings (pure_homo=True)."""
        X, A_true, _, _ = generate_synthetic_data(n=100)

        # Run LOVE with homogeneous approach
        delta_grid = np.arange(0.1, 1.2, 0.1)
        result = LOVE(X, pure_homo=True, delta=delta_grid)

        # Basic checks
        assert 'K' in result
        assert 'pureVec' in result
        assert 'A' in result
        assert 'C' in result
        assert 'group' in result

        print(f"Estimated K: {result['K']}")
        print(f"Pure variables: {result['pureVec']}")
        print(f"optDelta: {result['optDelta']}")

    def test_love_different_methods(self):
        """Test LOVE with different non-pure row estimation methods."""
        X, _, _, _ = generate_synthetic_data(n=100)

        methods = ["HT", "ST"]
        for method in methods:
            result = LOVE(X, pure_homo=False, est_non_pure_row=method)
            assert 'K' in result
            print(f"Method {method}: K = {result['K']}")


class TestScreenX:
    """Test cases for the Screen_X function."""

    def test_screen_x_basic(self):
        """Test Screen_X with default parameters."""
        # Generate data with a noise feature
        np.random.seed(42)
        n, p, K = 100, 7, 2

        A = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, 1],
            [1/3, 2/3],
            [1/2, -1/2],
            [0, 0]  # Pure noise feature
        ])

        Z = np.random.randn(n, K) * np.sqrt(2)
        E = np.random.randn(n, p)
        X = Z @ A.T + E

        # Run Screen_X
        result = Screen_X(X)

        assert 'noise_ind' in result
        assert 'thresh_min' in result
        assert 'cv_mean' in result

        print(f"Detected noise indices: {result['noise_ind']}")
        print(f"Optimal threshold: {result['thresh_min']}")

    def test_screen_x_single_threshold(self):
        """Test Screen_X with a single threshold value."""
        X, _, _, _ = generate_synthetic_data(n=100)

        # Single threshold should return array of indices
        result = Screen_X(X, thresh_grid=np.array([0.1]))
        assert isinstance(result, np.ndarray)


class TestUtilities:
    """Test utility functions."""

    def test_recover_group(self):
        """Test recoverGroup function."""
        from slide_py.love_python.love.utilities import recoverGroup

        A = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ])

        groups = recoverGroup(A)

        assert len(groups) == 2
        assert len(groups[0]['pos']) == 1  # Index 0
        assert len(groups[0]['neg']) == 1  # Index 1
        assert len(groups[1]['pos']) == 1  # Index 2
        assert len(groups[1]['neg']) == 1  # Index 3

    def test_offSum(self):
        """Test offSum function."""
        from slide_py.love_python.love.utilities import offSum

        M = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]
        ])
        weights = np.array([1, 1, 1])

        result = offSum(M, weights)
        # Upper triangular elements (excluding diagonal): 2, 3, 5
        expected = 2**2 + 3**2 + 5**2
        assert np.isclose(result, expected)


class TestScoreMatrix:
    """Test score matrix computation."""

    def test_score_mat(self):
        """Test Score_mat function."""
        from slide_py.love_python.love.score import Score_mat

        np.random.seed(42)
        X = np.random.randn(50, 5)
        R = np.corrcoef(X, rowvar=False)

        result = Score_mat(R, q=2, exact=False)

        assert 'score' in result
        assert 'moments' in result
        assert result['score'].shape == (5, 5)
        assert result['moments'].shape == (5, 5)


if __name__ == "__main__":
    # Run basic tests
    print("Testing LOVE package...")
    print("=" * 50)

    print("\n1. Testing LOVE (heterogeneous)...")
    test = TestLOVE()
    test.test_love_hetero()

    print("\n2. Testing LOVE (homogeneous)...")
    test.test_love_homo()

    print("\n3. Testing Screen_X...")
    test_screen = TestScreenX()
    test_screen.test_screen_x_basic()

    print("\n4. Testing utilities...")
    test_util = TestUtilities()
    test_util.test_recover_group()
    test_util.test_offSum()

    print("\n5. Testing Score_mat...")
    test_score = TestScoreMatrix()
    test_score.test_score_mat()

    print("\n" + "=" * 50)
    print("All basic tests passed!")
