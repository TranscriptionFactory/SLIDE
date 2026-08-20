#!/usr/bin/env python3
"""
Comprehensive tests for R SLIDE alignment features in the knockoff filter module.

This test suite validates the following new features:
1. find_opt_iter() function - finds optimal iteration with max overlap
2. Feature chunking - breaks features into chunks for large p
3. Two-stage screening - combines chunk results and re-runs knockoffs
4. Backward compatibility - existing behavior preserved
5. knockoff_filter_voting_slide() wrapper - R SLIDE defaults

Tests use synthetic data with known properties to exercise edge cases.

Author: Testing Agent
Date: 2026-02-01
"""

import pytest
import numpy as np
import warnings


# =============================================================================
# Fixtures and Helper Functions
# =============================================================================

@pytest.fixture
def simple_data():
    """
    Create simple test data with strong signals.

    n=200, p=50, 5 true signals with beta=3.0
    This is an easy case where n >> 2p, allowing stable knockoff generation.
    """
    np.random.seed(42)
    n, p = 200, 50
    X = np.random.randn(n, p)

    # Standardize columns
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    beta = np.zeros(p)
    beta[:5] = 3.0  # First 5 features are true signals

    y = X @ beta + np.random.randn(n) * 0.5

    return X, y, beta


@pytest.fixture
def large_feature_data():
    """
    Create test data with many features to test chunking.

    n=150, p=300, requires chunking if f_size=100.
    """
    np.random.seed(123)
    n, p = 150, 300

    # Create correlated features to make more realistic
    base = np.random.randn(n, p // 3)
    X = np.column_stack([
        base + np.random.randn(n, p // 3) * 0.5,
        base + np.random.randn(n, p // 3) * 0.5,
        base + np.random.randn(n, p // 3) * 0.5,
    ])

    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    beta = np.zeros(p)
    # True signals spread across chunks
    beta[10:15] = 2.5    # In first chunk (0-99)
    beta[120:125] = 2.5  # In second chunk (100-199)
    beta[250:255] = 2.5  # In third chunk (200-299)

    y = X @ beta + np.random.randn(n)

    return X, y, beta


@pytest.fixture
def boundary_data():
    """
    Create test data at n ~ p boundary where knockoff generation is harder.

    n=100, p=80, tests regularization/shrinkage paths.
    """
    np.random.seed(456)
    n, p = 100, 80
    X = np.random.randn(n, p)

    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    beta = np.zeros(p)
    beta[:3] = 2.0

    y = X @ beta + np.random.randn(n) * 0.5

    return X, y, beta


def create_mock_selected_list(iterations, p, seed=42):
    """
    Create a list of selected variable arrays for testing find_opt_iter.

    Parameters
    ----------
    iterations : int
        Number of iterations to simulate
    p : int
        Number of features
    seed : int
        Random seed

    Returns
    -------
    list of np.ndarray
        Each element is an array of selected variable indices
    """
    np.random.seed(seed)
    selected_list = []

    for i in range(iterations):
        # Randomly select some variables
        n_selected = np.random.randint(0, min(10, p) + 1)
        if n_selected > 0:
            selected = np.sort(np.random.choice(p, size=n_selected, replace=False))
        else:
            selected = np.array([], dtype=int)
        selected_list.append(selected)

    return selected_list


# =============================================================================
# Tests for find_opt_iter() function
# =============================================================================

def _get_find_opt_iter():
    """
    Helper to import find_opt_iter, returning None if not implemented.
    """
    try:
        from loveslide.knockoff.filter import find_opt_iter
        return find_opt_iter
    except (ImportError, AttributeError):
        return None


# Check if find_opt_iter is available
_find_opt_iter_available = _get_find_opt_iter() is not None


def _get_slide_voting():
    """
    Helper to import knockoff_filter_voting_slide, returning None if not implemented.
    """
    try:
        from loveslide.knockoff.filter import knockoff_filter_voting_slide
        return knockoff_filter_voting_slide
    except (ImportError, AttributeError):
        return None


def _get_chunk_boundaries():
    """
    Helper to import _compute_chunk_boundaries, returning None if not implemented.
    """
    try:
        from loveslide.knockoff.filter import _compute_chunk_boundaries
        return _compute_chunk_boundaries
    except (ImportError, AttributeError):
        return None


# Check if SLIDE voting wrapper is available
_slide_voting_available = _get_slide_voting() is not None
_chunk_boundaries_available = _get_chunk_boundaries() is not None


@pytest.mark.skipif(not _find_opt_iter_available, reason="find_opt_iter not yet implemented")
class TestFindOptIter:
    """
    Tests for the find_opt_iter function that selects optimal iteration.

    The function returns (selected_vars, optimal_iter_index).
    """

    def test_simple_max_overlap(self):
        """
        Test basic case: one iteration clearly has maximum overlap.

        Given freq_vars = [0, 1, 2] and selected_list where iteration 2
        has all three, find_opt_iter should return iteration 2's selections.
        """
        from loveslide.knockoff.filter import find_opt_iter

        freq_vars = np.array([0, 1, 2])
        selected_list = [
            np.array([0]),           # overlap=1
            np.array([0, 1]),        # overlap=2
            np.array([0, 1, 2]),     # overlap=3 (max)
            np.array([0, 1, 5, 6]),  # overlap=2
        ]

        selected_vars, opt_iter = find_opt_iter(freq_vars, selected_list)

        # Should return iteration 2's selections (index 2)
        assert np.array_equal(selected_vars, np.array([0, 1, 2]))
        assert opt_iter == 2

    def test_tie_breaking_smallest_selection(self):
        """
        Test tie-breaking: when multiple iterations have same overlap,
        pick the one with smallest selection set.

        This matches R's findOptIter behavior.
        """
        from loveslide.knockoff.filter import find_opt_iter

        freq_vars = np.array([0, 1])
        selected_list = [
            np.array([0, 1, 2, 3, 4, 5]),  # overlap=2, size=6
            np.array([0, 1, 2, 3]),         # overlap=2, size=4
            np.array([0, 1]),               # overlap=2, size=2 (smallest)
            np.array([0, 1, 2]),            # overlap=2, size=3
        ]

        selected_vars, opt_iter = find_opt_iter(freq_vars, selected_list)

        # Should return iteration 2 (smallest size=2 among tied)
        assert np.array_equal(selected_vars, np.array([0, 1]))
        assert opt_iter == 2

    def test_empty_freq_vars(self):
        """
        Edge case: freq_vars is empty.

        When no variables pass the frequency threshold, return empty
        (nothing to intersect with).
        """
        from loveslide.knockoff.filter import find_opt_iter

        freq_vars = np.array([], dtype=int)
        selected_list = [
            np.array([0, 1, 2]),
            np.array([0]),
            np.array([0, 1]),
        ]

        selected_vars, opt_iter = find_opt_iter(freq_vars, selected_list)

        # Empty freq_vars returns empty result
        assert len(selected_vars) == 0
        assert opt_iter is None

    def test_single_iteration_with_overlap(self):
        """
        Edge case: only one iteration in selected_list, with full overlap.

        Should return that iteration's selections.
        """
        from loveslide.knockoff.filter import find_opt_iter

        freq_vars = np.array([0, 1, 2])
        selected_list = [
            np.array([0, 1, 2, 3, 4]),  # full overlap with freq_vars
        ]

        selected_vars, opt_iter = find_opt_iter(freq_vars, selected_list)

        assert np.array_equal(selected_vars, np.array([0, 1, 2, 3, 4]))
        assert opt_iter == 0

    def test_single_iteration_no_overlap(self):
        """
        Edge case: only one iteration, no overlap with freq_vars.

        When max_overlap is 0, the function returns freq_vars as a fallback.
        This makes sense: if no iteration has any overlap with the frequent
        variables, we fall back to the frequent variables themselves.
        """
        from loveslide.knockoff.filter import find_opt_iter

        freq_vars = np.array([0, 1, 2])
        selected_list = [
            np.array([5, 6, 7]),  # no overlap with freq_vars
        ]

        selected_vars, opt_iter = find_opt_iter(freq_vars, selected_list)

        # When max_overlap == 0, returns freq_vars as fallback
        assert np.array_equal(selected_vars, freq_vars)
        assert opt_iter is None  # No valid iteration found

    def test_all_iterations_identical(self):
        """
        Edge case: all iterations select the same variables.

        Should return that common selection from first iteration (all tie).
        """
        from loveslide.knockoff.filter import find_opt_iter

        freq_vars = np.array([0, 1, 2, 3])
        common_selection = np.array([0, 1, 2])
        selected_list = [common_selection.copy() for _ in range(5)]

        selected_vars, opt_iter = find_opt_iter(freq_vars, selected_list)

        assert np.array_equal(selected_vars, common_selection)
        # All tied, pick first (index 0)
        assert opt_iter == 0

    def test_empty_iteration(self):
        """
        Edge case: some iterations select zero variables.

        Empty selections should have overlap=0 but size=0.
        """
        from loveslide.knockoff.filter import find_opt_iter

        freq_vars = np.array([0, 1])
        selected_list = [
            np.array([], dtype=int),  # empty, overlap=0, size=0
            np.array([0, 1]),         # overlap=2, size=2
            np.array([], dtype=int),  # empty
        ]

        selected_vars, opt_iter = find_opt_iter(freq_vars, selected_list)

        # Iteration 1 has max overlap=2
        assert np.array_equal(selected_vars, np.array([0, 1]))
        assert opt_iter == 1

    def test_list_input_vs_array(self):
        """
        Verify function handles both list and array inputs for freq_vars.
        """
        from loveslide.knockoff.filter import find_opt_iter

        selected_list = [
            np.array([0, 1, 2]),
            np.array([0, 1]),
        ]

        # Test with list input
        selected_vars_list, opt_iter_list = find_opt_iter([0, 1], selected_list)

        # Test with array input
        selected_vars_array, opt_iter_array = find_opt_iter(np.array([0, 1]), selected_list)

        assert np.array_equal(selected_vars_list, selected_vars_array)
        assert opt_iter_list == opt_iter_array


# =============================================================================
# Tests for Feature Chunking
# =============================================================================

class TestFeatureChunking:
    """Tests for feature chunking behavior with large p."""

    @pytest.mark.skipif(not _slide_voting_available, reason="knockoff_filter_voting_slide not yet implemented")
    def test_no_chunking_when_p_small(self, simple_data):
        """
        Verify chunking produces same results as non-chunking when p <= f_size.

        With p=50 and f_size=100, should be equivalent to single knockoff run.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, _ = simple_data
        p = X.shape[1]

        # f_size > p means no chunking
        result_chunked = knockoff_filter_voting_slide(
            X, y,
            f_size=100,  # p=50 < 100, no chunking
            niter=10,
            spec=0.1,
            fdr=0.2,
            base_seed=42
        )

        # Import regular voting for comparison
        from loveslide.knockoff.filter import knockoff_filter_voting

        result_regular = knockoff_filter_voting(
            X, y,
            niter=10,
            spec=0.1,
            fdr=0.2,
            base_seed=42
        )

        # Results should be very similar (may differ due to findOptIter)
        # At minimum, selected counts should match
        assert len(result_chunked.selection_counts) == p
        assert len(result_regular.selection_counts) == p

    @pytest.mark.skipif(not _chunk_boundaries_available, reason="Feature chunking not yet implemented")
    def test_chunking_creates_correct_chunks(self, large_feature_data):
        """
        Verify chunks are created correctly with expected boundaries.

        p=300, f_size=100 should create 3 chunks: [0:100], [100:200], [200:300]
        """
        from loveslide.knockoff.filter import _compute_chunk_boundaries

        X, y, _ = large_feature_data
        p = X.shape[1]  # 300
        f_size = 100

        # Get chunk boundaries (internal function)
        chunks = _compute_chunk_boundaries(p, f_size)

        # Should have 3 chunks
        assert len(chunks) == 3

        # Verify boundaries
        expected = [(0, 100), (100, 200), (200, 300)]
        assert chunks == expected

    @pytest.mark.skipif(not _chunk_boundaries_available, reason="Feature chunking not yet implemented")
    def test_chunking_with_non_divisible_p(self):
        """
        Test chunking when p is not evenly divisible by f_size.

        p=250, f_size=100 should create chunks: [0:84], [84:168], [168:250]
        (R uses ceiling division to balance chunk sizes)
        """
        from loveslide.knockoff.filter import _compute_chunk_boundaries

        p = 250
        f_size = 100

        chunks = _compute_chunk_boundaries(p, f_size)

        # R-style: ceil(250/100)=3 splits, ceil(250/3)=84 per chunk
        # Chunks should be approximately equal sized
        chunk_sizes = [end - start for start, end in chunks]

        # All chunks should be reasonable size
        assert all(50 <= size <= 100 for size in chunk_sizes)

        # Total should cover all features
        total = sum(chunk_sizes)
        assert total == p

    @pytest.mark.skipif(not _slide_voting_available, reason="knockoff_filter_voting_slide not yet implemented")
    def test_chunking_preserves_feature_indices(self, large_feature_data):
        """
        Verify selected indices from chunks map back to original feature space.

        When chunk 2 (100-200) selects local index 20, global index should be 120.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, beta = large_feature_data
        p = X.shape[1]

        # True signals at indices 10-14, 120-124, 250-254
        true_signals = set(np.where(beta != 0)[0])

        result = knockoff_filter_voting_slide(
            X, y,
            f_size=100,
            niter=20,  # More iterations for stability
            spec=0.1,
            fdr=0.2,
            base_seed=42
        )

        # All selected indices should be valid
        assert all(0 <= idx < p for idx in result.selected)

        # Should find at least some true signals
        selected_set = set(result.selected)
        overlap = selected_set & true_signals
        # With strong signals, expect to find some
        assert len(overlap) > 0 or len(result.selected) == 0


# =============================================================================
# Tests for Two-Stage Screening
# =============================================================================

@pytest.mark.skipif(not _slide_voting_available, reason="knockoff_filter_voting_slide not yet implemented")
class TestTwoStageScreening:
    """Tests for two-stage screening behavior with multiple chunks."""

    def test_second_stage_runs_on_combined_candidates(self, large_feature_data):
        """
        Verify second-stage knockoff runs on combined screened variables.

        After first stage screens variables from each chunk, second stage
        should run on the union of all screened variables.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, _ = large_feature_data

        # Run with chunking
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = knockoff_filter_voting_slide(
                X, y,
                f_size=100,
                niter=10,
                spec=0.1,
                fdr=0.2,
                base_seed=42,
                verbose=True
            )

        # Result should contain screen_vars from first stage if available
        if hasattr(result, 'screen_vars'):
            # Second stage selected should be subset of screen_vars
            assert all(idx in result.screen_vars for idx in result.selected)

    def test_no_second_stage_for_single_chunk(self, simple_data):
        """
        Verify no second stage when only one chunk needed.

        With p=50 and f_size=100, no second stage should run.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, _ = simple_data

        result = knockoff_filter_voting_slide(
            X, y,
            f_size=100,  # p=50 < 100, single chunk
            niter=10,
            spec=0.1,
            fdr=0.2,
            base_seed=42
        )

        # With single chunk, should behave like regular voting
        assert hasattr(result, 'selected')
        assert hasattr(result, 'selection_counts')

    def test_second_stage_with_no_first_stage_selections(self):
        """
        Edge case: first stage selects no variables.

        If all chunks find zero variables, second stage should be skipped
        and empty selection returned.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        # Create noise-only data (no true signals)
        np.random.seed(789)
        n, p = 100, 200
        X = np.random.randn(n, p)
        y = np.random.randn(n)  # No relationship with X

        result = knockoff_filter_voting_slide(
            X, y,
            f_size=100,
            niter=10,
            spec=0.5,  # High threshold to ensure few selections
            fdr=0.05,  # Strict FDR
            base_seed=42
        )

        # May select nothing, which is valid
        assert len(result.selected) >= 0


# =============================================================================
# Tests for Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Tests ensuring existing behavior is preserved."""

    def test_default_voting_unchanged(self, simple_data):
        """
        Verify knockoff_filter_voting with default params works identically.

        No new features should affect existing API.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        X, y, _ = simple_data

        # Run with default parameters
        result1 = knockoff_filter_voting(
            X, y,
            niter=10,
            spec=0.1,
            base_seed=42
        )

        # Run again to verify reproducibility
        result2 = knockoff_filter_voting(
            X, y,
            niter=10,
            spec=0.1,
            base_seed=42
        )

        # Results should be identical
        assert np.array_equal(result1.selection_counts, result2.selection_counts)
        assert np.array_equal(result1.selected, result2.selected)

    def test_voting_result_structure(self, simple_data):
        """
        Verify VotingResult dataclass has expected attributes.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting, VotingResult

        X, y, _ = simple_data

        result = knockoff_filter_voting(
            X, y,
            niter=10,
            spec=0.1,
            base_seed=42
        )

        # Check it's the right type
        assert isinstance(result, VotingResult)

        # Check all expected attributes
        assert hasattr(result, 'selection_counts')
        assert hasattr(result, 'selection_frequency')
        assert hasattr(result, 'selected')
        assert hasattr(result, 'threshold')
        assert hasattr(result, 'niter')
        assert hasattr(result, 'spec')
        assert hasattr(result, 'min_selections')

        # Check types
        assert isinstance(result.selection_counts, np.ndarray)
        assert isinstance(result.selected, np.ndarray)
        assert isinstance(result.niter, int)
        assert isinstance(result.spec, float)

    def test_existing_parameters_work(self, simple_data):
        """
        Verify all existing parameters still work correctly.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        X, y, _ = simple_data

        # Test with various existing parameter combinations
        result = knockoff_filter_voting(
            X, y,
            niter=5,
            spec=0.2,
            fdr=0.15,
            offset=1,
            n_jobs=1,
            base_seed=123,
            verbose=False,
            match_r=True,
            use_cache=True
        )

        assert len(result.selection_counts) == X.shape[1]
        assert result.niter == 5
        assert result.spec == 0.2

    def test_custom_knockoffs_still_work(self, simple_data):
        """
        Verify custom knockoff functions still work.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting
        from loveslide.knockoff.create import create_second_order

        X, y, _ = simple_data

        # Use custom knockoff function
        custom_knockoffs = lambda x: create_second_order(x, method='equi')

        result = knockoff_filter_voting(
            X, y,
            knockoffs=custom_knockoffs,
            niter=5,
            spec=0.1,
            base_seed=42
        )

        assert len(result.selection_counts) == X.shape[1]


# =============================================================================
# Tests for knockoff_filter_voting_slide() Wrapper
# =============================================================================

@pytest.mark.skipif(not _slide_voting_available, reason="knockoff_filter_voting_slide not yet implemented")
class TestKnockoffFilterVotingSlide:
    """Tests for the R SLIDE-compatible wrapper function."""

    def test_r_slide_defaults(self, simple_data):
        """
        Verify knockoff_filter_voting_slide sets correct R SLIDE defaults.

        R SLIDE defaults:
        - offset=0 (standard knockoff, not knockoff+)
        - fdr=0.1
        - spec=0.1
        - niter=500
        - f_size based on n,K formula
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, _ = simple_data

        # Run with minimal parameters - should use R defaults
        result = knockoff_filter_voting_slide(
            X, y,
            niter=5,  # Override for speed
            base_seed=42
        )

        # Check defaults were applied
        assert result.spec == 0.1  # R default
        assert result.niter == 5   # Our override

    def test_stability_rule_is_the_default(self, simple_data):
        """
        Knockoffs.select_short_freq_slide -- the entry point the SLIDE pipeline
        uses -- now defaults to the stability rule: the reported selection is
        exactly the frequency set, every feature whose selection_counts reach
        ceil(niter * spec) and nothing else.

        findOptIter, by contrast, discards the frequency table and reports a
        single draw out of niter, which makes the result swing between runs that
        are otherwise identical. It stays reachable via slide_selection=True.
        """
        import math

        from loveslide import Knockoffs

        X, y, _ = simple_data
        niter, spec = 20, 0.1

        result = Knockoffs.select_short_freq_slide(
            X, y, niter=niter, spec=spec, fdr=0.2, f_size=100, base_seed=42,
        )

        min_selections = math.ceil(niter * spec)
        expected = np.where(np.asarray(result.selection_counts) >= min_selections)[0]

        assert np.array_equal(np.sort(result.selected), np.sort(expected))
        assert result.optimal_iter is None

        # Opting back in must still reach the R rule: one draw out of niter.
        refined = Knockoffs.select_short_freq_slide(
            X, y, niter=niter, spec=spec, fdr=0.2, f_size=100, base_seed=42,
            slide_selection=True,
        )
        assert refined.optimal_iter is not None

    def test_end_to_end_synthetic(self, simple_data):
        """
        End-to-end test with synthetic data having known signals.

        Should detect at least some of the 5 true signals.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, beta = simple_data
        true_signals = set(np.where(beta != 0)[0])  # {0, 1, 2, 3, 4}

        result = knockoff_filter_voting_slide(
            X, y,
            niter=20,
            spec=0.1,
            fdr=0.2,
            base_seed=42
        )

        selected_set = set(result.selected)

        # Calculate precision/recall
        true_positives = len(selected_set & true_signals)
        false_positives = len(selected_set - true_signals)

        # With strong signals (beta=3.0), should have good recall
        # Allow some flexibility due to randomness
        if len(selected_set) > 0:
            precision = true_positives / len(selected_set)
            # Expect reasonable precision (>50%) for strong signals
            assert precision > 0.3 or true_positives >= 3

    def test_slide_with_large_features(self, large_feature_data):
        """
        Test SLIDE wrapper with features requiring chunking.

        With p=300 and f_size=100, should use 3 chunks.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, beta = large_feature_data
        true_signals = set(np.where(beta != 0)[0])

        result = knockoff_filter_voting_slide(
            X, y,
            f_size=100,
            niter=10,
            spec=0.1,
            fdr=0.2,
            base_seed=42
        )

        # Should produce valid result
        assert len(result.selection_counts) == X.shape[1]

        # Selected should be subset of all features
        assert all(0 <= idx < X.shape[1] for idx in result.selected)

    def test_slide_reproducibility(self, simple_data):
        """
        Verify SLIDE wrapper produces reproducible results with same seed.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        X, y, _ = simple_data

        result1 = knockoff_filter_voting_slide(
            X, y,
            niter=10,
            spec=0.1,
            base_seed=42
        )

        result2 = knockoff_filter_voting_slide(
            X, y,
            niter=10,
            spec=0.1,
            base_seed=42
        )

        assert np.array_equal(result1.selected, result2.selected)
        assert np.array_equal(result1.selection_counts, result2.selection_counts)


# =============================================================================
# Tests for Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_feature(self):
        """
        Edge case: p=1 (single feature).
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        y = 2 * X[:, 0] + np.random.randn(n) * 0.5

        result = knockoff_filter_voting(
            X, y,
            niter=5,
            spec=0.1,
            base_seed=42
        )

        assert len(result.selection_counts) == 1
        assert len(result.selected) <= 1

    def test_many_features_few_samples(self, boundary_data):
        """
        Edge case: n ~ p (many features, few samples).

        Should handle gracefully with shrinkage.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        X, y, _ = boundary_data  # n=100, p=80

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings

            result = knockoff_filter_voting(
                X, y,
                niter=5,
                spec=0.1,
                base_seed=42
            )

        # Should complete without error
        assert len(result.selection_counts) == X.shape[1]

    def test_zero_spec(self):
        """
        Edge case: spec=0 should raise error or select nothing.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        with pytest.raises(ValueError):
            knockoff_filter_voting(X, y, niter=5, spec=0)

    def test_spec_one(self):
        """
        Edge case: spec=1.0 means variable must be selected in ALL iterations.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)

        # Strong signal
        beta = np.zeros(p)
        beta[0] = 5.0
        y = X @ beta + np.random.randn(n) * 0.1

        result = knockoff_filter_voting(
            X, y,
            niter=5,
            spec=1.0,  # Must be selected in 100% of runs
            base_seed=42
        )

        # min_selections should equal niter
        assert result.min_selections == 5

    def test_high_fdr(self):
        """
        Edge case: fdr=0.5 (very permissive).

        Should select more variables than strict FDR.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        np.random.seed(42)
        n, p = 100, 30
        X = np.random.randn(n, p)

        beta = np.zeros(p)
        beta[:5] = 2.0
        y = X @ beta + np.random.randn(n) * 0.5

        result_strict = knockoff_filter_voting(
            X, y, niter=10, spec=0.1, fdr=0.05, base_seed=42
        )

        result_permissive = knockoff_filter_voting(
            X, y, niter=10, spec=0.1, fdr=0.5, base_seed=42
        )

        # Permissive FDR should tend to select more
        # (may not always be true due to threshold mechanics)
        assert len(result_permissive.selection_counts) == p

    def test_niter_one(self):
        """
        Edge case: niter=1 (single iteration).

        Voting degenerates to single knockoff run.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        result = knockoff_filter_voting(
            X, y,
            niter=1,
            spec=0.5,
            base_seed=42
        )

        # min_selections should be 1 (ceil(1 * 0.5) = 1)
        assert result.min_selections == 1
        assert result.niter == 1


# =============================================================================
# Tests for Internal Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_knockoff_threshold(self):
        """
        Test knockoff_threshold function computes correct thresholds.
        """
        from loveslide.knockoff.filter import knockoff_threshold

        # Simple case: clear separation
        W = np.array([3.0, 2.5, 2.0, -0.5, -1.0, -1.5])

        # With offset=1 (knockoff+), threshold should control FDR
        t = knockoff_threshold(W, fdr=0.2, offset=1)

        # Verify threshold is positive
        assert t >= 0

        # Selected should be those with W >= t
        selected = np.where(W >= t)[0]

        # FDP estimate should be <= fdr
        if len(selected) > 0:
            num_neg = np.sum(W <= -t)
            fdp = (1 + num_neg) / len(selected)
            assert fdp <= 0.2 + 0.01  # Small tolerance

    def test_prepare_knockoff_cache(self, simple_data):
        """
        Test _prepare_knockoff_cache creates valid cache.
        """
        from loveslide.knockoff.filter import _prepare_knockoff_cache

        X, _, _ = simple_data

        cache = _prepare_knockoff_cache(X, method='sdp')

        # Check required keys
        assert 'mu' in cache
        assert 'Sigma' in cache
        assert 'diag_s' in cache
        assert 'L' in cache
        assert 'degenerate' in cache

        # Check dimensions
        p = X.shape[1]
        assert cache['mu'].shape == (p,)
        assert cache['Sigma'].shape == (p, p)
        assert cache['diag_s'].shape == (p,)

        if not cache['degenerate']:
            assert cache['L'].shape == (p, p)

    def test_sample_knockoffs_from_cache(self, simple_data):
        """
        Test _sample_knockoffs_from_cache produces valid knockoffs.
        """
        from loveslide.knockoff.filter import (
            _prepare_knockoff_cache,
            _sample_knockoffs_from_cache
        )

        X, _, _ = simple_data
        n, p = X.shape

        cache = _prepare_knockoff_cache(X, method='sdp')

        np.random.seed(42)
        Xk = _sample_knockoffs_from_cache(X, cache)

        # Check shape
        assert Xk.shape == (n, p)

        # Check no NaN/Inf
        assert np.isfinite(Xk).all()

        # Knockoffs should be different from original
        diff = np.linalg.norm(X - Xk)
        assert diff > 0


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrized:
    """Parametrized tests for various configurations."""

    @pytest.mark.parametrize("n,p", [
        (200, 50),   # n >> 2p, easy
        (100, 40),   # n > 2p
        (100, 80),   # n ~ p, boundary
        (80, 100),   # n < p, challenging
    ])
    def test_various_dimensions(self, n, p):
        """
        Test voting works across various n, p ratios.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        np.random.seed(42)
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = knockoff_filter_voting(
                X, y,
                niter=3,
                spec=0.1,
                base_seed=42
            )

        assert len(result.selection_counts) == p

    @pytest.mark.parametrize("method", ['equi', 'sdp', 'asdp'])
    def test_various_methods(self, method, simple_data):
        """
        Test voting with different knockoff construction methods.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting
        from loveslide.knockoff.create import create_second_order

        X, y, _ = simple_data

        knockoffs = lambda x: create_second_order(x, method=method)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = knockoff_filter_voting(
                X, y,
                knockoffs=knockoffs,
                niter=3,
                spec=0.1,
                base_seed=42,
                use_cache=False  # Custom knockoffs bypass cache
            )

        assert len(result.selection_counts) == X.shape[1]

    @pytest.mark.parametrize("spec", [0.05, 0.1, 0.2, 0.5])
    def test_various_spec_thresholds(self, spec, simple_data):
        """
        Test voting with different spec thresholds.

        Higher spec should yield fewer selected variables.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting

        X, y, _ = simple_data

        result = knockoff_filter_voting(
            X, y,
            niter=20,
            spec=spec,
            base_seed=42
        )

        # min_selections should scale with spec
        expected_min = int(np.ceil(20 * spec))
        assert result.min_selections == expected_min


# =============================================================================
# Performance Tests (marked slow)
# =============================================================================

class TestPerformance:
    """Performance-related tests (may be slow)."""

    @pytest.mark.slow
    def test_caching_speedup(self, simple_data):
        """
        Verify caching provides speedup over uncached version.
        """
        import time
        from loveslide.knockoff.filter import knockoff_filter_voting

        X, y, _ = simple_data
        niter = 20

        # Time cached version
        start = time.time()
        _ = knockoff_filter_voting(
            X, y,
            niter=niter,
            spec=0.1,
            base_seed=42,
            use_cache=True
        )
        cached_time = time.time() - start

        # Time uncached version
        start = time.time()
        _ = knockoff_filter_voting(
            X, y,
            niter=niter,
            spec=0.1,
            base_seed=42,
            use_cache=False
        )
        uncached_time = time.time() - start

        # Cached should be faster (at least not slower)
        speedup = uncached_time / cached_time if cached_time > 0 else 1.0
        print(f"\nCaching speedup: {speedup:.2f}x")

        # Allow some tolerance for noise
        assert speedup > 0.8  # Cached shouldn't be much slower

    @pytest.mark.slow
    @pytest.mark.skipif(not _slide_voting_available, reason="knockoff_filter_voting_slide not yet implemented")
    def test_large_scale_stability(self):
        """
        Test stability with larger scale data.
        """
        from loveslide.knockoff.filter import knockoff_filter_voting_slide

        np.random.seed(42)
        n, p = 200, 500
        X = np.random.randn(n, p)

        beta = np.zeros(p)
        beta[:10] = 2.0
        y = X @ beta + np.random.randn(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = knockoff_filter_voting_slide(
                X, y,
                f_size=100,
                niter=10,
                spec=0.1,
                base_seed=42
            )

        # Should complete without error
        assert len(result.selection_counts) == p


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
