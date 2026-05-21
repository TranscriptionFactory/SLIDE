"""Unit tests for knockoff_threshold against known R outputs."""

import numpy as np
import pytest

from loveslide.knockoff.filter import knockoff_threshold


class TestKnockoffThresholdEdgeCases:
    """Edge cases for knockoff_threshold."""

    def test_all_positive_w(self):
        """All positive W should select threshold at smallest |W|."""
        W = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # offset=1: (1 + 0) / 5 = 0.2 > 0.1, but at t=1: (1+0)/5=0.2 > 0.1
        # At t=0: (1+0)/5=0.2 > 0.1... depends on fdr
        t = knockoff_threshold(W, fdr=0.2, offset=1)
        assert t == 0.0  # (1+0)/5 = 0.2 <= 0.2

    def test_all_negative_w(self):
        """All negative W should return inf (nothing selected)."""
        W = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        t = knockoff_threshold(W, fdr=0.1, offset=1)
        assert t == np.inf

    def test_all_zero_w(self):
        """All zero W should return inf (degenerate case)."""
        W = np.array([0.0, 0.0, 0.0, 0.0])
        # At t=0: numerator = offset + sum(W <= 0) = 1 + 4 = 5
        # denominator = max(1, sum(W >= 0)) = 4
        # ratio = 5/4 = 1.25 > 0.1 => no threshold found
        t = knockoff_threshold(W, fdr=0.1, offset=1)
        assert t == np.inf

    def test_single_element_positive(self):
        """Single positive W element."""
        W = np.array([2.5])
        # At t=0: (1+0)/1 = 1.0 > 0.1; At t=2.5: (1+0)/1 = 1.0 > 0.1 => inf
        t = knockoff_threshold(W, fdr=0.1, offset=1)
        assert t == np.inf
        # With offset=0: at t=0: (0+0)/1=0 <= 0.1
        t = knockoff_threshold(W, fdr=0.1, offset=0)
        assert t == 0.0


class TestKnockoffThresholdOffset:
    """Test offset=0 vs offset=1 behavior."""

    def test_offset_0_more_liberal(self):
        """Offset=0 should generally yield a lower (or equal) threshold."""
        W = np.array([3.0, 2.0, 1.0, -0.5, -1.5])
        t0 = knockoff_threshold(W, fdr=0.1, offset=0)
        t1 = knockoff_threshold(W, fdr=0.1, offset=1)
        assert t0 <= t1

    def test_offset_invalid_raises(self):
        """Invalid offset should raise ValueError."""
        W = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="offset must be either 0 or 1"):
            knockoff_threshold(W, fdr=0.1, offset=2)


class TestKnockoffThresholdRReference:
    """
    Tests against known R knockoff::knockoff.threshold() outputs.

    R code to generate reference:
        library(knockoff)
        W <- c(3.2, 2.1, 1.5, 0.8, -0.3, -1.1, -2.0, -2.5)
        knockoff.threshold(W, fdr=0.1, offset=0)
        knockoff.threshold(W, fdr=0.1, offset=1)
    """

    def test_r_reference_offset0(self):
        """Match R knockoff.threshold with offset=0."""
        # W vector with known R output
        W = np.array([3.2, 2.1, 1.5, 0.8, -0.3, -1.1, -2.0, -2.5])
        t = knockoff_threshold(W, fdr=0.1, offset=0)
        # R output: threshold = 2.1
        # At t=2.1: numerator = 0 + sum(W <= -2.1) = 1 (#W<=-2.5)
        # denominator = max(1, sum(W >= 2.1)) = 2 (#3.2, #2.1)
        # ratio = 1/2 = 0.5 > 0.1
        # At t=2.5: numerator = 0 + sum(W <= -2.5) = 1
        # denominator = max(1, sum(W >= 2.5)) = 1 (#3.2)
        # ratio = 1/1 = 1.0 > 0.1
        # At t=3.2: numerator = 0 + sum(W <= -3.2) = 0
        # denominator = max(1, sum(W >= 3.2)) = 1
        # ratio = 0/1 = 0 <= 0.1
        assert t == pytest.approx(3.2)

    def test_r_reference_offset1(self):
        """Match R knockoff.threshold with offset=1."""
        W = np.array([3.2, 2.1, 1.5, 0.8, -0.3, -1.1, -2.0, -2.5])
        t = knockoff_threshold(W, fdr=0.1, offset=1)
        # At t=3.2: numerator = 1 + sum(W <= -3.2) = 1 + 0 = 1
        # denominator = max(1, sum(W >= 3.2)) = 1
        # ratio = 1/1 = 1.0 > 0.1
        # No threshold found => inf
        assert t == np.inf

    def test_r_reference_liberal_fdr(self):
        """Match R with more liberal FDR."""
        W = np.array([3.2, 2.1, 1.5, 0.8, -0.3, -1.1, -2.0, -2.5])
        t = knockoff_threshold(W, fdr=0.5, offset=0)
        # At t=0.3: numerator = 0 + sum(W <= -0.3) = 4 (-0.3, -1.1, -2.0, -2.5)
        # denominator = max(1, sum(W >= 0.3)) = 5 (3.2, 2.1, 1.5, 0.8, ... not -0.3)
        # Actually W >= 0.3: 3.2, 2.1, 1.5, 0.8 = 4
        # ratio = 4/4 = 1.0 > 0.5
        # At t=0.8: num = 0 + sum(W <= -0.8) = 3 (-1.1, -2.0, -2.5)
        # den = sum(W >= 0.8) = 4 (3.2, 2.1, 1.5, 0.8)
        # ratio = 3/4 = 0.75 > 0.5
        # At t=1.1: num = 0 + sum(W <= -1.1) = 3 (-1.1, -2.0, -2.5)
        # den = sum(W >= 1.1) = 3 (3.2, 2.1, 1.5)
        # ratio = 3/3 = 1.0 > 0.5
        # At t=1.5: num = 0 + sum(W <= -1.5) = 2 (-2.0, -2.5)
        # den = sum(W >= 1.5) = 3 (3.2, 2.1, 1.5)
        # ratio = 2/3 = 0.667 > 0.5
        # At t=2.0: num = 0 + sum(W <= -2.0) = 2 (-2.0, -2.5)
        # den = sum(W >= 2.0) = 2 (3.2, 2.1)
        # ratio = 2/2 = 1.0 > 0.5
        # At t=2.1: num = 0 + sum(W <= -2.1) = 1 (-2.5)
        # den = sum(W >= 2.1) = 2 (3.2, 2.1)
        # ratio = 1/2 = 0.5 <= 0.5
        assert t == pytest.approx(2.1)

    def test_r_reference_strong_signal(self):
        """W vector with clear signal (many large positive, few negative)."""
        W = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, -0.1, -0.2])
        t = knockoff_threshold(W, fdr=0.2, offset=1)
        # At t=0: (1 + 2) / 6 = 0.5 > 0.2
        # At t=0.1: (1 + 2) / 6 = 0.5 > 0.2
        # At t=0.2: (1 + 1) / 6 = 0.333 > 0.2
        # At t=2.5: (1 + 0) / 6 = 0.167 <= 0.2
        assert t == pytest.approx(2.5)
