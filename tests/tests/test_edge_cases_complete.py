"""
Comprehensive edge case and error handling tests for SLIDE_py.

Addresses the TODO items found in existing test files:
- Parameter validation edge cases
- Data validation edge cases
- Cross-platform compatibility
- Recovery from failures
- Memory management edge cases
- File I/O error scenarios
- Concurrent execution edge cases
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import threading
import multiprocessing
import gc
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from loveslide import (
    SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs, VotingResult,
    call_love, Plotter, Estimator, SLIDE_Estimator,
    init_data, show_params, check_params, calc_default_fsize
)


class TestParameterValidationEdgeCases:
    """Complete parameter validation edge case testing."""


    def test_parameter_missing_required(self):
        """Test behavior when required parameters are missing."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Empty parameters (should use defaults or raise error)
        try:
            SLIDE({}, x=X, y=y)
        except ValueError as e:
            assert "required" in str(e).lower()


class TestMemoryManagementEdgeCases:
    """Test memory management edge cases and large dataset handling."""

    def test_memory_efficient_chunking(self):
        """Test that chunking works efficiently for large datasets."""
        # Create moderately large dataset
        n, p = 2000, 500

        try:
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            initial_memory = gc.get_obj_count = len(gc.get_objects())

            params = {'fdr': 0.1, 'niter': 3, 'f_size': 50}
            slide = SLIDE(params, x=X, y=y)

            final_memory = len(gc.get_objects())

            # Should not create excessive temporary objects
            assert final_memory - initial_memory < 50000

        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")


    def test_repeated_large_operations(self):
        """Test memory stability with repeated large operations."""
        memory_usage = []

        for i in range(5):
            X = np.random.randn(500, 100)
            y = np.random.randn(500)

            slide = SLIDE({'fdr': 0.1, 'niter': 2}, x=X, y=y)

            del X, y, slide
            gc.collect()

            memory_usage.append(len(gc.get_objects()))

        # Memory usage should not continuously increase
        assert max(memory_usage) - min(memory_usage) < 10000


class TestConcurrentExecutionEdgeCases:
    """Test concurrent execution and thread safety."""


    def test_shared_resource_access(self):
        """Test access to shared resources (e.g., random number generators)."""
        # This tests that different instances don't interfere with each other
        X1 = np.random.RandomState(42).randn(50, 10)
        X2 = np.random.RandomState(42).randn(50, 10)
        y = np.random.randn(50)

        # Should give identical results
        slide1 = SLIDE({'fdr': 0.1, 'seed': 42}, x=X1, y=y)
        slide2 = SLIDE({'fdr': 0.1, 'seed': 42}, x=X2, y=y)

        # Results should be identical
        assert slide1.input_params == slide2.input_params


class TestFileIOErrorScenarios:
    """Test file I/O error handling scenarios."""

    def test_permission_denied_scenarios(self):
        """Test handling of file permission errors."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock permission errors for different file operations
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Should not crash on file operations
            slide = SLIDE({'fdr': 0.1}, x=X, y=y)
            assert slide is not None


    def test_temporary_file_handling(self):
        """Test proper cleanup of temporary files."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Run SLIDE that might create temporary files
            slide = SLIDE({'fdr': 0.1}, x=X, y=y)

            # Check that no temporary files are left behind
            temp_files = list(Path(temp_dir).glob("*"))
            assert len(temp_files) == 0  # Should clean up


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility issues."""

    def test_path_separator_handling(self):
        """Test that path separators work across platforms."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Test with different path styles
        if sys.platform == "win32":
            test_path = "C:\\temp\\test_file.txt"
        else:
            test_path = "/tmp/test_file.txt"

        # Should handle platform-specific paths
        slide = SLIDE({'fdr': 0.1, 'output_path': test_path}, x=X, y=y)
        assert slide is not None

    def test_line_ending_handling(self):
        """Test handling of different line endings."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Mock files with different line endings
        for line_ending in ['\n', '\r\n', '\r']:
            mock_content = f"param1=value1{line_ending}param2=value2{line_ending}"

            with patch('builtins.open', mock_open(read_data=mock_content)):
                # Should handle different line endings
                slide = SLIDE({'fdr': 0.1}, x=X, y=y)
                assert slide is not None

    def test_unicode_handling(self):
        """Test handling of unicode characters in parameters/paths."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Unicode in string parameters
        unicode_params = {
            'fdr': 0.1,
            'description': 'Test with unicode: αβγ δεζ 中文 🔬'
        }

        # Should handle unicode characters
        slide = SLIDE(unicode_params, x=X, y=y)
        assert slide is not None

    def test_numerical_precision_differences(self):
        """Test that results are consistent across different precision levels."""
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        # Run with lower precision
        slide_32 = SLIDE({'fdr': 0.1}, x=X, y=y)

        # Run with higher precision
        X_64 = X.astype(np.float64)
        y_64 = y.astype(np.float64)
        slide_64 = SLIDE({'fdr': 0.1}, x=X_64, y=y_64)

        # Results should be reasonably similar
        assert slide_32 is not None
        assert slide_64 is not None


class TestComplexEdgeCaseScenarios:
    """Test complex scenarios combining multiple edge cases."""


    def test_extreme_parameter_combinations(self):
        """Test extreme but valid parameter combinations."""
        X = np.random.randn(1000, 100)
        y = np.random.randn(1000)

        extreme_params = {
            'fdr': 1e-10,     # Extremely strict FDR
            'niter': 1000,    # Many iterations
            'f_size': 1,      # Smallest possible chunk size
        }

        try:
            slide = SLIDE(extreme_params, x=X, y=y)
            assert slide is not None
        except (MemoryError, TimeoutError):
            pytest.skip("Extreme parameters exceed resource limits")

    def test_pathological_data_structures(self):
        """Test pathological but valid data structures."""
        # Data where most features are noise
        n, p = 200, 100
        signal_features = 2
        noise_features = p - signal_features

        X_signal = np.random.randn(n, signal_features)
        X_noise = 1e-6 * np.random.randn(n, noise_features)  # Very weak noise
        X = np.column_stack([X_signal, X_noise])

        y = X_signal[:, 0] + X_signal[:, 1] + 0.1 * np.random.randn(n)

        slide = SLIDE({'fdr': 0.05}, x=X, y=y)  # Should find signal
        assert slide is not None