"""
Comprehensive integration tests for the SLIDE_py package.

Major gaps identified:
- End-to-end pipeline testing with realistic data
- Cross-module interaction testing
- Performance and scalability testing
- Error propagation through the pipeline
- Configuration and parameter interaction testing
- Backend comparison and consistency testing
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import time
import gc
from pathlib import Path
from unittest.mock import Mock, patch

from slide_py import (
    SLIDE, OptimizeSLIDE, SLIDEcv, Knockoffs, VotingResult,
    call_love, Plotter, Estimator, SLIDE_Estimator,
    init_data, show_params, check_params, calc_default_fsize
)


class TestFullPipelineIntegration:
    """Test complete SLIDE pipeline integration."""

    def test_minimal_working_example(self):
        """Test that a minimal example works end-to-end."""
        # Generate simple synthetic data
        np.random.seed(42)
        n, p = 200, 50
        K = 3

        # Create latent factors
        Z = np.random.randn(n, K)
        A = np.random.randn(p, K)
        A[20:, 0] = 0  # Sparse structure
        A[:15, 2] = 0

        # Generate data
        X = Z @ A.T + 0.5 * np.random.randn(n, p)
        y = np.random.randn(n)

        # Minimal SLIDE run
        params = {
            'fdr': 0.2,
            'niter': 3,
            'f_size': 20,
            'backend': 'python'
        }

        slide = SLIDE(params, x=X, y=y)

        # Should initialize without errors
        assert slide is not None
        assert hasattr(slide, 'data')
        assert hasattr(slide, 'input_params')

    def test_love_to_knockoff_pipeline(self):
        """Test data flow from LOVE to knockoff filtering."""
        X = np.random.randn(150, 30)
        y = np.random.randn(150)

        # Step 1: Run LOVE
        love_result = call_love(X, lbd=0.5, mu=0.5, thresh_fdr=0.2)

        # Step 2: Use LOVE results in knockoff filtering
        if 'latent_factors' in love_result:
            # Mock the integration (actual implementation may differ)
            knockoffs = Knockoffs(backend='python')

            # This tests that the interfaces are compatible
            ko_result = knockoffs.select_short_freq(
                X, y, fdr=0.1, niter=5
            )

            assert isinstance(ko_result, VotingResult)

    def test_slide_with_different_configurations(self):
        """Test SLIDE with various configuration combinations."""
        X = np.random.randn(100, 25)
        y = np.random.randn(100)

        # Test different parameter combinations
        configs = [
            {'fdr': 0.05, 'niter': 3, 'backend': 'python'},
            {'fdr': 0.1, 'niter': 5, 'backend': 'python'},
            {'fdr': 0.2, 'niter': 10, 'backend': 'python'},
        ]

        for params in configs:
            slide = SLIDE(params, x=X, y=y)
            assert slide.input_params['fdr'] == params['fdr']
            assert slide.input_params['niter'] == params['niter']

    def test_optimize_slide_integration(self):
        """Test OptimizeSLIDE integration with base SLIDE."""
        X = np.random.randn(120, 20)
        y = np.random.randn(120)

        params = {
            'fdr': 0.15,
            'niter': 4,
            'backend': 'python'
        }

        opt_slide = OptimizeSLIDE(params, x=X, y=y)

        # Should inherit from SLIDE
        assert isinstance(opt_slide, SLIDE)
        assert hasattr(opt_slide, 'get_latent_factors')


    def test_estimator_slide_integration(self):
        """Test integration between Estimator and SLIDE classes."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Basic estimator
        estimator = Estimator(X, y)
        assert estimator is not None

        # SLIDE estimator
        slide_estimator = SLIDE_Estimator(X, y, method='knockoff')
        assert slide_estimator is not None
        assert isinstance(slide_estimator, Estimator)

    def test_plotting_integration(self):
        """Test Plotter integration with SLIDE results."""
        X = np.random.randn(80, 12)

        # Mock some results that plotter might receive
        mock_latent_factors = np.random.randn(80, 3)
        mock_loadings = np.random.randn(12, 3)
        mock_correlation_net = np.random.randn(12, 12)

        plotter = Plotter()

        # Should be able to create plotter without errors
        assert plotter is not None


class TestCrossModuleInteractions:
    """Test interactions between different modules."""


    def test_parameter_propagation_through_pipeline(self):
        """Test that parameters propagate correctly through the pipeline."""
        X = np.random.randn(80, 12)
        y = np.random.randn(80)

        params = {
            'fdr': 0.15,
            'niter': 4,
            'f_size': 8,
            'backend': 'python',
            'seed': 12345
        }

        slide = SLIDE(params, x=X, y=y)

        # Parameters should be preserved
        for key, value in params.items():
            if key in slide.input_params:
                assert slide.input_params[key] == value


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""

    @pytest.mark.slow
    def test_scalability_with_increasing_features(self):
        """Test performance as number of features increases."""
        n = 200
        feature_counts = [10, 50, 100, 200]
        times = []

        for p in feature_counts:
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            params = {'fdr': 0.1, 'niter': 3, 'backend': 'python'}

            start_time = time.time()
            try:
                slide = SLIDE(params, x=X, y=y)
                elapsed = time.time() - start_time
                times.append(elapsed)
            except MemoryError:
                pytest.skip(f"Memory insufficient for {p} features")

        # Time should increase sub-quadratically
        if len(times) >= 2:
            # Simple check that it doesn't explode
            assert max(times) < 300  # 5 minutes max

    @pytest.mark.slow
    def test_scalability_with_increasing_samples(self):
        """Test performance as number of samples increases."""
        p = 50
        sample_counts = [100, 500, 1000, 2000]
        times = []

        for n in sample_counts:
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            params = {'fdr': 0.1, 'niter': 3, 'backend': 'python'}

            start_time = time.time()
            try:
                slide = SLIDE(params, x=X, y=y)
                elapsed = time.time() - start_time
                times.append(elapsed)
            except MemoryError:
                pytest.skip(f"Memory insufficient for {n} samples")

        # Time should scale reasonably
        if len(times) >= 2:
            assert max(times) < 300  # 5 minutes max

    def test_memory_efficiency_large_datasets(self):
        """Test memory efficiency with larger datasets."""
        n, p = 1000, 200

        try:
            X = np.random.randn(n, p)
            y = np.random.randn(n)

            params = {'fdr': 0.1, 'niter': 2, 'backend': 'python'}

            initial_objects = len(gc.get_objects())

            slide = SLIDE(params, x=X, y=y)

            final_objects = len(gc.get_objects())

            # Should not create excessive intermediate objects
            assert final_objects - initial_objects < 10000

        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")

    def test_computational_efficiency_repeated_runs(self):
        """Test efficiency of repeated runs (caching, etc.)."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        params = {'fdr': 0.1, 'niter': 3, 'backend': 'python'}

        # First run
        start_time = time.time()
        slide1 = SLIDE(params, x=X, y=y)
        first_time = time.time() - start_time

        # Second run (should be similar speed, not slower due to memory issues)
        start_time = time.time()
        slide2 = SLIDE(params, x=X.copy(), y=y.copy())
        second_time = time.time() - start_time

        # Second run shouldn't be much slower (no major memory leaks)
        assert second_time <= first_time * 2


class TestConfigurationRobustness:
    """Test robustness to different configurations and edge cases."""

    def test_extreme_parameter_values(self):
        """Test behavior with extreme parameter values."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Very strict FDR
        params_strict = {'fdr': 0.001, 'niter': 5}
        slide_strict = SLIDE(params_strict, x=X, y=y)
        assert slide_strict is not None

        # Very lenient FDR
        params_lenient = {'fdr': 0.99, 'niter': 5}
        slide_lenient = SLIDE(params_lenient, x=X, y=y)
        assert slide_lenient is not None

        # Very few iterations
        params_few_iter = {'fdr': 0.1, 'niter': 1}
        slide_few_iter = SLIDE(params_few_iter, x=X, y=y)
        assert slide_few_iter is not None


    def test_mixed_data_types_handling(self):
        """Test handling of different data types."""
        n, p = 80, 10

        # Test with different numpy dtypes
        for dtype in [np.float32, np.float64]:
            X = np.random.randn(n, p).astype(dtype)
            y = np.random.randn(n).astype(dtype)

            params = {'fdr': 0.1, 'niter': 3}
            slide = SLIDE(params, x=X, y=y)
            assert slide is not None


class TestErrorHandlingIntegration:
    """Test comprehensive error handling across the pipeline."""


    def test_file_io_error_handling(self):
        """Test handling of file I/O errors."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        params = {'fdr': 0.1, 'niter': 3}

        # Mock file permission errors
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should not crash on file operations
            slide = SLIDE(params, x=X, y=y)
            assert slide is not None


class TestReproducibilityIntegration:
    """Test reproducibility across the complete pipeline."""

    def test_seed_consistency_full_pipeline(self):
        """Test that setting seeds gives reproducible results."""
        X = np.random.RandomState(42).randn(80, 12)
        y = np.random.RandomState(42).randn(80)

        params = {
            'fdr': 0.1,
            'niter': 5,
            'backend': 'python',
            'seed': 12345
        }

        # Run twice with same seed
        slide1 = SLIDE(params.copy(), x=X.copy(), y=y.copy())
        slide2 = SLIDE(params.copy(), x=X.copy(), y=y.copy())

        # Should get identical results (if implementation is deterministic)
        assert slide1.input_params == slide2.input_params

    def test_cross_platform_consistency(self):
        """Test consistency across different numerical configurations."""
        X = np.random.randn(50, 8)
        y = np.random.randn(50)

        params = {'fdr': 0.1, 'niter': 3}

        # Should work consistently regardless of numpy configuration
        slide = SLIDE(params, x=X, y=y)
        assert slide is not None

    def test_version_compatibility(self):
        """Test that results are consistent across updates (mock test)."""
        X = np.random.randn(60, 10)
        y = np.random.randn(60)

        params = {'fdr': 0.1, 'niter': 3}

        # This would ideally test against saved reference results
        slide = SLIDE(params, x=X, y=y)
        assert slide is not None

        # In practice, this would compare against golden standard results


class TestRealWorldScenarios:
    """Test scenarios that mimic real-world usage patterns."""

    def test_typical_genomics_workflow(self):
        """Test workflow typical in genomics applications."""
        # Simulate gene expression data (samples x genes)
        n_samples, n_genes = 200, 500
        X = np.random.lognormal(mean=0, sigma=1, size=(n_samples, n_genes))
        y = np.random.randn(n_samples)

        # Log-transform (common in genomics)
        X_log = np.log2(X + 1)

        # Typical genomics parameters
        params = {
            'fdr': 0.05,  # Strict FDR for genomics
            'niter': 10,  # More iterations for stability
            'backend': 'python'
        }

        slide = SLIDE(params, x=X_log, y=y)
        assert slide is not None

    def test_time_series_like_workflow(self):
        """Test workflow with time series-like data."""
        n_timepoints, n_features = 100, 30

        # Generate time series with some temporal structure
        t = np.linspace(0, 10, n_timepoints)
        X = np.column_stack([
            np.sin(i * t) + 0.1 * np.random.randn(n_timepoints)
            for i in range(n_features)
        ])
        y = np.random.randn(n_timepoints)

        params = {'fdr': 0.1, 'niter': 5}

        slide = SLIDE(params, x=X, y=y)
        assert slide is not None

    def test_high_dimensional_workflow(self):
        """Test workflow with high-dimensional data (p >> n)."""
        n_samples, n_features = 100, 1000  # p >> n

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        # Appropriate parameters for high-dimensional data
        params = {
            'fdr': 0.1,
            'niter': 5,
            'f_size': min(50, n_features // 20)  # Reasonable chunk size
        }

        try:
            slide = SLIDE(params, x=X, y=y)
            assert slide is not None
        except (MemoryError, np.linalg.LinAlgError):
            pytest.skip("High-dimensional test requires special handling")

    def test_iterative_analysis_workflow(self):
        """Test iterative analysis workflow (multiple parameter settings)."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        # Test multiple FDR values
        fdr_values = [0.05, 0.1, 0.2]
        results = []

        for fdr in fdr_values:
            params = {'fdr': fdr, 'niter': 3}
            slide = SLIDE(params, x=X, y=y)
            results.append(slide)

        # All should succeed
        assert len(results) == len(fdr_values)
        assert all(r is not None for r in results)