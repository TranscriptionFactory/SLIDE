"""
Test coverage for loveslide.tools utility functions.

Major gaps:
- init_data parameter parsing and validation
- Data loading from various file formats
- Parameter checking and validation logic
- Default value calculations
- Error handling for invalid inputs
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, mock_open

from loveslide.tools import init_data, show_params, check_params, calc_default_fsize


class TestInitData:
    """Test init_data function for data initialization."""

    def test_init_data_with_arrays(self):
        """Test init_data with numpy arrays."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        data, processed_params = init_data(params, x=X, y=y)

        assert hasattr(data, 'X')
        assert hasattr(data, 'y')
        assert data.X.shape == (100, 50)
        assert data.y.shape == (100,)

    def test_init_data_with_dataframes(self):
        """Test init_data with pandas DataFrames."""
        X = pd.DataFrame(np.random.randn(100, 50))
        y = pd.Series(np.random.randn(100))
        params = {"fdr": 0.1}

        data, processed_params = init_data(params, x=X, y=y)

        # Should convert to appropriate format
        assert data.X.shape == (100, 50)
        assert data.y.shape == (100,)

    def test_init_data_from_file_paths(self):
        """Test init_data loading from file paths."""
        # TODO: Create temporary files and test loading
        pass

    def test_init_data_mismatched_dimensions(self):
        """Test init_data error handling for mismatched X, y dimensions."""
        X = np.random.randn(100, 50)
        y = np.random.randn(90)  # Wrong length
        params = {"fdr": 0.1}

        with pytest.raises(ValueError):
            init_data(params, x=X, y=y)

    def test_init_data_missing_data(self):
        """Test init_data with missing/NaN values."""
        X = np.random.randn(100, 50)
        X[10, 5] = np.nan  # Introduce missing value
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        # Should either handle or raise informative error
        with pytest.raises((ValueError, RuntimeError)):
            init_data(params, x=X, y=y)

    def test_init_data_parameter_defaults(self):
        """Test that init_data sets appropriate parameter defaults."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        params = {}  # Empty params

        data, processed_params = init_data(params, x=X, y=y)

        # Should have reasonable defaults
        assert "fdr" in processed_params
        assert "niter" in processed_params
        # TODO: Verify actual default values

    def test_init_data_parameter_validation(self):
        """Test init_data validates parameters."""
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Invalid FDR
        params = {"fdr": -0.1}  # Negative FDR
        with pytest.raises(ValueError):
            init_data(params, x=X, y=y)

        # Invalid niter
        params = {"fdr": 0.1, "niter": 0}  # Zero iterations
        with pytest.raises(ValueError):
            init_data(params, x=X, y=y)


class TestInitDataYFactor:
    """init_data(y_factor=True) encodes y of any container type as 0..k-1 labels."""

    @pytest.mark.parametrize(
        "make_y",
        [
            lambda idx: np.array(["case", "ctrl", "case", "ctrl"]),
            lambda idx: np.array([3, 5, 3, 5]),
            lambda idx: np.array([[3], [5], [3], [5]]),
            lambda idx: pd.Series(["case", "ctrl", "case", "ctrl"], index=idx),
            lambda idx: pd.DataFrame({"y": [3.0, 5.0, 3.0, 5.0]}, index=idx),
        ],
        ids=["numpy-str", "numpy-int", "numpy-column", "series", "dataframe"],
    )
    def test_y_factor_encodes_any_y_container(self, make_y):
        idx = ["a", "b", "c", "d"]
        X = pd.DataFrame(np.random.randn(4, 3), index=idx)

        data, params = init_data({}, x=X, y=make_y(idx))

        assert params["y_factor"] is True
        assert isinstance(data.Y, (pd.Series, pd.DataFrame))
        assert data.Y.values.ravel().tolist() == [0, 1, 0, 1]
        assert list(data.Y.index) == idx

    def test_numpy_x_and_y_become_aligned_frames(self):
        data, _ = init_data({}, x=np.random.randn(4, 3), y=np.array([1, 0, 1, 0]))

        assert isinstance(data.X, pd.DataFrame) and isinstance(data.Y, pd.DataFrame)
        assert data.X.index.equals(data.Y.index)
        assert data.Y.values.ravel().tolist() == [1, 0, 1, 0]


class TestShowParams:
    """Test show_params function for parameter display."""

    def test_show_params_basic(self, capsys):
        """Test show_params displays parameters correctly."""
        params = {"fdr": 0.1, "niter": 10}

        # Mock data object
        class MockData:
            X = np.random.randn(100, 50)
            y = np.random.randn(100)

        data = MockData()

        show_params(params, data)

        captured = capsys.readouterr()
        assert "fdr" in captured.out
        assert "0.1" in captured.out
        assert "niter" in captured.out
        assert "10" in captured.out

    def test_show_params_empty_params(self, capsys):
        """Test show_params with empty parameters."""
        params = {}

        class MockData:
            X = np.random.randn(10, 5)
            y = np.random.randn(10)

        data = MockData()

        # Should not crash with empty params
        show_params(params, data)


class TestCheckParams:
    """Test check_params function for parameter validation."""

    def test_check_params_valid(self):
        """Test check_params with valid parameters."""
        params = {
            "fdr": 0.1,
            "niter": 10,
            "fsize": 50,
            "n_workers": 1
        }

        class MockData:
            X = np.random.randn(100, 80)
            y = np.random.randn(100)

        data = MockData()

        # Should not raise exception
        check_params(params, data)

    def test_check_params_invalid_fdr(self):
        """Test check_params catches invalid FDR values."""
        class MockData:
            X = np.random.randn(100, 50)
            y = np.random.randn(100)

        data = MockData()

        # FDR out of range
        params = {"fdr": 1.5}  # > 1.0
        with pytest.raises(ValueError):
            check_params(params, data)

        params = {"fdr": -0.1}  # < 0.0
        with pytest.raises(ValueError):
            check_params(params, data)

    def test_check_params_invalid_niter(self):
        """Test check_params catches invalid niter values."""
        params = {"fdr": 0.1, "niter": -5}  # Negative

        class MockData:
            X = np.random.randn(100, 50)
            y = np.random.randn(100)

        data = MockData()

        with pytest.raises(ValueError):
            check_params(params, data)

    def test_check_params_fsize_vs_p_relationship(self):
        """Test check_params validates fsize relative to number of features."""
        class MockData:
            X = np.random.randn(100, 20)  # 20 features
            y = np.random.randn(100)

        data = MockData()

        # fsize larger than total features
        params = {"fdr": 0.1, "fsize": 50}  # > 20 features
        # Should either adjust automatically or warn/error
        # TODO: Determine expected behavior


class TestCalcDefaultFsize:
    """Test calc_default_fsize function."""

    def test_calc_default_fsize_basic(self):
        """Test basic functionality of calc_default_fsize."""
        n_rows = 100
        K = 5

        fsize = calc_default_fsize(n_rows, K)

        assert isinstance(fsize, int)
        assert fsize > 0
        assert fsize > K  # Should be larger than number of factors

    def test_calc_default_fsize_edge_cases(self):
        """Test calc_default_fsize with edge cases."""
        # Very small n
        fsize_small = calc_default_fsize(10, 2)
        assert fsize_small > 0

        # Very large n
        fsize_large = calc_default_fsize(10000, 10)
        assert fsize_large > 0

        # Large K relative to n
        fsize_large_k = calc_default_fsize(50, 40)
        assert fsize_large_k > 0

    def test_calc_default_fsize_relationship(self):
        """Test that fsize scales appropriately with inputs."""
        n1, K = 100, 5
        n2, K = 200, 5

        fsize1 = calc_default_fsize(n1, K)
        fsize2 = calc_default_fsize(n2, K)

        # Larger n should generally allow larger fsize
        # TODO: Verify the actual relationship based on implementation


class TestToolsIntegration:
    """Integration tests for tools module functions."""

    def test_tools_workflow_integration(self):
        """Test complete workflow using tools functions."""
        # Create data
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        params = {"fdr": 0.1}

        # Initialize
        data, processed_params = init_data(params, x=X, y=y)

        # Check parameters
        check_params(processed_params, data)  # Should not raise

        # Calculate default fsize
        fsize = calc_default_fsize(100, 5)
        processed_params["fsize"] = fsize

        # Show parameters
        # show_params(processed_params, data)  # Would print output

        assert True  # If we get here, workflow completed

    def test_tools_error_propagation(self):
        """Test that errors propagate correctly through tools."""
        # TODO: Test error handling throughout the tools workflow
        pass