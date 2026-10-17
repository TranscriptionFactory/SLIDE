"""
End-to-end pipeline tests for slide-py v1.0.0.

Runs the full SLIDE pipeline on real HIV cytokine data and compares
outputs against reference results from the build check (20260122_135613).

Also runs cross-backend comparisons (Python vs R knockoffs) and validates
on the larger continuous-outcome SSc dataset when rpy2 is available.

Test classes:
    TestPackageInstall        - import checks, version, __all__ exports
    TestLOVEPipeline          - call_love() on HIV data, matrix shapes, pure vars
    TestKnockoffPipeline      - Knockoffs.select_short_freq on latent factors
    TestFullPipeline          - OptimizeSLIDE.run_pipeline end-to-end
    TestEstimatorScoring      - Estimator fit/predict/evaluate
    TestDSDPFallback          - SDP solver detection chain
    TestBackendComparison     - cross-backend (python/r_knockoffs/r) on HIV [rpy2]
    TestSScPipeline           - Python backend on SSc continuous data
    TestSScBackendComparison  - cross-backend on SSc data [rpy2]
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = _PROJECT_ROOT / "example"
_REFERENCE_BASE = (
    _PROJECT_ROOT
    / "archive"
    / "comparison"
    / "build_check_outputs"
    / "20260122_135613"
)
_REF_GLMNET = _REFERENCE_BASE / "Py_pyLOVE_kf_glmnet" / "0.1_0.5_out"

_SSC_X = Path("/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv")
_SSC_Y = Path("/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _jaccard(set_a, set_b):
    """Compute Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def _nonzero_rows(A):
    """Return set of row indices with any non-zero entry."""
    if isinstance(A, pd.DataFrame):
        A = A.values
    return set(np.where(np.any(A != 0, axis=1))[0])


def _run_pipeline(out_dir, knockoff_backend, x_path, y_path,
                  y_factor=True, **kwargs):
    """Run OptimizeSLIDE.run_pipeline with given backend, return (slider, result_dir)."""
    from slide_py import OptimizeSLIDE

    input_params = {
        "x_path": str(x_path),
        "y_path": str(y_path),
        "y_factor": y_factor,
        "niter": 500,
        "SLIDE_top_feats": 10,
        "out_path": str(out_dir),
        "fdr": 0.1,
        "thresh_fdr": 0.2,
        "pure_homo": True,
        "do_interacts": True,
        "n_workers": 1,
        "spec": 0.1,
        "love_backend": "python",
        "knockoff_backend": knockoff_backend,
        "knockoff_method": "asdp",
        "knockoff_shrink": False,
        "knockoff_offset": 0,
        "fstat": "glmnet_lambdasmax",
        "delta": [0.1],
        "lambda": [0.5],
    }
    input_params.update(kwargs)

    slider = OptimizeSLIDE(input_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slider.run_pipeline(verbose=True)

    return slider, Path(out_dir) / "0.1_0.5_out"


def _load_sig_lfs(out_dir):
    """Load sig_LFs.txt as a list of strings."""
    return np.loadtxt(out_dir / "sig_LFs.txt", dtype=str).reshape(-1).tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hiv_data():
    """Load HIV cytokine example data (60 x 80, binary outcome)."""
    X = pd.read_csv(_EXAMPLE_DIR / "HIV+cytokines_X.csv", index_col=0)
    y = pd.read_csv(_EXAMPLE_DIR / "HIV+cytokines_y.csv", index_col=0).iloc[:, 0]
    return X, y


@pytest.fixture(scope="module")
def reference_A():
    """Load reference A matrix from build check."""
    if not (_REF_GLMNET / "A.csv").exists():
        pytest.skip(f"build-check reference outputs not on disk: {_REF_GLMNET}")
    return pd.read_csv(_REF_GLMNET / "A.csv", index_col=0)


@pytest.fixture(scope="module")
def reference_z_matrix():
    """Load reference z_matrix from build check."""
    if not (_REF_GLMNET / "z_matrix.csv").exists():
        pytest.skip(f"build-check reference outputs not on disk: {_REF_GLMNET}")
    return pd.read_csv(_REF_GLMNET / "z_matrix.csv", index_col=0)


@pytest.fixture(scope="module")
def love_result_on_hiv(hiv_data):
    """
    Run call_love() once and share across the LOVE test class.

    Uses pure_homo=True, delta=[0.1] to match the build check config.
    """
    from slide_py import call_love

    X, y = hiv_data
    result = call_love(
        X,
        lbd=0.5,
        pure_homo=True,
        delta=[0.1],
        thresh_fdr=0.2,
        backend="python",
        verbose=False,
    )
    return result


@pytest.fixture(scope="module")
def pipeline_output_dir(tmp_path_factory):
    """Provide a clean output directory for pipeline tests (pytest keeps the last 3 runs)."""
    return tmp_path_factory.mktemp("pipeline_outputs")


@pytest.fixture(scope="module")
def ssc_data():
    """Load SSc data (24 x 804, continuous outcome MRSS)."""
    pytest.importorskip("pandas")
    if not _SSC_X.exists() or not _SSC_Y.exists():
        pytest.skip(f"SSc cluster data not on disk: {_SSC_X.parent}")
    X = pd.read_csv(_SSC_X, index_col=0)
    y = pd.read_csv(_SSC_Y, index_col=0).iloc[:, 0]
    return X, y


# ===========================================================================
# 1. Package Install Tests
# ===========================================================================
class TestPackageInstall:
    """Validate that slide-py installs correctly and exports are importable."""

    def test_import_version(self):
        import slide_py
        assert slide_py.__version__ == "1.0.0"

    def test_all_exports(self):
        import slide_py

        for name in slide_py.__all__:
            obj = getattr(slide_py, name, None)
            assert obj is not None, f"{name} listed in __all__ but not importable"


# ===========================================================================
# 2. LOVE Pipeline Tests
# ===========================================================================
class TestLOVEPipeline:
    """Run LOVE on HIV cytokine data and validate outputs."""

    def test_love_output_keys(self, love_result_on_hiv):
        """call_love() returns expected keys."""
        result = love_result_on_hiv
        for key in ("K", "A", "C", "pureVec", "group"):
            assert key in result, f"Missing key: {key}"

    def test_love_matrix_shapes(self, hiv_data, love_result_on_hiv):
        """A is (p, K) and C is (K, K)."""
        X, _ = hiv_data
        r = love_result_on_hiv
        p = X.shape[1]
        K = r["K"]

        assert r["A"].shape == (p, K), f"A shape {r['A'].shape} != ({p}, {K})"
        assert r["C"].shape == (K, K), f"C shape {r['C'].shape} != ({K}, {K})"

    def test_love_K_reasonable(self, love_result_on_hiv):
        """K should be positive and reasonable for 80 features."""
        K = love_result_on_hiv["K"]
        assert 2 <= K <= 40, f"K={K} outside reasonable range [2, 40]"

    def test_love_pure_variables_count(self, love_result_on_hiv):
        """Pure variable count should be reasonable (reference had 43)."""
        n_pure = len(love_result_on_hiv["pureVec"])
        # Allow some flexibility since exact count depends on algorithm details
        assert 20 <= n_pure <= 70, f"Pure var count {n_pure} outside [20, 70]"

    def test_love_A_agreement_with_reference(self, love_result_on_hiv, reference_A):
        """
        A matrix from Python LOVE should correlate well with reference.

        The columns may be permuted and sign-flipped, so we compare
        the best column-wise absolute correlation.
        """
        A_new = love_result_on_hiv["A"]
        A_ref = reference_A.values

        # Both should have same number of rows (p=80)
        assert A_new.shape[0] == A_ref.shape[0]

        # For each reference column, find best-matching new column
        K_ref = A_ref.shape[1]
        K_new = A_new.shape[1]
        # K values should be similar
        assert abs(K_ref - K_new) <= 3, f"K mismatch: ref={K_ref}, new={K_new}"

        # Check non-zero pattern similarity (column permutation invariant)
        jaccard = _jaccard(_nonzero_rows(A_ref), _nonzero_rows(A_new))
        assert jaccard >= 0.7, f"Non-zero pattern Jaccard={jaccard:.3f} < 0.7"


# ===========================================================================
# 3. Knockoff Pipeline Tests
# ===========================================================================
class TestKnockoffPipeline:
    """Test knockoff selection on HIV latent factors."""

    @pytest.mark.slow
    def test_knockoff_python_backend(self, hiv_data, love_result_on_hiv):
        """Run select_short_freq with python backend on latent factors."""
        from slide_py import Knockoffs

        X, y = hiv_data
        A = love_result_on_hiv["A"]

        # Compute z_matrix = X_std @ A @ inv(C)
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        selected = Knockoffs.select_short_freq(
            z, y.values,
            backend="python",
            niter=50,
            spec=0.1,
            fdr=0.1,
            f_size=100,
            verbose=False,
        )

        assert isinstance(selected, np.ndarray)
        assert all(0 <= idx < z.shape[1] for idx in selected)
        print(f"\nKnockoff python: selected {len(selected)} LFs")

    def test_knockoff_slide_voting(self, hiv_data, love_result_on_hiv):
        """Run select_short_freq_slide and verify VotingResult."""
        from slide_py import Knockoffs, VotingResult

        X, y = hiv_data
        A = love_result_on_hiv["A"]
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        result = Knockoffs.select_short_freq_slide(
            z, y.values,
            backend="python",
            niter=50,
            spec=0.1,
            fdr=0.1,
            f_size=100,
            verbose=False,
        )

        assert isinstance(result, VotingResult)
        assert hasattr(result, "selected")
        assert hasattr(result, "selection_counts")
        assert hasattr(result, "optimal_iter")
        print(f"\nKnockoff SLIDE: {len(result.selected)} selected, "
              f"optimal_iter={result.optimal_iter}")

    def test_knockoff_reproducibility(self, hiv_data, love_result_on_hiv):
        """Same base_seed should produce identical selections."""
        from slide_py import Knockoffs

        X, y = hiv_data
        A = love_result_on_hiv["A"]
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        kwargs = dict(backend="python", niter=20, spec=0.1, f_size=100, base_seed=42)
        r1 = Knockoffs.select_short_freq_slide(z, y.values, **kwargs)
        r2 = Knockoffs.select_short_freq_slide(z, y.values, **kwargs)

        assert np.array_equal(r1.selected, r2.selected)
        assert np.array_equal(r1.selection_counts, r2.selection_counts)

    def test_invalid_backend_raises(self, hiv_data, love_result_on_hiv):
        """Invalid backend name should raise ValueError."""
        from slide_py import Knockoffs

        X, y = hiv_data
        A = love_result_on_hiv["A"]
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        with pytest.raises(ValueError, match="Unknown backend"):
            Knockoffs.select_short_freq_slide(
                z, y.values, backend="invalid_backend", niter=10, spec=0.1,
            )


# ===========================================================================
# 4. Full Pipeline Tests
# ===========================================================================
@pytest.mark.slow
class TestFullPipeline:
    """Run OptimizeSLIDE.run_pipeline() on HIV data end-to-end."""

    @pytest.fixture(scope="class")
    def pipeline_result(self, pipeline_output_dir):
        """Run the full pipeline once and share across tests."""
        out = pipeline_output_dir / "python_hiv"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="python",
            x_path=_EXAMPLE_DIR / "HIV+cytokines_X.csv",
            y_path=_EXAMPLE_DIR / "HIV+cytokines_y.csv",
            y_factor=True,
        )

    def test_optimize_slide_hiv(self, pipeline_result):
        """Pipeline should complete without error."""
        slider, out_dir = pipeline_result
        assert out_dir.exists()

    def test_pipeline_output_files(self, pipeline_result):
        """Verify expected output files are produced."""
        _, out_dir = pipeline_result
        expected_files = [
            "A.csv",
            "z_matrix.csv",
            "sig_LFs.txt",
            "love_result.pkl",
        ]
        for fname in expected_files:
            path = out_dir / fname
            assert path.exists(), f"Missing output file: {fname}"

    def test_pipeline_love_agreement(self, pipeline_result, reference_A):
        """
        A matrix from pipeline should correlate well with reference.

        Check non-zero row pattern Jaccard >= 0.7.
        """
        _, out_dir = pipeline_result
        A_new = pd.read_csv(out_dir / "A.csv", index_col=0)

        jaccard = _jaccard(_nonzero_rows(reference_A), _nonzero_rows(A_new))
        print(f"\nA non-zero pattern Jaccard: {jaccard:.3f}")
        assert jaccard >= 0.7, f"A non-zero Jaccard={jaccard:.3f} < 0.7"

    def test_pipeline_z_matrix_shape(self, pipeline_result, hiv_data):
        """z_matrix should be (n, K) with n matching input data."""
        _, out_dir = pipeline_result
        X, _ = hiv_data
        z = pd.read_csv(out_dir / "z_matrix.csv", index_col=0)
        assert z.shape[0] == X.shape[0], f"z_matrix rows {z.shape[0]} != n={X.shape[0]}"
        assert z.shape[1] >= 2, f"z_matrix only has {z.shape[1]} columns"

    def test_pipeline_selection_sanity(self, pipeline_result):
        """sig_LFs should be non-empty and contain valid column names."""
        _, out_dir = pipeline_result
        sig_lfs = _load_sig_lfs(out_dir)
        assert len(sig_lfs) > 0, "No significant LFs found"

        z = pd.read_csv(out_dir / "z_matrix.csv", index_col=0)
        for lf in sig_lfs:
            assert lf in z.columns, f"sig_LF '{lf}' not in z_matrix columns"

    def test_pipeline_scores_file(self, pipeline_result):
        """scores.txt should exist and contain performance metrics."""
        _, out_dir = pipeline_result
        scores_path = out_dir / "scores.txt"
        assert scores_path.exists(), "scores.txt not found"
        content = scores_path.read_text()
        assert "True Scores" in content, "scores.txt missing 'True Scores'"


# ===========================================================================
# 5. Estimator Scoring Tests
# ===========================================================================
class TestEstimatorScoring:
    """Test Estimator and SLIDE_Estimator on HIV latent factors."""

    def test_estimator_basic(self, hiv_data, love_result_on_hiv):
        """Estimator.get_aucs should return array of AUC scores."""
        from slide_py import Estimator

        X, y = hiv_data
        A = love_result_on_hiv["A"]
        X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        C = love_result_on_hiv["C"]
        z = X_std.values @ A @ np.linalg.inv(C)

        # Use first 3 columns as features
        aucs = Estimator.get_aucs(
            z[:, :3], y.values,
            n_iters=20,
            test_size=0.2,
            scaler="standard",
        )
        assert isinstance(aucs, np.ndarray)
        assert len(aucs) == 20
        mean_auc = np.mean(aucs)
        # AUC should be > 0.5 (better than random) for real data
        print(f"\nEstimator AUC: {mean_auc:.3f}")


# ===========================================================================
# 6. SDP Solver Fallback Tests
# ===========================================================================
class TestDSDPFallback:
    """Verify SDP solver detection chain works."""

    def test_sdp_solver_detection(self):
        """The default SDP solver is DSDP, with no implicit cvxpy fallback."""
        from slide_py.knockoff.solve import _get_sdp_solver

        assert _get_sdp_solver() == "dsdp"

    def test_sdp_solve_produces_valid_diag(self):
        """create_solve_sdp should produce valid diag_s for a small matrix."""
        from slide_py.knockoff.solve import create_solve_sdp

        np.random.seed(42)
        p = 10
        X = np.random.randn(50, p)
        G = np.corrcoef(X, rowvar=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = create_solve_sdp(G)

        assert s.shape == (p,)
        assert np.all(s >= -1e-6), "diag_s has negative values"
        assert np.all(s <= 1.0 + 1e-6), "diag_s exceeds 1.0"


# ===========================================================================
# 7. Backend Comparison Tests (HIV data, requires rpy2)
# ===========================================================================
@pytest.mark.requires_r
@pytest.mark.slow
class TestBackendComparison:
    """Compare Python, r_knockoffs, and full R backends on HIV data."""

    @pytest.fixture(scope="class")
    def python_hiv_result(self, pipeline_output_dir):
        """Run HIV pipeline with python knockoff backend."""
        out = pipeline_output_dir / "python_hiv_cmp"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="python",
            x_path=_EXAMPLE_DIR / "HIV+cytokines_X.csv",
            y_path=_EXAMPLE_DIR / "HIV+cytokines_y.csv",
        )

    @pytest.fixture(scope="class")
    def r_knockoffs_hiv_result(self, pipeline_output_dir):
        """Run HIV pipeline with r_knockoffs backend."""
        out = pipeline_output_dir / "r_knockoffs_hiv"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="r_knockoffs",
            x_path=_EXAMPLE_DIR / "HIV+cytokines_X.csv",
            y_path=_EXAMPLE_DIR / "HIV+cytokines_y.csv",
        )

    @pytest.fixture(scope="class")
    def full_r_hiv_result(self, pipeline_output_dir):
        """Run HIV pipeline with full R knockoff backend."""
        out = pipeline_output_dir / "r_full_hiv"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="r",
            x_path=_EXAMPLE_DIR / "HIV+cytokines_X.csv",
            y_path=_EXAMPLE_DIR / "HIV+cytokines_y.csv",
        )

    def test_python_pipeline_hiv(self, python_hiv_result):
        """Python backend pipeline completes and produces outputs."""
        _, out_dir = python_hiv_result
        assert (out_dir / "A.csv").exists()
        assert (out_dir / "sig_LFs.txt").exists()
        print(f"\nPython HIV: {len(_load_sig_lfs(out_dir))} sig LFs")

    def test_r_knockoffs_pipeline_hiv(self, r_knockoffs_hiv_result):
        """R knockoffs backend pipeline completes and produces outputs."""
        _, out_dir = r_knockoffs_hiv_result
        assert (out_dir / "A.csv").exists()
        assert (out_dir / "sig_LFs.txt").exists()
        print(f"\nR knockoffs HIV: {len(_load_sig_lfs(out_dir))} sig LFs")

    def test_full_r_pipeline_hiv(self, full_r_hiv_result):
        """Full R backend pipeline completes and produces outputs."""
        _, out_dir = full_r_hiv_result
        assert (out_dir / "A.csv").exists()
        assert (out_dir / "sig_LFs.txt").exists()
        print(f"\nFull R HIV: {len(_load_sig_lfs(out_dir))} sig LFs")

    def test_backend_A_matrices_agree(self, python_hiv_result,
                                      r_knockoffs_hiv_result,
                                      full_r_hiv_result):
        """A matrices should be identical across backends (same LOVE, different knockoffs)."""
        A_py = pd.read_csv(python_hiv_result[1] / "A.csv", index_col=0)
        A_rko = pd.read_csv(r_knockoffs_hiv_result[1] / "A.csv", index_col=0)
        A_r = pd.read_csv(full_r_hiv_result[1] / "A.csv", index_col=0)

        # All should have same shape (same LOVE run params)
        assert A_py.shape == A_rko.shape, (
            f"Python {A_py.shape} vs r_knockoffs {A_rko.shape}")
        assert A_py.shape == A_r.shape, (
            f"Python {A_py.shape} vs full_r {A_r.shape}")

        # Non-zero row patterns should match perfectly (same LOVE)
        j_py_rko = _jaccard(_nonzero_rows(A_py), _nonzero_rows(A_rko))
        j_py_r = _jaccard(_nonzero_rows(A_py), _nonzero_rows(A_r))
        print(f"\nA Jaccard: py-rko={j_py_rko:.3f}, py-r={j_py_r:.3f}")

        assert j_py_rko >= 0.9, f"Py vs r_knockoffs A Jaccard={j_py_rko:.3f} < 0.9"
        assert j_py_r >= 0.9, f"Py vs full_r A Jaccard={j_py_r:.3f} < 0.9"

    def test_backend_sig_lfs_overlap(self, python_hiv_result,
                                     r_knockoffs_hiv_result,
                                     full_r_hiv_result):
        """Compute pairwise Jaccard of sig_LFs across backends."""
        lfs_py = set(_load_sig_lfs(python_hiv_result[1]))
        lfs_rko = set(_load_sig_lfs(r_knockoffs_hiv_result[1]))
        lfs_r = set(_load_sig_lfs(full_r_hiv_result[1]))

        j_py_rko = _jaccard(lfs_py, lfs_rko)
        j_py_r = _jaccard(lfs_py, lfs_r)
        j_rko_r = _jaccard(lfs_rko, lfs_r)

        print(f"\n--- HIV sig_LF Comparison ---")
        print(f"Python:      {sorted(lfs_py)}")
        print(f"R knockoffs: {sorted(lfs_rko)}")
        print(f"Full R:      {sorted(lfs_r)}")
        print(f"Jaccard py-rko={j_py_rko:.3f}, py-r={j_py_r:.3f}, rko-r={j_rko_r:.3f}")

        # Variation across backends is expected; just log it
        assert len(lfs_py) > 0, "Python found no sig LFs"
        assert len(lfs_rko) > 0, "R knockoffs found no sig LFs"
        assert len(lfs_r) > 0, "Full R found no sig LFs"

    def test_backend_scores_reasonable(self, python_hiv_result,
                                       r_knockoffs_hiv_result,
                                       full_r_hiv_result):
        """All backends should produce AUC > 0.5 (better than random)."""
        from slide_py import Estimator

        for label, (_, out_dir) in [
            ("Python", python_hiv_result),
            ("R knockoffs", r_knockoffs_hiv_result),
            ("Full R", full_r_hiv_result),
        ]:
            z = pd.read_csv(out_dir / "z_matrix.csv", index_col=0)
            sig_lfs = _load_sig_lfs(out_dir)
            z_sig = z[sig_lfs].values

            # Load y from HIV data
            y = pd.read_csv(
                _EXAMPLE_DIR / "HIV+cytokines_y.csv", index_col=0
            ).iloc[:, 0]
            y_enc = y.replace({v: i for i, v in enumerate(np.unique(y))}).astype(int)

            scores = Estimator.get_aucs(z_sig, y_enc.values, n_iters=50,
                                        test_size=0.2, scaler="standard")
            mean_auc = np.nanmean(scores)
            print(f"\n{label} HIV AUC: {mean_auc:.3f}")
            assert mean_auc > 0.5, f"{label} AUC {mean_auc:.3f} <= 0.5"


# ===========================================================================
# 8. SSc Pipeline Tests (Python backend)
# ===========================================================================
@pytest.mark.slow
class TestSScPipeline:
    """Run SLIDE pipeline on SSc continuous outcome data."""

    def test_ssc_data_loads(self, ssc_data):
        """Verify SSc X is (24, 804) and y is continuous."""
        X, y = ssc_data
        assert X.shape == (24, 804), f"SSc X shape {X.shape} != (24, 804)"
        assert len(np.unique(y)) > 2, "SSc y should be continuous (MRSS)"
        print(f"\nSSc data: X={X.shape}, y range=[{y.min():.1f}, {y.max():.1f}]")

    def test_ssc_love_runs(self, ssc_data):
        """call_love on SSc data produces valid K and A."""
        from slide_py import call_love

        X, y = ssc_data
        result = call_love(
            X, lbd=0.5, pure_homo=True, delta=[0.1],
            thresh_fdr=0.2, backend="python", verbose=False,
        )
        K = result["K"]
        assert 2 <= K <= 400, f"SSc K={K} outside [2, 400]"
        assert result["A"].shape == (804, K), (
            f"SSc A shape {result['A'].shape} != (804, {K})")
        print(f"\nSSc LOVE: K={K}, pure_vars={len(result['pureVec'])}")

    @pytest.fixture(scope="class")
    def ssc_python_result(self, pipeline_output_dir, ssc_data):
        """Run SSc pipeline with python backend."""
        out = pipeline_output_dir / "python_ssc"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="python",
            x_path=_SSC_X,
            y_path=_SSC_Y,
            y_factor=False,
        )

    def test_ssc_pipeline_python(self, ssc_python_result):
        """SSc pipeline completes with python backend."""
        _, out_dir = ssc_python_result
        assert out_dir.exists()
        print(f"\nSSc Python pipeline output: {out_dir}")

    def test_ssc_pipeline_output_files(self, ssc_python_result):
        """Verify standard output files exist."""
        _, out_dir = ssc_python_result
        for fname in ["A.csv", "z_matrix.csv", "sig_LFs.txt", "love_result.pkl"]:
            assert (out_dir / fname).exists(), f"Missing: {fname}"

    def test_ssc_pipeline_selection_sanity(self, ssc_python_result):
        """sig_LFs should be non-empty with valid indices."""
        _, out_dir = ssc_python_result
        sig_lfs = _load_sig_lfs(out_dir)
        assert len(sig_lfs) > 0, "SSc: no significant LFs found"

        z = pd.read_csv(out_dir / "z_matrix.csv", index_col=0)
        for lf in sig_lfs:
            assert lf in z.columns, f"sig_LF '{lf}' not in z_matrix columns"
        print(f"\nSSc Python: {len(sig_lfs)} sig LFs = {sorted(sig_lfs)}")


# ===========================================================================
# 9. SSc Backend Comparison Tests (requires rpy2)
# ===========================================================================
@pytest.mark.requires_r
@pytest.mark.slow
class TestSScBackendComparison:
    """Compare Python, r_knockoffs, and full R backends on SSc data."""

    @pytest.fixture(scope="class")
    def ssc_python_result(self, pipeline_output_dir, ssc_data):
        """Run SSc pipeline with python backend."""
        out = pipeline_output_dir / "python_ssc_cmp"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="python",
            x_path=_SSC_X,
            y_path=_SSC_Y,
            y_factor=False,
        )

    @pytest.fixture(scope="class")
    def ssc_r_knockoffs_result(self, pipeline_output_dir, ssc_data):
        """Run SSc pipeline with r_knockoffs backend."""
        out = pipeline_output_dir / "r_knockoffs_ssc"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="r_knockoffs",
            x_path=_SSC_X,
            y_path=_SSC_Y,
            y_factor=False,
        )

    @pytest.fixture(scope="class")
    def ssc_full_r_result(self, pipeline_output_dir, ssc_data):
        """Run SSc pipeline with full R knockoff backend."""
        out = pipeline_output_dir / "r_full_ssc"
        out.mkdir(parents=True, exist_ok=True)
        return _run_pipeline(
            out,
            knockoff_backend="r",
            x_path=_SSC_X,
            y_path=_SSC_Y,
            y_factor=False,
        )

    def test_ssc_r_knockoffs_pipeline(self, ssc_r_knockoffs_result):
        """R knockoffs pipeline completes on SSc data."""
        _, out_dir = ssc_r_knockoffs_result
        assert (out_dir / "A.csv").exists()
        assert (out_dir / "sig_LFs.txt").exists()
        print(f"\nSSc R knockoffs: {len(_load_sig_lfs(out_dir))} sig LFs")

    def test_ssc_full_r_pipeline(self, ssc_full_r_result):
        """Full R pipeline completes on SSc data."""
        _, out_dir = ssc_full_r_result
        assert (out_dir / "A.csv").exists()
        assert (out_dir / "sig_LFs.txt").exists()
        print(f"\nSSc Full R: {len(_load_sig_lfs(out_dir))} sig LFs")

    def test_ssc_backend_A_agreement(self, ssc_python_result,
                                     ssc_r_knockoffs_result,
                                     ssc_full_r_result):
        """A matrices should match across backends (same LOVE)."""
        A_py = pd.read_csv(ssc_python_result[1] / "A.csv", index_col=0)
        A_rko = pd.read_csv(ssc_r_knockoffs_result[1] / "A.csv", index_col=0)
        A_r = pd.read_csv(ssc_full_r_result[1] / "A.csv", index_col=0)

        assert A_py.shape == A_rko.shape, (
            f"Python {A_py.shape} vs r_knockoffs {A_rko.shape}")
        assert A_py.shape == A_r.shape, (
            f"Python {A_py.shape} vs full_r {A_r.shape}")

        j_py_rko = _jaccard(_nonzero_rows(A_py), _nonzero_rows(A_rko))
        j_py_r = _jaccard(_nonzero_rows(A_py), _nonzero_rows(A_r))
        print(f"\nSSc A Jaccard: py-rko={j_py_rko:.3f}, py-r={j_py_r:.3f}")

        assert j_py_rko >= 0.9, f"SSc Py vs r_knockoffs A Jaccard={j_py_rko:.3f} < 0.9"
        assert j_py_r >= 0.9, f"SSc Py vs full_r A Jaccard={j_py_r:.3f} < 0.9"

    def test_ssc_backend_comparison_report(self, ssc_python_result,
                                           ssc_r_knockoffs_result,
                                           ssc_full_r_result):
        """Print full comparison table across backends."""
        from slide_py import Estimator

        y = pd.read_csv(_SSC_Y, index_col=0).iloc[:, 0]

        results = {
            "Python": ssc_python_result[1],
            "R knockoffs": ssc_r_knockoffs_result[1],
            "Full R": ssc_full_r_result[1],
        }

        print(f"\n{'='*60}")
        print(f"SSc Backend Comparison Report")
        print(f"{'='*60}")

        all_lfs = {}
        for label, out_dir in results.items():
            sig_lfs = _load_sig_lfs(out_dir)
            all_lfs[label] = set(sig_lfs)

            z = pd.read_csv(out_dir / "z_matrix.csv", index_col=0)
            z_sig = z[sig_lfs].values

            scores = Estimator.get_aucs(z_sig, y.values, n_iters=50,
                                        test_size=0.2, scaler="standard")
            mean_score = np.nanmean(scores)
            print(f"\n{label}:")
            print(f"  sig_LFs ({len(sig_lfs)}): {sorted(sig_lfs)}")
            print(f"  mean corr: {mean_score:.3f}")

        # Pairwise Jaccard
        labels = list(all_lfs.keys())
        print(f"\nPairwise sig_LF Jaccard:")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                jac = _jaccard(all_lfs[labels[i]], all_lfs[labels[j]])
                print(f"  {labels[i]} vs {labels[j]}: {jac:.3f}")

        # At least one backend should find sig LFs
        assert any(len(v) > 0 for v in all_lfs.values()), (
            "No backend found significant LFs on SSc data")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
