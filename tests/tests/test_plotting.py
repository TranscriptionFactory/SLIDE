"""
Test coverage for slide_py.plotting visualization functionality.

Major gaps:
- Plot generation and output validation
- Error handling for invalid data inputs
- File saving and directory creation
- Plot styling and formatting
- Edge cases with empty or malformed data
- Integration with different data types
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

from slide_py.plotting import Plotter


class TestPlotterInit:
    """Test Plotter initialization."""

    def test_plotter_init(self):
        """Test basic Plotter initialization."""
        plotter = Plotter()
        assert isinstance(plotter, Plotter)


class TestLatentFactorPlots:
    """Test latent factor plotting functionality."""

    @pytest.fixture
    def sample_lfs_data(self):
        """Create sample latent factors data for testing."""
        return {
            'LF1': np.random.randn(50),
            'LF2': np.random.randn(50),
            'LF3': np.random.randn(50)
        }


    def test_plot_latent_factors_empty_data(self):
        """Test plotting with empty latent factors."""
        empty_lfs = {}

        # Should handle gracefully or raise informative error
        with pytest.raises((ValueError, RuntimeError)):
            Plotter.plot_latent_factors(empty_lfs)


class TestInteractionPlots:
    """Test interaction plotting functionality."""

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction pairs data."""
        return [
            ('LF1', 'LF2'),
            ('LF2', 'LF3'),
            ('LF1', 'LF3')
        ]

    def test_plot_interactions_basic(self, sample_interactions):
        """Test basic interaction plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_interactions(sample_interactions, outdir=tmpdir)


    def test_plot_interactions_custom_title(self, sample_interactions):
        """Test interaction plot with custom title."""
        custom_title = "Custom Interaction Plot"

        with tempfile.TemporaryDirectory() as tmpdir:
            Plotter.plot_interactions(
                sample_interactions,
                outdir=tmpdir,
                title=custom_title
            )




def test_control_density_r_bandwidth_and_constant_fallback():
    from slide_py.plotting import _control_density
    reference = pd.read_csv(Path(__file__).parents[1] / 'fixtures/control_plot/expected.csv').iloc[0]
    for values, bandwidth in [([.2, .3, .5, .8], reference.bw), ([.7] * 4, reference.bw_constant)]:
        grid, density = _control_density(values)
        assert grid[0] == pytest.approx(min(values) - 3 * bandwidth)
        assert grid[-1] == pytest.approx(max(values) + 3 * bandwidth)
        assert np.isfinite(density).all()
        from scipy.integrate import trapezoid
        assert trapezoid(density, grid) == pytest.approx(1, abs=.003)
    grid, _ = _control_density(np.repeat(.6, 20))
    assert grid[0] == pytest.approx(.6 - 3 * .9 * .6 * 20 ** -.2)
    with pytest.raises(ValueError, match='two finite'):
        _control_density([np.nan, .5])


def test_control_plot_semantics_and_export(tmp_path):
    scores = {'real': .8, 'full_random': np.linspace(.3, .7, 20),
              'partial_random': np.repeat(.6, 20), 'condition': 'auc'}
    fig = Plotter.plot_controlplot(scores, outdir=tmp_path)
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'auc'
    assert ax.get_ylabel() == 'Density'
    assert [x.get_text() for x in ax.legend_.texts] == ['FullRandom', 'PartialRandom', 'Real']
    assert ax.lines[0].get_color() == 'red'
    assert ax.lines[0].get_linestyle() == '--'
    assert (tmp_path / 'ControlPerformancePlot.png').is_file()
    plt.close(fig)
    fig = Plotter.plot_controlplot(scores, outdir=tmp_path, file_format='pdf')
    assert (tmp_path / 'ControlPerformancePlot.pdf').is_file()
    plt.close(fig)


def test_gene_lists_match_r_union_spearman_and_raw_auc(tmp_path):
    from slide_py.slide import SLIDE
    y = pd.Series([0, 0, 0, 1, 1, 1])
    x = pd.DataFrame({'positive': [1, 2, 3, 4, 5, 6],
                      'negative': [6, 5, 4, 3, 2, 1],
                      'neutral': [1, 2, 3, 1, 2, 3]})
    a = pd.DataFrame({'Z0': [-.9, .7, .5]}, index=x.columns)
    genes = SLIDE.get_LF_genes(a, 'Z0', x, y, top_feats=4, outpath=tmp_path)
    assert list(genes.index) == ['positive', 'negative', 'neutral']
    assert genes.AUC.tolist() == [1, 0, .5]
    assert genes.color.tolist() == ['red', 'blue', 'white']
    assert genes.loading.tolist() == [.9, .7, .5]
    table = pd.read_csv(tmp_path / 'feature_list_Z0.txt', sep='\t')
    assert table.columns.tolist() == ['names', 'A_loading', 'AUCs', 'corrs', 'color']
    fig = Plotter.plot_latent_factors({'Z0': genes}, outdir=tmp_path, marginal_lfs=['Z0'])
    assert [text.get_text() for text in fig.axes[0].texts] == genes.index.tolist()
    assert fig.axes[0].texts[0].get_fontstyle() == 'italic'
    assert (tmp_path / 'plotSigGenes_marginals.pdf').is_file()
    plt.close(fig)


def test_gene_continuous_uses_spearman_and_independent_union(tmp_path):
    from scipy.stats import spearmanr
    from slide_py.slide import SLIDE
    y = pd.Series([1, 2, 3, 4, 5, 6])
    x = pd.DataFrame({'nonlinear': [1, 2, 3, 4, 5, 100],
                      'other': [6, 2, 3, 5, 1, 4]})
    a = pd.DataFrame({'Z0': [-.9, .1]}, index=x.columns)
    genes = SLIDE.get_LF_genes(a, 'Z0', x, y, top_feats=2, outpath=tmp_path)
    assert list(genes.index) == ['nonlinear']  # Both criteria choose it, not a forced second gene.
    assert genes.iloc[0]['corr'] == pytest.approx(spearmanr(x.nonlinear, y).statistic)
    assert genes.iloc[0]['corr'] != pytest.approx(np.corrcoef(x.nonlinear, y)[0, 1])
    assert 'AUCs' not in pd.read_csv(tmp_path / 'feature_list_Z0.txt', sep='\t').columns


def test_control_raw_scores_export_includes_real(tmp_path):
    from slide_py.slide import SLIDE
    instance = SLIDE.__new__(SLIDE)
    instance.latent_factors = pd.DataFrame(np.ones((3, 1)))
    instance.sig_LFs = ['Z0']
    instance.sig_interacts = []
    instance.show_params = lambda: None
    scores = {'real': .9, 'full_random': np.array([.2, .4]), 'condition': 'corr'}
    instance.save_params(tmp_path, scores)
    table = pd.read_csv(tmp_path / 'ControlPerformance.csv')
    assert table.to_dict('list') == {'group': ['FullRandom', 'FullRandom', 'Real'], 'value': [.2, .4, .9]}
    assert 'Partial Random: NA' in (tmp_path / 'scores.txt').read_text()
