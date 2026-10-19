import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import numpy as np
import pandas as pd
import networkx as nx

try:
    import seaborn as sns
    import seaborn.objects as so
except ImportError:
    sns = None
    so = None


def _control_density(values):
    """Gaussian KDE with R density's bw.nrd0, including its constant-data fallback."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("Control density requires at least two finite scores")
    spread = 0.0 if np.ptp(values) == 0 else np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25])) / 1.34
    scale = min(spread, iqr) or spread or abs(values[0]) or 1.0
    bandwidth = 0.9 * scale * values.size ** -0.2
    grid = np.linspace(values.min() - 3 * bandwidth, values.max() + 3 * bandwidth, 512)
    density = np.exp(-0.5 * ((grid[:, None] - values) / bandwidth) ** 2).mean(axis=1)
    density /= bandwidth * np.sqrt(2 * np.pi)
    return grid, density


class Plotter:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_latent_factors(lfs, outdir=None, title='plotSigGenes_marginals', marginal_lfs=None):
        """Render R's feature-list order and sign colors as a native PDF."""
        max_genes = max(len(df) for df in lfs.values())
        fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(lfs)), 7), dpi=300)
        for i, (lf_name, genes) in enumerate(lfs.items()):
            emphasize = marginal_lfs is not None and lf_name in marginal_lfs
            for height, (gene, row) in zip(range(len(genes), 0, -1), genes.iterrows()):
                ax.text(i, height, gene, color=row['color'].lower(), fontsize=10,
                        fontweight='bold' if emphasize else 'normal',
                        fontstyle='italic' if emphasize else 'normal', ha='center', va='center')
        ax.set_title('SLIDE Marginal Variables' if marginal_lfs is None else
                     'Significant Latent Factors — Marginals (bold/italic) and Interactions')
        ax.set_xlim(-0.5, len(lfs) - 0.5)
        ax.set_ylim(0, max_genes + 1)
        ax.set_xticks(range(len(lfs)), labels=lfs.keys())
        ax.set_yticks([])
        ax.set_xlabel('Significant Latent Factor')
        ax.set_ylabel('Genes Associated with Significant Latent Factors')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        fig.tight_layout()
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(os.path.join(outdir, f'{title}.pdf'), bbox_inches='tight', facecolor='white')
        return fig

    @staticmethod
    def plot_corr_network(X, lf_dict, outdir=None, minimum=0.25):

        colors = {'red': '#FF0000', 'white': '#FFFFFF', 'blue': '#0000FF'}

        for lf, lf_loadings in lf_dict.items():
            lf_genes = lf_loadings.index.tolist()
            color_dict = lf_loadings['color'].map(colors).to_dict()
            
            features = X[lf_genes]
            corr = features.corr().where(lambda x: abs(x) > minimum, 0)
            corr_vals = corr.to_numpy(copy=True)
            np.fill_diagonal(corr_vals, 0)
            corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)

            G = nx.from_pandas_adjacency(corr)

            for gene in G.nodes():
                G.nodes[gene]['color'] = color_dict[gene]

            fig, ax = plt.subplots(figsize=(5,5), facecolor='white')
            ax.grid(False)
            # pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
            pos = nx.circular_layout(G)
            nx.draw_networkx_nodes(G, pos, 
                                 node_color=[G.nodes[node]['color'] for node in G.nodes()],
                                 node_size=600,
                                 alpha=0.4,
                                 ax=ax)
            # Draw edges with alpha based on correlation strength
            for (node1, node2, data) in G.edges(data=True):
                weight = abs(data['weight'])
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[(node1, node2)],
                                     width=weight*5,
                                     alpha=min(weight, 1.0),
                                     edge_color=['green' if data['weight'] < 0 else 'pink'])
                
            nx.draw_networkx_labels(G, pos, font_size=10)
            # nx.draw_networkx_edge_labels(G, pos, font_size=10)

            plt.tight_layout()
            plt.gca().set_aspect('equal')
            plt.savefig(os.path.join(outdir, f'corr_{lf}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')

    @staticmethod
    def plot_controlplot(scores, outdir=None, title='ControlPerformancePlot', xlabel=None, file_format='png'):
        """Render the R control comparison independently and save its canonical PNG."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for key, label, color in [('full_random', 'FullRandom', '#0000FF'),
                                  ('partial_random', 'PartialRandom', '#00FF00')]:
            if key not in scores:
                continue
            values = np.asarray(scores[key], dtype=float)
            if np.isfinite(values).sum() < 2:
                continue
            grid, density = _control_density(values)
            ax.fill_between(grid, density, alpha=0.7, color=color, label=label)
        if np.isfinite(scores['real']):
            ax.axvline(scores['real'], color='red', linestyle='--', linewidth=2, label='Real')
        ax.set_xlabel(xlabel or scores.get('condition', 'corr'))
        ax.set_ylabel('Density')
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend()
        fig.tight_layout()
        if outdir:
            fig.savefig(os.path.join(outdir, f'{title}.{file_format}'), dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def plot_interactions(interaction_pairs, outdir=None, title='Interaction Pairs'):
        """
        Plot interaction pairs for different latent factor configurations.
        
        Parameters:
        - interaction_pairs: np array of [[marginal_lfs], [interaction_lfs]]
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        G = nx.Graph()
        marginal_lfs = set(interaction_pairs[0])
        interaction_lfs = set(interaction_pairs[1])
        G.add_nodes_from(marginal_lfs | interaction_lfs)
        for marginal_lf, interaction_lf in zip(interaction_pairs[0], interaction_pairs[1]):
            G.add_edge(marginal_lf, interaction_lf)

        node_colors = [
            'salmon' if node in marginal_lfs else 'gray'
            for node in G.nodes()
        ]

        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_size=600,
            font_size=10,
            node_color=node_colors,
            edge_color='#888',
            linewidths=1,
            ax=ax
        )

        legend_elements = [
            Patch(facecolor='salmon', edgecolor='k', label='Marginal LF'),
            Patch(facecolor='gray', edgecolor='k', label='Interacting LF')
        ]
        ax.legend(handles=legend_elements, loc='best')
        ax.set_axis_off()
        plt.tight_layout()

        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), dpi=300, bbox_inches='tight', facecolor='white')
        return fig







