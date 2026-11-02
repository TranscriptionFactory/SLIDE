"""Method-4 repeated CV on saved latent factors, with a training-y null."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .knockoffs import Knockoffs
from .sample_cv import _fit_ols
from .tools import calc_default_fsize

logger = logging.getLogger(__name__)


class SLIDEcv:
    """Reselect method-4 models within repeated caret-style stratified folds.

    Saved Z is globally sample-SD scaled, as in R/SLIDEcv.R. Continuous
    targets use training-fold mean/SD; binary AUC uses the original labels.
    Selection uses native Gaussian knockoffs, offset zero and findOptIter.
    Numerical seeds define native draws, not paired draws across languages.
    """

    def __init__(self, slide_obj, *, nrep=20, k=10, eval_type="corr", **kwargs):
        self.slide_obj = slide_obj
        self.nrep, self.k, self.eval_type = nrep, k, eval_type
        factors, response = slide_obj.latent_factors, slide_obj.data.Y
        self.z_matrix = np.asarray(factors, dtype=float)
        values = np.asarray(response, dtype=float)
        if (self.z_matrix.ndim != 2 or values.ndim not in (1, 2)
                or (values.ndim == 2 and values.shape[1] != 1)):
            raise ValueError("SLIDEcv requires matrix Z and a single response column")
        self.y = values.reshape(-1)
        self.n_samples, self.n_lfs = self.z_matrix.shape
        if (self.n_samples < 3 or self.n_lfs < 1 or len(self.y) != self.n_samples
                or not np.isfinite(self.z_matrix).all() or not np.isfinite(self.y).all()):
            raise ValueError("SLIDEcv requires finite row-aligned Z/y with >=3 rows and >=1 factor")
        self.sample_ids = (factors.index.to_numpy() if isinstance(factors, pd.DataFrame)
                           else np.arange(self.n_samples))
        if (pd.Index(self.sample_ids).has_duplicates or pd.isna(self.sample_ids).any()
                or (isinstance(response, (pd.DataFrame, pd.Series))
                    and not pd.Index(self.sample_ids).equals(response.index))):
            raise ValueError("SLIDEcv Z/y sample IDs must be unique, nonmissing and identically ordered")
        for name, value, upper in (("nrep", nrep, None), ("k", k, self.n_samples)):
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or value < (2 if name == "k" else 1)
                    or (upper is not None and value > upper)):
                raise ValueError(f"SLIDEcv {name}: invalid integer {value}")
        if eval_type not in ("corr", "auc"):
            raise ValueError("SLIDEcv eval_type must be corr or auc")
        self.is_classifier = eval_type == "auc"
        if np.ptp(self.y) == 0 or (self.is_classifier and len(np.unique(self.y)) != 2):
            raise ValueError("SLIDEcv requires variable y; auc requires exactly two levels")
        scales = self.z_matrix.std(axis=0, ddof=1)
        if (scales <= 0).any():
            raise ValueError("SLIDEcv saved Z has constant factors and cannot be sample-SD scaled")
        self.z_matrix = (self.z_matrix - self.z_matrix.mean(axis=0)) / scales
        self._f_size = kwargs.pop("f_size", calc_default_fsize(self.n_samples, self.n_lfs))
        params = slide_obj.input_params
        for name, expected in (("knockoff_backend", "python"), ("knockoff_method", "asdp"),
                               ("knockoff_shrink", False), ("knockoff_offset", 0),
                               ("fstat", "glmnet_lambdasmax")):
            if params.get(name, expected) != expected:
                raise ValueError(f"SLIDEcv native method-4 requires {name}={expected!r}")
        self._ko_params = dict(spec=params.get("spec", 0.2), fdr=params.get("fdr", 0.1),
                               niter=params.get("niter", 100), n_workers=params.get("n_workers", 1))
        self.do_interacts = params.get("do_interacts", True)
        if (isinstance(self._f_size, bool) or not isinstance(self._f_size, (int, np.integer))
                or self._f_size < 1):
            raise ValueError("SLIDEcv f_size must be a positive integer")
        if (not 0 < self._ko_params["spec"] <= 1 or not 0 < self._ko_params["fdr"] < 1
                or isinstance(self._ko_params["niter"], bool)
                or not isinstance(self._ko_params["niter"], (int, np.integer))
                or self._ko_params["niter"] < 1):
            raise ValueError("SLIDEcv requires spec in (0,1], fdr in (0,1), and positive integer niter")

    def run(self, *, outpath: Optional[Union[str, Path]] = None, seed=42, **kwargs):
        """Return tidy repeat scores; keep every held-out row in predictions_.

        metric_value is Spearman or direction-corrected AUC; raw_metric retains
        its direction, mse uses fold-standardized targets for corr, and status
        explicitly identifies an undefined metric. Both branches use Gaussian OLS.
        """
        if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("SLIDEcv seed must be a nonnegative integer")
        self._prediction_rows = []
        frames = []
        for rep in range(self.nrep):
            logger.info("SLIDEcv replicate %d / %d", rep + 1, self.nrep)
            frames.append(self._bench_cv(rep, seed=seed + rep, **kwargs))
        results = pd.concat(frames, ignore_index=True)
        self.predictions_ = pd.DataFrame(self._prediction_rows)
        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            results.to_csv(outpath / "slidecv_results.csv", index=False)
            self.predictions_.to_csv(outpath / "slidecv_predictions.csv", index=False)
            self._plot_boxplot(results, outpath)
        return results

    def _bench_cv(self, rep, *, seed=0, max_resplit=10, **kwargs):
        rng = np.random.default_rng(seed)
        folds = self._make_folds(rng, max_resplit)
        predictions = {method: np.full(self.n_samples, np.nan) for method in ("SLIDE", "SLIDE_y")}
        coverage = np.zeros(self.n_samples, dtype=int)
        true = np.empty(self.n_samples)
        rows = []
        for fold, (train, valid) in enumerate(folds):
            train_y, valid_y = self.y[train], self.y[valid]
            if not self.is_classifier:
                train_y, valid_y = self._standardize_fold(train_y, valid_y)
            true[valid] = valid_y
            coverage[valid] += 1
            permuted = train_y[rng.permutation(len(train_y))]
            for method, response in (("SLIDE", train_y), ("SLIDE_y", permuted)):
                pred = np.asarray(self._run_slide_fold(self.z_matrix[train], response,
                                                      self.z_matrix[valid], rng=rng, **kwargs))
                if pred.shape != (len(valid),):
                    raise ValueError(f"SLIDEcv repeat {rep} fold {fold} {method}: incomplete predictions")
                predictions[method][valid] = pred
                rows.extend(dict(replicate=rep, fold=fold, method=method, row_index=int(i),
                                 sample_id=str(self.sample_ids[i]), y_true=float(t), y_pred=float(p),
                                 model="gaussian_ols") for i, t, p in zip(valid, valid_y, pred))
        if not np.all(coverage == 1):
            raise ValueError(f"SLIDEcv repeat {rep}: incomplete or duplicate held-out sample coverage")
        if hasattr(self, "_prediction_rows"):
            self._prediction_rows.extend(rows)
        scores = []
        for method, predicted in predictions.items():
            metric, raw, mse = self._score(true, predicted)
            scores.append(dict(method=method, metric_value=metric, replicate=rep,
                               raw_metric=raw, mse=mse,
                               imputed_predictions=int(np.isnan(predicted).sum()) if self.is_classifier and np.isfinite(predicted).any() else 0,
                               status="ok" if np.isfinite(metric) else "undefined_metric"))
        return pd.DataFrame(scores)

    def _select(self, z, y, spec, f_size, rng):
        # The shared selector seeds legacy NumPy internals; preserve caller RNG state.
        state = np.random.get_state()
        try:
            return np.asarray(Knockoffs.select_short_freq(
                z=z, y=y, spec=spec, f_size=f_size, backend="python",
                fdr=self._ko_params["fdr"], niter=self._ko_params["niter"],
                n_workers=self._ko_params["n_workers"],
                base_seed=int(rng.integers(0, 2**31)), slide_selection=True), dtype=int)
        finally:
            np.random.set_state(state)

    @staticmethod
    def _increase_spec(spec):
        if spec + 0.1 > 1 + 1e-12:
            raise ValueError("SLIDEcv model remains oversized at maximal selection specificity")
        return min(1.0, spec + 0.1)

    def _run_slide_fold(self, train_z, train_y, valid_z, *, rng=None, **kwargs):
        rng = np.random.default_rng(0) if rng is None else rng
        spec = self._ko_params["spec"]
        while True:
            marginals = self._select(train_z, train_y, spec, self._f_size, rng)
            if len(marginals) < len(train_y):
                break
            spec = self._increase_spec(spec)
        if not len(marginals):
            marginals = np.array([rng.integers(train_z.shape[1])])
        train_x, valid_x = train_z[:, marginals], valid_z[:, marginals]
        if self.do_interacts and len(marginals) < len(train_y):
            pairs = self._find_interactions_fold(train_z, train_y, marginals, self._f_size, rng=rng)
            ups_train, ups_valid = self._build_prediction_features(
                train_z, valid_z, marginals, pairs, train_y=train_y)
            selected = (np.array([0]) if len(marginals) == 1 else
                        self._select(ups_train, train_y, self._ko_params["spec"], self._f_size, rng))
            if len(selected):
                train_x, valid_x = ups_train[:, selected], ups_valid[:, selected]
        return self._gaussian_predict(train_x, train_y, valid_x)

    def _find_interactions_fold(self, train_z, train_y, marginal_idxs, f_size, *, rng=None, **kwargs):
        rng = np.random.default_rng(0) if rng is None else rng
        spec = self._ko_params["spec"]
        while True:
            pairs = []
            for marginal in marginal_idxs:
                # R's numeric already_made names miss Z-prefixed pairs, so keep reverse pairs.
                candidates = [j for j in range(train_z.shape[1]) if j != marginal]
                fitted = self._gaussian_predict(train_z[:, [marginal]], train_y, train_z[:, [marginal]])
                selected = (self._select(train_z[:, [marginal]] * train_z[:, candidates],
                                         train_y - fitted, spec, 100, rng)
                            if candidates else [])
                if len(selected) > len(train_y) - len(marginal_idxs):
                    spec = self._increase_spec(spec)
                    break
                pairs.extend((int(marginal), candidates[j]) for j in selected)
            else:
                return pairs

    @staticmethod
    def _gaussian_predict(train_x, train_y, valid_x):
        design = np.column_stack((np.ones(len(train_x)), train_x))
        kept, coefficients = _fit_ols(design, train_y)
        predicted = np.column_stack((np.ones(len(valid_x)), valid_x))[:, kept] @ coefficients
        if not np.isfinite(predicted).all():
            raise ValueError("SLIDEcv Gaussian OLS produced nonfinite predictions")
        return predicted

    @staticmethod
    def _build_prediction_features(train_z, valid_z, marginal_idxs, interaction_pairs, *, train_y):
        """Fit each marginal/interaction Gaussian submodel and predict its upsilon."""
        ups_train, ups_valid = [], []
        for marginal in marginal_idxs:
            partners = [j for m, j in interaction_pairs if m == marginal]
            train_x = np.column_stack((train_z[:, marginal],
                                       train_z[:, [marginal]] * train_z[:, partners]))
            valid_x = np.column_stack((valid_z[:, marginal],
                                       valid_z[:, [marginal]] * valid_z[:, partners]))
            fitted = SLIDEcv._gaussian_predict(train_x, train_y, np.vstack((train_x, valid_x)))
            ups_train.append(fitted[:len(train_z)])
            ups_valid.append(fitted[len(train_z):])
        return np.column_stack(ups_train), np.column_stack(ups_valid)

    def _score(self, y_true, y_pred):
        predicted = np.asarray(y_pred, dtype=float).copy()
        if self.eval_type == "auc":
            if np.isinf(predicted).any():
                raise ValueError("SLIDEcv AUC predictions must be finite or missing")
            present = ~np.isnan(predicted)
            if not present.any() or len(np.unique(y_true)) != 2:
                return np.nan, np.nan, np.nan
            predicted[~present] = np.median(predicted[present])
            raw = float(roc_auc_score(y_true, predicted))
            return max(raw, 1 - raw), raw, np.nan
        mse = float(np.mean((predicted - y_true)**2))
        if (not np.isfinite(predicted).all() or not np.isfinite(y_true).all()
                or np.ptp(predicted) == 0 or np.ptp(y_true) == 0):
            return np.nan, np.nan, mse
        corr = float(spearmanr(y_true, predicted).statistic)
        return corr, corr, mse

    def _compute_metric(self, y_true, y_pred):
        return self._score(np.asarray(y_true), np.asarray(y_pred))[0]

    @staticmethod
    def _standardize_fold(train, valid):
        """Use training sample-SD scaling for a continuous response or matrix."""
        center, scale = train.mean(axis=0), train.std(axis=0, ddof=1)
        return (train - center) / scale, (valid - center) / scale

    def _make_folds(self, rng, max_resplit):
        if (isinstance(max_resplit, bool) or not isinstance(max_resplit, (int, np.integer))
                or max_resplit < 1):
            raise ValueError("SLIDEcv max_resplit must be a positive integer")
        indices = np.arange(self.n_samples)
        if self.k == self.n_samples:
            folds = [(indices[indices != i], np.array([i])) for i in indices]
            if self._folds_valid(self.y, folds):
                return folds
        else:
            if self.is_classifier:
                labels = self.y
            else:
                cuts = min(5, max(2, self.n_samples // self.k))
                breaks = np.unique(np.quantile(self.y, np.linspace(0, 1, cuts)))
                labels = np.searchsorted(breaks[1:-1], self.y, side="left")
            for _ in range(max_resplit):
                assignments = np.empty(self.n_samples, dtype=int)
                for label in np.unique(labels):
                    rows = np.flatnonzero(labels == label)
                    assigned = np.concatenate((np.tile(np.arange(self.k), len(rows) // self.k),
                                               rng.permutation(self.k)[:len(rows) % self.k]))
                    assignments[rows] = rng.permutation(assigned)
                folds = [(indices[assignments != i], indices[assignments == i]) for i in range(self.k)]
                if self._folds_valid(self.y, folds):
                    return folds
        raise ValueError(f"SLIDEcv cannot generate nondegenerate folds after {max_resplit} attempts")

    def _folds_valid(self, y, folds):
        if any(len(valid) == 0 or len(train) < 2 or np.ptp(y[train]) == 0 for train, valid in folds):
            return False
        return (self.is_classifier or any(len(valid) == 1 for _, valid in folds)
                or all(np.ptp(y[valid]) > 0 for _, valid in folds))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_boxplot(
        self,
        results: pd.DataFrame,
        outpath: Path,
        **kwargs,
    ) -> None:
        """Generate a boxplot comparing SLIDE vs SLIDE_y metrics.

        Args:
            results: Tidy DataFrame from ``run()``.
            outpath: Directory to write the figure.
            **kwargs: Reserved.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping boxplot")
            return

        try:
            import seaborn as sns
        except ImportError:
            sns = None

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        fig.patch.set_facecolor("white")

        metric_label = "Spearman r" if self.eval_type == "corr" else "Direction-corrected AUC"

        if sns is not None:
            sns.set_style("whitegrid")
            sns.boxplot(
                data=results,
                x="method",
                y="metric_value",
                ax=ax,
                palette={"SLIDE": "#e74c3c", "SLIDE_y": "#3498db"},
            )
        else:
            slide_vals = results.loc[results["method"] == "SLIDE", "metric_value"]
            slidey_vals = results.loc[results["method"] == "SLIDE_y", "metric_value"]
            ax.boxplot(
                [slide_vals.values, slidey_vals.values],
                labels=["SLIDE", "SLIDE_y"],
            )

        ax.set_ylabel(metric_label)
        ax.set_title("SLIDEcv: SLIDE vs permuted-y null")
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            outpath / "slidecv_boxplot.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        logger.info("Boxplot saved to %s", outpath / "slidecv_boxplot.png")
