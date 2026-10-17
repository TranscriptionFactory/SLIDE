"""Cross-validation module for SLIDE latent-factor models.

Implements R-style SLIDEcv: repeated stratified k-fold CV that benchmarks
a fitted SLIDE model against a permuted-y null, reporting Spearman
correlation (regression) or ROC-AUC (classification).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .knockoffs import Knockoffs
from .tools import calc_default_fsize

logger = logging.getLogger(__name__)


class SLIDEcv:
    """Repeated k-fold cross-validation benchmark for a fitted SLIDE model.

    Runs knockoff variable selection *within* each CV fold on the pre-computed
    latent-factor Z matrix, then scores predictions against a permuted-y
    null (``SLIDE_y``).

    Args:
        slide_obj: A fitted ``OptimizeSLIDE`` instance (must have
            ``latent_factors``, ``data.Y``, ``input_params``, ``marginal_idxs``).
        nrep: Number of independent replicates.
        k: Number of CV folds per replicate.
        eval_type: ``'corr'`` for Spearman correlation (regression) or
            ``'auc'`` for ROC-AUC (classification).
        **kwargs: Reserved for forward compatibility.
    """

    def __init__(
        self,
        slide_obj,
        *,
        nrep: int = 20,
        k: int = 10,
        eval_type: str = "corr",
        **kwargs,
    ) -> None:
        self.slide_obj = slide_obj
        self.nrep = nrep
        self.k = k
        self.eval_type = eval_type

        # Extract data from the fitted SLIDE object
        self.z_matrix: np.ndarray = np.asarray(slide_obj.latent_factors.values)
        self.y: np.ndarray = np.asarray(slide_obj.data.Y.values).flatten()
        self.n_samples, self.n_lfs = self.z_matrix.shape

        # Determine if classification
        self.is_classifier: bool = len(np.unique(self.y)) == 2

        if self.eval_type == "auc" and not self.is_classifier:
            logger.warning(
                "eval_type='auc' requested but y is not binary; falling back to 'corr'"
            )
            self.eval_type = "corr"

        # Compute f_size once from full dataset (matches R SLIDEcv behavior)
        self._f_size: int = kwargs.pop(
            "f_size", calc_default_fsize(self.n_samples, self.n_lfs)
        )

        # Knockoff parameters forwarded from the fitted model
        params = slide_obj.input_params
        self._ko_params: dict = dict(
            spec=params.get("spec", 0.2),
            fdr=params.get("fdr", 0.1),
            niter=params.get("niter", 100),
            backend=params.get("knockoff_backend", "python"),
            method=params.get("knockoff_method", "asdp"),
            shrink=params.get("knockoff_shrink", False),
            offset=params.get("knockoff_offset", 0),
            fstat=params.get("fstat", "glmnet_lambdasmax"),
            n_workers=params.get("n_workers", 1),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        outpath: Optional[Union[str, Path]] = None,
        seed: int = 42,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute repeated CV and return tidy results.

        Args:
            outpath: If given, saves a CSV and boxplot to this directory.
            seed: Base random seed (incremented per replicate).
            **kwargs: Forwarded to ``_bench_cv``.

        Returns:
            DataFrame with columns ``[method, metric_value, replicate]``.
        """
        frames: list[pd.DataFrame] = []
        for rep in range(self.nrep):
            logger.info("SLIDEcv replicate %d / %d", rep + 1, self.nrep)
            rep_seed = seed + rep
            df = self._bench_cv(rep, seed=rep_seed, **kwargs)
            frames.append(df)

        results = pd.concat(frames, ignore_index=True)

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            results.to_csv(outpath / "slidecv_results.csv", index=False)
            self._plot_boxplot(results, outpath)

        return results

    # ------------------------------------------------------------------
    # Per-replicate logic
    # ------------------------------------------------------------------

    def _bench_cv(
        self,
        rep: int,
        *,
        seed: int = 0,
        max_resplit: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Run one replicate of k-fold CV.

        Args:
            rep: Replicate index (used in output only).
            seed: Random seed for this replicate.
            max_resplit: Maximum re-splits when a fold has zero-variance y.
            **kwargs: Forwarded to ``_run_slide_fold``.

        Returns:
            DataFrame with one row per method (``SLIDE``, ``SLIDE_y``).
        """
        rng = np.random.RandomState(seed)

        z = self.z_matrix
        y = self.y

        # Choose fold splitter
        if self.is_classifier:
            splitter_cls = StratifiedKFold
        else:
            splitter_cls = KFold

        # Split, re-splitting if a fold has zero-variance y (matches R while-loop)
        for attempt in range(max_resplit):
            splitter = splitter_cls(
                n_splits=self.k, shuffle=True, random_state=rng.randint(0, 2**31)
            )
            folds = list(splitter.split(z, y))
            if self._folds_valid(y, folds):
                break
        else:
            logger.warning(
                "Rep %d: could not find valid folds after %d attempts", rep, max_resplit
            )

        slide_preds = np.full(self.n_samples, np.nan)
        slide_y_preds = np.full(self.n_samples, np.nan)

        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
            logger.debug("  Rep %d fold %d / %d", rep, fold_idx + 1, self.k)

            z_train, z_valid = z[train_idx], z[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # Standardize within fold (train stats applied to valid)
            z_train_s, z_valid_s = self._standardize_fold(z_train, z_valid)

            # Permuted y for null distribution
            y_perm = y_train[rng.permutation(len(y_train))]

            # True SLIDE predictions
            preds = self._run_slide_fold(z_train_s, y_train, z_valid_s, **kwargs)
            slide_preds[valid_idx] = preds

            # Null (permuted y) predictions
            preds_null = self._run_slide_fold(z_train_s, y_perm, z_valid_s, **kwargs)
            slide_y_preds[valid_idx] = preds_null

        # Compute metric across all held-out samples
        slide_metric = self._compute_metric(y, slide_preds)
        slide_y_metric = self._compute_metric(y, slide_y_preds)

        return pd.DataFrame(
            {
                "method": ["SLIDE", "SLIDE_y"],
                "metric_value": [slide_metric, slide_y_metric],
                "replicate": [rep, rep],
            }
        )

    # ------------------------------------------------------------------
    # Per-fold knockoff + prediction
    # ------------------------------------------------------------------

    def _run_slide_fold(
        self,
        train_z: np.ndarray,
        train_y: np.ndarray,
        valid_z: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Select features via knockoffs on train, predict on valid.

        Mirrors R SLIDEcv fold logic:
        1. Run knockoff filter for marginal LFs on train_z.
        2. For each marginal, correct y and test interactions.
        3. Build a prediction model (with or without interactions).
        4. Return predictions for valid set.

        Args:
            train_z: Standardized training Z matrix ``(n_train, K)``.
            train_y: Training response ``(n_train,)``.
            valid_z: Standardized validation Z matrix ``(n_valid, K)``.
            **kwargs: Reserved.

        Returns:
            Predicted values for the validation set ``(n_valid,)``.
        """
        n_train, n_lfs = train_z.shape
        f_size = self._f_size
        ko = self._ko_params

        # --- Step 1: marginal selection ---
        marginal_idxs = Knockoffs.select_short_freq(
            z=train_z,
            y=train_y,
            spec=ko["spec"],
            fdr=ko["fdr"],
            niter=ko["niter"],
            f_size=f_size,
            n_workers=ko["n_workers"],
            backend=ko["backend"],
            method=ko["method"],
            shrink=ko["shrink"],
            offset=ko["offset"],
            fstat=ko["fstat"],
        )

        # If nothing found, pick one random feature (matches R fallback)
        if len(marginal_idxs) == 0:
            marginal_idxs = np.array(
                [np.random.randint(0, n_lfs)]
            )
            logger.debug("No marginals selected; using random feature %d", marginal_idxs[0])

        # --- Step 2: interaction selection (R-style per-marginal) ---
        interaction_pairs = self._find_interactions_fold(
            train_z, train_y, marginal_idxs, f_size
        )

        # --- Step 3: build prediction features ---
        train_X, valid_X = self._build_prediction_features(
            train_z, valid_z, marginal_idxs, interaction_pairs
        )

        # --- Step 4: fit model on train, predict on valid ---
        if self.is_classifier:
            model = LogisticRegression(max_iter=1000, solver="lbfgs")
        else:
            model = LinearRegression()

        model.fit(train_X, train_y)

        if self.is_classifier:
            return model.predict_proba(valid_X)[:, 1]
        return model.predict(valid_X)

    # ------------------------------------------------------------------
    # Interaction detection within a fold
    # ------------------------------------------------------------------

    def _find_interactions_fold(
        self,
        train_z: np.ndarray,
        train_y: np.ndarray,
        marginal_idxs: np.ndarray,
        f_size: int,
        **kwargs,
    ) -> list[tuple[int, int]]:
        """Find interaction pairs within a fold (R-style per-marginal logic).

        For each marginal:
        1. Build interaction terms with candidate LFs.
        2. Correct y for the marginal's effect.
        3. Run knockoff filter on interaction terms vs corrected y.

        Args:
            train_z: Training Z matrix.
            train_y: Training response.
            marginal_idxs: Indices of marginal LFs found.
            f_size: Feature chunk size for knockoffs.
            **kwargs: Reserved.

        Returns:
            List of ``(marginal_idx, interacting_idx)`` tuples.
        """
        ko = self._ko_params
        n_lfs = train_z.shape[1]
        used_marginals: set[int] = set()
        pairs: list[tuple[int, int]] = []

        for marg_idx in marginal_idxs:
            used_marginals.add(int(marg_idx))
            z_marginal = train_z[:, marg_idx]

            # Candidates: all LFs except already-used marginals
            candidate_idxs = [i for i in range(n_lfs) if i not in used_marginals]
            if len(candidate_idxs) == 0:
                continue

            z_candidates = train_z[:, candidate_idxs]

            # Interaction terms: marginal x each candidate
            interaction_terms = z_marginal[:, np.newaxis] * z_candidates

            # Correct y for marginal effect
            corrected_y = Knockoffs.correct_y(z_marginal, train_y)

            # Knockoff filter on interaction terms
            int_f_size = self._f_size

            sig_local = Knockoffs.select_short_freq(
                z=interaction_terms,
                y=corrected_y,
                spec=ko["spec"],
                fdr=ko["fdr"],
                niter=ko["niter"],
                f_size=int_f_size,
                n_workers=ko["n_workers"],
                backend=ko["backend"],
                method=ko["method"],
                shrink=ko["shrink"],
                offset=ko["offset"],
                fstat=ko["fstat"],
            )

            for local_idx in sig_local:
                pairs.append((int(marg_idx), candidate_idxs[local_idx]))

        return pairs

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prediction_features(
        train_z: np.ndarray,
        valid_z: np.ndarray,
        marginal_idxs: np.ndarray,
        interaction_pairs: list[tuple[int, int]],
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the feature matrices for prediction.

        If interactions are found, for each marginal the method builds
        an upsilon (fitted value from ``lm(y ~ marginal + interactions)``),
        then stacks the marginal columns alongside the interaction columns.
        If no interactions, uses marginal columns directly.

        Args:
            train_z: Training Z matrix.
            valid_z: Validation Z matrix.
            marginal_idxs: Marginal LF indices.
            interaction_pairs: ``(marginal, interacting)`` tuples.
            **kwargs: Reserved.

        Returns:
            ``(train_X, valid_X)`` feature matrices for the final model.
        """
        train_marginals = train_z[:, marginal_idxs]
        valid_marginals = valid_z[:, marginal_idxs]

        if len(interaction_pairs) == 0:
            return train_marginals, valid_marginals

        # Build interaction columns
        train_interactions = np.column_stack(
            [train_z[:, m] * train_z[:, j] for m, j in interaction_pairs]
        )
        valid_interactions = np.column_stack(
            [valid_z[:, m] * valid_z[:, j] for m, j in interaction_pairs]
        )

        train_X = np.column_stack([train_marginals, train_interactions])
        valid_X = np.column_stack([valid_marginals, valid_interactions])

        return train_X, valid_X

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the evaluation metric over all held-out predictions.

        Args:
            y_true: True response values.
            y_pred: Predicted values (probabilities for classification).

        Returns:
            Spearman correlation or ROC-AUC, depending on ``self.eval_type``.
        """
        # Drop NaN entries (shouldn't happen, but defensive)
        mask = ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if self.eval_type == "auc":
            if len(np.unique(y_true)) < 2:
                return np.nan
            return roc_auc_score(y_true, y_pred)

        # Default: Spearman correlation
        if len(y_true) < 3:
            return np.nan
        corr, _ = spearmanr(y_true, y_pred)
        return corr

    # ------------------------------------------------------------------
    # Fold utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize_fold(
        z_train: np.ndarray, z_valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardize using training statistics (matches R ``standCV``).

        Args:
            z_train: Training feature matrix.
            z_valid: Validation feature matrix.

        Returns:
            ``(z_train_scaled, z_valid_scaled)``.
        """
        scaler = StandardScaler()
        z_train_s = scaler.fit_transform(z_train)
        z_valid_s = scaler.transform(z_valid)
        return z_train_s, z_valid_s

    @staticmethod
    def _folds_valid(y: np.ndarray, folds: list[tuple]) -> bool:
        """Check that no fold has zero-variance y (matches R validation loop).

        Args:
            y: Full response vector.
            folds: List of ``(train_idx, valid_idx)`` tuples.

        Returns:
            ``True`` if all folds are valid.
        """
        for train_idx, valid_idx in folds:
            if len(np.unique(y[train_idx])) < 2:
                return False
            if len(np.unique(y[valid_idx])) < 2:
                return False
        return True

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_boxplot(
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

        metric_label = "Spearman r" if "corr" in results.columns else "metric_value"

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
