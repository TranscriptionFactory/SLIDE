from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr


def _mean_valid_scores(scores: np.ndarray) -> float:
    """Average evaluable resamples, ignoring undefined AUC/correlation splits."""
    valid = [score for score in np.asarray(scores, dtype=object).ravel()
             if score is not None and np.isfinite(score)]
    return float(np.mean(valid)) if valid else np.nan


class Estimator():
    def __init__(self, model: str = 'auto', scaler: str = 'standard', **kwargs):
        """Initialize estimator with automatic model selection for binary Y.

        Args:
            model: Model type - 'auto' (detect from Y), 'linear', or 'logistic'
            scaler: Scaler type - 'standard' or 'minmax'
            **kwargs: Additional arguments for forward compatibility
        """
        self.model_type = model
        self.model = None
        self.is_classifier = None
        self.scaler = scaler

    def _init_model(self, y: np.ndarray) -> None:
        """Initialize model based on Y values (auto-detect binary vs continuous)."""
        if self.model is not None:
            return  # Already initialized

        y_flat = np.array(y).flatten()
        n_unique = len(np.unique(y_flat))

        if self.model_type == 'auto':
            # Auto-detect: use logistic for binary, linear for continuous
            if n_unique == 2:
                self.model = LogisticRegression(max_iter=1000, solver='lbfgs')
                self.is_classifier = True
            else:
                self.model = LinearRegression()
                self.is_classifier = False
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, solver='lbfgs')
            self.is_classifier = True
        elif self.model_type == 'linear':
            self.model = LinearRegression()
            self.is_classifier = False
        else:
            raise ValueError(f"Invalid model: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Estimator':
        self._init_model(y)
        return self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for classifiers."""
        if self.is_classifier:
            return self.model.predict_proba(X)[:, 1]  # Probability of class 1
        return self.predict(X)

    def train_test_split(self, X, y, test_size=0.2, seed=1334):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test

    def score(self, yhat: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Compute score: ROC-AUC for classifiers, correlation for regression."""
        if self.is_classifier:
            # Classification: use ROC-AUC with probability scores
            if len(np.unique(y)) == 1:
                return None
            return roc_auc_score(y, yhat)
        else:
            # Regression: use correlation
            if len(y) < 3:
                return None
            corr, _ = pearsonr(np.array(yhat).flatten(), np.array(y).flatten())
            return corr

    @staticmethod
    def scale_features(X, scaler, feature_range=(-1, 1)):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        if scaler == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler == 'standard':
            scaler = StandardScaler()
        else:
            return X
            
        scaler.fit(X)
        return scaler.transform(X)

    def evaluate(self, X, y, n_iters: int = 10, test_size: float = 0.15, **kwargs) -> np.ndarray:
        """Evaluate model performance over multiple train/test splits.

        Args:
            X: Feature matrix
            y: Target variable
            n_iters: Number of iterations
            test_size: Proportion of data for testing
            **kwargs: Additional arguments for forward compatibility

        Returns:
            Array of scores from each iteration
        """
        scores = []
        X = X.copy()

        X = self.scale_features(X, scaler=self.scaler)

        for iter in range(n_iters):
            X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size, seed=iter)
            self.fit(X_train, y_train)
            # Use predict_proba for classifiers to get ROC-AUC compatible scores
            if self.is_classifier:
                y_pred = self.predict_proba(X_test)
            else:
                y_pred = self.predict(X_test)
            scores.append(self.score(y_pred, y_test))

        return np.array(scores)
    
    @staticmethod
    def get_aucs(
        X: np.ndarray,
        y: np.ndarray,
        n_iters: int = 10,
        test_size: float = 0.2,
        scaler: str = 'standard',
        **kwargs
    ) -> np.ndarray:
        """Compute AUC/correlation scores over multiple iterations.

        Args:
            X: Feature matrix
            y: Target variable (binary for ROC-AUC, continuous for correlation)
            n_iters: Number of iterations
            test_size: Proportion of data for testing
            scaler: Scaler type
            **kwargs: Additional arguments for forward compatibility

        Returns:
            Array of scores
        """
        estimator = Estimator(model='auto', scaler=scaler)
        return estimator.evaluate(X, y, n_iters, test_size)
    
def _control_score(X, y, condition):
    """R calcControlPerformance: in-sample OLS, including binary responses."""
    design = np.column_stack((np.ones(len(y)), X))
    if not np.isfinite(design).all():
        raise ValueError("Control model features must be finite; interaction products may have overflowed")
    with np.errstate(over='ignore', invalid='ignore'):
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if not np.isfinite(fitted).all():
        raise ValueError("Control model predictions must be finite")
    if condition == 'auc':
        levels = np.unique(y)
        auc = roc_auc_score(y == levels[1], fitted)
        score = float(auc if np.median(fitted[y == levels[0]]) <=
                      np.median(fitted[y == levels[1]]) else 1 - auc)
    else:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            model_ss = np.sum((fitted - fitted.mean()) ** 2)
            residual_ss = np.sum((y - fitted) ** 2)
            total_ss = model_ss + residual_ss
            if not np.isfinite(total_ss):
                raise ValueError("Control model score must be finite; sums of squares overflowed")
            score = float(np.sqrt(model_ss / total_ss))
    if not np.isfinite(score):
        raise ValueError("Control model score must be finite")
    return score


class SLIDE_Estimator(Estimator):

    @staticmethod
    def score_performance(
            sig_LFs, sig_interacts, y, latent_factors, latent_factors_z2=None,
            n_iters=1000, *,
            interaction_pairs=None, seed=None, condition=None):
        """R control scores from in-sample OLS without feature scaling.

        interaction_pairs contains zero-based column positions, shape (2, L).
        Random streams are local and are not identical to R's random streams.
        """
        z = np.asarray(latent_factors, dtype=float)
        z2 = z if latent_factors_z2 is None else np.asarray(latent_factors_z2, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if (z.ndim != 2 or z2.ndim != 2 or z.shape[0] != len(y)
                or z2.shape[0] != len(y) or len(y) < 2
                or not all(np.isfinite(a).all() for a in (z, z2, y))):
            raise ValueError("Control scoring requires finite, row-aligned matrices and response")
        if np.unique(y).size < 2:
            raise ValueError("Control scoring requires a nonconstant response")
        condition = condition or ('auc' if np.unique(y).size == 2 else 'corr')
        if condition not in ('auc', 'corr') or (condition == 'auc' and np.unique(y).size != 2):
            raise ValueError("condition must be corr, or auc with a binary response")
        if not isinstance(n_iters, (int, np.integer)) or n_iters < 1:
            raise ValueError("n_iters must be a positive integer")
        marginal_idxs = latent_factors.columns.get_indexer(sig_LFs)
        if (len(marginal_idxs) == 0 or (marginal_idxs < 0).any()
                or len(set(marginal_idxs)) != len(marginal_idxs)):
            raise ValueError("Control scoring requires distinct selected marginal factors")
        if interaction_pairs is None and len(sig_interacts):
            raise ValueError("Exact interaction_pairs are required for selected interactions")
        pairs = np.asarray(interaction_pairs if interaction_pairs is not None else [])
        if pairs.size == 0:
            pairs = np.empty((2, 0), dtype=int)
        if (pairs.ndim != 2 or pairs.shape[0] != 2
                or not np.issubdtype(pairs.dtype, np.integer)
                or (pairs < 0).any() or (pairs[0] >= z.shape[1]).any()
                or (pairs[1] >= z2.shape[1]).any()
                or not np.isin(pairs[0], marginal_idxs).all()):
            raise ValueError("interaction_pairs must be valid 2-by-L marginal/partner indices")
        k, count = len(marginal_idxs), pairs.shape[1]
        if count > k * z2.shape[1]:
            raise ValueError("Selected interactions exceed the random interaction pool")
        with np.errstate(over='ignore', invalid='ignore'):
            real = np.column_stack((z[:, marginal_idxs], z[:, pairs[0]] * z2[:, pairs[1]]))
        scores = {'real': _control_score(real, y, condition), 'condition': condition}
        rng = np.random.default_rng(seed)
        for group in ('full_random', 'partial_random'):
            if group == 'partial_random' and count == 0:
                continue
            values = []
            for _ in range(n_iters):
                selected = (rng.choice(z.shape[1], k, replace=False)
                            if group == 'full_random' else marginal_idxs)
                # R expand.grid lists all partners for each selected marginal.
                terms = rng.choice(k * z2.shape[1], count, replace=False)
                with np.errstate(over='ignore', invalid='ignore'):
                    interactions = z[:, selected[terms // z2.shape[1]]] * z2[:, terms % z2.shape[1]]
                design = np.column_stack((z[:, selected], interactions))
                values.append(_control_score(design, y, condition))
            scores[group] = np.asarray(values)
        return scores
