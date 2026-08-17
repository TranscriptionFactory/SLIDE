from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr
from .knockoffs import Knockoffs


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
    
class SLIDE_Estimator(Estimator):

    @staticmethod
    def score_performance(
            sig_LFs, sig_interacts, y, latent_factors, latent_factors_z2=None,
            n_iters=100, test_size=0.2, scaler='standard'):
        """
        Score the performance of the given latent factors relative to each other and to random
        selection of marginals and interactions.
        This visualization is similar to the control plot in original R code.

        s1: standalone LFs 
        s2: interacting LF terms (marginal * interacting)
        """

        if latent_factors_z2 is None: # This is only for SLIDE_PLM
            latent_factors_z2 = latent_factors

        scores = defaultdict(list)
        n = y.shape[0]
        
        s1 = latent_factors.loc[:, sig_LFs] # standalone LFs
        s2 = Knockoffs.get_interaction_terms(s1, latent_factors_z2.loc[:, sig_interacts]).reshape(n, -1)        # interaction LF terms
        s3 = np.concatenate([s1, s2], axis=1) if len(s2) > 0 else s1
        scores['real'] = _mean_valid_scores(
            Estimator.get_aucs(s3, y, n_iters, test_size, scaler)
        )

        n_marginals = s1.shape[1]
        n_interactions = s2.shape[1]
        sig_terms = np.concatenate([sig_LFs, sig_interacts], axis=0)
        non_marginal_terms = latent_factors.columns[~latent_factors.columns.isin(sig_LFs)]
        non_interaction_terms = latent_factors_z2.columns[~latent_factors_z2.columns.isin(sig_terms)]

        # To prevent out of bounds when doing np.random.choice
        n_interactions = min(n_interactions, len(non_interaction_terms))
        n_marginals = min(n_marginals, len(non_marginal_terms))

        # Skip random comparison if not enough non-significant terms to sample
        if n_marginals == 0:
            scores['full_random'] = np.array([np.nan])
            scores['partial_random'] = np.array([np.nan])
            return scores

        for _ in range(n_iters):
            # Get random marginal terms
            s1_random = latent_factors.loc[:, np.random.choice(
                non_marginal_terms, 
                size=n_marginals, 
                replace=False
            )].values

            # Get random interaction terms
            s2_random = Knockoffs.get_interaction_terms(
                s1_random,
                latent_factors_z2.loc[:, np.random.choice(
                    non_interaction_terms,
                    size=n_interactions,
                    replace=False
                )]
            ).reshape(n, -1)

            # Combine random terms
            s3_random = (np.concatenate([s1_random, s2_random], axis=1) 
                        if len(s2_random) > 0 else s1_random)
            
            # Calculate scores for fully random selection
            scores['full_random'].append(
                Estimator.get_aucs(s3_random, y, 1, test_size, scaler)
            )

            # Calculate scores for partial random selection
            s2_real = Knockoffs.get_interaction_terms(
                s1,
                latent_factors_z2.loc[:, np.random.choice(
                    non_interaction_terms,
                    size=n_interactions,
                    replace=False
                )]
            ).reshape(n, -1)

            s3_partial = (np.concatenate([s1, s2_real], axis=1) 
                        if len(s2_real) > 0 else s1)
            
            scores['partial_random'].append(
                Estimator.get_aucs(s3_partial, y, 1, test_size, scaler)
            )

        scores['partial_random'] = np.array(scores['partial_random']).flatten()
        scores['full_random'] = np.array(scores['full_random']).flatten()

        return scores
