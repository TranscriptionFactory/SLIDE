"""Fixed-design repeated holdouts for screening, separate from rigorous SLIDEcv."""

import math

import numpy as np
from scipy.linalg import lstsq
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _fit_ols(design, response):
    """Keep the first independent terms, as R lm's ordered QR does."""
    kept, basis = [], []
    for column in range(design.shape[1]):
        residual = design[:, column].copy()
        for _ in range(2):
            for vector in basis:
                residual -= vector * np.dot(vector, residual)
        norm = np.linalg.norm(residual)
        if norm > 1e-7 * np.linalg.norm(design[:, column]):
            kept.append(column)
            basis.append(residual / norm)
    coefficients = lstsq(design[:, kept], response, lapack_driver='gelsy')[0]
    return kept, coefficients


def sample_cv(z, y, marginal_indices, interaction_pairs=None, *, condition=None,
              k=4, iterations=100, seed=None, train_splits=None, sample_ids=None):
    """Pool held-out OLS predictions without reselection or standardization.

    Indices are zero-based; pairs have shape (2, L). Explicit train_splits
    replace random draws and must each contain ceil((1-1/k)*n) distinct rows.
    The saved coefficients belong to the LAST split, not a full-data fit.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if (z.ndim != 2 or len(y) != z.shape[0] or len(y) < 2
            or not np.isfinite(z).all() or not np.isfinite(y).all()):
        raise ValueError('sample_cv requires finite z and row-aligned y with at least two rows')
    if condition is None:
        condition = 'auc' if len(np.unique(y)) <= 2 else 'corr'
    if condition not in ('auc', 'corr'):
        raise ValueError('sample_cv condition must be auc or corr')
    levels = np.unique(y)
    if condition == 'auc' and len(levels) != 2:
        raise ValueError('sample_cv auc requires exactly two response levels')
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 2 <= k <= len(y):
        raise ValueError('sample_cv k must be an integer between 2 and n')
    if (isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer))
            or iterations < 1):
        raise ValueError('sample_cv iterations must be a positive integer')
    marginals = np.asarray(marginal_indices)
    pairs = np.asarray(interaction_pairs if interaction_pairs is not None else [])
    if marginals.size == 0:
        marginals = np.array([], dtype=int)
    if pairs.size == 0:
        pairs = np.empty((2, 0), dtype=int)
    if (marginals.ndim != 1 or not np.issubdtype(marginals.dtype, np.integer)
            or len(np.unique(marginals)) != len(marginals)
            or (marginals < 0).any() or (marginals >= z.shape[1]).any()):
        raise ValueError('sample_cv marginal_indices must be distinct valid column indices')
    if (pairs.ndim != 2 or pairs.shape[0] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
            or (pairs < 0).any() or (pairs >= z.shape[1]).any()
            or not np.isin(pairs[0], marginals).all()):
        raise ValueError('sample_cv interaction_pairs must be valid 2-by-L marginal/partner indices')
    ids = list(range(len(y))) if sample_ids is None else list(sample_ids)
    if len(ids) != len(y):
        raise ValueError('sample_cv sample_ids must match the number of rows')
    train_n = math.ceil((1 - 1 / k) * len(y))
    result = dict(metric=None, predictions=[], last_coefficients=None, last_rank=None,
                  last_aliased_terms=[], last_train_indices=[], last_train_ids=[],
                  term_order=['(Intercept)'] + [f'Z{i+1}' for i in marginals]
                  + [f'Z{i+1}.Z{j+1}' for i, j in pairs.T], guard='ok', status='ok',
                  settings=dict(condition=condition, k=int(k), iterations=int(iterations),
                                n=len(y), train_n=train_n, test_n=len(y)-train_n,
                                seed=None if seed is None else int(seed),
                                model='last_split_ols', score_provenance='pooled_holdout'))
    if not len(marginals):
        result['guard'] = result['status'] = 'no_marginals'
        return result
    if len(np.unique(np.concatenate((marginals, pairs.ravel())))) > train_n:
        result['guard'] = result['status'] = 'model_too_large'
        return result
    if train_n >= len(y):
        raise ValueError('sample_cv split must leave at least one held-out row')
    design = np.column_stack((np.ones(len(y)), z[:, marginals],
                              z[:, pairs[0]] * z[:, pairs[1]]))
    if not np.isfinite(design).all():
        raise ValueError('sample_cv interaction products must be finite')
    rng = np.random.default_rng(seed)
    splits = (train_splits if train_splits is not None else
              (rng.choice(len(y), train_n, replace=False) for _ in range(iterations)))
    if train_splits is not None and len(train_splits) != iterations:
        raise ValueError('sample_cv train_splits length must equal iterations')
    for iteration, indices in enumerate(splits):
        train = np.asarray(indices)
        if (train.ndim != 1 or len(train) != train_n
                or not np.issubdtype(train.dtype, np.integer)
                or len(np.unique(train)) != train_n
                or (train < 0).any() or (train >= len(y)).any()):
            raise ValueError(f'sample_cv split {iteration}: expected {train_n} distinct valid train indices')
        test = np.setdiff1d(np.arange(len(y)), train)
        kept, coefficients = _fit_ols(design[train], y[train])
        predicted = design[test][:, kept] @ coefficients
        if not np.isfinite(coefficients).all() or not np.isfinite(predicted).all():
            raise ValueError(f'sample_cv split {iteration}: nonfinite OLS coefficients or predictions')
        result['predictions'].extend(
            dict(iteration=iteration, row_index=int(row), sample_id=str(ids[row]),
                 y_true=float(y[row]), y_pred=float(value))
            for row, value in zip(test, predicted))
        result['last_coefficients'] = [None] * design.shape[1]
        for column, value in zip(kept, coefficients):
            result['last_coefficients'][column] = float(value)
        result['last_rank'] = len(kept)
        result['last_aliased_terms'] = [i for i in range(design.shape[1]) if i not in kept]
        result['last_train_indices'] = train.tolist()
        result['last_train_ids'] = [str(ids[i]) for i in train]
    true = np.array([p['y_true'] for p in result['predictions']])
    pred = np.array([p['y_pred'] for p in result['predictions']])
    if condition == 'auc':
        result['metric'] = float(roc_auc_score(true == levels[1], pred))
    elif (np.isfinite(true).all() and np.ptp(true) > 0 and np.ptp(pred) > 0):
        result['metric'] = float(np.corrcoef(rankdata(true), rankdata(pred))[0, 1])
    if result['metric'] is None or not np.isfinite(result['metric']):
        result['metric'] = None
        result['status'] = 'missing_metric'
    return result
