"""Method-4 CV: fitted Gaussian submodels, held-out scoring and native selection."""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slide_py import SLIDEcv


def make_cv(*, y=None, z=None, **kwargs):
    rng = np.random.default_rng(21)
    z = rng.normal(size=(32, 3)) if z is None else np.asarray(z, dtype=float)
    y = 2 * z[:, 0] + rng.normal(size=len(z)) if y is None else np.asarray(y, dtype=float)
    ids = [f"sample-{i}" for i in range(len(z))]
    slide = SimpleNamespace(latent_factors=pd.DataFrame(z, index=ids),
                            data=SimpleNamespace(Y=pd.DataFrame(y, index=ids)),
                            input_params=dict(niter=3, n_workers=1, spec=0.2, fdr=0.1))
    return SLIDEcv(slide, nrep=2, k=4, **kwargs)


@pytest.fixture(scope="module")
def rscript():
    executable = shutil.which(os.environ.get('SLIDE_RSCRIPT', 'Rscript'))
    if executable is None:
        pytest.skip("R reference requires Rscript")
    probe = subprocess.run([executable, '--vanilla', '-e',
                            'quit(status=if(requireNamespace("dplyr",quietly=TRUE))0 else 42)'],
                           capture_output=True, text=True)
    if probe.returncode == 42:
        pytest.skip("R reference requires dplyr")
    probe.check_returncode()
    return executable


@pytest.mark.parametrize("binary", [False, True])
def test_fixed_selected_submodels_and_final_predictions_match_real_r(binary, rscript):
    train = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2], [3, 1]], dtype=float)
    valid = np.array([[.5, 1], [1, 1.5], [2.5, 2], [3, 2]], dtype=float)
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1] if binary else [1, 2, 1, 4, 3, 5, 4, 8], dtype=float)
    ups_train, ups_valid = SLIDEcv._build_prediction_features(
        train, valid, [0, 1], [(0, 1), (1, 0)], train_y=y)
    predicted = SLIDEcv._gaussian_predict(ups_train, y, ups_valid)
    code = r'''
        suppressPackageStartupMessages(library(dplyr))
        args <- commandArgs(TRUE)
        for (f in c('interUnion.R','getUpsilon.R','runSLIDEBeta.R')) source(file.path(args[1], f))
        z <- matrix(c(0,0,1,0,0,1,1,1,2,1,1,2,2,2,3,1),ncol=2,byrow=TRUE)
        v <- matrix(c(.5,1,1,1.5,2.5,2,3,2),ncol=2,byrow=TRUE)
        colnames(z) <- colnames(v) <- c('Z1','Z2')
        binary <- args[2] == 'True'
        y <- if(binary) c(0,0,1,0,1,1,0,1) else c(1,2,1,4,3,5,4,8)
        SLIDE <- function(...) {
            u <- lapply(1:2,function(i) {
                terms <- interUnion(i,z)$interaction
                getUpsilon(colnames(terms),terms,z,y,i)
            })
            list(marginal_vars=1:2,interaction_vars=c('Z1.Z2','Z2.Z1'),
                 upsilon=cbind(u[[1]]$upsilon,u[[2]]$upsilon),
                 upsilon_mods=list(ups1=u[[1]]$upsilon_mod,ups2=u[[2]]$upsilon_mod))
        }
        invisible(capture.output(res <- runSLIDEBeta(y,NULL,z,v,4,.2,3,.1,100,NULL,NULL,NULL,FALSE,1,binary,TRUE)))
        emit <- function(label,x) cat(label,paste(format(x,digits=17),collapse=','),'\n',sep='')
        for(i in 1:2) {
            emit(paste0('TRAIN',i,':'),res$model$upsilon[,i])
            emit(paste0('VALID',i,':'),predict(res$model$upsilon_mods[[i]],
                 newdata=setNames(data.frame(v[,i],v[,1]*v[,2]),c(as.character(i),if(i==1)'Z1.Z2' else 'Z2.Z1'))))
        }
        emit('FINAL:',res$valid_pred)
        a <- data.frame(first=z[,1],alias=2*z[,1],other=z[,2])
        av <- data.frame(first=v[,1],alias=c(99,98,97,96),other=v[,2])
        emit('ALIAS:',suppressWarnings(predict(lm(y~.,a),av)))
    '''
    root = Path(__file__).resolve().parents[2]
    reference = subprocess.run([rscript, '--vanilla', '-e', code,
                                str(root / 'R'), str(binary)], capture_output=True, text=True, check=True)
    values = {line.split(':', 1)[0]: np.fromstring(line.split(':', 1)[1], sep=',')
              for line in reference.stdout.splitlines() if line.startswith(('TRAIN', 'VALID', 'FINAL:', 'ALIAS:'))}
    for i in range(2):
        np.testing.assert_allclose(ups_train[:, i], values[f'TRAIN{i+1}'], atol=1e-9)
        np.testing.assert_allclose(ups_valid[:, i], values[f'VALID{i+1}'], atol=1e-9)
    np.testing.assert_allclose(predicted, values['FINAL'], atol=1e-9)
    aliased = SLIDEcv._gaussian_predict(np.column_stack((train[:, 0], 2*train[:, 0], train[:, 1])), y,
                                        np.column_stack((valid[:, 0], [99, 98, 97, 96], valid[:, 1])))
    np.testing.assert_allclose(aliased, values['ALIAS'], atol=1e-9)


@pytest.mark.parametrize("binary", [False, True])
def test_native_selection_reproducibility_coverage_and_observed_null_separation(binary, tmp_path):
    cv = make_cv()
    if binary:
        cv = make_cv(z=cv.slide_obj.latent_factors.values, y=(cv.y > np.median(cv.y)).astype(float), eval_type='auc')
    np.random.seed(173)
    state = np.random.get_state()
    first = cv.run(seed=9, outpath=tmp_path / "cv")
    predictions = cv.predictions_.copy()
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    pd.testing.assert_frame_equal(first, cv.run(seed=9))
    pd.testing.assert_frame_equal(predictions, cv.predictions_)
    assert set(first.method) == {'SLIDE', 'SLIDE_y'}
    assert set(first.status) == {'ok'}
    assert set(predictions.model) == {'gaussian_ols'}
    assert len(predictions) == cv.nrep * 2 * cv.n_samples
    assert predictions.groupby(['replicate', 'method', 'row_index']).size().eq(1).all()
    branches = predictions.pivot(index=['replicate', 'row_index'], columns='method', values='y_pred')
    assert not np.allclose(branches.SLIDE, branches.SLIDE_y)
    for (_, _), group in predictions.groupby(['replicate', 'fold']):
        np.testing.assert_array_equal(group[group.method == 'SLIDE'].y_true,
                                      group[group.method == 'SLIDE_y'].y_true)
    if binary:
        assert first.metric_value.ge(.5).all()
        np.testing.assert_allclose(first.metric_value, np.maximum(first.raw_metric, 1-first.raw_metric))
    assert (tmp_path / "cv" / "slidecv_predictions.csv").is_file()


def test_fallback_uses_fresh_native_rng(monkeypatch):
    cv = make_cv()
    cv.do_interacts = False
    monkeypatch.setattr(cv, '_select', lambda *args: np.array([], dtype=int))
    train, valid = cv.z_matrix[:24], cv.z_matrix[24:]
    rng = np.random.default_rng(6)
    selected = np.random.default_rng(6).integers(cv.n_lfs)
    actual = cv._run_slide_fold(train, cv.y[:24], valid, rng=rng)
    expected = cv._gaussian_predict(train[:, [selected]], cv.y[:24], valid[:, [selected]])
    np.testing.assert_allclose(actual, expected)


def test_interaction_reverse_pairs_residuals_and_default_chunk(monkeypatch):
    cv = make_cv()
    seen = []
    def select(z, y, spec, f_size, rng):
        seen.append((z.copy(), y.copy(), f_size))
        return np.array([0])
    monkeypatch.setattr(cv, '_select', select)
    pairs = cv._find_interactions_fold(cv.z_matrix, cv.y, [0, 1], 2)
    assert pairs == [(0, 1), (1, 0)]
    for marginal, (terms, response, chunk) in enumerate(seen):
        assert chunk == 100
        assert abs(response.mean()) < 1e-12
        assert abs(response @ cv.z_matrix[:, marginal]) < 1e-10
        np.testing.assert_allclose(terms[:, 0], cv.z_matrix[:, 0] * cv.z_matrix[:, 1])


def test_no_selected_upsilons_falls_back_to_all_marginals(monkeypatch):
    cv = make_cv()
    selections = iter(([0, 1], [0], [0], []))
    monkeypatch.setattr(cv, '_select', lambda *args: np.asarray(next(selections), dtype=int))
    actual = cv._run_slide_fold(cv.z_matrix[:24], cv.y[:24], cv.z_matrix[24:])
    expected = cv._gaussian_predict(cv.z_matrix[:24, :2], cv.y[:24], cv.z_matrix[24:, :2])
    np.testing.assert_allclose(actual, expected)


def test_selected_upsilon_and_full_data_chunk_policy(monkeypatch):
    cv = make_cv(z=np.random.default_rng(4).normal(size=(10, 12)), y=np.arange(10))
    assert cv._f_size == 10
    selections = iter(([0, 1], [0], [0], [1]))
    chunks = []
    def select(z, y, spec, f_size, rng):
        chunks.append(f_size)
        return np.asarray(next(selections), dtype=int)
    monkeypatch.setattr(cv, '_select', select)
    train, valid, y = cv.z_matrix[:8], cv.z_matrix[8:], cv.y[:8]
    actual = cv._run_slide_fold(train, y, valid)
    ups_train, ups_valid = cv._build_prediction_features(train, valid, [0, 1], [(0, 1), (1, 0)], train_y=y)
    expected = cv._gaussian_predict(ups_train[:, [1]], y, ups_valid[:, [1]])
    np.testing.assert_allclose(actual, expected)
    assert chunks == [10, 100, 100, 10]
