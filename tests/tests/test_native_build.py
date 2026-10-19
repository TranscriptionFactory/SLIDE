"""Regression checks for BLAS discovery without compiling a target toolchain."""

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import setuptools


@pytest.fixture
def native_setup(monkeypatch):
    for name in ('BLAS_LIB_DIR', 'LAPACK_LIB_DIR', 'BLAS_LIB', 'LAPACK_LIB', 'LD_LIBRARY_PATH'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(setuptools, 'setup', lambda **kwargs: None)
    monkeypatch.setattr(np.__config__, 'get_info', lambda name: {}, raising=False)
    with monkeypatch.context() as bootstrap:
        bootstrap.setenv('BLAS_LIB', 'openblas')
        return runpy.run_path(str(Path(__file__).resolve().parents[2] / 'setup.py'))


@pytest.mark.parametrize('filename, library', [
    ('openblas.lib', 'openblas'), ('libopenblas.lib', 'libopenblas'),
    ('libopenblas.dylib', 'openblas'),
])
def test_openblas_library_discovery(native_setup, monkeypatch, tmp_path, filename, library):
    monkeypatch.setattr(native_setup['sys'], 'platform', 'darwin' if filename.endswith('.dylib') else 'win32')
    (tmp_path / filename).touch()
    monkeypatch.setenv('BLAS_LIB_DIR', str(tmp_path))
    directories, libraries, _ = native_setup['get_blas_info']()
    assert directories[0] == str(tmp_path)
    assert libraries == [library]


def test_explicit_libraries_override_numpy(native_setup, monkeypatch, tmp_path):
    monkeypatch.setattr(np.__config__, 'get_info', lambda name: {'libraries': ['wrong']})
    monkeypatch.setenv('BLAS_LIB_DIR', str(tmp_path))
    monkeypatch.setenv('LAPACK_LIB_DIR', str(tmp_path))
    monkeypatch.setenv('BLAS_LIB', 'custom_blas,custom_blas')
    monkeypatch.setenv('LAPACK_LIB', 'custom_lapack')
    directories, libraries, _ = native_setup['get_blas_info']()
    assert directories.count(str(tmp_path)) == 1
    assert libraries == ['custom_blas', 'custom_lapack']


def test_windows_does_not_guess_unix_link_libraries(native_setup, monkeypatch):
    monkeypatch.setattr(native_setup['sys'], 'platform', 'win32')
    with pytest.raises(RuntimeError, match='MSVC-compatible BLAS/LAPACK'):
        native_setup['get_blas_info']()


@pytest.mark.skipif(sys.platform == 'win32' or shutil.which('bash') is None, reason='Unix wrapper')
@pytest.mark.parametrize('active', [True, False])
@pytest.mark.parametrize('layout', ['lib', 'Library/lib'])
def test_pixi_runtime_library_precedence(tmp_path, active, layout):
    prefix = tmp_path / 'pixi-env'
    library = prefix / layout
    runtime = prefix / 'Library/bin' if layout == 'Library/lib' else library
    library.mkdir(parents=True)
    runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    variables = ['LIBRARY_PATH', 'LD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'OPENBLAS_TARGET_DIR']
    for name in [*variables, 'PIXI_ENVIRONMENT_NAME']:
        env.pop(name, None)
    env['CONDA_PREFIX'] = str(prefix)
    env['TMPDIR'] = str(tmp_path)
    if active:
        env['PIXI_ENVIRONMENT_NAME'] = 'rust'
    wrapper = Path(__file__).resolve().parents[2] / 'scripts/with-rust-blas-env.sh'
    command = f'import json, os; print(json.dumps([os.environ.get(k, "") for k in {variables!r}]))'
    result = subprocess.run(
        ['bash', str(wrapper), sys.executable, '-c', command],
        env=env, check=True, capture_output=True, text=True,
    )
    paths = json.loads(result.stdout)
    if active:
        assert [path.split(os.pathsep)[0] for path in paths] == [
            str(library), str(runtime), str(runtime), str(library),
        ]
    else:
        assert all(str(prefix) not in path for path in paths)
