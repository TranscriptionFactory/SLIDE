# Manual Python source installation

Prefer the [Pixi installation](../README.md#source-checkout-and-platform-coverage).
The recipes below use system compilers and libraries. Run commands from the
repository root.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. These
examples use Python 3.11 and a new `.venv`; do not overwrite an environment you
want to keep.
On Debian/Ubuntu:

```bash
sudo apt-get install build-essential gfortran python3-dev python3-venv libopenblas-dev liblapack-dev
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python .       # native Python backend, including plots
# uv pip install --python .venv/bin/python ".[r]"  # also needs R and R packages
source .venv/bin/activate
slide-py --version
```

Source installation builds the bundled C DSDP and Fortran glmnet extensions.
It needs C/C++/Fortran compilers, Python headers, BLAS/LAPACK; the isolated
Python build environment installs Meson/Ninja. R is not needed for the native
Python backend. Installation fails if either native extension cannot build; it
does not silently install a different fitting backend. Local Linux source-build
checks cover Python 3.11 only; repaired wheels and other interpreters await CI.
The Windows Pixi recipe supplies Flang, but its mixed C/Fortran linking and
runtime behavior still require native CI; no verified Windows Python recipe is available. Dependency-wheel checks do not
prove that the SLIDE wheel builds.

The uv commands have not been tested end-to-end here; the local installation
checks recorded in [CONTRIBUTING.md](../CONTRIBUTING.md#recorded-validation-status-2026-10-19) used pip. uv does not replace the native compiler prerequisites.

For an isolated source-distribution build, build the sdist from the checkout or
download `slide-py-sdist`, then extract it and build its wheel in a fresh
environment. PEP 517 isolation installs the declared build dependencies; install
runtime dependencies as binary wheels when checking an install. Examples are
repository files and are not guaranteed to be included in the sdist. With the
`build` frontend installed, from the checkout root:

```bash
python -m build --sdist --outdir dist
mkdir -p build/sdist
tar -xzf dist/*.tar.gz -C build/sdist
python -m build --wheel --outdir dist/wheels build/sdist/*
```

For macOS 15+, install Xcode Command Line Tools and a CPython 3.9–3.12
interpreter first. The following source recipe is **untested on native macOS**:

```bash
brew install openblas gcc
export MACOSX_DEPLOYMENT_TARGET=15.0
export FC="$(brew --prefix gcc)/bin/gfortran"
export BLAS_LIB_DIR="$(brew --prefix openblas)/lib"
export LAPACK_LIB_DIR="$BLAS_LIB_DIR"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python .
source .venv/bin/activate
slide-py --version
```

For the R-enabled development environment on Linux x86-64, install Pixi and run:

```bash
pixi install --locked -e dev
pixi run --locked -e dev slide-py --version
```

This environment intentionally includes CVXPY, R, `rpy2`, and R backend
packages. Only this R-enabled environment is Linux-only; the default and Rust
environments have four-target locks. Do not add R or
`rpy2` as native Python build requirements.
For a native uv or pip installation, select `knockoff_backend: python` as in the [quick start](../README.md#python-usage) to avoid requiring R at runtime; some lower-level Python methods default
to the R backend. The `[r]` extra installs the bridge, not R or its packages.
