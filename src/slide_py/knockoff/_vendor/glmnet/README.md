# Vendored python-glmnet

This is a vendored copy of [python-glmnet](https://github.com/civisanalytics/python-glmnet)
for use in knockoff-filter. The Fortran-based glmnet provides ~5-10x speedup over sklearn
for lasso path computation.

## Building the Fortran Extension

The Fortran extension requires:
- gfortran (GNU Fortran compiler)
- gcc (C compiler for f2py wrapper)
- meson + ninja (build system)

### On HPC (CRC clusters)

```bash
# Load compilers
module load gcc/12.2.0

# Activate conda environment with meson/ninja
conda activate slide-py_env  # or any env with meson, ninja, numpy

# Build the extension
cd knockoff/_vendor/glmnet
meson setup builddir
ninja -C builddir

# Copy the .so file to the package directory
cp builddir/_glmnet*.so .
```

### General installation

```bash
# Install build dependencies
pip install meson ninja numpy

# Build the extension
cd knockoff/_vendor/glmnet
meson setup builddir
ninja -C builddir
cp builddir/_glmnet*.so .
```

## Fallback Behavior

If the Fortran extension is not available, knockoff-filter automatically falls back to
sklearn's pure-Python implementation. The fallback is transparent and produces equivalent
results, but may be slower for large datasets.

## License

The vendored code is licensed under GPL-2.0 (see LICENSE file).
Original source: https://github.com/civisanalytics/python-glmnet
