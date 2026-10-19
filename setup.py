"""
Build both native solvers required by the default SLIDE backend.

Environment variables for customization:
    BLAS_LIB_DIR    - Directory containing BLAS library
    LAPACK_LIB_DIR  - Directory containing LAPACK library
    BLAS_LIB        - BLAS library name (default: openblas or blas)
    LAPACK_LIB      - LAPACK library name (default: openblas or lapack)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from glob import glob
from importlib.machinery import EXTENSION_SUFFIXES
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class NativeBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name != 'slide_py.knockoff._vendor.glmnet._glmnet':
            if self.compiler.compiler_type == 'msvc':
                ext.extra_compile_args = ['/O2']
            else:
                warning = ('-Wno-sometimes-uninitialized' if sys.platform == 'darwin'
                           else '-Wno-maybe-uninitialized')
                ext.extra_compile_args = ['-O3', '-fPIC', '-Wno-unused-function', warning]
            return super().build_extension(ext)

        source = Path('src/slide_py/knockoff/_vendor/glmnet').resolve()
        build = Path(self.build_temp).resolve() / 'glmnet'
        build.mkdir(parents=True, exist_ok=True)
        native_file = build / 'python.ini'
        native_file.write_text("[binaries]\npython = " + repr(Path(sys.executable).as_posix()) + "\n")
        subprocess.run(
            [sys.executable, '-m', 'mesonbuild.mesonmain', 'setup',
             str(build), str(source), '--native-file', str(native_file)],
            check=True,
        )
        subprocess.run(
            [sys.executable, '-m', 'mesonbuild.mesonmain', 'compile', '-C', str(build)],
            check=True,
        )
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidates = [build / ('_glmnet' + suffix) for suffix in EXTENSION_SUFFIXES]
        artifacts = [path for path in candidates if path.is_file()]
        if len(artifacts) != 1:
            raise RuntimeError(f'Expected one glmnet extension in {build}, found {artifacts}')
        shutil.copy2(artifacts[0], destination)


def get_numpy_include():
    """Get NumPy include directory."""
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        return None


def get_blas_info():
    """
    Try to detect BLAS/LAPACK library paths.

    Priority:
    1. Environment variables (BLAS_LIB_DIR, LAPACK_LIB_DIR)
    2. NumPy's BLAS configuration
    3. Standard system locations
    """
    library_dirs = []
    libraries = []
    extra_link_args = []

    # Check environment variables first
    if os.environ.get('BLAS_LIB_DIR'):
        library_dirs.append(os.environ['BLAS_LIB_DIR'])
    if os.environ.get('LAPACK_LIB_DIR'):
        library_dirs.append(os.environ['LAPACK_LIB_DIR'])

    # Explicit names take priority over NumPy and automatic discovery.
    for variable in ('BLAS_LIB', 'LAPACK_LIB'):
        libraries.extend(name for name in os.environ.get(variable, '').split(',') if name)

    # Try to get info from NumPy
    try:
        import numpy as np

        # NumPy 2.0+ uses np.show_config() differently
        if not libraries and hasattr(np.__config__, 'get_info'):
            blas_info = np.__config__.get_info('blas_opt_info') or {}
            lapack_info = np.__config__.get_info('lapack_opt_info') or {}

            library_dirs.extend(blas_info.get('library_dirs', []))
            library_dirs.extend(lapack_info.get('library_dirs', []))
            libraries.extend(blas_info.get('libraries', []))
            libraries.extend(lapack_info.get('libraries', []))
            extra_link_args.extend(blas_info.get('extra_link_args', []))
    except Exception:
        pass

    # Add standard locations as fallback
    standard_dirs = [
        '/usr/lib',
        '/usr/lib64',
        '/usr/local/lib',
        '/usr/lib/x86_64-linux-gnu',  # Debian/Ubuntu
        '/opt/homebrew/opt/openblas/lib',  # macOS ARM
        '/usr/local/opt/openblas/lib',  # macOS Intel
    ]

    # Check for module-loaded libraries (HPC)
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    for path in ld_library_path.split(os.pathsep):
        if path and os.path.isdir(path):
            library_dirs.append(path)

    library_dirs.extend([d for d in standard_dirs if os.path.isdir(d)])

    # OpenBLAS includes LAPACK; Windows needs an MSVC-compatible import library.
    if not libraries:
        filenames = [('openblas.lib', 'openblas'), ('libopenblas.lib', 'libopenblas')] if sys.platform == 'win32' else [
            ('libopenblas' + suffix, 'openblas') for suffix in ('.so', '.dylib', '.a')
        ]
        for lib_dir in library_dirs:
            for filename, name in filenames:
                if os.path.isfile(os.path.join(lib_dir, filename)):
                    libraries = [name]
                    break
            if libraries:
                break
        if not libraries:
            if sys.platform == 'win32':
                raise RuntimeError(
                    'Set BLAS_LIB_DIR and LAPACK_LIB_DIR to MSVC-compatible BLAS/LAPACK '
                    'import libraries; set BLAS_LIB and LAPACK_LIB for non-OpenBLAS names.'
                )
            libraries = ['blas', 'lapack']

    # Remove duplicates while preserving order
    library_dirs = list(dict.fromkeys(library_dirs))
    libraries = list(dict.fromkeys(libraries))

    return library_dirs, libraries, extra_link_args


def get_dsdp_solverension():
    """
    Create the pydsdp5 Extension object.
    """
    # Get source files
    base_dir = 'src/slide_py/dsdp_solver'
    c_dir = os.path.join(base_dir, 'dsdp', 'C')

    sources = [os.path.join(c_dir, 'pyreadsdpa.c')]
    sources.extend(glob(os.path.join(c_dir, 'allc', '*.c')))

    if not sources or len(sources) < 2:
        raise RuntimeError("Could not find dsdp_solver C source files")

    # Get include directories
    include_dirs = [os.path.join(c_dir, 'allinclude')]

    numpy_include = get_numpy_include()
    if numpy_include:
        include_dirs.append(numpy_include)

    # Get BLAS/LAPACK configuration
    library_dirs, libraries, extra_link_args = get_blas_info()

    return Extension(
        'slide_py.dsdp_solver.pydsdp5',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    )


# Setup (metadata comes from pyproject.toml)
setup(
    ext_modules=[
        get_dsdp_solverension(),
        Extension('slide_py.knockoff._vendor.glmnet._glmnet', sources=[]),
    ],
    cmdclass={'build_ext': NativeBuildExt},
)
