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
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class NativeBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name != 'slide_py.knockoff._vendor.glmnet._glmnet':
            return super().build_extension(ext)

        source = Path('src/slide_py/knockoff/_vendor/glmnet').resolve()
        build = Path(self.build_temp).resolve() / 'glmnet'
        build.mkdir(parents=True, exist_ok=True)
        native_file = build / 'python.ini'
        native_file.write_text("[binaries]\npython = " + repr(sys.executable) + "\n")
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
        shutil.copy2(build / destination.name, destination)


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

    # Try to get info from NumPy
    try:
        import numpy as np

        # NumPy 2.0+ uses np.show_config() differently
        if hasattr(np.__config__, 'get_info'):
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
    for path in ld_library_path.split(':'):
        if path and os.path.isdir(path):
            library_dirs.append(path)

    library_dirs.extend([d for d in standard_dirs if os.path.isdir(d)])

    # Determine library names
    if not libraries:
        # Check environment variables for library names
        blas_lib = os.environ.get('BLAS_LIB', '').split(',')
        lapack_lib = os.environ.get('LAPACK_LIB', '').split(',')

        if blas_lib and blas_lib[0]:
            libraries.extend(blas_lib)
        if lapack_lib and lapack_lib[0]:
            libraries.extend(lapack_lib)

        # Default: try openblas first (includes LAPACK), then separate libs
        if not libraries:
            # Check if openblas is available
            for lib_dir in library_dirs:
                if any(os.path.exists(os.path.join(lib_dir, f'libopenblas{ext}'))
                       for ext in ['.so', '.dylib', '.a']):
                    libraries = ['openblas']
                    break

            # Fallback to separate blas/lapack
            if not libraries:
                if sys.platform == 'darwin':
                    # macOS Accelerate framework
                    libraries = ['blas', 'lapack']
                else:
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

    # Compiler flags
    extra_compile_args = ['-O3', '-fPIC']
    if sys.platform == 'darwin':
        extra_compile_args.extend(['-Wno-unused-function', '-Wno-sometimes-uninitialized'])
    else:
        extra_compile_args.extend(['-Wno-unused-function', '-Wno-maybe-uninitialized'])

    return Extension(
        'slide_py.dsdp_solver.pydsdp5',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
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
