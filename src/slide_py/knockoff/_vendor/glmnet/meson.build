# Meson build file for glmnet Fortran extension

project('glmnet', 'c', 'fortran',
  version: '2.2.1',
  default_options: [
    'buildtype=release'
  ])

# Get Python and NumPy for f2py
py = import('python').find_installation('python', pure: false)
py_dep = py.dependency()

# Find NumPy include directories for f2py
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Fortran compile flags (fixed-form source with 8-byte reals)
fc = meson.get_compiler('fortran')
if fc.get_id() == 'gcc'
  add_project_arguments('-std=legacy', '-ffixed-form', '-fdefault-real-8', language: 'fortran')
elif fc.get_id() == 'llvm-flang'
  add_project_arguments('-ffixed-form', '-fdefault-real-8', language: 'fortran')
else
  error('glmnet requires GNU Fortran or LLVM Flang (fixed form, 8-byte default real)')
endif
if host_machine.system() == 'windows' and (fc.get_id() != 'llvm-flang' or meson.get_compiler('c').get_argument_syntax() != 'msvc')
  error('Windows glmnet requires MSVC-compatible C and LLVM Flang; native feasibility testing is required')
endif

# Fortran source and f2py interface
glmnet_source = files('src/glmnet/glmnet5.f90')
glmnet_pyf = files('_glmnet.pyf')

# Generate the wrapper from the pyf file
glmnet_wrapper = custom_target('_glmnetmodule',
  input: glmnet_pyf,
  output: ['_glmnetmodule.c'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@']
)

# Get fortranobject.c path
fortranobject_c = incdir_f2py / 'fortranobject.c'

# Build the extension module
py.extension_module('_glmnet',
  [glmnet_wrapper, glmnet_source, fortranobject_c],
  include_directories: inc_np,
  dependencies: py_dep,
  install: true,
  subdir: 'slide_py/knockoff/_vendor/glmnet'
)
