#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

append_path() {
  local var_name="$1"
  local value="$2"
  local current="${!var_name:-}"
  if [[ -z "${value}" || ! -d "${value}" ]]; then
    return 0
  fi

  case ":${current}:" in
    *":${value}:"*) return 0 ;;
  esac

  if [[ -n "${current}" ]]; then
    export "${var_name}=${value}:${current}"
  else
    export "${var_name}=${value}"
  fi
}

find_first_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

# Active Pixi libraries take precedence over cached builds and host installations.
if [[ -n "${PIXI_ENVIRONMENT_NAME:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  lib_dir="${CONDA_PREFIX}/lib"
  runtime_dir="${lib_dir}"
  if [[ -d "${CONDA_PREFIX}/Library/lib" ]]; then
    lib_dir="${CONDA_PREFIX}/Library/lib"
    runtime_dir="${CONDA_PREFIX}/Library/bin"
  fi
  append_path LIBRARY_PATH "${lib_dir}"
  append_path LD_LIBRARY_PATH "${runtime_dir}"
  append_path DYLD_FALLBACK_LIBRARY_PATH "${runtime_dir}"
  export OPENBLAS_TARGET_DIR="${lib_dir}"
  exec "$@"
fi

openblas_dir=""
if [[ -d "${repo_root}/target/debug/build" ]]; then
  while IFS= read -r dir; do
    if [[ -e "${dir}/libopenblas.so" || -e "${dir}/libopenblas.so.0" || -e "${dir}/libopenblas.a" ]]; then
      openblas_dir="${dir}"
      break
    fi
  done < <(
    find "${repo_root}/target/debug/build" \
      -path '*/out/OpenBLAS-0.3.32' \
      -type d \
      2>/dev/null
  )
fi

if [[ -z "${openblas_dir}" ]]; then
  openblas_dir="$(
    find_first_dir \
      /usr/lib/x86_64-linux-gnu \
      /usr/lib64 \
      /home/node/linuxbrew/lib \
      /home/linuxbrew/.linuxbrew/lib \
      || true
  )"
fi

gfortran_dir="$(
  find_first_dir \
    /home/node/linuxbrew/lib \
    /home/node/linuxbrew/Cellar/gcc/15.2.0_1/lib \
    /home/node/linuxbrew/Cellar/gcc/15.2.0_1/lib/gcc/current \
    /home/linuxbrew/.linuxbrew/lib \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib64 \
    || true
)"

quadmath_dir="${gfortran_dir}"

append_path LD_LIBRARY_PATH "${openblas_dir}"
append_path LD_LIBRARY_PATH "${gfortran_dir}"
append_path LD_LIBRARY_PATH "${quadmath_dir}"
append_path LIBRARY_PATH "${openblas_dir}"
append_path LIBRARY_PATH "${gfortran_dir}"
append_path LIBRARY_PATH "${quadmath_dir}"

if [[ -n "${openblas_dir}" ]]; then
  export OPENBLAS_TARGET_DIR="${openblas_dir}"
fi

exec "$@"
