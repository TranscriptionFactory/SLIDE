export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:/ihome/djishnu/aar126/.nvm/versions/node/v25.7.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path:/ihome/djishnu/aar126/.codex/tmp/arg0/codex-arg0OSIJzp:/ihome/djishnu/aar126/.nvm/versions/node/v25.7.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path:/ihome/djishnu/aar126/.bun/bin:/ihome/djishnu/aar126/go/bin:/ihome/djishnu/aar126/linuxbrew/bin:/ihome/djishnu/aar126/linuxbrew/sbin:/scratch/slurm-10955239/tmp/rmux-shim-53a8d3fa31fc069b:/ihome/djishnu/aar126/.bun/bin:/ihome/djishnu/aar126/go/bin:/ihome/djishnu/aar126/.local/share/mamba/condabin:/ihome/djishnu/aar126/src/shellconfig/devtools/.venv/bin:/ihome/djishnu/aar126/src/shellconfig/bin/git-subcommands:/ihome/djishnu/aar126/.nvm/versions/node/v25.7.0/bin:/ihome/djishnu/aar126/.cargo/bin:/ix/djishnu/Aaron/1_general_use/special_scripts/JishnuLab/:/ihome/djishnu/aar126/.local/bin:/ihome/djishnu/aar126/src/shellconfig/bin:/ihome/djishnu/aar126/.pixi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/ihome/crc/pipx/bin/:/ihome/djishnu/aar126/bin:/ihome/djishnu/aar126/.local/bin:/ihome/djishnu/aar126/bin:/ihome/crc/pipx/bin/"
export CONDA_SHLVL=1
export CONDA_PREFIX=/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev
export PIXI_PROJECT_ROOT=/ix/djishnu/Aaron/1_general_use/omni-SLIDE
export PIXI_EXE=/ihome/djishnu/aar126/.pixi/bin/pixi
export PIXI_PROJECT_VERSION=1.0.0
export PIXI_IN_SHELL=1
export PIXI_PROJECT_MANIFEST=/ix/djishnu/Aaron/1_general_use/omni-SLIDE/pixi.toml
export PIXI_PROJECT_NAME=slide-py
export CONDA_DEFAULT_ENV=slide-py:dev
export PIXI_ENVIRONMENT_NAME=dev
export PIXI_ENVIRONMENT_PLATFORMS=linux-64
export PIXI_PROMPT='(slide-py:dev) '
. /ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/etc/conda/activate.d/activate-r-base.sh
. /ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/etc/conda/activate.d/libglib_activate.sh
. /ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/etc/conda/activate.d/libxml2-split_activate.sh
export CONDA_ENV_SHLVL_1_PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
export RPY2_CFFI_MODE=ABI
source /ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/share/bash-completion/completions/*

# shellcheck shell=bash
pixi() {
    local first_arg="${1-}"

    "${PIXI_EXE-}" "$@" || return $?

    case "${first_arg-}" in
    add | a | remove | rm | install | i)
        eval "$("$PIXI_EXE" shell-hook --change-ps1 false)"
        hash -r
        ;;
    esac || :

    return 0
}

export PS1="(slide-py:dev) ${PS1:-}"
