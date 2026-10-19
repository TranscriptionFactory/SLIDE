# Repository instructions

## Git LFS

If `git-lfs` is unavailable, activate its Conda environment in the same shell
before running Git operations that invoke LFS hooks:

```bash
source $HOME/.bashrc && conda activate git-lfs
git lfs version
```

Continue only after `git lfs version` succeeds. Keep the existing LFS hooks.
