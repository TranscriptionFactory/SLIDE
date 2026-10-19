# Repository instructions

## Git LFS

If `git-lfs` is unavailable, activate its Conda environment in the same shell
before running Git operations that invoke LFS hooks:

```bash
module load anaconda3/2025.7.0-2-python_3.11
eval "$(conda shell.bash hook)"
conda activate git-lfs
git lfs version
```

Continue only after `git lfs version` succeeds. Keep the existing LFS hooks.
