# Rust vs Python knockoff cache internal comparison

Checkpoint: `benchmark_results/20260525_191227_rust_rerun/rust/0.1_0.1_out/checkpoint.json`
Screened Z shape: `24 x 14`
Stage-one union: `[0, 5, 10, 25, 33, 38, 39, 44, 53, 55, 61, 62, 66, 69]`

| quantity | max abs diff | relative diff | correlation |
|---|---:|---:|---:|
| diag_s | 0.860193 | 1 | -0.175737 |
| mu_k | 2.08209 | 0.827067 | 0.722005 |
| factor_l_trans_vs_python_lower_T | 0.79415 | 0.923489 | 0.569765 |
| shared_noise_xk | 3.01115 | 1.19607 | 0.451494 |
| empirical_cov_shared_xk | 0.605546 | 0.491363 | 0.765307 |
| rust_sampler_seed_42_vs_python_shared_noise_xk | 3.54768 | 1.40919 | 0.270541 |

## Conclusion

The forced shared-noise Xk matrices do **not** match. Because the same `Z_noise` is used with `Xk = mu_k + Z_noise @ factor`, this is not an RNG-only/noise-stream divergence. The large `diag_s`, `mu_k`, factor, and empirical-covariance differences indicate upstream cache/math divergence, dominated by Rust and Python constructing the knockoff cache from different covariance estimates for the same screened matrix.

Artifacts: `screened_z.csv`, `shared_noise.csv`, `rust_cache.json`, `python_cache_and_shared_noise.npz`, `summary.json`.
