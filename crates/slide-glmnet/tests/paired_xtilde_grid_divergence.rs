use ndarray::Array1;
use slide_glmnet::{
    stat_glmnet_lambdasmax_prebuilt_diagnostics, GaussianLassoPathConfig, GlmnetFamily,
};
use slide_types::{Matrix, Vector};

#[test]
fn gaussian_grid_matches_r_denominator() {
    let x = Matrix::from_rows(vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![-1.0, 0.0, 1.0, 0.0],
        vec![0.0, -1.0, 0.0, 1.0],
    ])
    .expect("valid fixture");
    let y = Vector::from_ndarray(Array1::from_vec(vec![1.0, -1.0, -1.0, 1.0]));
    let nlambda = 4;
    let diagnostics = stat_glmnet_lambdasmax_prebuilt_diagnostics(
        &x,
        2,
        &y,
        GlmnetFamily::Gaussian,
        &GaussianLassoPathConfig {
            nlambda,
            ..GaussianLassoPathConfig::default()
        },
    )
    .expect("valid knockoff lasso path");
    let rust = diagnostics.path.lambda;
    let lambda_max = rust.as_ndarray()[0];
    let lambda_min = lambda_max / 2000.0;
    let r_compatible: Vec<_> = (0..nlambda)
        .map(|index| lambda_max * (lambda_min / lambda_max).powf(index as f64 / nlambda as f64))
        .collect();

    assert_eq!(rust.len(), nlambda);
    assert_eq!(rust.as_ndarray()[0], lambda_max);
    assert_eq!(rust.as_ndarray().as_slice().unwrap(), r_compatible);
    assert!(rust
        .as_ndarray()
        .windows(2)
        .into_iter()
        .all(|pair| pair[0] > pair[1]));
    assert!(r_compatible[nlambda - 1] > lambda_min);

    let diagnostics = stat_glmnet_lambdasmax_prebuilt_diagnostics(
        &x,
        2,
        &y,
        GlmnetFamily::Gaussian,
        &GaussianLassoPathConfig {
            nlambda: 1,
            ..GaussianLassoPathConfig::default()
        },
    )
    .expect("valid one-value knockoff lasso path");
    let lambda = diagnostics.path.lambda;
    assert_eq!(lambda.as_ndarray().as_slice().unwrap(), &[lambda_max]);
}
