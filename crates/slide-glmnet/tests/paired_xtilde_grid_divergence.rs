use ndarray::Array1;
use slide_glmnet::build_lambda_grid;
use slide_types::{Matrix, Vector};

#[test]
fn rust_grid_reproduces_paired_xtilde_r_denominator_divergence() {
    let x = Matrix::from_rows(vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![-1.0, 0.0, 1.0, 0.0],
        vec![0.0, -1.0, 0.0, 1.0],
    ])
    .expect("valid fixture");
    let y = Vector::from_ndarray(Array1::from_vec(vec![1.0, -1.0, -1.0, 1.0]));
    let nlambda = 4;
    let rust = build_lambda_grid(&x, &y, nlambda, 1.0).expect("valid lambda grid");
    let lambda_max = rust.as_ndarray()[0];
    let lambda_min = lambda_max / 2000.0;
    let r_compatible: Vec<_> = (0..nlambda)
        .map(|index| lambda_max * (lambda_min / lambda_max).powf(index as f64 / nlambda as f64))
        .collect();

    assert_eq!(rust.as_ndarray()[0], r_compatible[0]);
    assert_ne!(rust.as_ndarray()[1], r_compatible[1]);
    assert_eq!(rust.as_ndarray()[nlambda - 1], lambda_min);
    assert!(r_compatible[nlambda - 1] > lambda_min);
}
