use ndarray::{concatenate, Array1, Array2, Axis};
use slide_glmnet::{
    binomial_lasso_path, cross_validated_lambda, gaussian_lasso_path,
    gaussian_lasso_path_with_left_cache, stat_glmnet_coefdiff, stat_glmnet_lambdadiff,
    stat_glmnet_lambdasmax, GaussianCdWorkspace, GaussianLassoPathConfig, GlmnetFamily,
    HalfPreparedData,
};
use slide_types::{Matrix, Vector};
use std::{fs, path::PathBuf};

const EXACT_ABS_TOL: f64 = 1.0e-10;
const EXACT_REL_TOL: f64 = 1.0e-10;
const PATH_ABS_TOL: f64 = 1.0e-4;
const PATH_REL_TOL: f64 = 3.0e-2;
const GRID_ABS_TOL: f64 = 1.0e-3;
const GRID_REL_TOL: f64 = 1.0e-2;
const CORRELATION_FLOOR: f64 = 0.97;

#[test]
fn gaussian_schedule_matches_saved_r_signs_with_fixed_preparation() {
    let fixture = fixture_dir("gaussian_schedule");
    let x = read_matrix(fixture.join("input_x.csv"));
    let xtilde = read_matrix(fixture.join("input_xtilde.csv"));
    let y = read_vector(fixture.join("input_y.csv"));
    let lambda = read_vector(fixture.join("lambda.csv"));
    let expected = read_vector(fixture.join("expected_sign.csv"));
    let statistic = stat_glmnet_lambdasmax(
        &x, &xtilde, &y, GlmnetFamily::Gaussian,
        &GaussianLassoPathConfig {
            standardize: false,
            nlambda: lambda.len(),
            lambda: Some(lambda),
            tolerance: 1e-7,
            max_iter: 100000,
            ..GaussianLassoPathConfig::default()
        },
    ).unwrap_or_else(|error| panic!("saved scheduling replay failed: {error}"));
    let signs = statistic.as_ndarray().mapv(|v| if v == 0.0 { 0.0 } else { v.signum() });
    assert_eq!(signs, *expected.as_ndarray());
}

#[test]
fn gaussian_covariance_matches_r_entry_for_collinear_saved_draw() {
    let fixture = fixture_dir("gaussian_covariance");
    let x = read_matrix(fixture.join("input_x.csv"));
    let xtilde = read_matrix(fixture.join("input_xtilde.csv"));
    let y = read_vector(fixture.join("input_y.csv"));
    let expected_entry = read_vector(fixture_dir("gaussian_covariance").join("expected_entry.csv"));
    let swap = read_vector(fixture.join("swap.csv"))
        .as_ndarray()
        .iter()
        .map(|value| *value != 0.0)
        .collect::<Vec<_>>();
    let y_mean = y.as_ndarray().mean().unwrap();
    let y_centered = y.as_ndarray().mapv(|value| value - y_mean);
    let x_cache = HalfPreparedData::new(&x, &y_centered, true);
    let xtilde_cache = HalfPreparedData::new(&xtilde, &y_centered, true);
    let config = GaussianLassoPathConfig {
        nlambda: 500,
        standardize: true,
        tolerance: 1e-7,
        max_iter: 100000,
        scale_ddof: 1,
        ..GaussianLassoPathConfig::default()
    };
    let mut workspace = GaussianCdWorkspace::new(x.shape().0, 2 * x.shape().1);
    let result = gaussian_lasso_path_with_left_cache(
        &x_cache,
        &xtilde_cache,
        &swap,
        &y,
        y_mean,
        &y_centered,
        &config,
        &mut workspace,
    )
    .unwrap_or_else(|error| panic!("collinear covariance path failed: {error}"));
    let beta = result.beta.as_ndarray();
    let entry = beta
        .row(5)
        .iter()
        .position(|value| *value != 0.0)
        .expect("R-selected column should enter the path");
    assert_eq!(entry as f64, expected_entry.as_ndarray()[0]);
}

#[test]
fn gaussian_path_matches_r_glmnet_fixture() {
    let fixture = fixture_dir("gaussian_path");
    let x = read_matrix(fixture.join("input_x.csv"));
    let y = read_vector(fixture.join("input_y.csv"));
    let expected_lambda = read_vector(fixture.join("expected_lambda.csv"));
    let expected_beta = read_matrix(fixture.join("expected_beta.csv"));
    let expected_intercept = read_vector(fixture.join("expected_intercept.csv"));

    let result = gaussian_lasso_path(
        &x,
        &y,
        &GaussianLassoPathConfig {
            nlambda: expected_lambda.len(),
            ..GaussianLassoPathConfig::default()
        },
    )
    .unwrap_or_else(|error| panic!("gaussian_lasso_path failed: {error}"));

    assert_vector_close_exact(result.lambda.as_ndarray(), expected_lambda.as_ndarray());
    assert_matrix_close_with_tolerance(
        result.beta.as_ndarray(),
        expected_beta.as_ndarray(),
        PATH_ABS_TOL,
        PATH_REL_TOL,
    );
    assert_vector_close_with_tolerance(
        result.intercept.as_ndarray(),
        expected_intercept.as_ndarray(),
        PATH_ABS_TOL,
        PATH_REL_TOL,
    );
}

#[test]
fn binomial_path_matches_r_glmnet_fixture() {
    let fixture = fixture_dir("binomial_path");
    let x = read_matrix(fixture.join("input_x.csv"));
    let y = read_vector(fixture.join("input_y.csv"));
    let expected_lambda = read_vector(fixture.join("expected_lambda.csv"));
    let expected_beta = read_matrix(fixture.join("expected_beta.csv"));
    let expected_intercept = read_vector(fixture.join("expected_intercept.csv"));

    let result = binomial_lasso_path(
        &x,
        &y,
        &GaussianLassoPathConfig {
            nlambda: expected_lambda.len(),
            ..GaussianLassoPathConfig::default()
        },
    )
    .unwrap_or_else(|error| panic!("binomial_lasso_path failed: {error}"));

    assert_vector_close_exact(result.lambda.as_ndarray(), expected_lambda.as_ndarray());
    assert_matrix_close_with_tolerance(
        result.beta.as_ndarray(),
        expected_beta.as_ndarray(),
        PATH_ABS_TOL,
        PATH_REL_TOL,
    );
    assert_vector_close_with_tolerance(
        result.intercept.as_ndarray(),
        expected_intercept.as_ndarray(),
        PATH_ABS_TOL,
        PATH_REL_TOL,
    );
}

#[test]
fn gaussian_knockoff_statistics_match_r_glmnet_fixture() {
    assert_knockoff_fixture_matches("gaussian_knockoff", GlmnetFamily::Gaussian);
}

#[test]
fn binomial_knockoff_statistics_match_r_glmnet_fixture() {
    assert_knockoff_fixture_matches("binomial_knockoff", GlmnetFamily::Binomial);
}

fn assert_knockoff_fixture_matches(name: &str, family: GlmnetFamily) {
    let fixture = fixture_dir(name);
    let x = read_matrix(fixture.join("input_x.csv"));
    let x_tilde = read_matrix(fixture.join("input_x_tilde.csv"));
    let y = read_vector(fixture.join("input_y.csv"));
    let expected_lambda = read_vector(fixture.join("expected_lambda.csv"));
    let expected_lambdasmax = read_vector(fixture.join("expected_lambdasmax.csv"));
    let expected_lambdadiff = read_vector(fixture.join("expected_lambdadiff.csv"));
    let expected_coefdiff = read_vector(fixture.join("expected_coefdiff.csv"));
    let config = GaussianLassoPathConfig {
        nlambda: expected_lambda.len(),
        cv_folds: 4,
        ..GaussianLassoPathConfig::default()
    };

    let lambdasmax = stat_glmnet_lambdasmax(&x, &x_tilde, &y, family, &config)
        .unwrap_or_else(|error| panic!("stat_glmnet_lambdasmax failed: {error}"));
    let lambdadiff = stat_glmnet_lambdadiff(&x, &x_tilde, &y, family, &config)
        .unwrap_or_else(|error| panic!("stat_glmnet_lambdadiff failed: {error}"));
    let coefdiff = stat_glmnet_coefdiff(&x, &x_tilde, &y, family, &config)
        .unwrap_or_else(|error| panic!("stat_glmnet_coefdiff failed: {error}"));
    let augmented = augment_features(&x, &x_tilde);
    let cv = cross_validated_lambda(&augmented, &y, family, &config)
        .unwrap_or_else(|error| panic!("cross_validated_lambda failed: {error}"));

    assert_correlation_at_least(
        lambdasmax.as_ndarray(),
        expected_lambdasmax.as_ndarray(),
        "lambdasmax",
    );
    assert_correlation_at_least(
        lambdadiff.as_ndarray(),
        expected_lambdadiff.as_ndarray(),
        "lambdadiff",
    );
    assert_correlation_at_least(
        coefdiff.as_ndarray(),
        expected_coefdiff.as_ndarray(),
        "coefdiff",
    );
    assert_vector_close_with_tolerance(
        cv.lambda.as_ndarray(),
        expected_lambda.as_ndarray(),
        GRID_ABS_TOL,
        GRID_REL_TOL,
    );
}

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/glmnet")
        .join(name)
}

fn read_matrix(path: PathBuf) -> Matrix {
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let rows = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(',')
                .map(|value| {
                    value.trim().parse::<f64>().unwrap_or_else(|error| {
                        panic!("failed to parse value in {}: {error}", path.display())
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let data = rows.into_iter().flatten().collect::<Vec<_>>();
    Matrix::from_ndarray(
        Array2::from_shape_vec((nrows, ncols), data)
            .unwrap_or_else(|error| panic!("failed to shape {}: {error}", path.display())),
    )
}

fn read_vector(path: PathBuf) -> Vector {
    let matrix = read_matrix(path);
    let array = matrix.as_ndarray();
    let data = if array.ncols() == 1 {
        array.column(0).to_vec()
    } else if array.nrows() == 1 {
        array.row(0).to_vec()
    } else {
        panic!("expected vector-shaped fixture");
    };
    Vector::from_ndarray(Array1::from_vec(data))
}

fn augment_features(x: &Matrix, x_tilde: &Matrix) -> Matrix {
    Matrix::from_ndarray(
        concatenate(
            Axis(1),
            &[x.as_ndarray().view(), x_tilde.as_ndarray().view()],
        )
        .unwrap_or_else(|error| panic!("failed to concatenate augmented features: {error}")),
    )
}

fn assert_matrix_close_with_tolerance(
    actual: &Array2<f64>,
    expected: &Array2<f64>,
    abs_tol: f64,
    rel_tol: f64,
) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        assert_close(*actual_value, *expected_value, abs_tol, rel_tol);
    }
}

fn assert_vector_close_exact(actual: &Array1<f64>, expected: &Array1<f64>) {
    assert_vector_close_with_tolerance(actual, expected, EXACT_ABS_TOL, EXACT_REL_TOL);
}

fn assert_vector_close_with_tolerance(
    actual: &Array1<f64>,
    expected: &Array1<f64>,
    abs_tol: f64,
    rel_tol: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (&actual_value, &expected_value) in actual.iter().zip(expected.iter()) {
        assert_close(actual_value, expected_value, abs_tol, rel_tol);
    }
}

fn assert_correlation_at_least(actual: &Array1<f64>, expected: &Array1<f64>, label: &str) {
    assert_eq!(actual.len(), expected.len());
    let correlation = pearson_correlation(actual, expected);
    assert!(
        correlation >= CORRELATION_FLOOR,
        "{label} correlation {correlation:.6} fell below {CORRELATION_FLOOR:.2}"
    );
}

fn pearson_correlation(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    let n = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / n;
    let right_mean = right.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut left_ss = 0.0;
    let mut right_ss = 0.0;
    for (&left_value, &right_value) in left.iter().zip(right.iter()) {
        let left_centered = left_value - left_mean;
        let right_centered = right_value - right_mean;
        numerator += left_centered * right_centered;
        left_ss += left_centered * left_centered;
        right_ss += right_centered * right_centered;
    }
    numerator / (left_ss.sqrt() * right_ss.sqrt())
}

fn assert_close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) {
    let scale = expected.abs().max(1.0);
    let diff = (actual - expected).abs();
    assert!(
        diff <= abs_tol.max(rel_tol * scale),
        "expected {expected:.12e}, got {actual:.12e}, diff={diff:.12e}"
    );
}
