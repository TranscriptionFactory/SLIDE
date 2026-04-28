use ndarray::{Array1, Array2};
use slide_sdp::{create_solve_asdp, create_solve_equi, create_solve_sdp};
use slide_types::Matrix;
use std::{fs, path::PathBuf};

const ABS_TOL: f64 = 1.0e-6;
const REL_TOL: f64 = 1.0e-6;
const FEASIBILITY_FLOOR: f64 = 1.0e-10;

#[test]
fn equi_matches_r_knockoff_fixture() {
    let fixture = fixture_dir("corr3");
    let sigma = read_matrix(fixture.join("input_sigma.csv"));
    let expected = read_vector(fixture.join("expected_equi.csv"));

    let actual =
        create_solve_equi(&sigma).unwrap_or_else(|error| panic!("equi solve failed: {error}"));

    assert_vector_close(actual.as_ndarray(), &expected);
    assert_shifted_matrix_feasible(&sigma, actual.as_ndarray());
}

#[test]
fn sdp_matches_r_knockoff_fixtures() {
    for name in ["corr3", "block4", "cov4"] {
        let fixture = fixture_dir(name);
        let sigma = read_matrix(fixture.join("input_sigma.csv"));
        let expected = read_vector(fixture.join("expected_sdp.csv"));

        let actual = create_solve_sdp(&sigma)
            .unwrap_or_else(|error| panic!("sdp solve failed for {name}: {error}"));

        assert_vector_close(actual.as_ndarray(), &expected);
        assert_shifted_matrix_feasible(&sigma, actual.as_ndarray());
    }
}

#[test]
fn asdp_matches_r_knockoff_fixtures() {
    for (name, max_block_size) in [("corr3", 3_usize), ("block4", 2)] {
        let fixture = fixture_dir(name);
        let sigma = read_matrix(fixture.join("input_sigma.csv"));
        let expected = read_vector(fixture.join("expected_asdp.csv"));

        let actual = create_solve_asdp(&sigma, max_block_size)
            .unwrap_or_else(|error| panic!("asdp solve failed for {name}: {error}"));

        assert_vector_close(actual.as_ndarray(), &expected);
        assert_shifted_matrix_feasible(&sigma, actual.as_ndarray());
    }
}

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/sdp")
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

fn read_vector(path: PathBuf) -> Array1<f64> {
    let matrix = read_matrix(path);
    let array = matrix.as_ndarray();
    if array.ncols() == 1 {
        Array1::from_vec(array.column(0).to_vec())
    } else if array.nrows() == 1 {
        Array1::from_vec(array.row(0).to_vec())
    } else {
        panic!("expected vector-shaped fixture");
    }
}

fn assert_vector_close(actual: &Array1<f64>, expected: &Array1<f64>) {
    assert_eq!(actual.len(), expected.len());
    for (&actual_value, &expected_value) in actual.iter().zip(expected.iter()) {
        assert_close(actual_value, expected_value);
    }
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = ABS_TOL.max(REL_TOL * expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.12e}, got {actual:.12e}, tolerance {tolerance:.12e}"
    );
}

fn assert_shifted_matrix_feasible(sigma: &Matrix, diag_s: &Array1<f64>) {
    assert!(diag_s.iter().all(|value| *value >= -ABS_TOL));
    let sigma_array = sigma.as_ndarray();
    let dimension = sigma_array.nrows();
    let mut shifted = Array2::zeros((dimension, dimension));
    for row in 0..dimension {
        for column in 0..dimension {
            shifted[(row, column)] = 2.0 * sigma_array[(row, column)];
        }
        shifted[(row, row)] -= diag_s[row];
    }
    let min_eigenvalue = Matrix::from_ndarray(shifted)
        .symmetric_eigendecomposition()
        .unwrap_or_else(|error| panic!("shifted eigendecomposition failed: {error}"))
        .eigenvalues()
        .as_ndarray()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_eigenvalue >= FEASIBILITY_FLOOR - 1.0e-8,
        "shifted matrix min eigenvalue {min_eigenvalue:.12e} violated feasibility floor"
    );
}
