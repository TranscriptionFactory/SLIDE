use ndarray::{Array1, Array2};
use slide_glmnet::{gaussian_lasso_path, GaussianLassoPathConfig};
use slide_types::{Matrix, Vector};
use std::fs;

fn load_csv_matrix(path: &str) -> Array2<f64> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    let rows: Vec<Vec<f64>> = content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse::<f64>().unwrap()).collect())
        .collect();
    let n = rows.len();
    let m = rows[0].len();
    let mut arr = Array2::zeros((n, m));
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            arr[(i, j)] = val;
        }
    }
    arr
}

fn load_csv_vector(path: &str) -> Array1<f64> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    Array1::from_iter(
        content
            .lines()
            .filter(|l| !l.is_empty())
            .map(|v| v.trim().parse::<f64>().unwrap()),
    )
}

fn entry_lambda(beta: &ndarray::Array2<f64>, lambda: &ndarray::Array1<f64>, threshold: f64) -> ndarray::Array1<f64> {
    ndarray::Array1::from_iter(beta.rows().into_iter().map(|row| {
        row.iter()
            .zip(lambda.iter())
            .find_map(|(&c, &l)| (c.abs() > threshold).then_some(l))
            .unwrap_or(0.0)
    }))
}

fn count_ties(entries: &ndarray::Array1<f64>, p: usize) -> usize {
    (0..p).filter(|&j| entries[j] == entries[p + j] && entries[j] != 0.0).count()
}

#[test]
fn compare_tolerances() {
    let x_arr = load_csv_matrix("/tmp/augmented_chunk0.csv");
    let y_arr = load_csv_vector("/tmp/y_vector.csv");
    let (_n, two_p) = (x_arr.nrows(), x_arr.ncols());
    let p = two_p / 2;

    let augmented = Matrix::from_ndarray(x_arr);
    let y = Vector::from_ndarray(y_arr);

    for (tol, max_iter, early_stop, label) in [
        (1e-4, 1000, true, "Rust default (1e-4, 1K, early_stop)"),
        (1e-4, 1000, false, "Rust no-early-stop (1e-4, 1K)"),
        (1e-7, 10000, true, "Tight (1e-7, 10K, early_stop)"),
        (1e-7, 10000, false, "Tight no-early-stop (1e-7, 10K)"),
    ] {
        let config = GaussianLassoPathConfig {
            nlambda: 250,
            tolerance: tol,
            max_iter,
            early_stop_all_entered: early_stop,
            scale_ddof: 1,
            ..GaussianLassoPathConfig::default()
        };
        let path = gaussian_lasso_path(&augmented, &y, &config).expect("failed");
        let beta = path.beta.as_ndarray();
        let lambda = path.lambda.as_ndarray();

        let entries = entry_lambda(beta, lambda, 1e-10);
        let ties = count_ties(&entries, p);
        let nonzero_w: usize = (0..p)
            .filter(|&j| {
                let orig = entries[j];
                let knock = entries[p + j];
                let w = orig.max(knock) * (orig - knock).signum();
                w.abs() > 1e-15
            })
            .count();

        println!("{}: path_len={}, ties={}/{}, nonzero_W={}/{}", 
            label, lambda.len(), ties, p, nonzero_w, p);
    }
}
