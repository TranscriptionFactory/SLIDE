use ndarray::Array2;
use slide_sdp::create_solve_sdp;
use slide_types::Matrix;
use std::fs;

#[test]
#[ignore = "requires externally generated CSVs: set SLIDE_COMPARE_DIR to a directory with sigma_chunk0.csv"]
fn compare_diag_s_with_python() {
    let dir = std::env::var("SLIDE_COMPARE_DIR")
        .expect("SLIDE_COMPARE_DIR must point to a directory holding sigma_chunk0.csv");
    let path = std::path::Path::new(&dir).join("sigma_chunk0.csv");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let rows: Vec<Vec<f64>> = content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.split(',')
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    let n = rows.len();
    let mut arr = Array2::<f64>::zeros((n, n));
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            arr[(i, j)] = val;
        }
    }

    let sigma = Matrix::from_ndarray(arr);
    let diag_s = create_solve_sdp(&sigma).expect("SDP solver failed");
    let values = diag_s.as_ndarray();

    println!("\nRust SDP diag_s for chunk 0 (p={}):", n);
    for (i, &v) in values.iter().enumerate() {
        println!("  [{:2}] = {:.10}", i, v);
    }
    println!("Sum: {:.6}", values.sum());
    println!(
        "Nonzero (>1e-6): {}/{}",
        values.iter().filter(|&&v| v > 1e-6).count(),
        n
    );
}
