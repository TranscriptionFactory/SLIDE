use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2};
use serde::Serialize;
use sha2::{Digest, Sha256};
use slide_glmnet::{
    gaussian_lasso_path_with_left_cache, stat_glmnet_lambdasmax_prebuilt_diagnostics,
    GaussianCdWorkspace, GaussianLassoPathConfig, GlmnetFamily, HalfPreparedData,
};
use slide_types::{Matrix, Vector};
use std::{env, fs, path::Path};

#[derive(Serialize)]
struct Output {
    backend: &'static str,
    preparation: &'static str,
    entry_tie_count: usize,
    tolerance: f64,
    max_iter: usize,
    a_sha256: String,
    lf_names_sha256: String,
    xtilde_sha256: String,
    swap_sha256: String,
    z_sha256: String,
    y_sha256: String,
    shape: [usize; 2],
    standardized_design: StandardizedDesign,
    lambda: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    entry_original: Vec<f64>,
    entry_knockoff: Vec<f64>,
    w: Vec<f64>,
    threshold: Option<f64>,
    selected: Vec<usize>,
    total_cd_iters: usize,
    wall_ms: f64,
}

#[derive(Serialize)]
struct StandardizedDesign {
    column_mean: Vec<f64>,
    column_sd_ddof1: Vec<f64>,
    response_mean: f64,
    response_sd_ddof1: f64,
}

fn read_f64(path: &Path, expected: usize) -> Result<(Vec<f64>, String)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() != expected * 8 {
        bail!(
            "{} has {} bytes; expected {}",
            path.display(),
            bytes.len(),
            expected * 8
        );
    }
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let values = bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
        .collect();
    Ok((values, hash))
}

fn threshold(w: &[f64], fdr: f64) -> Option<f64> {
    let mut candidates: Vec<_> = w
        .iter()
        .map(|value| value.abs())
        .filter(|value| *value > 0.0)
        .collect();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    candidates.into_iter().find(|candidate| {
        let negative = w.iter().filter(|value| **value <= -*candidate).count();
        let positive = w.iter().filter(|value| **value >= *candidate).count();
        positive > 0 && negative as f64 / positive as f64 <= fdr
    })
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 16 {
        bail!(
            "usage: paired_xtilde <z.raw> <xtilde.raw> <y.raw> <a.raw> <swap.raw> <n> <p> <a_n> <a_p> <lf_names> <fdr> <nlambda> <tolerance> <max_iter> <output.json>"
        );
    }
    let n: usize = args[6].parse()?;
    let p: usize = args[7].parse()?;
    let a_n: usize = args[8].parse()?;
    let a_p: usize = args[9].parse()?;
    let fdr: f64 = args[11].parse()?;
    let nlambda: usize = args[12].parse()?;
    let tolerance: f64 = args[13].parse()?;
    let max_iter: usize = args[14].parse()?;
    env::set_var("SLIDE_KNOCKOFF_TOL", tolerance.to_string());
    env::set_var("SLIDE_KNOCKOFF_MAX_ITER", max_iter.to_string());
    let cached = env::var("SLIDE_PAIRED_CACHED").is_ok_and(|value| value == "1");
    let (z_values, z_sha256) = read_f64(Path::new(&args[1]), n * p)?;
    let (xtilde_values, xtilde_sha256) = read_f64(Path::new(&args[2]), n * p)?;
    let (y_values, y_sha256) = read_f64(Path::new(&args[3]), n)?;
    let (_, a_sha256) = read_f64(Path::new(&args[4]), a_n * a_p)?;
    if a_p != p {
        bail!("A has {a_p} columns; expected {p}");
    }
    let lf_names_bytes = fs::read(&args[10])?;
    let lf_names_sha256 = format!("{:x}", Sha256::digest(&lf_names_bytes));
    let lf_names = String::from_utf8(lf_names_bytes)?;
    if lf_names.lines().count() != p {
        bail!("LF name file has wrong number of names");
    }
    let swap_bytes = fs::read(&args[5])?;
    let swap_sha256 = format!("{:x}", Sha256::digest(&swap_bytes));
    if swap_bytes.len() != p {
        bail!("swap mask has {} bytes; expected {p}", swap_bytes.len());
    }

    let z = Array2::from_shape_vec((n, p), z_values)?;
    let xtilde = Array2::from_shape_vec((n, p), xtilde_values)?;
    let mut augmented = Array2::zeros((n, 2 * p));
    for j in 0..p {
        for i in 0..n {
            if swap_bytes[j] == 0 {
                augmented[(i, j)] = z[(i, j)];
                augmented[(i, p + j)] = xtilde[(i, j)];
            } else {
                augmented[(i, j)] = xtilde[(i, j)];
                augmented[(i, p + j)] = z[(i, j)];
            }
        }
    }
    let raw_column_mean: Vec<_> = (0..2 * p)
        .map(|j| (0..n).map(|i| augmented[(i, j)]).sum::<f64>() / n as f64)
        .collect();
    let raw_column_sd_ddof1: Vec<_> = (0..2 * p)
        .map(|j| {
            ((0..n)
                .map(|i| (augmented[(i, j)] - raw_column_mean[j]).powi(2))
                .sum::<f64>()
                / (n - 1) as f64)
                .sqrt()
        })
        .collect();
    let column_mean: Vec<_> = (0..2 * p)
        .map(|j| {
            (0..n)
                .map(|i| (augmented[(i, j)] - raw_column_mean[j]) / raw_column_sd_ddof1[j])
                .sum::<f64>()
                / n as f64
        })
        .collect();
    let column_sd_ddof1: Vec<_> = (0..2 * p)
        .map(|j| {
            let standardized_mean = column_mean[j];
            ((0..n)
                .map(|i| {
                    let value = (augmented[(i, j)] - raw_column_mean[j]) / raw_column_sd_ddof1[j];
                    (value - standardized_mean).powi(2)
                })
                .sum::<f64>()
                / (n - 1) as f64)
                .sqrt()
        })
        .collect();
    let response_mean = y_values.iter().sum::<f64>() / n as f64;
    let response_sd_ddof1 = (y_values
        .iter()
        .map(|value| (value - response_mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64)
        .sqrt();
    let config = GaussianLassoPathConfig {
        nlambda,
        tolerance,
        max_iter,
        standardize: true,
        scale_ddof: 1,
        ..GaussianLassoPathConfig::default()
    };
    let started = std::time::Instant::now();
    let y = Vector::from_ndarray(Array1::from_vec(y_values));
    let path = if cached {
        let y_centered = y.as_ndarray().mapv(|value| value - response_mean);
        let x_cache = HalfPreparedData::new(&Matrix::from_ndarray(z), &y_centered, true);
        let tilde_cache = HalfPreparedData::new(&Matrix::from_ndarray(xtilde), &y_centered, true);
        let swap: Vec<_> = swap_bytes.iter().map(|value| *value != 0).collect();
        let mut workspace = GaussianCdWorkspace::new(n, 2 * p);
        gaussian_lasso_path_with_left_cache(
            &x_cache,
            &tilde_cache,
            &swap,
            &y,
            response_mean,
            &y_centered,
            &config,
            &mut workspace,
        )?
    } else {
        stat_glmnet_lambdasmax_prebuilt_diagnostics(
            &Matrix::from_ndarray(augmented),
            p,
            &y,
            GlmnetFamily::Gaussian,
            &config,
        )?
        .path
    };
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let entries: Vec<_> = path
        .beta
        .as_ndarray()
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(path.lambda.as_ndarray())
                .find_map(|(&coefficient, &lambda)| (coefficient != 0.0).then_some(lambda))
                .unwrap_or(0.0)
        })
        .collect();
    let mut entry_original = vec![0.0; p];
    let mut entry_knockoff = vec![0.0; p];
    let mut w: Vec<_> = (0..p)
        .map(|j| entries[j].max(entries[p + j]) * (entries[j] - entries[p + j]).signum())
        .collect();
    for j in 0..p {
        if swap_bytes[j] == 0 {
            entry_original[j] = entries[j];
            entry_knockoff[j] = entries[p + j];
        } else {
            entry_original[j] = entries[p + j];
            entry_knockoff[j] = entries[j];
            w[j] = -w[j];
        }
    }
    let cutoff = threshold(&w, fdr);
    let selected = cutoff.map_or_else(Vec::new, |value| {
        w.iter()
            .enumerate()
            .filter_map(|(index, statistic)| (*statistic >= value).then_some(index))
            .collect()
    });
    let coefficients = path
        .beta
        .as_ndarray()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    let output = Output {
        backend: "rust",
        entry_tie_count: (0..p)
            .filter(|&j| entries[j] > 0.0 && entries[j] == entries[p + j])
            .count(),
        preparation: if cached { "cached" } else { "uncached" },
        tolerance,
        max_iter,
        a_sha256,
        lf_names_sha256,
        xtilde_sha256,
        swap_sha256,
        z_sha256,
        y_sha256,
        shape: [n, p],
        standardized_design: StandardizedDesign {
            column_mean,
            column_sd_ddof1,
            response_mean,
            response_sd_ddof1,
        },
        lambda: path.lambda.as_ndarray().to_vec(),
        coefficients,
        entry_original,
        entry_knockoff,
        w,
        threshold: cutoff,
        selected,
        total_cd_iters: path.total_cd_iters,
        wall_ms,
    };
    fs::write(&args[15], serde_json::to_vec(&output)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_signed_max_reports_positive_tied_entries() {
        let x = Matrix::from_ndarray(ndarray::array![
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ]);
        let y = Vector::from_ndarray(ndarray::array![2.0, 0.0, 0.0, -2.0]);
        let diagnostics = stat_glmnet_lambdasmax_prebuilt_diagnostics(
            &x,
            1,
            &y,
            GlmnetFamily::Gaussian,
            &GaussianLassoPathConfig {
                scale_ddof: 1,
                ..GaussianLassoPathConfig::default()
            },
        )
        .unwrap();
        let entries = diagnostics.entry_lambda.as_ndarray();
        assert!(entries[0] > 0.0);
        assert_eq!(entries[0], entries[1]);
        // Preserve production behavior in replay; mathematical sign(0) would give W=0.
        assert_eq!(diagnostics.statistic.as_ndarray()[0], entries[0]);
    }
}
