//! Profile actual voting on saved little-endian row-major float64 inputs.
//! Usage: profile_voting Z.raw y.raw n p niter [Rscript]
//! Set SLIDE_KNOCKOFF_PROFILE=1 for per-draw stage events on stderr.

use ndarray::{Array1, Array2};
use slide_knockoff::{
    generate_knockoffs_r, knockoff_filter_voting_slide, prepare_knockoff_cache, KnockoffBackend,
    KnockoffConfig,
};
use slide_types::{Matrix, Vector};
use std::{error::Error, path::Path, time::Instant};

fn read_raw(path: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err("raw input size must be a multiple of eight".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut value = [0; 8];
            value.copy_from_slice(chunk);
            f64::from_le_bytes(value)
        })
        .collect())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(6..=7).contains(&args.len()) {
        return Err("usage: profile_voting Z.raw y.raw n p niter [Rscript]".into());
    }
    let n = args[3].parse()?;
    let p = args[4].parse()?;
    let x = Matrix::from_ndarray(Array2::from_shape_vec((n, p), read_raw(&args[1])?)?);
    let y = Vector::from_ndarray(Array1::from_vec(read_raw(&args[2])?));
    let mut config = KnockoffConfig {
        niter: args[5].parse()?,
        parallel: false,
        ..KnockoffConfig::default()
    };
    let backend = if args.len() == 7 {
        "r_generated"
    } else {
        "native"
    };
    let started = Instant::now();
    if args.len() == 7 {
        let rscript = Path::new(&args[6]);
        let samples = generate_knockoffs_r(
            &x,
            &config.knockoff_method,
            config.niter,
            config.base_seed,
            rscript,
        )?;
        std::hint::black_box(&samples);
        config.knockoff_backend = KnockoffBackend::R {
            rscript: rscript.into(),
        };
    } else {
        let cache = prepare_knockoff_cache(&x, &config.knockoff_method)?;
        std::hint::black_box(&cache);
    }
    println!(
        "{{\"event\":\"standalone_setup\",\"backend\":\"{backend}\",\"seconds\":{}}}",
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let result = knockoff_filter_voting_slide(&x, &y, &config)?;
    println!("{{\"event\":\"voting_total\",\"backend\":\"{backend}\",\"n\":{n},\"p\":{p},\"niter\":{},\"nlambda\":{},\"parallel\":false,\"selected_count\":{},\"seconds\":{}}}", config.niter, config.nlambda, result.selected.len(), started.elapsed().as_secs_f64());
    Ok(())
}
