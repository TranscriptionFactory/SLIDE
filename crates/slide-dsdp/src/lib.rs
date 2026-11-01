//! Narrow safe interface to the repository's vendored DSDP C implementation.
use std::sync::Mutex;

pub const MAX_DIMENSION: usize = 4096;

// DSDP has process-global logging/event state, even for distinct solver instances.
static SOLVER_LOCK: Mutex<()> = Mutex::new(());

extern "C" {
    fn slide_dsdp_diagonal(
        n: i32,
        covariance: *mut f64,
        maxit: i32,
        gap: f64,
        feas: f64,
        solution: *mut f64,
        stats: *mut f64,
        status: *mut i32,
    ) -> i32;
}

pub struct Solution {
    pub x: Vec<f64>,
    pub objectives: [f64; 2],
    pub errors: [f64; 6],
    pub status: [i32; 3],
}

/// Solve max sum(s), 0 <= s <= 1, packed_covariance - diag(s) >= 0.
/// The input is the packed lower triangle of twice the correlation matrix.
pub fn solve(
    n: usize,
    mut packed_covariance: Vec<f64>,
    maxit: u32,
    gap: f64,
    feas: f64,
) -> Result<Solution, String> {
    if n == 0
        || n > MAX_DIMENSION
        || packed_covariance.len() != n * (n + 1) / 2
        || packed_covariance.iter().any(|v| !v.is_finite())
        || maxit == 0
        || maxit > i32::MAX as u32
        || !gap.is_finite()
        || gap <= 0.0
        || !feas.is_finite()
        || feas <= 0.0
    {
        return Err("invalid DSDP dimensions, finite matrix, iteration limit or tolerances".into());
    }
    let _lock = SOLVER_LOCK.lock().map_err(|_| "DSDP lock poisoned")?;
    let mut x = vec![0.0; n];
    let mut stats = [0.0; 8];
    let mut status = [0; 3];
    // SAFETY: dimensions and lengths are validated above. The shim retains no
    // pointers after returning, writes n/8/3 entries, and is serialized by the lock.
    let rc = unsafe {
        slide_dsdp_diagonal(
            n as i32,
            packed_covariance.as_mut_ptr(),
            maxit as i32,
            gap,
            feas,
            x.as_mut_ptr(),
            stats.as_mut_ptr(),
            status.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("DSDP native error {rc}"));
    }
    Ok(Solution {
        x,
        objectives: [stats[0], stats[1]],
        errors: [stats[2], stats[3], stats[4], stats[5], stats[6], stats[7]],
        status,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_invalid_ffi_inputs_before_allocation() {
        for (n, values) in [
            (0, vec![]),
            (4097, vec![]),
            (2, vec![1.0]),
            (1, vec![f64::NAN]),
        ] {
            assert!(super::solve(n, values, 200, 1e-6, 1e-6).is_err());
        }
    }
}
