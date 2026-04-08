#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod glasso;

pub use glasso::{default_solver_settings, est_omega, solve_row, EstOmegaError};
