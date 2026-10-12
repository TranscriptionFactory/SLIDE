#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod cv;
mod glasso;
mod hetero_cv;
mod love;
mod nonpure;
mod prescreen;
mod pure_hetero;
mod pure_homo;
mod z_matrix;

pub use cv::{cal_fitted_sigma, cv_delta, cv_lbd, CalFittedSigmaResult, CvError};
pub use glasso::{default_solver_settings, est_omega, solve_row, EstOmegaError};
pub use hetero_cv::{
    kfold_cv_delta, lp_score, score_mat, HeteroCvError, KfoldCvDeltaResult, ScoreMatResult,
    ScoreNorm,
};
pub use love::{love, love_cached, LoveCache, LoveConfig, LoveError, LoveResult, NonPureMethod};
pub use nonpure::{dantzig, est_aj_dant, est_aj_inv, est_y, lp, NonPureError};
pub use prescreen::{screen_x, ScreenXCvResult, ScreenXError, ScreenXOutput};
pub use pure_hetero::{
    est_bi_c, est_k, est_pure, post_est_pure, re_est_pure, EstBiCResult, EstPureResult,
    PureHeteroError,
};
pub use pure_homo::{
    est_ai, est_c, find_sign_pure_node, recover_ai, EstAiResult, MergeStrategy, PureHomoError,
    SignedPureGroup,
};
pub use z_matrix::{compute_z_map, map_estimate_z, ZMatrixError};
