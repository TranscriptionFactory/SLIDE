use ndarray::Array1;
use serde::Deserialize;
use slide_core::{
    optimize_slide, MarginalLfSelectionConfig, OptimizeResponseType, OptimizeSlideConfig,
};
use slide_data_io::{
    read_feature_matrix, read_response_vector, MatrixReadOptions, ResponseReadOptions, ResponseType,
};
use slide_glmnet::GlmnetFamily;
use slide_knockoff::{KnockoffBackend, KnockoffMethod, KnockoffStatistic};
use slide_love::{LoveConfig, MergeStrategy, NonPureMethod, ScoreNorm};
use slide_types::{Scalar, Vector};

#[derive(Debug, Deserialize)]
struct FixtureConfig {
    delta: Scalar,
    lambda: Scalar,
    thresh_fdr: Scalar,
    fdr: Scalar,
    spec: Scalar,
    niter: usize,
    do_interacts: bool,
    knockoff_method: String,
    knockoff_offset: usize,
    response_type: String,
}

fn fixture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/slide_core/hiv_delta_0_10_lambda_0_50")
}

fn load_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> T {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn jaccard(left: &[usize], right: &[usize]) -> Scalar {
    let left = left
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let right = right
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let intersection = left.intersection(&right).count();
    let union = left.union(&right).count();
    if union == 0 {
        1.0
    } else {
        intersection as Scalar / union as Scalar
    }
}

#[test]
fn optimize_slide_matches_committed_r_hiv_fixture() {
    let fixture_dir = fixture_dir();
    let config_path = fixture_dir.join("config.json");
    let fixture_config: FixtureConfig = load_json(&config_path);
    let expected_k: usize = load_json(&fixture_dir.join("expected_k.json"));
    let expected_sig_lfs: Vec<usize> = load_json(&fixture_dir.join("expected_sig_lfs.json"));

    let x = read_feature_matrix(
        fixture_dir.join("input_x.csv"),
        &MatrixReadOptions::default(),
    )
    .unwrap_or_else(|error| panic!("failed to load HIV X fixture: {error}"))
    .matrix;
    let y = read_response_vector(
        fixture_dir.join("input_y.csv"),
        &ResponseReadOptions {
            delimiter: None,
            response_type: Some(ResponseType::Binary),
        },
    )
    .unwrap_or_else(|error| panic!("failed to load HIV y fixture: {error}"));

    let knockoff_method = match fixture_config.knockoff_method.as_str() {
        "equi" => KnockoffMethod::Equi,
        other => panic!("unsupported fixture knockoff_method: {other}"),
    };
    let response_type = match fixture_config.response_type.as_str() {
        "binary" => OptimizeResponseType::Binary,
        other => panic!("unsupported fixture response_type: {other}"),
    };

    let config = OptimizeSlideConfig {
        do_interacts: fixture_config.do_interacts,
        response_type,
        love: LoveConfig {
            lbd: Vector::from_ndarray(Array1::from_vec(vec![fixture_config.lambda])),
            mu: 0.5,
            est_non_pure_row: NonPureMethod::Dantzig,
            pure_homo: true,
            diagonal: false,
            delta: Some(Vector::from_ndarray(Array1::from_vec(vec![
                fixture_config.delta,
            ]))),
            merge_strategy: MergeStrategy::Intersection,
            rep_cv: 5,
            ndelta: 5,
            q: ScoreNorm::L2,
            exact: false,
            max_pure: None,
            nfolds: 10,
            thresh_fdr: Some(fixture_config.thresh_fdr),
        },
        knockoff: MarginalLfSelectionConfig {
            fdr: fixture_config.fdr,
            spec: fixture_config.spec,
            niter: fixture_config.niter,
            f_size: None,
            knockoff_method,
            knockoff_backend: KnockoffBackend::default(),
            knockoff_offset: fixture_config.knockoff_offset,
            statistic: KnockoffStatistic::GlmnetLambdasMax,
            nlambda: 250,
            base_seed: 42,
            slide_selection: false,
            family: GlmnetFamily::Binomial,
        },
        final_cv: None,
    };

    let result = optimize_slide(
        &x,
        &y,
        &[fixture_config.delta],
        &[fixture_config.lambda],
        &config,
    )
    .unwrap_or_else(|error| panic!("optimize_slide failed on HIV fixture: {error}"));

    let actual_sig_lfs = &result.best.marginal.sig_lfs;
    let marginal_jaccard = jaccard(actual_sig_lfs, &expected_sig_lfs);

    assert_eq!(result.evaluations.len(), 1);
    assert_eq!(result.best.love.k, expected_k);
    assert_eq!(result.best.z.nrows(), x.nrows());
    assert_eq!(result.best.z.ncols(), expected_k);
    assert!(result.best.score.is_finite(), "score should be finite");
    assert_eq!(
        result.best.marginal.voting.stage_one_union,
        Some(actual_sig_lfs.clone()),
        "n > K should skip stage-one chunking"
    );
    assert!(
        marginal_jaccard >= 0.5,
        "Rust vs R marginal LF Jaccard too low: {marginal_jaccard:.3} (actual={actual_sig_lfs:?}, expected={expected_sig_lfs:?})"
    );
}
