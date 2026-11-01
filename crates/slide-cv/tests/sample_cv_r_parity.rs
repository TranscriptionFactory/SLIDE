use ndarray::Array1;
use serde_json::Value;
use slide_cv::{run_sample_cv_with_splits, SampleCvCondition, SampleCvConfig, SampleCvStatus};
use slide_types::{Matrix, Vector};

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/sample_cv/2026-11-01_r-reference/sample_cv.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("fixture array").iter()
        .map(|item| item.as_f64().expect("fixture number")).collect()
}

#[test]
fn sample_cv_matches_r_reference_predictions_and_metrics() {
    let data = fixture();
    for case in data["cases"].as_array().expect("cases") {
        let z = Matrix::from_rows(case["z"].as_array().expect("z rows").iter().map(numbers).collect())
            .expect("rectangular z");
        let y = Vector::from_ndarray(Array1::from_vec(numbers(&case["y"])));
        let marginals = case["marginal_indices"].as_array().expect("marginals")
            .iter().map(|value| value.as_u64().expect("marginal index") as usize).collect::<Vec<_>>();
        let interactions = case["interaction_pairs"].as_array().expect("pairs").iter().map(|pair| {
            let pair = pair.as_array().expect("pair");
            (pair[0].as_u64().expect("left") as usize, pair[1].as_u64().expect("right") as usize)
        }).collect::<Vec<_>>();
        let train_splits = if case["splits"].as_array().expect("splits").is_empty() {
            vec![vec![0, 1, 2]]
        } else {
            case["splits"].as_array().expect("splits").iter().map(|split| {
                split["train_indices"].as_array().expect("train indices").iter()
                    .map(|value| value.as_u64().expect("train index") as usize).collect()
            }).collect()
        };
        let condition = if case["condition"] == "auc" { SampleCvCondition::Auc } else { SampleCvCondition::Corr };
        let config = SampleCvConfig {
            condition: Some(condition), k: 4, iterations: train_splits.len(), seed: 42,
        };
        let result = run_sample_cv_with_splits(&z, &y, &marginals, &interactions, &config, &train_splits)
            .unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
        let expected = &case["expected"];
        let expected_guard = expected["guard"].as_str().expect("guard");
        match expected_guard {
            "model_too_large" => assert_eq!(result.status, SampleCvStatus::ModelTooLarge, "{}", case["name"]),
            "ok" if expected["metric"].is_null() => assert_eq!(result.status, SampleCvStatus::MissingMetric, "{}", case["name"]),
            "ok" => assert_eq!(result.status, SampleCvStatus::Ok, "{}", case["name"]),
            other => panic!("unexpected fixture guard {other}"),
        }
        match expected["metric"].as_f64() {
            Some(metric) => assert!((result.metric.expect("R metric") - metric).abs() < 1e-9, "{}", case["name"]),
            None => assert!(result.metric.is_none(), "{}", case["name"]),
        }
        let predictions = expected["predictions"].as_array().expect("expected predictions");
        assert_eq!(result.predictions.len(), predictions.len(), "{}", case["name"]);
        for (actual, expected) in result.predictions.iter().zip(predictions) {
            assert_eq!(actual.iteration, expected["iteration"].as_u64().expect("iteration") as usize);
            assert_eq!(actual.row_index, expected["row_index"].as_u64().expect("row") as usize);
            assert_eq!(actual.y_true, expected["y_true"].as_f64().expect("truth"));
            assert!((actual.y_pred - expected["y_pred"].as_f64().expect("prediction")).abs() < 1e-9,
                    "{} row {}", case["name"], actual.row_index);
        }
        if expected_guard == "ok" {
            let model = result.last_model.as_ref().expect("model for unguarded fixture");
            assert_eq!(model.rank, expected["last_rank"].as_u64().expect("rank") as usize, "{}", case["name"]);
            let expected_terms = expected["term_order"].as_array().expect("term order").iter()
                .map(|term| term.as_str().expect("term name")).collect::<Vec<_>>();
            assert_eq!(model.terms, expected_terms, "{}", case["name"]);
            assert_eq!(model.train_indices, expected["last_train_indices"].as_array().expect("train").iter()
                .map(|value| value.as_u64().expect("train row") as usize).collect::<Vec<_>>(), "{}", case["name"]);
            let coefficient_map = expected["last_coefficients"].as_object().expect("coefficients");
            let coefficients = expected["r_term_order"].as_array().expect("R term order").iter()
                .map(|term| coefficient_map.get(term.as_str().expect("term name")).expect("term coefficient"))
                .collect::<Vec<_>>();
            assert_eq!(model.coefficients.len(), coefficients.len(), "{}", case["name"]);
            for (actual, expected) in model.coefficients.iter().zip(coefficients) {
                match expected.as_f64() {
                    Some(value) => assert!((actual.expect("coefficient") - value).abs() < 1e-9, "{}", case["name"]),
                    None => assert!(actual.is_none(), "{}", case["name"]),
                }
            }
        } else {
            assert!(result.last_model.is_none(), "{}", case["name"]);
        }
    }
}

#[test]
fn sample_cv_empty_selection_has_no_screening_metric() {
    let z = Matrix::from_rows((0..8).map(|row| vec![row as f64]).collect()).expect("matrix");
    let y = Vector::from_ndarray(Array1::from_vec((0..8).map(|row| row as f64).collect()));
    let config = SampleCvConfig { condition: Some(SampleCvCondition::Corr), k: 4, iterations: 1, seed: 42 };
    let result = slide_cv::run_sample_cv(&z, &y, &[], &[], &config).expect("empty selection status");
    assert_eq!(result.status, SampleCvStatus::NoMarginals);
    assert!(result.metric.is_none());
    assert!(result.predictions.is_empty());
}
