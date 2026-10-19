use std::path::Path;

use anyhow::{ensure, Result};
use ndarray::Array1;
use slide_core::{OptimizeSlideConfig, OptimizeSlideResult};
use slide_data_io::{write_params_json, LoadedMatrix, OutputWriteOptions};
use slide_score::{auc_score, control_performance, pearson_correlation, ResponseType};
use slide_types::{Matrix, Vector};

#[derive(Clone, Debug)]
pub(crate) struct Feature {
    pub name: String,
    pub loading: f64,
    pub auc: Option<f64>,
    pub corr: Option<f64>,
}
impl Feature {
    pub fn color(&self) -> &'static str {
        match self.corr {
            Some(c) if c > 0.0 => "red",
            Some(c) if c < 0.0 => "blue",
            _ => "white",
        }
    }
}

fn ranks(values: &Array1<f64>) -> Array1<f64> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut result = Array1::zeros(values.len());
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && values[order[i]] == values[order[j]] {
            j += 1;
        }
        for &index in &order[i..j] {
            result[index] = (i + j + 1) as f64 / 2.0;
        }
        i = j;
    }
    result
}

pub(crate) fn top_features(
    x: &Matrix,
    feature_names: &[String],
    y: &Vector,
    a: &Matrix,
    lf: usize,
    binary: bool,
) -> Result<Vec<Feature>> {
    ensure!(
        x.nrows() == y.len()
            && x.ncols() == a.nrows()
            && feature_names.len() == x.ncols()
            && lf < a.ncols(),
        "feature plot dimensions do not match"
    );
    let yr = ranks(y.as_ndarray());
    let high = y
        .as_ndarray()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let low = y.as_ndarray().iter().copied().fold(f64::INFINITY, f64::min);
    ensure!(
        !binary
            || (low.is_finite()
                && high.is_finite()
                && low < high
                && y.as_ndarray().iter().all(|&v| v == low || v == high)),
        "binary feature plot requires exactly two finite response levels"
    );
    let binary_y = y.as_ndarray().mapv(|v| if v == high { 1.0 } else { 0.0 });
    let mut features = Vec::new();
    for (i, name) in feature_names.iter().enumerate() {
        let loading = a.as_ndarray()[[i, lf]].abs();
        if loading == 0.0 {
            continue;
        }
        let x = x.as_ndarray().column(i).to_owned();
        features.push(Feature {
            name: name.clone(),
            loading,
            auc: if binary {
                Some(auc_score(&binary_y, &x)?)
            } else {
                None
            },
            corr: pearson_correlation(&ranks(&x), &yr).ok(),
        });
    }
    let mut loading: Vec<_> = (0..features.len()).collect();
    loading.sort_by(|&a, &b| features[b].loading.total_cmp(&features[a].loading));
    let mut association = loading.clone();
    loading.truncate(10);
    let score = |i: usize| {
        if binary {
            features[i].auc
        } else {
            features[i].corr.map(f64::abs)
        }
        .unwrap_or(f64::NEG_INFINITY)
    };
    association.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
    association.truncate(10);
    for i in association {
        if !loading.contains(&i) {
            loading.push(i);
        }
    }
    Ok(loading
        .into_iter()
        .map(|i| Feature {
            name: features[i].name.clone(),
            loading: features[i].loading,
            auc: features[i].auc,
            corr: features[i].corr,
        })
        .collect())
}

// Existing matrix writers omit row labels; public artifacts must retain input identities.
fn labelled_matrix(path: &Path, matrix: &Matrix, labels: &[String]) -> Result<()> {
    ensure!(
        matrix.nrows() == labels.len(),
        "row labels do not match matrix at {}",
        path.display()
    );
    ensure!(
        matrix.as_ndarray().iter().all(|v| v.is_finite()),
        "non-finite exported matrix at {}",
        path.display()
    );
    let mut writer = csv::Writer::from_path(path)?;
    let mut header = vec![String::new()];
    header.extend((0..matrix.ncols()).map(|i| format!("Z{i}")));
    writer.write_record(header)?;
    for (label, row) in labels.iter().zip(matrix.as_ndarray().rows()) {
        let mut record = vec![label.clone()];
        record.extend(row.iter().map(ToString::to_string));
        writer.write_record(record)?;
    }
    writer.flush()?;
    Ok(())
}

pub(crate) fn export_results(
    path: &Path,
    loaded: &LoadedMatrix,
    y: &Vector,
    results: &OptimizeSlideResult,
    config: &OptimizeSlideConfig,
) -> Result<()> {
    ensure!(
        !path.join("summary_table.csv").exists(),
        "results already exist at {}; choose a new out_path",
        path.display()
    );
    for result in results.evaluations.iter().flatten() {
        let out = path.join(format!("{}_{}_out", result.delta, result.lambda));
        ensure!(!out.exists(), "result directory already exists at {}; choose a new out_path to avoid mixing stale artifacts", out.display());
    }
    std::fs::create_dir_all(path)?;
    ensure!(
        loaded
            .matrix
            .as_ndarray()
            .iter()
            .chain(y.as_ndarray().iter())
            .all(|v| v.is_finite()),
        "result export requires finite feature and response values"
    );
    let sample_ids = loaded
        .sample_ids
        .clone()
        .unwrap_or_else(|| (0..loaded.matrix.nrows()).map(|i| i.to_string()).collect());
    let mut response_levels = y.as_ndarray().to_vec();
    response_levels.sort_by(f64::total_cmp);
    response_levels.dedup();
    let binary = response_levels.len() == 2;
    // R AUC treats the sorted higher response level as positive; selection keeps original y.
    let output_y = if binary {
        Vector::from_ndarray(y.as_ndarray().mapv(|value| {
            if value == response_levels[1] {
                1.0
            } else {
                0.0
            }
        }))
    } else {
        y.clone()
    };
    let y = &output_y;
    let mut summary = csv::Writer::from_path(path.join("summary_table.csv"))?;
    summary.write_record([
        "delta",
        "lambda",
        "num_of_LFs",
        "num_of_Sig_LFs",
        "num_of_Interactors",
        "Control_Performance",
    ])?;
    for result in results.evaluations.iter().flatten() {
        let out = path.join(format!("{}_{}_out", result.delta, result.lambda));
        std::fs::create_dir_all(&out)?;
        labelled_matrix(&out.join("A.csv"), &result.love.a, &loaded.feature_names)?;
        labelled_matrix(&out.join("z_matrix.csv"), &result.z, &sample_ids)?;
        let marginals = &result.marginal.sig_lfs;
        let interactions = &result.interactions.sig_interacts;
        std::fs::write(
            out.join("sig_LFs.txt"),
            marginals
                .iter()
                .map(|i| format!("Z{i}\n"))
                .collect::<String>(),
        )?;
        std::fs::write(
            out.join("sig_interacts.txt"),
            interactions
                .iter()
                .map(|(i, j)| format!("Z{i}.Z{j}\n"))
                .collect::<String>(),
        )?;
        let mut selected = marginals.clone();
        for &(i, j) in interactions {
            selected.extend([i, j]);
        }
        selected.sort_unstable();
        selected.dedup();
        let mut columns = Vec::new();
        for lf in selected {
            let features = top_features(
                &loaded.matrix,
                &loaded.feature_names,
                y,
                &result.love.a,
                lf,
                binary,
            )?;
            let mut writer = csv::WriterBuilder::new()
                .delimiter(b'\t')
                .from_path(out.join(format!("feature_list_Z{lf}.txt")))?;
            writer.write_record(if binary {
                vec!["names", "A_loading", "AUCs", "corrs", "color"]
            } else {
                vec!["names", "A_loading", "corrs", "color"]
            })?;
            for f in &features {
                let mut row = vec![f.name.clone(), f.loading.to_string()];
                if binary {
                    row.push(f.auc.map(|v| v.to_string()).unwrap_or_default());
                }
                row.push(f.corr.map(|v| v.to_string()).unwrap_or_default());
                row.push(
                    match f.color() {
                        "red" => "Red",
                        "blue" => "Blue",
                        _ => "White",
                    }
                    .to_string(),
                );
                writer.write_record(row)?;
            }
            writer.flush()?;
            columns.push((lf, features));
        }
        crate::plotting::gene_plot(
            &out.join("plotSigGenes_marginals.pdf"),
            &columns,
            marginals,
            false,
        )?;
        if config.do_interacts {
            crate::plotting::gene_plot(&out.join("plotSigGenes.pdf"), &columns, marginals, true)?;
        }
        let mut control_score = String::new();
        let mut full_random = "NA".to_string();
        let mut partial_random = "NA".to_string();
        if !marginals.is_empty() {
            let control = control_performance(
                &result.z,
                y,
                marginals,
                interactions,
                if binary {
                    ResponseType::Binary
                } else {
                    ResponseType::Continuous
                },
                1000,
                config.knockoff.base_seed,
            )?;
            control_score = control.real.to_string();
            full_random = (control.full_random.iter().sum::<f64>()
                / control.full_random.len() as f64)
                .to_string();
            if !control.partial_random.is_empty() {
                partial_random = (control.partial_random.iter().sum::<f64>()
                    / control.partial_random.len() as f64)
                    .to_string();
            }
            let mut writer = csv::Writer::from_path(out.join("ControlPerformance.csv"))?;
            writer.write_record(["group", "value"])?;
            for (group, values) in [
                ("FullRandom", &control.full_random),
                ("PartialRandom", &control.partial_random),
            ] {
                for value in values {
                    writer.write_record([group, &value.to_string()])?;
                }
            }
            writer.write_record(["Real", &control.real.to_string()])?;
            writer.flush()?;
            crate::plotting::control_plot(
                &out.join(if config.do_interacts {
                    "ControlPerformancePlot.png"
                } else {
                    "ControlPerformancePlot.pdf"
                }),
                &control,
            )?;
        }
        std::fs::write(out.join("scores.txt"), format!("Score contract: R in-sample OLS control performance (not held-out CV)\nTrue Scores: {}\nPartial Random: {}\nFull Random: {}\nNumber of latent factors: {}\nNumber of marginals: {}\nNumber of interactions: {}\n", if control_score.is_empty() { "NA" } else { &control_score }, partial_random, full_random, result.z.ncols(), marginals.len(), interactions.len()))?;
        write_params_json(
            out.join("run_params.txt"),
            &serde_json::json!({"delta":result.delta,"lambda":result.lambda,"config":config,"control_random_draws":1000,"control_score_contract":"R in-sample OLS; not held-out CV"}),
            &OutputWriteOptions::default(),
        )?;
        summary.write_record([
            result.delta.to_string(),
            result.lambda.to_string(),
            result.z.ncols().to_string(),
            marginals.len().to_string(),
            interactions.len().to_string(),
            control_score,
        ])?;
    }
    summary.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn labelled_csv_escapes_identifiers_and_features_follow_r_union() -> Result<()> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmpfiles/2026-10-17_native-output-fixture");
        std::fs::create_dir_all(&dir)?;
        let names = vec![
            "gene,\"one\"\nnext".to_string(),
            "negative".to_string(),
            "constant".to_string(),
        ];
        let a = Matrix::from_rows(vec![vec![-0.8], vec![0.2], vec![0.1]])?;
        labelled_matrix(&dir.join("A.csv"), &a, &names)?;
        let mut reader = csv::Reader::from_path(dir.join("A.csv"))?;
        let records: Vec<_> = reader.records().collect::<std::result::Result<_, _>>()?;
        assert_eq!(&records[0][0], names[0]);
        assert_eq!(&reader.headers()?[1], "Z0");
        let loaded = LoadedMatrix {
            matrix: Matrix::from_rows(vec![
                vec![1.0, 4.0, 1.0],
                vec![2.0, 3.0, 1.0],
                vec![3.0, 2.0, 1.0],
                vec![4.0, 1.0, 1.0],
            ])?,
            feature_names: names,
            sample_ids: None,
        };
        let y = Vector::from_ndarray(Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]));
        let features = top_features(&loaded.matrix, &loaded.feature_names, &y, &a, 0, true)?;
        assert_eq!(features.len(), 3);
        assert_eq!(features[0].loading, 0.8);
        assert_eq!(features[0].auc, Some(1.0));
        assert_eq!(features[0].color(), "red");
        assert_eq!(features[1].color(), "blue");
        assert_eq!(features[2].color(), "white");
        let shifted_y = Vector::from_ndarray(y.as_ndarray().mapv(|v| v + 2.0));
        let shifted = top_features(
            &loaded.matrix,
            &loaded.feature_names,
            &shifted_y,
            &a,
            0,
            true,
        )?;
        assert_eq!(shifted[0].auc, features[0].auc);
        assert_eq!(shifted[1].color(), features[1].color());
        let invalid_y = Vector::from_ndarray(Array1::from_vec(vec![0.0, 1.0, 2.0, 2.0]));
        assert!(top_features(
            &loaded.matrix,
            &loaded.feature_names,
            &invalid_y,
            &a,
            0,
            true
        )
        .is_err());
        assert!(top_features(&loaded.matrix, &loaded.feature_names, &y, &a, 1, true).is_err());
        assert_eq!(
            ranks(&Array1::from_vec(vec![2.0, 1.0, 1.0])).to_vec(),
            vec![3.0, 1.5, 1.5]
        );
        Ok(())
    }
    #[test]
    fn native_export_with_and_without_selections() -> Result<()> {
        let matrix = Matrix::from_rows(vec![
            vec![1.0, 4.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
        ])?;
        let a = Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]])?;
        let vector = Vector::from_ndarray(Array1::from_vec(vec![0.0, 0.0]));
        let loaded = LoadedMatrix {
            matrix: matrix.clone(),
            feature_names: vec!["gene1".into(), "gene2".into()],
            sample_ids: Some(vec!["s1".into(), "s2".into(), "s3".into(), "s4".into()]),
        };
        let y = Vector::from_ndarray(Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]));
        let run: slide_core::SlideRunResult = serde_json::from_value(serde_json::json!({
            "delta":0.1,"lambda":0.2,"z":matrix,"score":0.5,
            "love":{"k":2,"pure_vec":[0,1],"pure_ind":[],"group":[],"a":a,"c":a,"omega":null,"gamma":vector,"opt_delta":0.1},
            "marginal":{"sig_lfs":[0],"voting":{"selection_counts":[1,0],"selection_frequency":vector,"selected":[0],"freq_selected":[0],"threshold":0.5,"niter":1,"spec":0.5,"min_selections":1,"selected_list":[[0]],"optimal_iter":null,"stage_one_union":null}},
            "interactions":{"sig_interacts":[],"interaction_terms":matrix,"voting":[]}
        }))?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmpfiles/2026-10-17_native-output-fixture");
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        for selected in [true, false] {
            let mut run = run.clone();
            if !selected {
                run.marginal.sig_lfs.clear();
            }
            let results = OptimizeSlideResult {
                best: run.clone(),
                evaluations: vec![Some(run)],
                final_cv: None,
            };
            let path = root.join(format!("export-{unique}-{selected}"));
            export_results(
                &path,
                &loaded,
                &y,
                &results,
                &OptimizeSlideConfig::default(),
            )?;
            let out = path.join("0.1_0.2_out");
            assert_eq!(
                std::fs::read_to_string(out.join("sig_LFs.txt"))?,
                if selected { "Z0\n" } else { "" }
            );
            if selected {
                let alternate_path = root.join(format!("export-{unique}-labels-2-3"));
                let alternate_y = Vector::from_ndarray(y.as_ndarray().mapv(|value| value + 2.0));
                export_results(
                    &alternate_path,
                    &loaded,
                    &alternate_y,
                    &results,
                    &OptimizeSlideConfig::default(),
                )?;
                let alternate_out = alternate_path.join("0.1_0.2_out");
                for name in ["ControlPerformance.csv", "feature_list_Z0.txt"] {
                    assert_eq!(
                        std::fs::read(out.join(name))?,
                        std::fs::read(alternate_out.join(name))?
                    );
                }
                assert_eq!(alternate_y.as_ndarray().to_vec(), vec![2.0, 2.0, 3.0, 3.0]);
                assert_eq!(y.as_ndarray().to_vec(), vec![0.0, 0.0, 1.0, 1.0]);
            }
            assert!(out.join("scores.txt").exists());
            assert_eq!(out.join("ControlPerformance.csv").exists(), selected);
            assert_eq!(out.join("plotSigGenes_marginals.pdf").exists(), selected);
            assert!(export_results(
                &path,
                &loaded,
                &y,
                &results,
                &OptimizeSlideConfig::default()
            )
            .is_err());
        }
        Ok(())
    }
}
