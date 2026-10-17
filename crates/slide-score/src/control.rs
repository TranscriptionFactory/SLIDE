use rand::{rngs::StdRng, seq::index::sample, SeedableRng};
use serde::{Deserialize, Serialize};
use slide_types::{Matrix, Scalar, Vector};

use crate::{
    auc_score, build_selected_feature_matrix, fit_linear_predict, ResponseType, ScoreError,
    ScoreMetric,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlPerformance {
    pub real: Scalar,
    pub full_random: Vec<Scalar>,
    pub partial_random: Vec<Scalar>,
    pub metric: ScoreMetric,
}

/// In-sample OLS controls following SLIDE R's calcControlPerformance, not CV scores.
/// Native seeded draws are reproducible but do not reproduce R's RNG stream.
/// Near-singular fits can differ from R's QR solver: the shared fitter uses SVD.
pub fn control_performance(
    z: &Matrix,
    y: &Vector,
    marginal_lfs: &[usize],
    interaction_pairs: &[(usize, usize)],
    response_type: ResponseType,
    n_random: usize,
    seed: u64,
) -> Result<ControlPerformance, ScoreError> {
    if y.len() != z.nrows() {
        return Err(ScoreError::FeatureResponseLengthMismatch {
            expected: z.nrows(),
            actual: y.len(),
        });
    }
    if !z
        .as_ndarray()
        .iter()
        .chain(y.as_ndarray().iter())
        .all(|v| v.is_finite())
    {
        return Err(ScoreError::NonFiniteControlScore);
    }
    if marginal_lfs.is_empty() {
        return Err(ScoreError::EmptyControlMarginals);
    }
    if n_random == 0 {
        return Err(ScoreError::EmptyControlIterations);
    }
    let real_features = build_selected_feature_matrix(z, marginal_lfs, interaction_pairs)?;
    for (kind, requested, available) in [
        ("marginals", marginal_lfs.len(), z.ncols()),
        (
            "interactions",
            interaction_pairs.len(),
            marginal_lfs.len() * z.ncols(),
        ),
    ] {
        if requested > available {
            return Err(ScoreError::InvalidControlSampleSize {
                kind,
                requested,
                available,
            });
        }
    }
    match response_type {
        ResponseType::Binary => {
            auc_score(y.as_ndarray(), y.as_ndarray())?;
        }
        ResponseType::Continuous => {
            if y.len() < 2 || y.as_ndarray().iter().all(|v| *v == y.as_ndarray()[0]) {
                return Err(ScoreError::DegenerateContinuousScore);
            }
        }
    }
    let real = score_features(&real_features, y, response_type)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut full_random = Vec::with_capacity(n_random);
    let mut partial_random = Vec::new();
    for _ in 0..n_random {
        let marginals = sample(&mut rng, z.ncols(), marginal_lfs.len()).into_vec();
        let pairs = random_pairs(&mut rng, &marginals, z.ncols(), interaction_pairs.len());
        let features = build_selected_feature_matrix(z, &marginals, &pairs)?;
        full_random.push(score_features(&features, y, response_type)?);
    }
    if !interaction_pairs.is_empty() {
        for _ in 0..n_random {
            let pairs = random_pairs(&mut rng, marginal_lfs, z.ncols(), interaction_pairs.len());
            let features = build_selected_feature_matrix(z, marginal_lfs, &pairs)?;
            partial_random.push(score_features(&features, y, response_type)?);
        }
    }
    Ok(ControlPerformance {
        real,
        full_random,
        partial_random,
        metric: match response_type {
            ResponseType::Binary => ScoreMetric::Auc,
            ResponseType::Continuous => ScoreMetric::Correlation,
        },
    })
}

fn random_pairs(
    rng: &mut StdRng,
    marginals: &[usize],
    p: usize,
    count: usize,
) -> Vec<(usize, usize)> {
    // R's selected-major K × P pool includes self-products and reverse pairs.
    sample(rng, marginals.len() * p, count)
        .iter()
        .map(|i| (marginals[i / p], i % p))
        .collect()
}

fn score_features(
    features: &Matrix,
    y: &Vector,
    response_type: ResponseType,
) -> Result<Scalar, ScoreError> {
    if !features.as_ndarray().iter().all(|v| v.is_finite()) {
        return Err(ScoreError::NonFiniteControlScore);
    }
    let predicted = fit_linear_predict(features, y, features)?;
    if !predicted.as_ndarray().iter().all(|v| v.is_finite()) {
        return Err(ScoreError::NonFiniteControlScore);
    }
    let score = match response_type {
        ResponseType::Binary => auto_direction_auc(y, &predicted)?,
        ResponseType::Continuous => {
            let mean = predicted.as_ndarray().sum() / y.len() as Scalar;
            let mss = predicted
                .as_ndarray()
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<Scalar>();
            let rss = y
                .as_ndarray()
                .iter()
                .zip(predicted.as_ndarray())
                .map(|(y, p)| (y - p).powi(2))
                .sum::<Scalar>();
            if !(mss + rss).is_finite() {
                return Err(ScoreError::NonFiniteControlScore);
            }
            // summary.lm uses fitted and residual sums of squares; a constant fit scores zero.
            (mss / (mss + rss)).sqrt()
        }
    };
    if !score.is_finite() {
        return Err(ScoreError::NonFiniteControlScore);
    }
    Ok(score)
}

fn auto_direction_auc(y: &Vector, predicted: &Vector) -> Result<Scalar, ScoreError> {
    let auc = auc_score(y.as_ndarray(), predicted.as_ndarray())?;
    let median = |label| {
        let mut values: Vec<_> = y
            .as_ndarray()
            .iter()
            .zip(predicted.as_ndarray())
            .filter_map(|(&y, &p)| (y == label).then_some(p))
            .collect();
        values.sort_by(Scalar::total_cmp);
        let mid = values.len() / 2;
        values[mid - usize::from(values.len() % 2 == 0)] / 2.0 + values[mid] / 2.0
    };
    Ok(if median(0.0) <= median(1.0) {
        auc
    } else {
        1.0 - auc
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn fixture() -> (Matrix, Vector) {
        (
            Matrix::from_ndarray(arr2(&[
                [0., 1.],
                [1., 0.],
                [2., 2.],
                [3., -1.],
                [4., 3.],
                [5., 1.],
            ])),
            Vector::from_ndarray(arr1(&[1., 2., 1., 4., 3., 6.])),
        )
    }

    #[test]
    fn analytic_ols_and_exact_pairs() -> Result<(), ScoreError> {
        let (z, _) = fixture();
        let y = Vector::from_ndarray(arr1(&[2., 5., 16., 5., 38., 27.])); // 2 + 3*z0 + 2*z0*z1
        let f = build_selected_feature_matrix(&z, &[0], &[(0, 1)])?;
        assert_eq!(f.ncols(), 2);
        let controls =
            control_performance(&z, &y, &[0], &[(0, 1)], ResponseType::Continuous, 8, 42)?;
        assert!((controls.real - 1.).abs() < 1e-12);
        assert_eq!(controls.full_random.len(), 8);
        assert_eq!(controls.partial_random.len(), 8);
        assert_eq!(
            controls,
            control_performance(&z, &y, &[0], &[(0, 1)], ResponseType::Continuous, 8, 42)?
        );
        Ok(())
    }

    #[test]
    fn continuous_matches_analytic_r_squared_and_omits_partial() -> Result<(), ScoreError> {
        let z = Matrix::from_ndarray(arr2(&[[0.], [1.], [2.], [3.]]));
        let y = Vector::from_ndarray(arr1(&[0., 1., 1., 3.]));
        let c = control_performance(&z, &y, &[0], &[], ResponseType::Continuous, 3, 2)?;
        assert!((c.real - (4.05_f64 / 4.75).sqrt()).abs() < 1e-12);
        assert!(c.partial_random.is_empty());
        assert!(c.full_random.iter().all(|v| (*v - c.real).abs() < 1e-12));
        Ok(())
    }

    #[test]
    fn binary_uses_ols_and_median_direction_not_max_auc() -> Result<(), ScoreError> {
        let y = Vector::from_ndarray(arr1(&[0., 0., 0., 1., 1., 1.]));
        let predictions = Vector::from_ndarray(arr1(&[0., 1., 100., -2., 2., 3.]));
        assert!((auto_direction_auc(&y, &predictions)? - 4. / 9.).abs() < 1e-12);
        let reversed = Vector::from_ndarray(-predictions.as_ndarray());
        assert!((auto_direction_auc(&y, &reversed)? - 4. / 9.).abs() < 1e-12);
        let z = Matrix::from_ndarray(arr2(&[[0.], [1.], [2.], [3.]]));
        let y = Vector::from_ndarray(arr1(&[0., 1., 0., 1.]));
        assert!(
            (control_performance(&z, &y, &[0], &[], ResponseType::Binary, 2, 1)?.real - 0.75).abs()
                < 1e-12
        );
        Ok(())
    }

    #[test]
    fn sampling_pool_includes_self_and_reverse() {
        let pairs = random_pairs(&mut StdRng::seed_from_u64(1), &[1, 0], 2, 4);
        for pair in [(1, 0), (1, 1), (0, 0), (0, 1)] {
            assert!(pairs.contains(&pair));
        }
    }

    #[test]
    fn constant_fit_scores_zero_and_inputs_are_validated() -> Result<(), ScoreError> {
        let z = Matrix::from_ndarray(arr2(&[[1.], [1.], [1.], [1.]]));
        let y = Vector::from_ndarray(arr1(&[0., 1., 2., 3.]));
        assert!(
            control_performance(&z, &y, &[0], &[], ResponseType::Continuous, 1, 1)?.real < 1e-12
        );
        let run = |z: &Matrix, y: &Vector, m: &[usize], p: &[(usize, usize)], n| {
            control_performance(z, y, m, p, ResponseType::Continuous, n, 1)
        };
        assert!(matches!(
            run(&z, &y, &[], &[], 1),
            Err(ScoreError::EmptyControlMarginals)
        ));
        assert!(matches!(
            run(&z, &y, &[0], &[], 0),
            Err(ScoreError::EmptyControlIterations)
        ));
        assert!(matches!(
            run(&z, &y, &[1], &[], 1),
            Err(ScoreError::SelectedLfOutOfBounds { .. })
        ));
        assert!(matches!(
            run(&z, &y, &[0], &[(0, 1)], 1),
            Err(ScoreError::InteractionOutOfBounds { .. })
        ));
        assert!(matches!(
            run(&z, &y, &[0, 0], &[], 1),
            Err(ScoreError::InvalidControlSampleSize { .. })
        ));
        assert!(matches!(
            run(&z, &y, &[0], &[(0, 0), (0, 0)], 1),
            Err(ScoreError::InvalidControlSampleSize { .. })
        ));
        assert!(matches!(
            run(&z, &Vector::from_ndarray(arr1(&[1.])), &[0], &[], 1),
            Err(ScoreError::FeatureResponseLengthMismatch { .. })
        ));
        assert!(matches!(
            run(&z, &Vector::from_ndarray(arr1(&[1.; 4])), &[0], &[], 1),
            Err(ScoreError::DegenerateContinuousScore)
        ));
        let bad = Matrix::from_ndarray(arr2(&[[f64::NAN], [1.], [1.], [1.]]));
        assert!(matches!(
            run(&bad, &y, &[0], &[], 1),
            Err(ScoreError::NonFiniteControlScore)
        ));
        assert!(matches!(
            control_performance(&z, &y, &[0], &[], ResponseType::Binary, 1, 1),
            Err(ScoreError::InvalidBinaryResponse)
        ));
        Ok(())
    }
}
