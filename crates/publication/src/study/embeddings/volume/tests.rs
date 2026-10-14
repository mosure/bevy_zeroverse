use super::*;

/// Balanced orthogonal Walsh directions with one constant coordinate, so every
/// row has unit norm while its variable axes have an analytically known spread.
fn cloud(amplitudes: &[f64]) -> Vec<Vec<f64>> {
    let constant = (1. - amplitudes.iter().map(|v| v * v).sum::<f64>()).sqrt();
    assert!(constant.is_finite());
    (0..64usize)
        .map(|i| {
            let mut v: Vec<_> = amplitudes
                .iter()
                .enumerate()
                .map(|(j, &a)| {
                    if (i & (j + 1)).count_ones().is_multiple_of(2) {
                        a
                    } else {
                        -a
                    }
                })
                .collect();
            v.push(constant);
            v
        })
        .collect()
}
fn valid_rank(result: &Value, rank: usize) -> &Value {
    let value = result["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["rank"] == rank)
        .unwrap();
    assert_eq!(value["status"], "valid", "{result}");
    value
}

#[test]
fn identical_unit_centroids_have_unit_volume_at_all_requested_ranks() {
    let before = cloud(&[0.05; 16]);
    let result = analyze(&before, &before).unwrap();
    for rank in RANKS {
        let value = valid_rank(&result, rank);
        assert!(value["geometric_mean_standard_deviation_ratio"]
            .as_f64()
            .unwrap()
            .is_finite());
        assert!((value["sqrt_covariance_determinant_ratio"].as_f64().unwrap() - 1.).abs() < 1e-10);
        assert!(value["per_axis_standard_deviation_ratios"]
            .as_array()
            .unwrap()
            .iter()
            .all(|x| (x.as_f64().unwrap() - 1.).abs() < 1e-10));
        for bound in ["lower", "upper"] {
            assert!(
                (value["paired_bootstrap_95_percent"][bound]
                    .as_f64()
                    .unwrap()
                    - 1.)
                    .abs()
                    < 1e-10
            );
        }
    }
    assert_eq!(
        result,
        analyze(&before, &before).unwrap(),
        "bootstrap and PCA must replay"
    );
    assert_eq!(result["bootstrap"]["resamples"], 400);
    assert_eq!(result["basis"]["pooled_rooms"], 128);
}

#[test]
fn known_double_axis_spread_has_two_to_rank_conditional_volume() {
    let before = cloud(&[0.05; 16]);
    let after = cloud(&[0.10; 16]);
    let result = analyze(&before, &after).unwrap();
    for rank in RANKS {
        let value = valid_rank(&result, rank);
        assert!(value["geometric_mean_standard_deviation_ratio"]
            .as_f64()
            .unwrap()
            .is_finite());
        let expected = 2f64.powi(rank as i32);
        assert!(
            (value["sqrt_covariance_determinant_ratio"].as_f64().unwrap() / expected - 1.).abs()
                < 1e-6
        );
        assert!(value["per_axis_standard_deviation_ratios"]
            .as_array()
            .unwrap()
            .iter()
            .all(|x| (x.as_f64().unwrap() - 2.).abs() < 1e-8));
        assert!(
            value.get("per_axis_standard_deviation_ratio").is_none(),
            "array field must be unambiguously plural"
        );
        assert!(
            (value["geometric_mean_standard_deviation_ratio"]
                .as_f64()
                .unwrap()
                - 2.)
                .abs()
                < 1e-6
        );
        for bound in ["lower", "upper"] {
            // Resampling can lose a Walsh direction. The shared ridge then
            // contributes unit spread on that collapsed axis, rather than 2x.
            let measured = value["paired_bootstrap_95_percent"][bound]
                .as_f64()
                .unwrap();
            assert!(
                measured > 0. && measured <= expected * (1. + 1e-6),
                "rank {rank}: {value}"
            );
        }
        if rank == 16 {
            assert!(
                value["paired_bootstrap_95_percent"]["singular_before_resamples"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        }
    }
    assert!(result["ridge"]["variance"].as_f64().unwrap() > 0.);
    assert!(
        result["basis"]["maximum_remaining_off_diagonal_over_trace"]
            .as_f64()
            .unwrap()
            < 1e-10
    );
}

#[test]
fn rotation_preserves_full_projected_volume() {
    let before = cloud(&[0.02, 0.04, 0.06, 0.08]);
    let mut after = before.clone();
    let angle = 0.63f64;
    let (s, c) = angle.sin_cos();
    for row in &mut after {
        let a = row[0];
        let b = row[3];
        row[0] = c * a - s * b;
        row[3] = s * a + c * b;
    }
    let result = analyze(&before, &after).unwrap();
    let value = valid_rank(&result, 4);
    assert!((value["sqrt_covariance_determinant_ratio"].as_f64().unwrap() - 1.).abs() < 1e-6);
    assert!(value["per_axis_standard_deviation_ratios"]
        .as_array()
        .unwrap()
        .iter()
        .all(|x| x.as_f64().unwrap().is_finite() && x.as_f64().unwrap() > 0.));
    assert_eq!(
        result["ranks"][1]["status"],
        "insufficient_pooled_rank_or_rooms"
    );
}

#[test]
fn degenerate_and_singular_cohorts_are_not_given_inflated_regularized_ratios() {
    let constant = vec![vec![1., 0., 0., 0., 0.]; 64];
    let result = analyze(&constant, &constant).unwrap();
    assert_eq!(result["status"], "degenerate_pooled_centroids");
    assert!(result["ranks"].as_array().unwrap().is_empty());
    let before = cloud(&[0.05, 0.05, 0.05, 0.]);
    let after = cloud(&[0.05, 0.05, 0.05, 0.05]);
    let result = analyze(&before, &after).unwrap();
    assert_eq!(result["ranks"][0]["status"], "singular_observed_cohort");
    assert!(result["ranks"][0]
        .get("sqrt_covariance_determinant_ratio")
        .is_none());
    let short = cloud(&[0.05; 4]);
    let result = analyze(&short[..3], &short[..3]).unwrap();
    assert!(result["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|x| x["status"] == "insufficient_pooled_rank_or_rooms"));
}

#[test]
fn invalid_shapes_precision_and_unpaired_inputs_fail_without_panic() {
    let valid = cloud(&[0.05; 4]);
    assert!(analyze(&valid, &valid[..63]).is_err());
    assert!(analyze(&valid[..1], &valid[..1]).is_err());
    let mut wrong = valid.clone();
    wrong[2].pop();
    assert!(analyze(&valid, &wrong).is_err());
    wrong = valid.clone();
    wrong[2][1] = f64::NAN;
    assert!(analyze(&valid, &wrong).is_err());
    wrong = valid.clone();
    wrong[2][1] += 0.5;
    assert!(analyze(&valid, &wrong).is_err());
}

#[test]
fn symmetric_jacobi_retains_eigenvector_orthogonality_and_reconstructs_matrix() {
    let original = vec![4., 1., 0.5, 1., 3., 0.2, 0.5, 0.2, 2.];
    let eigen = symmetric_eigen(original.clone(), 3).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let reconstructed = (0..3)
                .map(|k| eigen.vectors[i * 3 + k] * eigen.values[k] * eigen.vectors[j * 3 + k])
                .sum::<f64>();
            assert!((original[i * 3 + j] - reconstructed).abs() < 1e-10);
            let inner = (0..3)
                .map(|k| eigen.vectors[k * 3 + i] * eigen.vectors[k * 3 + j])
                .sum::<f64>();
            assert!((inner - if i == j { 1. } else { 0. }).abs() < 1e-10);
        }
    }
}

#[test]
fn larger_cohorts_use_a_bounded_fixed_basis_but_all_covariance_rows() {
    let base = cloud(&[0.05; 16]);
    let rows: Vec<_> = (0..513).map(|i| base[i % base.len()].clone()).collect();
    let result = analyze(&rows, &rows).unwrap();
    assert_eq!(result["room_pairs"], 513);
    assert_eq!(result["basis"]["pooled_rooms"], 512);
    assert_eq!(
        result["basis"]["fit_room_indices"]
            .as_array()
            .unwrap()
            .len(),
        256
    );
    for rank in RANKS {
        assert!(
            (valid_rank(&result, rank)["sqrt_covariance_determinant_ratio"]
                .as_f64()
                .unwrap()
                - 1.)
                .abs()
                < 1e-10
        );
    }
}
