//! Occupancy statistics. Shared-room pairs are never binomial observations.
use anyhow::{ensure, Result};
use bevy_zeroverse_capture::capacity::*;
use std::collections::BTreeMap;

pub fn finite_occupancy(probabilities: &[f64], draws: u64) -> Result<f64> {
    ensure!(
        !probabilities.is_empty()
            && probabilities
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.).contains(p)),
        "invalid finite probability law"
    );
    ensure!(
        (probabilities.iter().sum::<f64>() - 1.).abs() < 1e-10,
        "probabilities must sum to one"
    );
    Ok(probabilities
        .iter()
        .map(|&p| {
            if draws == 0 {
                0.
            } else if p == 1. {
                1.
            } else {
                -((draws as f64) * (-p).ln_1p()).exp_m1()
            }
        })
        .sum())
}

/// Exact without-replacement expectation in a *known* finite pool. This is
/// also the observed-cohort rarefaction oracle, not an unseen-support model.
pub fn finite_pool_occupancy(counts: &[u64], draws: u64) -> Result<f64> {
    let total = counts
        .iter()
        .try_fold(0_u64, |a, b| a.checked_add(*b))
        .ok_or_else(|| anyhow::anyhow!("count overflow"))?;
    ensure!(draws <= total, "draw count exceeds the known pool");
    Ok(counts
        .iter()
        .map(|&count| {
            if count == 0 || draws == 0 {
                return 0.;
            }
            if draws > total - count {
                return 1.;
            }
            // Choose the shorter equivalent product, retaining accuracy near zero.
            let steps = draws.min(count);
            let excluded = draws.max(count);
            let log_absence: f64 = (0..steps)
                .map(|i| (-(excluded as f64) / (total - i) as f64).ln_1p())
                .sum();
            -log_absence.exp_m1()
        })
        .sum())
}

fn log_add(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        high
    } else {
        high + (-(a - b).abs()).exp().ln_1p()
    }
}

fn binomial_cdf(n: u32, k: u32, p: f64) -> f64 {
    if k >= n || p == 0. {
        return 1.;
    }
    if p == 1. {
        return 0.;
    }
    let mut term = f64::from(n) * (-p).ln_1p();
    let mut total = term;
    let log_odds = p.ln() - (-p).ln_1p();
    for i in 0..k {
        term += f64::from(n - i).ln() - f64::from(i + 1).ln() + log_odds;
        total = log_add(total, term);
    }
    total.exp().min(1.)
}

fn invert_cdf(n: u32, k: u32, target: f64) -> f64 {
    let (mut lo, mut hi) = (0., 1.);
    for _ in 0..64 {
        let mid = (lo + hi) * 0.5;
        if binomial_cdf(n, k, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

/// Clopper-Pearson bounds for prespecified disjoint independent pairs only.
/// Use the same two-sided tail allocation for every observed count. Selecting
/// a one-sided interval only after observing zero would under-cover near that
/// selection boundary. Zero still has an unknown reciprocal upper bound.
pub fn collision_interval(matches: u32, pairs: u32, confidence: f64) -> Result<[f64; 2]> {
    ensure!(
        pairs > 0 && matches <= pairs && (0.5..1.).contains(&confidence),
        "invalid binomial experiment"
    );
    let alpha = 1. - confidence;
    if matches == 0 {
        return Ok([0., -((alpha * 0.5).ln() / f64::from(pairs)).exp_m1()]);
    }
    let lo = invert_cdf(pairs, matches - 1, 1. - alpha * 0.5);
    let hi = if matches == pairs {
        1.
    } else {
        invert_cdf(pairs, matches, alpha * 0.5)
    };
    Ok([lo, hi])
}

fn scalar(
    point: Option<f64>,
    lower: Option<f64>,
    upper: Option<f64>,
    confidence: Option<f64>,
    method: &str,
    scope: &str,
) -> ScalarEstimate {
    ScalarEstimate {
        point,
        lower,
        upper,
        confidence,
        method: method.into(),
        scope: scope.into(),
    }
}

pub(super) fn estimate(
    definition: &DiversityDefinition,
    labels: &[String],
    request: &CapacityRequest,
) -> Result<DefinitionEstimate> {
    let n = labels.len() as u32;
    let mut counts = BTreeMap::<&str, u64>::new();
    for label in labels {
        *counts.entry(label).or_default() += 1;
    }
    let counts_vec: Vec<_> = counts.values().copied().collect();
    let pairs = n / 2;
    let matches = labels
        .as_chunks::<2>()
        .0
        .iter()
        .filter(|p| p[0] == p[1])
        .count() as u32;
    let c2 = (n >= 2).then(|| {
        counts
            .values()
            .map(|&c| (c as f64) * (c.saturating_sub(1) as f64))
            .sum::<f64>()
            / (f64::from(n) * f64::from(n - 1))
    });
    let iid = matches!(
        request.seed_policy,
        SeedPolicy::UniformWithReplacement { .. }
    );
    // Simultaneous coverage for the prespecified definitions, not uncorrected
    // repeated looks chosen after inspecting the audit.
    let adjusted_confidence = 1. - (1. - request.confidence) / (request.definitions.len() as f64);
    let interval = if iid && pairs > 0 {
        Some(collision_interval(matches, pairs, adjusted_confidence)?)
    } else {
        None
    };
    let entropy = if n == 0 {
        0.
    } else {
        (-counts
            .values()
            .map(|&c| {
                let p = c as f64 / f64::from(n);
                p * p.ln()
            })
            .sum::<f64>())
        .exp()
    };
    let mut rarefaction = Vec::new();
    let mut sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, u64::from(n)];
    sizes.sort_unstable();
    sizes.dedup();
    for draws in sizes.into_iter().filter(|&s| s > 0 && s <= u64::from(n)) {
        rarefaction.push(OccupancyPoint {
            draws,
            distinct: scalar(
                Some(finite_pool_occupancy(&counts_vec, draws)?),
                None,
                None,
                None,
                "exact without-replacement rarefaction",
                "observed cohort only",
            ),
        });
    }
    let mut nested_validation = Vec::new();
    for discovery in [n / 8, n / 4, n / 2].into_iter().filter(|&v| v >= 16) {
        let prefix = &labels[..discovery as usize];
        let holdout = &labels[discovery as usize..(2 * discovery) as usize];
        let mut seen = BTreeMap::<&str, u32>::new();
        for key in prefix {
            *seen.entry(key).or_default() += 1;
        }
        let new: std::collections::BTreeSet<_> = holdout
            .iter()
            .filter(|k| !seen.contains_key(k.as_str()))
            .collect();
        let unseen = holdout
            .iter()
            .filter(|k| !seen.contains_key(k.as_str()))
            .count() as u32;
        nested_validation.push(NoveltyValidation {
            discovery_rooms: discovery,
            holdout_rooms: holdout.len() as u32,
            discovery_classes: seen.len() as u32,
            holdout_new_classes: new.len() as u32,
            holdout_unseen_draws: unseen,
            predicted_unseen_fraction: seen.values().filter(|&&v| v == 1).count() as f64
                / f64::from(discovery),
            observed_unseen_fraction: f64::from(unseen) / holdout.len() as f64,
        });
    }
    let horizon = request.requested_draws as f64;
    let collision_scope = if iid {
        "successful population draws"
    } else {
        "observed diagnostic cohort; no IID population inference"
    };
    Ok(DefinitionEstimate {
        definition: definition.clone(),
        status: if n == 0 {
            EstimateStatus::NotQualified {
                reasons: vec!["no completed observations for this representation".into()],
            }
        } else if interval.is_some() {
            EstimateStatus::Bounded
        } else {
            EstimateStatus::ObservedOnly
        },
        rooms: n,
        observed_classes: counts.len() as u32,
        singletons: counts.values().filter(|&&c| c == 1).count() as u32,
        doubletons: counts.values().filter(|&&c| c == 2).count() as u32,
        block_distinct_counts: BTreeMap::new(),
        disjoint_pairs: pairs,
        matching_disjoint_pairs: matches,
        collision_probability: scalar(
            c2,
            interval.map(|x| x[0]),
            interval.map(|x| x[1]),
            interval.map(|_| adjusted_confidence),
            if iid {
                "unbiased all-room U-statistic point; disjoint-pair exact bounds"
            } else {
                "empirical all-room pair frequency; no population interval"
            },
            collision_scope,
        ),
        collision_diversity: scalar(
            c2.filter(|&c| c > 0.).map(|c| 1. / c),
            interval.map(|x| 1. / x[1]),
            interval.and_then(|x| (x[0] > 0.).then(|| 1. / x[0])),
            interval.map(|_| adjusted_confidence),
            "reciprocal collision probability; plug-in point is biased",
            collision_scope,
        ),
        empirical_entropy_diversity: entropy,
        rarefaction,
        requested_occupancy: OccupancyPoint {
            draws: request.requested_draws,
            distinct: scalar(
                None,
                interval.map(|x| horizon / (1. + (horizon - 1.) * x[1])),
                interval.map(|_| horizon),
                interval.map(|_| adjusted_confidence),
                "distribution-free collision lower bound; no unseen-tail point forecast",
                "independent successful draws from the declared population",
            ),
        },
        max_qualified_forecast_draws: 0,
        nested_validation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_finite_oracles_and_correlated_factors() {
        assert!((finite_occupancy(&[0.5, 0.5], 3).unwrap() - 1.75).abs() < 1e-12);
        assert!((finite_pool_occupancy(&[2, 2], 2).unwrap() - 5. / 3.).abs() < 1e-12);
        assert_eq!(finite_pool_occupancy(&[2, 2], 4).unwrap(), 2.);
        // Identical D2 does not identify occupancy: perfect joint correlation
        // cannot be replaced by a product of the two marginal cardinalities.
        assert!(
            (finite_occupancy(&[0.5, 0.5], 10).unwrap()
                - finite_occupancy(&[2. / 3., 1. / 6., 1. / 6.], 10).unwrap())
            .abs()
                > 0.5
        );
        assert!(finite_pool_occupancy(&[u64::MAX, 1], 1).is_err());
        assert!(finite_occupancy(&[0.4, 0.4], 1).is_err());
    }
    #[test]
    fn zero_matches_does_not_claim_infinite_capacity() {
        let bounds = collision_interval(0, 256, 0.95).unwrap();
        assert_eq!(bounds[0], 0.);
        assert!((1. / bounds[1] - 1. / -(0.025_f64.ln() / 256.).exp_m1()).abs() < 1e-6);
        let request = CapacityRequest::default();
        let e = estimate(
            &request.definitions[0],
            &(0..512).map(|i| i.to_string()).collect::<Vec<_>>(),
            &request,
        )
        .unwrap();
        assert!(e.collision_diversity.point.is_none());
        assert!(e.collision_diversity.upper.is_none());
        assert!(e.requested_occupancy.distinct.point.is_none());
        assert_eq!(e.max_qualified_forecast_draws, 0);
    }
    #[test]
    fn exact_binomial_coverage_including_rare_tails() {
        // Enumerating the binomial law is stronger and cheaper than noisy Monte
        // Carlo coverage for these oracle sizes.
        for n in [8, 32, 128] {
            let transition = -(0.05_f64.ln() / f64::from(n)).exp_m1();
            for p in [0.0001, 0.01, 0.1, 0.5, 0.95, 1., transition + 1e-6] {
                let mut coverage = 0.;
                let mut previous = 0.;
                for k in 0..=n {
                    let cdf = binomial_cdf(n, k, p);
                    let [lo, hi] = collision_interval(k, n, 0.95).unwrap();
                    if p >= lo - 1e-12 && p <= hi + 1e-12 {
                        coverage += cdf - previous;
                    }
                    previous = cdf;
                }
                assert!(coverage >= 0.949999, "n={n} p={p} coverage={coverage}");
            }
        }
    }
}
