//! Conditional covariance-ellipsoid diagnostics in a common, measured PCA basis.
//! Rooms, not their correlated views, are the paired statistical sampling unit.
use super::*;

const RANKS: [usize; 3] = [4, 8, 16];
const RESAMPLES: usize = 400;
const RIDGE_FRACTION: f64 = 1e-8;
const BOOTSTRAP_SEED: u64 = 0x923cad8ed58041b7;

pub(super) fn analyze(before: &[Vec<f64>], after: &[Vec<f64>]) -> Result<Value> {
    ensure!(
        before.len() == after.len() && (2..=1024).contains(&before.len()),
        "volume diagnostic requires 2..1024 paired rooms"
    );
    let dimensions = before[0].len();
    ensure!(
        (1..=4096).contains(&dimensions)
            && before.iter().chain(after).all(|v| v.len() == dimensions
                && v.iter().all(|x| x.is_finite())
                && (dot(v, v) - 1.).abs() < 1e-5),
        "invalid volume diagnostic centroids"
    );
    let n = before.len();
    // Fit a bounded shared basis from prespecified stratified room indices.
    // Project and bootstrap every room; selection does not inspect embeddings.
    let fit_count = n.min(256);
    let fit_indices = basis_indices(n, fit_count);
    let pooled: Vec<_> = fit_indices
        .iter()
        .map(|&i| &before[i])
        .chain(fit_indices.iter().map(|&i| &after[i]))
        .collect();
    let mean: Vec<_> = (0..dimensions)
        .map(|d| pooled.iter().map(|v| v[d]).sum::<f64>() / pooled.len() as f64)
        .collect();
    let centered: Vec<Vec<f64>> = pooled
        .iter()
        .map(|v| v.iter().zip(&mean).map(|(x, m)| x - m).collect())
        .collect();
    let count = centered.len();
    let mut gram = vec![0.; count * count];
    for i in 0..count {
        for j in 0..=i {
            let value = dot(&centered[i], &centered[j]) / (count - 1) as f64;
            gram[i * count + j] = value;
            gram[j * count + i] = value;
        }
    }
    let trace: f64 = (0..count).map(|i| gram[i * count + i]).sum();
    ensure!(trace.is_finite(), "nonfinite pooled covariance");
    if trace <= 1e-24 {
        return Ok(
            json!({"schema_version":1,"status":"degenerate_pooled_centroids",
            "room_pairs":n,"ranks":[],"interpretation":interpretation()}),
        );
    }
    let eigen = symmetric_eigen(gram, count)?;
    let mut ordered: Vec<_> = (0..count).collect();
    ordered.sort_by(|&a, &b| eigen.values[b].total_cmp(&eigen.values[a]));
    let retained: Vec<_> = ordered
        .into_iter()
        .filter(|&i| eigen.values[i] > trace * 1e-10)
        .collect();
    let max_rank = RANKS[2].min(n - 1).min(dimensions).min(retained.len());
    let ridge = trace / dimensions.min(count - 1) as f64 * RIDGE_FRACTION;
    // Reconstruct right singular vectors from the small centered Gram matrix.
    // No 768/1152-dimensional covariance factorization or BLAS runtime is needed.
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(max_rank);
    for &axis in retained.iter().take(max_rank) {
        let scale = (eigen.values[axis] * (count - 1) as f64).sqrt();
        let mut vector = vec![0.; dimensions];
        for i in 0..count {
            let factor = eigen.vectors[i * count + axis] / scale;
            for (x, &value) in vector.iter_mut().zip(&centered[i]) {
                *x += factor * value;
            }
        }
        // Remove only floating-point residuals; the fitted eigenspace is shared.
        for prior in &basis {
            let residual = dot(&vector, prior);
            for (x, p) in vector.iter_mut().zip(prior) {
                *x -= residual * p;
            }
        }
        let norm = dot(&vector, &vector).sqrt();
        ensure!(
            norm.is_finite() && norm > 0.99,
            "unstable pooled PCA reconstruction"
        );
        vector.iter_mut().for_each(|x| *x /= norm);
        basis.push(vector);
    }
    let project = |rows: &[Vec<f64>]| -> Vec<Vec<f64>> {
        rows.iter()
            .map(|v| {
                let centered: Vec<_> = v.iter().zip(&mean).map(|(x, m)| x - m).collect();
                basis.iter().map(|axis| dot(&centered, axis)).collect()
            })
            .collect()
    };
    let before = project(before);
    let after = project(after);
    let basis_bytes: Vec<_> = basis
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let indices: Vec<_> = (0..n).collect();
    let bootstrap_indices = paired_resamples(n);
    let mut ranks = Vec::new();
    for rank in RANKS {
        if rank > max_rank {
            ranks.push(json!({"rank":rank,"status":"insufficient_pooled_rank_or_rooms"}));
            continue;
        }
        let covariance_before = covariance(&before, &indices, rank);
        let covariance_after = covariance(&after, &indices, rank);
        let projected_rank = |matrix: &[f64]| -> Result<usize> {
            Ok(symmetric_eigen(matrix.to_vec(), rank)?
                .values
                .into_iter()
                .filter(|&v| v > trace * 1e-10)
                .count())
        };
        let before_rank = projected_rank(&covariance_before)?;
        let after_rank = projected_rank(&covariance_after)?;
        if before_rank < rank || after_rank < rank {
            ranks.push(json!({"rank":rank,"status":"singular_observed_cohort",
                "before_projected_rank":before_rank,"after_projected_rank":after_rank}));
            continue;
        }
        let before_logdet = log_determinant(&covariance_before, rank, ridge)?;
        let after_logdet = log_determinant(&covariance_after, rank, ridge)?;
        let ratio = (0.5 * (after_logdet - before_logdet)).exp();
        ensure!(ratio.is_finite(), "nonfinite shared PCA volume ratio");
        let axis_ratios: Vec<_> = (0..rank)
            .map(|i| (covariance_after[i * rank + i] / covariance_before[i * rank + i]).sqrt())
            .collect();
        let mut bootstrap = Vec::with_capacity(RESAMPLES);
        let (mut singular_before, mut singular_after) = (0usize, 0usize);
        for sample in &bootstrap_indices {
            let b = covariance(&before, sample, rank);
            let a = covariance(&after, sample, rank);
            singular_before += usize::from(projected_rank(&b)? < rank);
            singular_after += usize::from(projected_rank(&a)? < rank);
            let ratio = (0.5
                * (log_determinant(&a, rank, ridge)? - log_determinant(&b, rank, ridge)?))
            .exp();
            ensure!(ratio.is_finite(), "nonfinite bootstrap volume ratio");
            bootstrap.push(ratio);
        }
        bootstrap.sort_by(f64::total_cmp);
        ranks.push(json!({"rank":rank,"status":"valid",
            "sqrt_covariance_determinant_ratio":ratio,
            "before_regularized_log_determinant":before_logdet,
            "after_regularized_log_determinant":after_logdet,
            "before_projected_rank":before_rank,"after_projected_rank":after_rank,
            "per_axis_standard_deviation_ratios":axis_ratios,
            "geometric_mean_standard_deviation_ratio":ratio.powf(1. / rank as f64),
            "before_per_axis_variance":(0..rank).map(|i|covariance_before[i*rank+i]).collect::<Vec<_>>(),
            "after_per_axis_variance":(0..rank).map(|i|covariance_after[i*rank+i]).collect::<Vec<_>>(),
            "paired_bootstrap_95_percent":{"lower":percentile(&bootstrap,0.025),
                "upper":percentile(&bootstrap,0.975),"median":percentile(&bootstrap,0.5),"resamples":RESAMPLES,
                "singular_before_resamples":singular_before,"singular_after_resamples":singular_after,
                "singular_policy":"retain every draw with the same recorded numerical ridge"}}));
    }
    Ok(
        json!({"schema_version":1,"status":"computed","room_pairs":n,
        "input":"equal-room-weight unit-normalized centroids of RGB view embeddings",
        "basis":{"fit":"pooled_before_and_after_equal_room_weight","pooled_rooms":count,
            "fit_room_indices":fit_indices,"fit_policy":"all rooms up to 256 pairs; otherwise deterministic stratified index subset, independent of image/embedding values; every room contributes to projected covariance and paired bootstrap",
            "centering":"pooled_mean","dimensions":dimensions,"pooled_nonzero_rank":retained.len(),
            "retained_max_rank":max_rank,"pooled_total_variance":trace,
            "float64_basis_sha256":sha(&basis_bytes),
            "retained_pooled_variance_eigenvalues":retained.iter().take(max_rank).map(|&i|eigen.values[i]).collect::<Vec<_>>(),
            "eigenvalue_relative_trace_threshold":1e-10,"jacobi_sweeps":eigen.sweeps,
            "maximum_remaining_off_diagonal_over_trace":eigen.residual},
        "covariance":"sample_covariance_with_each_cohort_mean_and_n_minus_one_denominator",
        "ridge":{"variance":ridge,"fraction_of_pooled_variance_per_possible_axis":RIDGE_FRACTION,
            "reference_axis_count":dimensions.min(count-1),"shared_across_cohorts_ranks_and_resamples":true},
        "bootstrap":{"resamples":RESAMPLES,"seed":format!("{BOOTSTRAP_SEED:016x}"),
            "unit":"paired_room_indices","basis":"fixed_from_observed_pooled_centroids",
            "ridge":"fixed_from_observed_pooled_covariance","interval":"percentile_with_linear_interpolation"},
        "ranks":ranks,"interpretation":interpretation()}),
    )
}

fn interpretation() -> &'static str {
    "Conditional covariance-ellipsoid volume in ranks 4/8/16 of one pooled equal-room-weight PCA basis; sqrt(det(C_after + ridge I)/det(C_before + ridge I)). These rank-specific ratios are not manifold volumes, a global 10M-scene diversity estimate, photographic realism, or learning utility. Paired room bootstrap keeps the basis and tiny shared ridge fixed, so its interval excludes encoder/PCA-fit uncertainty. Singular observed cohorts are not assigned inflated ridge-driven ratios."
}

struct Eigen {
    values: Vec<f64>,
    vectors: Vec<f64>,
    sweeps: usize,
    residual: f64,
}

/// Bounded cyclic Jacobi, retaining eigenvectors for the dual PCA reconstruction.
/// The bounded fit subset (256 matched rooms) produces at most a 512-square matrix.
fn symmetric_eigen(mut matrix: Vec<f64>, n: usize) -> Result<Eigen> {
    ensure!(
        n > 0 && n <= 512 && matrix.len() == n * n && matrix.iter().all(|x| x.is_finite()),
        "invalid symmetric eigenproblem"
    );
    let trace: f64 = (0..n).map(|i| matrix[i * n + i].abs()).sum();
    let mut vectors = vec![0.; n * n];
    for i in 0..n {
        vectors[i * n + i] = 1.;
    }
    if trace == 0. {
        return Ok(Eigen {
            values: vec![0.; n],
            vectors,
            sweeps: 0,
            residual: 0.,
        });
    }
    let mut sweeps = 0;
    let mut residual = 1.;
    for sweep in 0..64 {
        for p in 0..n {
            for q in p + 1..n {
                let off = matrix[p * n + q];
                if off.abs() <= trace * 1e-14 {
                    continue;
                }
                let tau = (matrix[q * n + q] - matrix[p * n + p]) / (2. * off);
                let t = if tau == 0. {
                    1.
                } else {
                    tau.signum() / (tau.abs() + tau.hypot(1.))
                };
                let c = (1. + t * t).sqrt().recip();
                let s = c * t;
                matrix[p * n + p] -= t * off;
                matrix[q * n + q] += t * off;
                matrix[p * n + q] = 0.;
                matrix[q * n + p] = 0.;
                for k in 0..n {
                    if k != p && k != q {
                        let a = matrix[k * n + p];
                        let b = matrix[k * n + q];
                        matrix[k * n + p] = c * a - s * b;
                        matrix[p * n + k] = matrix[k * n + p];
                        matrix[k * n + q] = s * a + c * b;
                        matrix[q * n + k] = matrix[k * n + q];
                    }
                    let a = vectors[k * n + p];
                    let b = vectors[k * n + q];
                    vectors[k * n + p] = c * a - s * b;
                    vectors[k * n + q] = s * a + c * b;
                }
            }
        }
        sweeps = sweep + 1;
        let max_off = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| matrix[p * n + q].abs())
            .fold(0., f64::max);
        residual = max_off / trace;
        if residual <= 1e-12 {
            break;
        }
    }
    ensure!(
        residual <= 1e-10,
        "bounded Jacobi eigenproblem did not converge"
    );
    Ok(Eigen {
        values: (0..n).map(|i| matrix[i * n + i]).collect(),
        vectors,
        sweeps,
        residual,
    })
}

fn covariance(rows: &[Vec<f64>], indices: &[usize], rank: usize) -> Vec<f64> {
    let mut mean = vec![0.; rank];
    for &i in indices {
        for (m, &v) in mean.iter_mut().zip(&rows[i]) {
            *m += v;
        }
    }
    mean.iter_mut().for_each(|m| *m /= indices.len() as f64);
    let mut matrix = vec![0.; rank * rank];
    for &i in indices {
        for a in 0..rank {
            for b in 0..=a {
                matrix[a * rank + b] += (rows[i][a] - mean[a]) * (rows[i][b] - mean[b]);
            }
        }
    }
    for a in 0..rank {
        for b in 0..=a {
            matrix[a * rank + b] /= (indices.len() - 1) as f64;
            matrix[b * rank + a] = matrix[a * rank + b];
        }
    }
    matrix
}

/// Positive regularized determinants evaluated in log space by Cholesky.
fn log_determinant(matrix: &[f64], n: usize, ridge: f64) -> Result<f64> {
    let mut lower = vec![0.; n * n];
    let mut result = 0.;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j] + if i == j { ridge } else { 0. };
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                ensure!(
                    sum > 0. && sum.is_finite(),
                    "nonpositive regularized covariance pivot"
                );
                lower[i * n + j] = sum.sqrt();
                result += sum.ln();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    Ok(result)
}

fn paired_resamples(n: usize) -> Vec<Vec<usize>> {
    let mut state = BOOTSTRAP_SEED;
    (0..RESAMPLES)
        .map(|_| {
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as usize % n
                })
                .collect()
        })
        .collect()
}
fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * probability;
    let index = position.floor() as usize;
    sorted[index]
        + (sorted[(index + 1).min(sorted.len() - 1)] - sorted[index]) * (position - index as f64)
}

#[cfg(test)]
mod tests;
