//! Exact bounded RBF sample diversity. This is an uncentered PSD kernel, not
//! covariance rank, occupancy support or an extrapolated population estimate.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KernelConfig {
    /// RBF length scale on unit-normalized, equally weighted room centroids.
    pub bandwidth: f64,
    /// Hard resource ceiling; no post-hoc sample selection or silent subsampling.
    pub max_rooms: usize,
}
impl KernelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.bandwidth.is_finite() && self.bandwidth > 0.,
            "positive finite kernel bandwidth required"
        );
        ensure!(
            (1..=512).contains(&self.max_rooms),
            "exact kernel budget must be 1..512 rooms"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KernelSummary {
    pub rooms: usize,
    pub vendi: f64,
    pub participation: f64,
    pub eigenvalues: Vec<f64>,
    pub off_diagonal_residual: f64,
    pub nearest_cosine_distance: Vec<Option<f64>>,
    /// Indices into the ordered room list; null for a singleton cohort.
    pub nearest_room: Vec<Option<usize>>,
}

pub fn analyze(vectors: &[Vec<f64>], config: &KernelConfig) -> Result<KernelSummary> {
    config.validate()?;
    ensure!(
        !vectors.is_empty() && vectors.len() <= config.max_rooms,
        "kernel cohort exceeds its declared budget or is empty"
    );
    let n = vectors.len();
    let d = vectors[0].len();
    ensure!((1..=4096).contains(&d), "unsupported embedding dimension");
    for row in vectors {
        ensure!(
            row.len() == d && row.iter().all(|x| x.is_finite()),
            "nonfinite or inconsistent embedding"
        );
        ensure!(
            (row.iter().map(|v| v * v).sum::<f64>() - 1.).abs() < 1e-4,
            "kernel embeddings must be unit normalized"
        );
    }
    let mut gram = vec![vec![0.; n]; n];
    let mut nearest = vec![None; n];
    let mut distances = vec![2.; n];
    for i in 0..n {
        gram[i][i] = 1. / n as f64;
        for j in 0..i {
            let squared = vectors[i]
                .iter()
                .zip(&vectors[j])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            let cosine = (squared / 2.).clamp(0., 2.);
            if nearest[i].is_none() || cosine < distances[i] {
                distances[i] = cosine;
                nearest[i] = Some(j);
            }
            if nearest[j].is_none() || cosine < distances[j] {
                distances[j] = cosine;
                nearest[j] = Some(i);
            }
            let entry = (-squared / (2. * config.bandwidth.powi(2))).exp() / n as f64;
            gram[i][j] = entry;
            gram[j][i] = entry;
        }
    }
    let participation = 1. / gram.iter().flatten().map(|v| v * v).sum::<f64>();
    // Symmetric cyclic Jacobi, with convergence checked from the actual remaining
    // off-diagonal Frobenius norm rather than assuming a sweep count suffices.
    let mut residual = f64::INFINITY;
    for _ in 0..64 {
        for p in 0..n {
            for q in p + 1..n {
                let off = gram[p][q];
                if off.abs() <= 1e-15 {
                    continue;
                }
                let tau = (gram[q][q] - gram[p][p]) / (2. * off);
                let t = if tau >= 0. { 1. } else { -1. } / (tau.abs() + tau.hypot(1.));
                let c = 1. / (1. + t * t).sqrt();
                let s = c * t;
                gram[p][p] -= t * off;
                gram[q][q] += t * off;
                gram[p][q] = 0.;
                gram[q][p] = 0.;
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let x = gram[k][p];
                    let y = gram[k][q];
                    gram[k][p] = c * x - s * y;
                    gram[p][k] = gram[k][p];
                    gram[k][q] = s * x + c * y;
                    gram[q][k] = gram[k][q];
                }
            }
        }
        residual = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| gram[i][j].powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            .sqrt();
        if residual <= 1e-10 {
            break;
        }
    }
    ensure!(residual <= 1e-10, "kernel eigensolver did not converge");
    let mut values: Vec<_> = (0..n).map(|i| gram[i][i]).collect();
    ensure!(
        values.iter().all(|&x| x >= -1e-9) && (values.iter().sum::<f64>() - 1.).abs() < 1e-8,
        "kernel is not a trace-one PSD matrix"
    );
    for v in &mut values {
        *v = v.max(0.);
    }
    let sum: f64 = values.iter().sum();
    for v in &mut values {
        *v /= sum;
    }
    values.sort_by(|a, b| b.total_cmp(a));
    let vendi = (-values
        .iter()
        .filter(|&&v| v > 0.)
        .map(|v| v * v.ln())
        .sum::<f64>())
    .exp();
    ensure!(
        vendi.is_finite() && vendi >= 1. - 1e-8 && vendi <= n as f64 + 1e-8,
        "invalid kernel diversity"
    );
    Ok(KernelSummary {
        rooms: n,
        vendi,
        participation,
        eigenvalues: values,
        off_diagonal_residual: residual,
        nearest_cosine_distance: distances
            .into_iter()
            .zip(&nearest)
            .map(|(d, peer)| peer.map(|_| d))
            .collect(),
        nearest_room: nearest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg(bandwidth: f64) -> KernelConfig {
        KernelConfig {
            bandwidth,
            max_rooms: 512,
        }
    }
    #[test]
    fn collapsed_and_separated_populations_have_known_diversity() {
        let collapsed = analyze(&vec![vec![1., 0.]; 32], &cfg(0.4)).unwrap();
        assert!((collapsed.vendi - 1.).abs() < 1e-8);
        assert!((collapsed.participation - 1.).abs() < 1e-8);
        let basis: Vec<_> = (0..16)
            .map(|i| (0..16).map(|j| f64::from(i == j)).collect())
            .collect();
        assert!((analyze(&basis, &cfg(0.05)).unwrap().vendi - 16.).abs() < 1e-8);
    }
    #[test]
    fn repeated_modes_do_not_multiply_diversity_and_order_is_irrelevant() {
        let mut rows = vec![vec![1., 0.]; 3];
        rows.push(vec![0., 1.]);
        let expected = (-0.75_f64 * 0.75_f64.ln() - 0.25_f64 * 0.25_f64.ln()).exp();
        let result = analyze(&rows, &cfg(0.05)).unwrap();
        assert!((result.vendi - expected).abs() < 1e-8);
        rows.reverse();
        assert!((analyze(&rows, &cfg(0.05)).unwrap().vendi - expected).abs() < 1e-8);
        let duplicated = rows.iter().chain(&rows).cloned().collect::<Vec<_>>();
        assert!((analyze(&duplicated, &cfg(0.05)).unwrap().vendi - expected).abs() < 1e-8);
    }
    #[test]
    fn two_point_spectrum_matches_analytic_rbf_and_rejects_invalid_inputs() {
        let single = analyze(&[vec![1., 0.]], &cfg(1.)).unwrap();
        assert_eq!(single.nearest_cosine_distance, vec![None]);
        assert_eq!(single.nearest_room, vec![None]);
        let result = analyze(&[vec![1., 0.], vec![0., 1.]], &cfg(1.)).unwrap();
        let k = (-1_f64).exp();
        assert!((result.eigenvalues[0] - (1. + k) / 2.).abs() < 1e-12);
        assert!(analyze(&[vec![1., 0.]], &cfg(f64::NAN)).is_err());
        assert!(analyze(&[vec![0., 0.]], &cfg(1.)).is_err());
        assert!(analyze(&[vec![1., f64::NAN]], &cfg(1.)).is_err());
    }
}
