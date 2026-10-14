//! Reconstruction diagnostics from measured directed first-surface visibility.
//! No inference scores or image similarity are used to accept or discard views.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViewGraph {
    pub schema_version: u32,
    pub camera_count: usize,
    pub minimum_directed_overlap: f64,
    /// An edge requires the threshold in BOTH directions. Diagonal is false.
    pub adjacency: Vec<Vec<bool>>,
    pub components: Vec<Vec<usize>>,
    pub degree: Vec<usize>,
    pub directed_overlap: Vec<Vec<f64>>,
    pub valid_pixels: Vec<u64>,
    pub mean_directed_overlap: Option<f64>,
    /// Fraction of a source view seen by exactly k OTHER cameras. This is
    /// pixel coverage, not unique world surface area or an independence claim.
    pub peer_count_fractions: Vec<Vec<f64>>,
}

/// All rows must represent the same scene and physical/trajectory time.
/// Zero-valid views remain explicit isolated nodes rather than fabricated overlap.
pub fn view_graph(
    valid: &[u64],
    shared: &[Vec<u64>],
    cardinality: &[Vec<u64>],
    threshold: f64,
) -> Result<ViewGraph, String> {
    let n = valid.len();
    if n == 0
        || n > 16
        || !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
        || shared.len() != n
        || cardinality.len() != n
    {
        return Err(
            "view graph requires 1..=16 cameras, finite threshold in [0,1] and square counts"
                .into(),
        );
    }
    let mut overlap = vec![vec![0.; n]; n];
    let mut fractions = vec![vec![0.; n]; n];
    for i in 0..n {
        if shared[i].len() != n
            || cardinality[i].len() != n
            || shared[i][i] != 0
            || shared[i].iter().any(|&c| c > valid[i])
            || cardinality[i]
                .iter()
                .try_fold(0u64, |a, &b| a.checked_add(b))
                != Some(valid[i])
        {
            return Err(format!("invalid visibility counts for camera {i}"));
        }
        // Membership multiplicity must agree with the sum of directed counts.
        let memberships = cardinality[i]
            .iter()
            .enumerate()
            .try_fold(0u64, |sum, (k, &c)| {
                c.checked_mul(k as u64).and_then(|c| sum.checked_add(c))
            });
        if memberships.is_none()
            || memberships != shared[i].iter().try_fold(0u64, |a, &b| a.checked_add(b))
        {
            return Err(format!(
                "membership cardinality differs from peer counts for camera {i}"
            ));
        }
        if valid[i] > 0 {
            for j in 0..n {
                overlap[i][j] = shared[i][j] as f64 / valid[i] as f64;
                fractions[i][j] = cardinality[i][j] as f64 / valid[i] as f64;
            }
        }
    }
    let adjacency: Vec<Vec<bool>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    i != j
                        && valid[i] > 0
                        && valid[j] > 0
                        && shared[i][j] > 0
                        && shared[j][i] > 0
                        && overlap[i][j].min(overlap[j][i]) >= threshold
                })
                .collect()
        })
        .collect();
    let mut visited = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut group = vec![start];
        let mut at = 0;
        while at < group.len() {
            let current = group[at];
            for (peer, seen) in visited.iter_mut().enumerate() {
                if !*seen && adjacency[current][peer] {
                    *seen = true;
                    group.push(peer);
                }
            }
            at += 1;
        }
        group.sort_unstable();
        components.push(group);
    }
    let mean = (n > 1).then(|| overlap.iter().flatten().sum::<f64>() / (n * (n - 1)) as f64);
    Ok(ViewGraph {
        schema_version: 1,
        camera_count: n,
        minimum_directed_overlap: threshold,
        degree: adjacency
            .iter()
            .map(|row| row.iter().filter(|&&x| x).count())
            .collect(),
        adjacency,
        components,
        directed_overlap: overlap,
        valid_pixels: valid.to_vec(),
        mean_directed_overlap: mean,
        peer_count_fractions: fractions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn connected_chain_does_not_require_reference_star() {
        let graph = view_graph(
            &[100; 3],
            &[vec![0, 60, 0], vec![60, 0, 60], vec![0, 60, 0]],
            &[vec![40, 60, 0], vec![0, 80, 20], vec![40, 60, 0]],
            0.5,
        )
        .unwrap();
        assert_eq!(graph.components, vec![vec![0, 1, 2]]);
        assert_eq!(graph.degree, [1, 2, 1]);
        assert!(!graph.adjacency[0][2]);
    }
    #[test]
    fn asymmetric_overlap_and_empty_views_remain_disconnected() {
        let graph = view_graph(
            &[100, 1000, 0],
            &[vec![0, 60, 0], vec![60, 0, 0], vec![0, 0, 0]],
            &[vec![40, 60, 0], vec![940, 60, 0], vec![0, 0, 0]],
            0.1,
        )
        .unwrap();
        assert_eq!(graph.components.len(), 3);
        assert_eq!(graph.directed_overlap[1][0], 0.06);
        let one = view_graph(&[4], &[vec![0]], &[vec![4]], 0.).unwrap();
        assert_eq!(one.mean_directed_overlap, None);
    }
    #[test]
    fn corrupt_memberships_and_denominators_are_rejected() {
        assert!(view_graph(
            &[10; 2],
            &[vec![0, 5], vec![5, 0]],
            &[vec![5, 5], vec![5, 5]],
            f64::NAN
        )
        .is_err());
        assert!(view_graph(
            &[10; 2],
            &[vec![0, 5], vec![5, 0]],
            &[vec![4, 6], vec![5, 5]],
            0.1
        )
        .is_err());
        assert!(view_graph(&[10], &[vec![0]], &[vec![9]], 0.1).is_err());
    }
}
