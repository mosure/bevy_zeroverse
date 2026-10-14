//! Measured geometry diagnostics, outside the generation hot path. All membership
//! comes from rendered co-visibility; placement proxies remain separate.
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{sample::View, scene::procedural_indoor::layout::IndoorManifest};
use serde::Serialize;

#[derive(Serialize)]
pub struct PairGeometry {
    pub cameras: [usize; 2],
    pub baseline_metres: f32,
    pub primary_shared_source_pixels: u64,
    /// Pixel-weighted source-view angles, quantized into half-degree bins.
    /// Null for no shared primary geometry; never replaced by a zero-angle pair.
    pub angle_degrees_p05_p50_p95: Option<[f32; 3]>,
}

#[derive(Serialize)]
pub struct PrimaryGeometry {
    pub membership: &'static str,
    pub graph: bevy_zeroverse_capture::reconstruction::ViewGraph,
    pub pairs: Vec<PairGeometry>,
}

pub fn measure(
    views: &[View],
    scene: &IndoorManifest,
    aabb: [[f32; 3]; 2],
) -> Result<PrimaryGeometry> {
    let n = views.len();
    ensure!((1..=16).contains(&n), "invalid reconstruction group size");
    let origins: Vec<_> = views
        .iter()
        .map(|v| {
            Mat4::from_cols_array_2d(&v.world_from_view)
                .w_axis
                .truncate()
        })
        .collect();
    let mut valid = vec![0u64; n];
    let mut shared = vec![vec![0u64; n]; n];
    let mut cardinality = vec![vec![0u64; n]; n];
    let mut angles = vec![vec![[0u64; 360]; n]; n];
    let low = Vec3::from_array(aabb[0]);
    let size = Vec3::from_array(aabb[1]) - low;
    let inverse = Quat::from_rotation_y(-scene.world_yaw);
    for (source, view) in views.iter().enumerate() {
        ensure!(
            view.position.len() == view.co_visibility.len()
                && view.position.len().is_multiple_of(16),
            "reconstruction position/membership dimensions differ"
        );
        for (p, m) in view
            .position
            .as_chunks::<16>()
            .0
            .iter()
            .zip(view.co_visibility.as_chunks::<16>().0)
        {
            let m: [f32; 4] = bytemuck::pod_read_unaligned(m);
            if m[2] != 1. {
                continue;
            }
            let p: [f32; 4] = bytemuck::pod_read_unaligned(p);
            let world = Vec3::new(p[0], p[1], p[2]) * size + low;
            let local = inverse * world;
            if !scene.in_primary_room(local, 0.)
                || local.y < -0.30
                || local.y > scene.room_size.y + 0.03
            {
                continue;
            }
            let mask = m[0] as u16;
            valid[source] += 1;
            cardinality[source][mask.count_ones() as usize] += 1;
            for peer in 0..n {
                if mask & (1 << peer) == 0 {
                    continue;
                }
                shared[source][peer] += 1;
                if source < peer {
                    let a = world - origins[source];
                    let b = world - origins[peer];
                    let angle = a.cross(b).length().atan2(a.dot(b)).to_degrees();
                    ensure!(angle.is_finite(), "nonfinite rendered triangulation angle");
                    angles[source][peer][((angle * 2.).floor() as usize).min(359)] += 1;
                }
            }
        }
    }
    let pairs = (0..n)
        .flat_map(|source| (source + 1..n).map(move |peer| (source, peer)))
        .map(|(source, peer)| PairGeometry {
            cameras: [source, peer],
            baseline_metres: origins[source].distance(origins[peer]),
            primary_shared_source_pixels: shared[source][peer],
            angle_degrees_p05_p50_p95: percentiles(&angles[source][peer]),
        })
        .collect();
    Ok(PrimaryGeometry {
        membership: "Rendered point inside the exact primary X/Z region, -0.30m <= local height <= room roof bound +0.03m; camera-image pixel weighting, no duplicate-surface correction. Boundary pixels can be excluded by finite position precision.",
        graph: bevy_zeroverse_capture::reconstruction::view_graph(&valid, &shared, &cardinality, 0.10).map_err(anyhow::Error::msg)?,
        pairs,
    })
}

fn percentiles(bins: &[u64; 360]) -> Option<[f32; 3]> {
    let count: u64 = bins.iter().sum();
    (count > 0).then(|| {
        [0.05, 0.5, 0.95].map(|q| {
            let target = ((count - 1) as f64 * q).round() as u64;
            let mut cumulative = 0;
            let i = bins
                .iter()
                .position(|&n| {
                    cumulative += n;
                    cumulative > target
                })
                .unwrap();
            (i as f32 + 0.5) * 0.5
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_correspondences_are_not_zero_angle_observations() {
        let mut bins = [0; 360];
        assert_eq!(percentiles(&bins), None);
        bins[60] = 100;
        assert_eq!(percentiles(&bins), Some([30.25; 3]));
    }
}
