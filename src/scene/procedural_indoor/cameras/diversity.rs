//! Group geometry at synchronized times, independent of the visibility proxy.
use super::{multiview::MultiViewSettings, IndoorCamera, IndoorManifest};
use bevy::prelude::*;
use serde::Serialize;

pub const PATH_SAMPLES: usize = 33;

pub(super) struct Track(pub [Vec3; PATH_SAMPLES]);
impl Track {
    pub fn new(camera: &IndoorCamera) -> Self {
        let mut path = camera.runtime_trajectory();
        Self(std::array::from_fn(|i| {
            path.sample(i as f32 / (PATH_SAMPLES - 1) as f32)
                .translation
        }))
    }
}

/// Worst synchronized geometry across 33 uniformly spaced trajectory samples.
/// Spread is the horizontal minor/major standard-deviation ratio (0=line, 1=isotropic).
/// Relative motion is RMS displacement difference after removing each start,
/// divided by the larger RMS travel. It is absent for an entirely static group.
#[derive(Debug, Clone, Serialize)]
pub struct CameraGroupGeometry {
    pub min_pairwise_baseline_m: f32,
    pub max_reference_baseline_m: f32,
    pub min_horizontal_spread: Option<f32>,
    pub min_relative_motion: Option<f32>,
}

fn horizontal_spread(points: impl Iterator<Item = Vec3>) -> f32 {
    // Subtract the first camera before accumulating to keep the covariance stable
    // under scene translation, including cameras very close to each other.
    let mut points = points.map(|p| p.xz());
    let Some(origin) = points.next() else {
        return 0.0;
    };
    let mut center = Vec2::ZERO;
    let mut count = 1.0;
    let (mut xx, mut xz, mut zz) = (0.0, 0.0, 0.0);
    for point in points {
        let d = point - origin - center;
        count += 1.0;
        center += d / count;
        let updated = point - origin - center;
        xx += d.x * updated.x;
        xz += d.x * updated.y;
        zz += d.y * updated.y;
    }
    let major = 0.5 * (xx + zz + ((xx - zz).powi(2) + 4.0 * xz * xz).sqrt());
    ((xx * zz - xz * xz).max(0.0).sqrt() / major.max(1e-12)).clamp(0.0, 1.0)
}

fn relative_motion(a: &Track, b: &Track) -> Option<f32> {
    let (mut residual, mut travel_a, mut travel_b) = (0.0, 0.0, 0.0);
    for (pa, pb) in a.0.iter().zip(b.0) {
        let da = *pa - a.0[0];
        let db = pb - b.0[0];
        residual += (da - db).length_squared();
        travel_a += da.length_squared();
        travel_b += db.length_squared();
    }
    let travel = travel_a.max(travel_b);
    (travel > 1e-8).then(|| (residual / travel).sqrt())
}

impl CameraGroupGeometry {
    pub fn accepts(&self, policy: &MultiViewSettings) -> bool {
        self.min_pairwise_baseline_m + 1e-5 >= policy.min_baseline
            && self.max_reference_baseline_m <= policy.max_baseline + 1e-5
            && self
                .min_horizontal_spread
                .is_none_or(|s| s + 1e-5 >= policy.min_spread)
            && self
                .min_relative_motion
                .is_none_or(|s| s + 1e-5 >= 0.15 * policy.trajectory_variation)
    }
}

pub(super) fn geometry(tracks: &[Track], extra: Option<&Track>) -> Option<CameraGroupGeometry> {
    let tracks: Vec<_> = tracks.iter().chain(extra).collect();
    if tracks.len() < 2 {
        return None;
    }
    let mut result = CameraGroupGeometry {
        min_pairwise_baseline_m: f32::INFINITY,
        max_reference_baseline_m: 0.0,
        min_horizontal_spread: None,
        min_relative_motion: None,
    };
    for (i, a) in tracks.iter().enumerate() {
        for b in &tracks[i + 1..] {
            for (pa, pb) in a.0.iter().zip(b.0) {
                let distance = pa.distance(pb);
                result.min_pairwise_baseline_m = result.min_pairwise_baseline_m.min(distance);
                if i == 0 {
                    result.max_reference_baseline_m = result.max_reference_baseline_m.max(distance);
                }
            }
            if let Some(relative) = relative_motion(a, b) {
                result.min_relative_motion =
                    Some(result.min_relative_motion.unwrap_or(relative).min(relative));
            }
        }
    }
    if tracks.len() >= 3 {
        result.min_horizontal_spread = Some(
            (0..PATH_SAMPLES)
                .map(|i| horizontal_spread(tracks.iter().map(|t| t.0[i])))
                .fold(1.0, f32::min),
        );
    }
    Some(result)
}

impl IndoorManifest {
    pub fn camera_group_geometry(&self) -> Option<CameraGroupGeometry> {
        geometry(
            &self.cameras.iter().map(Track::new).collect::<Vec<_>>(),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(start: Vec3, end: Vec3) -> Track {
        Track(std::array::from_fn(|i| start.lerp(end, i as f32 / 32.0)))
    }

    #[test]
    fn distinguishes_collinear_groups_rigid_paths_and_crossing_paths() {
        let policy = MultiViewSettings::default();
        let group: Vec<_> = (0..4)
            .map(|i| line(Vec3::X * i as f32 * 0.6, Vec3::X * i as f32 * 0.6 + Vec3::Z))
            .collect();
        let stats = geometry(&group, None).unwrap();
        assert_eq!(stats.min_horizontal_spread, Some(0.0));
        assert_eq!(stats.min_relative_motion, Some(0.0));
        assert!(!stats.accepts(&policy));
        let crossing = [line(Vec3::ZERO, Vec3::X), line(Vec3::X, Vec3::ZERO)];
        assert_eq!(
            geometry(&crossing, None).unwrap().min_pairwise_baseline_m,
            0.0
        );
        assert!(!geometry(&crossing, None).unwrap().accepts(&policy));
        let static_square: Vec<_> = [Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::X + Vec3::Z]
            .into_iter()
            .map(|p| line(p, p))
            .collect();
        let stats = geometry(&static_square, None).unwrap();
        assert!((stats.min_horizontal_spread.unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(stats.min_relative_motion, None);
        assert!(stats.accepts(&policy));
        assert!(geometry(&[], None).is_none());
        assert!(geometry(&static_square[..1], None).is_none());
    }

    #[test]
    fn spread_is_translation_rotation_and_scale_invariant() {
        let points = [Vec3::ZERO, Vec3::X, Vec3::Z * 0.5];
        let expected = horizontal_spread(points.into_iter());
        let rotation = Quat::from_rotation_y(0.7);
        let transformed = points.map(|p| rotation * p * 2.0 + Vec3::new(13.0, 2.0, -8.0));
        assert!((horizontal_spread(transformed.into_iter()) - expected).abs() < 1e-5);
    }
}
