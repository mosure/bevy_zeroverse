//! Count-aware proposal ranking uses the existing geometric oracle, never an
//! image model. Scores are preferences; they cannot bypass acceptance gates.
use super::{coverage::VisibilityView, IndoorCamera, IndoorManifest};
use bevy::prelude::*;
use rand::Rng;
use std::collections::BTreeSet;

pub(super) struct ViewSummary {
    origin: Vec3,
    forward: Vec3,
    cells: BTreeSet<[i32; 3]>,
}
impl ViewSummary {
    pub fn new(camera: &IndoorCamera, view: &VisibilityView, scene: &IndoorManifest) -> Self {
        let pose = camera.transform_at(0.0);
        let region = scene.primary_region();
        let cell = ((region.max - region.min).length() / 24.).max(0.15);
        Self {
            origin: pose.translation,
            forward: pose.rotation * Vec3::NEG_Z,
            cells: view
                .surfaces()
                .filter(|p| region.contains_projection_with_tolerance(p.xz(), 1e-4))
                .map(|p| (p / cell).floor().as_ivec3().to_array())
                .collect(),
        }
    }

    pub fn score(&self, peers: &[Self], count: usize, scene: &IndoorManifest) -> f32 {
        // With no peers, distance/angle/novelty all saturated to the same score.
        // Prefer anchors exposing more distinct primary-room surfaces instead
        // of an arbitrary first view into a disconnected neighboring room.
        if peers.is_empty() {
            return self.cells.len() as f32;
        }
        let (lo, hi) = scene.primary_room_bounds();
        let spacing = ((hi - lo).length() / (count.max(2) as f32).sqrt()).max(0.5);
        let nearest = peers
            .iter()
            .map(|p| self.origin.distance(p.origin))
            .fold(f32::INFINITY, f32::min);
        let angle = peers
            .iter()
            .map(|p| self.forward.dot(p.forward).clamp(-1., 1.).acos())
            .fold(f32::INFINITY, f32::min);
        let novel = self
            .cells
            .iter()
            .filter(|cell| peers.iter().all(|p| !p.cells.contains(*cell)))
            .count() as f32
            / self.cells.len().max(1) as f32;
        0.45 * (nearest / spacing).min(1.) + 0.25 * (angle / 0.9).min(1.) + 0.30 * novel
    }
}

pub(super) fn retarget(
    scene: &IndoorManifest,
    camera: &mut IndoorCamera,
    reference_views: &[VisibilityView],
    variation: f32,
    count: usize,
    rng: &mut impl Rng,
) {
    if variation <= 0. {
        return;
    }
    let probability = variation * (1. - 0.65 / count.max(2) as f32);
    if !rng.random_bool(probability as f64) {
        return;
    }
    let region = scene.primary_region();
    let points: Vec<_> = reference_views[0]
        .surfaces()
        .filter(|p| region.contains_projection(p.xz(), 0.) && p.distance(camera.start) >= 1.6)
        .collect();
    if points.is_empty() {
        return;
    }
    let surface = points[rng.random_range(0..points.len())];
    let aim = camera.target.lerp(surface, variation);
    let delta = aim - camera.target;
    camera.target = aim;
    if let Some(motion) = &mut camera.motion {
        motion.target_end += delta;
    }
}

/// Reuse the full proposal budget, with a small local window after the first
/// feasible view. Narrow rigs and explicit older policies retain first-fit cost.
pub(super) fn pool_size(variation: f32, count: usize) -> usize {
    if variation <= 0. {
        1
    } else {
        2 + usize::from(count > 4) + usize::from(count > 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn count_aware_score_rewards_spacing_angle_and_new_surfaces() {
        let mut scene =
            IndoorManifest::generate(1, super::super::super::layout::IndoorLayout::Mixed, 0.65, 0)
                .unwrap();
        scene.program = None;
        scene.room_size = Vec3::new(10., 3., 10.);
        let peers = [ViewSummary {
            origin: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            cells: BTreeSet::from([[0, 0, 0]]),
        }];
        let repeat = ViewSummary {
            origin: Vec3::X * 0.1,
            forward: Vec3::NEG_Z,
            cells: BTreeSet::from([[0, 0, 0]]),
        };
        let diverse = ViewSummary {
            origin: Vec3::X * 3.,
            forward: Vec3::X,
            cells: BTreeSet::from([[4, 0, 0], [5, 1, 0]]),
        };
        assert!(diverse.score(&peers, 12, &scene) > repeat.score(&peers, 12, &scene) + 0.5);
        assert!(diverse.score(&[], 8, &scene) > repeat.score(&[], 8, &scene));
        assert_eq!(pool_size(0., 12), 1);
        assert!(pool_size(1., 12) > pool_size(1., 2));
    }
}
