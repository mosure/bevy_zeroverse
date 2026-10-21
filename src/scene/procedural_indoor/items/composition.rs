//! Area-aware loose belongings: unequal task clusters, a scattered component,
//! and a clear working patch. Proposals still pass the scene's physical gate.
use super::*;
use crate::scene::procedural_indoor::placement::rotated_half;
use rand_chacha::ChaCha8Rng;
use std::f32::consts::PI;

#[derive(Debug, Clone, Serialize)]
pub struct ItemCluster {
    /// Metres in the supporting object's local X/Z plane.
    pub center: Vec2,
    pub spread: Vec2,
    pub heading: f32,
    pub weight: f32,
    pub task: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClutterProgram {
    /// Total supported objects, including equipment placed earlier.
    pub target_items: usize,
    /// Sum of object footprint areas / rectangular support area, not visibility.
    pub target_coverage: f32,
    pub max_attempts: usize,
    pub clusters: Vec<ItemCluster>,
    pub clear_center: Vec2,
    pub clear_half: Vec2,
    pub scatter_fraction: f32,
}

pub fn parameters(scene: &IndoorManifest, support: &IndoorObject) -> ClutterProgram {
    let p = super::parameters(scene, support);
    let mut r = stream(support.seed, 5511);
    let area = support.size.x * support.size.z;
    let amount = scene.domain().map_or(0.5, |d| d.clutter) * scene.density * p.occupancy;
    let target_items = (area * amount * r.random_range(12.0..22.0))
        .round()
        .clamp(0., 56.) as usize;
    let clusters = (0..((area * r.random_range(0.8..1.6)).ceil() as usize).clamp(1, 7))
        .map(|_| {
            let mut choice = r.random::<f32>();
            let task = p
                .task_weights
                .iter()
                .position(|w| {
                    choice -= w;
                    choice <= 0.
                })
                .unwrap_or(5);
            ItemCluster {
                center: support.size.xz()
                    * Vec2::new(r.random_range(-0.34..0.34), r.random_range(-0.34..0.34)),
                spread: Vec2::new(r.random_range(0.13..0.50), r.random_range(0.13..0.50)),
                heading: r.random_range(-PI..PI),
                weight: r.random_range(0.15..1.0),
                task,
            }
        })
        .collect();
    ClutterProgram {
        target_items,
        target_coverage: (0.10 + amount * r.random_range(0.25..0.50)).min(0.55),
        max_attempts: (target_items * 6).min(288),
        clusters,
        clear_center: support.size.xz()
            * Vec2::new(r.random_range(-0.23..0.23), r.random_range(-0.23..0.23)),
        clear_half: (support.size.xz()
            * Vec2::new(r.random_range(0.10..0.20), r.random_range(0.10..0.20)))
        .min(Vec2::splat(0.38)),
        scatter_fraction: r.random_range(0.20..0.62),
    }
}

impl ClutterProgram {
    pub(crate) fn cluster(&self, r: &mut ChaCha8Rng) -> &ItemCluster {
        let mut value = r.random::<f32>() * self.clusters.iter().map(|c| c.weight).sum::<f32>();
        self.clusters
            .iter()
            .find(|c| {
                value -= c.weight;
                value <= 0.
            })
            .unwrap_or(&self.clusters[0])
    }

    pub(crate) fn proposal(
        &self,
        c: &ItemCluster,
        support: &IndoorObject,
        size: Vec3,
        aligned: f32,
        r: &mut ChaCha8Rng,
    ) -> Option<(Vec3, f32)> {
        let yaw = if r.random_bool(aligned as f64) {
            c.heading + r.random_range(-0.70..0.70)
        } else {
            r.random_range(-PI..PI)
        };
        let half = rotated_half(size.xz(), yaw);
        let limit = support.size.xz() * 0.5 - half - Vec2::splat(0.018);
        if limit.cmple(Vec2::ZERO).any() {
            return None;
        }
        let center = if r.random_bool(self.scatter_fraction as f64) {
            limit * Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0))
        } else {
            // A triangular lobe has no hard edge density spike from clamping.
            c.center
                + c.spread
                    * Vec2::new(
                        r.random::<f32>() - r.random::<f32>(),
                        r.random::<f32>() - r.random::<f32>(),
                    )
        };
        if center.abs().cmpgt(limit).any()
            || (center - self.clear_center)
                .abs()
                .cmplt(self.clear_half + half)
                .all()
        {
            return None;
        }
        Some((Vec3::new(center.x, 0., center.y), yaw))
    }
}
