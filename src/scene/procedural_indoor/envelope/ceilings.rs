//! Piecewise ceiling support shared by construction, camera clearance and rays.
use super::*;
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;

impl EnvelopeProgram {
    pub(super) fn sample_soffits(&mut self, scene: &IndoorManifest) {
        let Some(program) = &scene.program else {
            return;
        };
        let mut r = stream(scene.seed, 4117);
        for zone in &program.zones {
            if !r.random_bool(0.48) {
                continue;
            }
            let extent = zone.max - zone.min;
            let center = (zone.min + zone.max) * 0.5;
            let axis = usize::from(r.random_bool(0.5));
            let mut half = extent * r.random_range(0.22..0.38);
            half[axis] = r.random_range(0.18..0.65);
            let lo = center - half;
            let hi = center + half;
            let roof = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
                .map(|p| self.roof_height(scene.room_size, p))
                .into_iter()
                .fold(f32::INFINITY, f32::min);
            let height = roof - r.random_range(0.13..0.36);
            if height < 2.55
                || !self.feature_clear(scene, lo, hi, 0.2)
                || self
                    .mezzanine
                    .as_ref()
                    .is_some_and(|m| m.deck.overlaps(lo, hi) && height < m.deck.height + 2.3)
                || self.pillars.iter().any(|p| {
                    p.center.clamp(lo - Vec2::splat(0.4), hi + Vec2::splat(0.4)) == p.center
                })
            {
                continue;
            }
            self.soffits.push(FloorPatch {
                min: lo,
                max: hi,
                height,
            });
        }
    }
}

/// Add connected, offset returns on multiple eligible exterior edges. Accepted
/// outlines retain the complete shared +Z wall and at least 78% of their area.
pub(super) fn articulate(seed: u64, original: Vec<Vec2>, size: Vec3) -> Vec<Vec2> {
    let mut r = stream(seed, 4116);
    let mut result = original.clone();
    let mut accepted = 0;
    for (a, b) in polygon::edges(&original) {
        let d = b - a;
        if accepted >= 2
            || d.length() < 4.5
            || (a.y - b.y).abs() < 0.01
            || a.y.min(b.y) > size.z * 0.5 - 0.1
            || !r.random_bool(0.42)
        {
            continue;
        }
        let Some(index) = result.iter().position(|p| *p == a) else {
            continue;
        };
        let normal = Vec2::new(-d.y, d.x).normalize();
        let start = r.random_range(0.15..0.30);
        let end = r.random_range(0.55..0.79);
        let depth = r.random_range(0.3..(size.x.min(size.z) * 0.12).min(1.6));
        let bevel = r.random_range(0.03..0.10);
        let mut proposal = result.clone();
        proposal.splice(
            index + 1..index + 1,
            [
                a + d * start,
                a + d * (start + bevel) + normal * depth,
                a + d * (end - bevel) + normal * depth,
                a + d * end,
            ],
        );
        if polygon::validate(&proposal).is_ok()
            && polygon::area(&proposal) > polygon::area(&original) * 0.78
        {
            result = proposal;
            accepted += 1;
        }
    }
    result
}
