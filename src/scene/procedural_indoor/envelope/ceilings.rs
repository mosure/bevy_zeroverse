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
                || !self.soffit_headroom_clear(lo, hi, height)
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
        self.sample_trays(scene);
    }

    /// A lowered perimeter around an uncut higher field. The existing slab
    /// union builds the returns and removes internal faces; every height query
    /// (camera, furniture, rays and voxel export) sees the same raised section.
    fn sample_trays(&mut self, scene: &IndoorManifest) {
        let Some(program) = &scene.program else {
            return;
        };
        let mut r = stream(scene.seed, 7593);
        for zone in &program.zones {
            if self.soffits.len() > 8 || !r.random_bool(0.62) {
                continue;
            }
            let center = (zone.min + zone.max) * 0.5;
            let half = (zone.max - zone.min) * r.random_range(0.24..0.38);
            if half.min_element() < 1.0 {
                continue;
            }
            let lo = center - half;
            let hi = center + half;
            if !self.feature_clear(scene, lo, hi, 0.25)
                || self.soffits.iter().any(|p| p.overlaps(lo, hi))
                || self
                    .mezzanine
                    .as_ref()
                    .is_some_and(|m| m.deck.overlaps(lo, hi))
                || self.pillars.iter().any(|p| {
                    p.center
                        .clamp(lo - Vec2::splat(p.radius), hi + Vec2::splat(p.radius))
                        == p.center
                })
            {
                continue;
            }
            let roof = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
                .map(|p| self.roof_height(scene.room_size, p))
                .into_iter()
                .fold(f32::INFINITY, f32::min);
            let height = roof - r.random_range(0.14..0.42);
            if height < 2.62 || !self.soffit_headroom_clear(lo, hi, height) {
                continue;
            }
            let band = half.min_element() * r.random_range(0.15..0.28);
            // Four non-overlapping strips, with shared edges built once by the
            // ceiling union. Continuous rim width/depth/aspect vary by zone.
            for (min, max) in [
                (lo, Vec2::new(hi.x, lo.y + band)),
                (Vec2::new(lo.x, hi.y - band), hi),
                (
                    Vec2::new(lo.x, lo.y + band),
                    Vec2::new(lo.x + band, hi.y - band),
                ),
                (
                    Vec2::new(hi.x - band, lo.y + band),
                    Vec2::new(hi.x, hi.y - band),
                ),
            ] {
                self.soffits.push(FloorPatch { min, max, height });
            }
        }
    }

    /// Lowered ceilings must clear every walking surface below them, including
    /// the stair flight and its landings. A check against the deck alone misses
    /// beams immediately in front of the upper landing. Use actual overlapping
    /// tread heights so lower steps still admit realistic ceiling articulation.
    fn soffit_headroom_clear(&self, lo: Vec2, hi: Vec2, height: f32) -> bool {
        self.mezzanine.as_ref().is_none_or(|m| {
            m.deck
                .overlaps(lo, hi)
                .then_some(m.deck.height)
                .into_iter()
                .chain(m.stair_highest_in(lo, hi))
                .all(|floor| height >= floor + 2.3)
        })
    }
}

#[cfg(test)]
#[test]
fn lowered_ceilings_clear_stairs_without_removing_the_mezzanine() {
    use crate::scene::procedural_indoor::{
        layout::IndoorLayout,
        validation::{validate_geometry, validate_layout},
    };
    let scene =
        IndoorManifest::generate_with_humans(66_242_909, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let e = scene.envelope.as_ref().unwrap();
    let m = e
        .mezzanine
        .as_ref()
        .expect("retain the generated mezzanine");
    validate_layout(&scene).unwrap();
    validate_geometry(&scene).unwrap();
    // This footprint intersects the top steps but not the deck. The previous
    // deck-only admission allowed a 4.89021 m soffit over a 2.90788 m tread.
    let lo = Vec2::new(-2.6564257, -0.59188634);
    let hi = Vec2::new(2.6564257, 0.59188634);
    assert!(!m.deck.overlaps(lo, hi));
    assert!(!e.soffit_headroom_clear(lo, hi, 4.89021));
    assert!(e.soffit_headroom_clear(lo, hi, m.deck.height + 2.31));
    for (a, b) in m.treads() {
        assert!(!e.soffit_headroom_clear(
            a.xz() + Vec2::splat(1e-4),
            b.xz() - Vec2::splat(1e-4),
            b.y + 2.29
        ));
        assert!(e.soffit_headroom_clear(
            a.xz() + Vec2::splat(1e-4),
            b.xz() - Vec2::splat(1e-4),
            b.y + 2.31
        ));
    }
    assert_eq!(
        scene,
        IndoorManifest::generate_with_humans(66_242_909, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap()
    );
}

#[cfg(test)]
#[test]
fn raised_fields_have_four_connected_returns_and_match_the_roof_oracle() {
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    let mut accepted = 0;
    for seed in 0..64 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut envelope = scene.envelope.clone().unwrap();
        envelope.soffits.clear();
        envelope.sample_trays(&scene);
        if envelope.soffits.is_empty() {
            continue;
        }
        accepted += 1;
        envelope.validate(&scene).unwrap();
        assert_eq!(envelope.soffits.len() % 4, 0);
        for ring in envelope.soffits.as_chunks::<4>().0 {
            let center = (ring[0].min + ring[1].max) * 0.5;
            assert!(envelope.ceiling_height(scene.room_size, center) > ring[0].height + 0.13);
            for p in ring {
                assert_eq!(
                    envelope.ceiling_height(scene.room_size, (p.min + p.max) * 0.5),
                    p.height
                );
            }
        }
        // Actual slab construction is also covered by the existing ray/roof
        // regression; this test guards the sampling topology and finite bounds.
    }
    assert!(
        accepted >= 8,
        "raised-field population too sparse: {accepted}"
    );
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
