//! Continuous pinboard placements, bounded by the frame and raised divider.
use super::*;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

#[derive(Clone, Debug, serde::Serialize)]
pub struct Pinup {
    pub center: Vec2,
    pub size: Vec2,
    pub yaw: f32,
    pub seed: u64,
}
pub fn placements(o: &IndoorObject) -> Vec<Pinup> {
    let p = parameters(o);
    let s = o.size;
    let frame = s.x.min(s.y) * p.board_frame_fraction;
    let mut r = stream(o.seed, 5730);
    let inset = Vec2::new(r.random_range(0.0..0.10), r.random_range(0.0..0.12));
    let bounds = (
        Vec2::new(-s.x * 0.5 + frame * 1.8, frame * 1.8) + inset * s.xy(),
        Vec2::new(s.x * 0.5 - frame * 1.8, s.y - frame * 1.8)
            - Vec2::new(r.random_range(0.0..0.10), r.random_range(0.0..0.12)) * s.xy(),
    );
    let area = (bounds.1 - bounds.0).element_product();
    let divider = p
        .board_divider_fraction
        .map(|f| (f - 0.5) * (s.x - frame * 2.));
    let coverage = r.random_range(0.22..0.48);
    let pattern = SurfacePattern::sample(&mut r);
    let heading = r.random_range(-0.16..0.16);
    let spread = r.random_range(0.04..0.38);
    let mut occupied: Vec<Footprint> = Vec::with_capacity(p.note_count);
    let mut notes = Vec::with_capacity(p.note_count);
    let mut used = 0.;
    for _ in 0..96 {
        if notes.len() >= p.note_count {
            break;
        }
        let size = Vec2::new(
            s.x * r.random_range(0.09..0.42),
            s.y * r.random_range(0.09..0.43),
        );
        if used + size.element_product() > area * coverage {
            continue;
        }
        let yaw = heading + r.random_range(-spread..spread);
        let Some(proposal) = pattern.propose(bounds, size, yaw, frame * 0.15, &mut r) else {
            continue;
        };
        if divider.is_some_and(|x| (proposal.center.x - x).abs() < proposal.half.x + frame * 0.32)
            || occupied
                .iter()
                .any(|old| old.overlaps(proposal, frame * 0.15))
        {
            continue;
        }
        occupied.push(proposal);
        used += size.element_product();
        notes.push(Pinup {
            center: proposal.center,
            size,
            yaw,
            seed: r.random(),
        });
    }
    notes
}
