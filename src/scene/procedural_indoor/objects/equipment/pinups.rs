//! Continuous pinboard placements, bounded by the frame and raised divider.
use super::*;
pub(super) mod sheet;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

#[derive(Clone, Debug, serde::Serialize)]
pub struct Pinup {
    pub center: Vec2,
    pub size: Vec2,
    pub yaw: f32,
    pub seed: u64,
}

/// Activity and stationery are drawn independently of the finish and wall size.
/// Sizes stay in metres; a sheet that cannot fit leaves room for a smaller item.
struct Posting {
    coverage: f32,
    landscape: f64,
    slip_share: f32,
}

pub(super) fn requested_count(o: &IndoorObject) -> usize {
    let activity = stream(o.seed, u64::from_le_bytes(*b"Postings")).random::<f32>();
    if activity < 0.065 {
        0
    } else {
        (1. + activity.powf(1.7) * 18.).round() as usize
    }
}

impl Posting {
    fn sample(o: &IndoorObject, count: usize) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"PostForm"));
        Self {
            coverage: 0.14 + 0.54 * (count as f32 / 19.).sqrt(),
            landscape: r.random_range(0.12..0.60),
            slip_share: r.random_range(0.12..0.56),
        }
    }

    fn sheet_size(&self, r: &mut impl Rng) -> Vec2 {
        let format = r.random::<f32>();
        let size = if format < self.slip_share {
            let width = r.random_range(0.065..0.142);
            Vec2::new(width, width * r.random_range(0.59..1.12))
        } else if format > 0.88 {
            Vec2::new(0.297, 0.420)
        } else {
            // Small flyers, full-page notices and photographs retain their aspect.
            let width = r.random_range(0.126..0.223);
            Vec2::new(width, width * r.random_range(1.28..1.56))
        } * r.random_range(0.92..1.08);
        if r.random_bool(self.landscape) {
            size.yx()
        } else {
            size
        }
    }
}

pub fn placements(o: &IndoorObject) -> Vec<Pinup> {
    placements_at(o, &parameters(o), bevy_zeroverse_capture::GENERATOR_VERSION)
}

pub(crate) fn placements_at(o: &IndoorObject, p: &EquipmentProgram, version: u32) -> Vec<Pinup> {
    let count = p.note_count;
    if count == 0 {
        return Vec::new();
    }
    let posting =
        (version >= super::board_section::TACKBOARD_VERSION).then(|| Posting::sample(o, count));
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
    let coverage = posting
        .as_ref()
        .map_or_else(|| r.random_range(0.22..0.48), |v| v.coverage);
    let pattern = SurfacePattern::sample(&mut r);
    let heading = r.random_range(-0.16..0.16);
    let spread = r.random_range(0.04..0.38);
    let mut occupied: Vec<Footprint> = Vec::with_capacity(count);
    let mut notes = Vec::with_capacity(count);
    let mut used = 0.;
    for _ in 0..96 {
        if notes.len() >= count {
            break;
        }
        let size = match &posting {
            Some(activity) => activity.sheet_size(&mut r),
            None => Vec2::new(
                s.x * r.random_range(0.09..0.42),
                s.y * r.random_range(0.09..0.43),
            ),
        };
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
