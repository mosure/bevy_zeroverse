use super::*;
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;

impl Mezzanine {
    pub(in super::super) fn sample(scene: &IndoorManifest, e: &EnvelopeProgram) -> Option<Self> {
        if scene.room_size.y <= 5.8 {
            return None;
        }
        let (zone_lo, zone_hi) = scene.primary_room_bounds();
        let zone = zone_hi - zone_lo;
        let mut r = stream(scene.seed, 0x4d455a5a504c414e);
        // Bounded admission. A failed wide/landed proposal can be followed by
        // a narrower straight flight; it does not lock out all mezzanines.
        for _ in 0..40 {
            let axis = usize::from(r.random_bool(0.5));
            let across = 1 - axis;
            if zone[axis] < 7.5 || zone[across] < 4.5 {
                continue;
            }
            let height = r.random_range(2.70_f32..3.42);
            let thickness = r.random_range(0.15..0.29);
            let beam_depth = r.random_range(0.11..0.24_f32.min(height - thickness - 2.2));
            let steps = (height / r.random_range(0.145..0.179)).ceil() as u32;
            let tread = r.random_range(0.275..0.35);
            let width = r.random_range(0.95..1.65);
            let landing = r.random_bool(0.35).then(|| Landing {
                after_riser: r.random_range(steps / 3..=steps * 2 / 3),
                extra_run: r.random_range(0.85..1.55),
            });
            let run = steps as f32 * tread + landing.as_ref().map_or(0., |l| l.extra_run);
            let depth = r.random_range(2.1..3.7);
            let breadth = r.random_range(2.9..6.0_f32.min(zone[across] - 0.7));
            let mut total = Vec2::ZERO;
            total[axis] = depth + run;
            total[across] = breadth;
            let slack = zone - total - Vec2::splat(0.7);
            if slack.min_element() < 0. {
                continue;
            }
            let origin = zone_lo
                + Vec2::splat(0.35)
                + slack * Vec2::new(r.random_range(0.0..1.), r.random_range(0.0..1.));
            let ascending_positive = r.random_bool(0.5);
            let mut lo = origin;
            let mut hi = origin + total;
            if ascending_positive {
                lo[axis] += run;
            } else {
                hi[axis] = lo[axis] + depth;
            }
            let margin = r.random_range(0.13..0.32);
            let across_start =
                lo[across] + margin + r.random_range(0.0..1.) * (breadth - width - 2. * margin);
            let mut stair_min = lo;
            let mut stair_max = hi;
            if ascending_positive {
                stair_max[axis] = lo[axis];
                stair_min[axis] = lo[axis] - run;
            } else {
                stair_min[axis] = hi[axis];
                stair_max[axis] = hi[axis] + run;
            }
            stair_min[across] = across_start;
            stair_max[across] = across_start + width;
            let m = Self {
                deck: FloorPatch {
                    min: lo,
                    max: hi,
                    height,
                },
                thickness,
                stair_min,
                stair_max,
                stair_axis: axis,
                steps,
                rail_height: r.random_range(1.03..1.22),
                construction: None,
            };
            if !e.feature_clear(scene, lo, hi, 0.30)
                || !e.feature_clear(scene, stair_min, stair_max, 0.25)
                || e.minimum_ceiling_height(scene.room_size, lo, hi) < height + 2.3
                || e.minimum_ceiling_height(scene.room_size, stair_min, stair_max) < height + 2.25
                || e.floor_patches
                    .iter()
                    .any(|p| p.overlaps(lo, hi) || p.overlaps(stair_min, stair_max))
            {
                continue;
            }
            let mut detail = stream(scene.seed, 0x4d455a5a44455441);
            let frame_surface =
                [Surface::Metal, Surface::Wood, Surface::Concrete][detail.random_range(0..3)];
            let concrete = frame_surface == Surface::Concrete;
            let support_width =
                detail.random_range(if concrete { 0.20..0.26_f32 } else { 0.14..0.23 });
            let support_depth =
                detail.random_range(if concrete { 0.20..0.28_f32 } else { 0.14..0.25 });
            let c = Construction {
                ascending_positive,
                landing,
                stair_structure: if concrete {
                    StairStructure::Solid
                } else {
                    [
                        StairStructure::Solid,
                        StairStructure::ClosedStringer,
                        StairStructure::OpenStringer,
                    ][detail.random_range(0..3)]
                },
                tread_thickness: detail.random_range(0.035..0.075),
                nosing_depth: detail.random_range(0.018..0.060),
                frame_surface,
                deck_surface: [Surface::Wood, Surface::Floor, Surface::Concrete]
                    [detail.random_range(0..3)],
                tread_surface: [Surface::Wood, Surface::Concrete, Surface::Metal]
                    [detail.random_range(0..3)],
                support_section: if frame_surface == Surface::Metal {
                    [
                        SupportSection::Square,
                        SupportSection::Round,
                        SupportSection::ISection,
                    ][detail.random_range(0..3)]
                } else if detail.random_bool(0.45) {
                    SupportSection::Round
                } else {
                    SupportSection::Square
                },
                support_width,
                support_depth,
                support_inset: detail
                    .random_range((support_width.max(support_depth) * 0.5 + 0.02).max(0.15)..0.29),
                support_pitch: detail.random_range(2.0..3.5),
                beam_depth,
                guard: super::super::guardrail::GuardSection::sample(scene.seed.wrapping_add(4129)),
            };
            let m = Self {
                construction: Some(c),
                ..m
            };
            debug_assert!(m.construction.as_ref().unwrap().validate(&m).is_ok());
            return Some(m);
        }
        None
    }
}
