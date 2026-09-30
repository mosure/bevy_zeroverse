use super::*;
use crate::scene::procedural_indoor::{
    architecture::facade::{ExteriorProgram, FacadeSide},
    layout::stream,
};
use rand::Rng;

impl EnvelopeProgram {
    pub fn room_height(seed: u64, size: Vec3) -> f32 {
        let mut r = stream(seed, 4100);
        if size.xz().min_element() > 7.5 && r.random_bool(0.25) {
            r.random_range(5.9..7.4)
        } else {
            size.y
        }
    }

    pub fn sample(scene: &mut IndoorManifest) -> Self {
        let mut r = stream(scene.seed, 4101);
        let size = scene.room_size;
        let half = size.xz() * 0.5;
        let taper = if r.random_bool(0.68) {
            r.random_range(0.02..0.18) * size.x
        } else {
            0.
        };
        let left = taper * r.random_range(0.1..0.9);
        let right = taper - left;
        let chamfer = if r.random_bool(0.62) {
            r.random_range(0.25..1.25_f32.min(size.x.min(size.z) * 0.16))
        } else {
            0.
        };
        let a = Vec2::new(-half.x + left + chamfer, -half.y);
        let b = Vec2::new(half.x - right - chamfer, -half.y);
        let mut footprint = vec![a];
        if b.x - a.x > 3.0 && r.random_bool(0.42) {
            let width = (b.x - a.x) * r.random_range(0.13..0.30);
            let center = (a.x + b.x) * 0.5 + (b.x - a.x) * r.random_range(-0.10..0.10);
            let depth = r.random_range(0.45..(size.z * 0.15).min(2.2));
            footprint.extend([
                Vec2::new(center - width * 0.5, -half.y),
                Vec2::new(center - width * 0.5, -half.y + depth),
                Vec2::new(center + width * 0.5, -half.y + depth),
                Vec2::new(center + width * 0.5, -half.y),
            ]);
        }
        footprint.push(b);
        if chamfer > 0. {
            footprint.push(Vec2::new(half.x - right, -half.y + chamfer));
        }
        footprint.extend([Vec2::new(half.x, half.y), Vec2::new(-half.x, half.y)]);
        if chamfer > 0. {
            footprint.push(Vec2::new(-half.x + left, -half.y + chamfer));
        }
        let slope = if r.random_bool(0.70) {
            r.random_range(0.1..(size.y - 2.5).clamp(0.11, 1.3))
        } else {
            0.
        };
        let fraction = r.random_range(0.0..1.0);
        let ceiling_drop = Vec2::new(
            slope * fraction * if r.random_bool(0.5) { 1. } else { -1. },
            slope * (1. - fraction) * if r.random_bool(0.5) { 1. } else { -1. },
        );
        let mut result = Self {
            footprint,
            ceiling_drop,
            floor_patches: vec![],
            pillars: vec![],
            mezzanine: None,
            walls: vec![],
        };

        // Fit the existing functional partition program to the actual outline.
        // Zones remain proposal domains; their accepted contents are clipped by
        // this envelope. Every retained partition still has a usable portal.
        if let Some(program) = &mut scene.program {
            program.partitions.retain_mut(|p| {
                let intervals = polygon::line_intervals(&result.footprint, p.axis, p.coordinate);
                let (lo, hi) = intervals
                    .into_iter()
                    .map(|(lo, hi)| (lo.max(p.start), hi.min(p.end)))
                    .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
                    .unwrap_or((0., 0.));
                if hi - lo < 2.2 {
                    return false;
                }
                p.start = lo;
                p.end = hi;
                p.door_width = p.door_width.min((hi - lo - 0.6).max(0.9));
                p.door_center = p.door_center.clamp(
                    lo + p.door_width * 0.5 + 0.26,
                    hi - p.door_width * 0.5 - 0.26,
                );
                let top = result.ceiling_height(size, p.position(p.door_center, 0.).xz());
                p.door_height = p.door_height.min(top - 0.15);
                if r.random_bool(0.48) {
                    p.door_height = (2.05_f32 + r.random_range(0.30..0.65)).min(top - 0.15);
                    p.arch_rise = (p.door_height - 2.05).max(0.);
                }
                true
            });
        }
        let (zone_lo, zone_hi) = scene.primary_room_bounds();
        let zone = zone_hi - zone_lo;
        // Floor offsets are shallow enough for a short stair, with every tread
        // serialized separately. Keep the room's portals at the base level.
        if zone.min_element() > 4.6 && r.random_bool(0.43) {
            for _ in 0..24 {
                let width = r.random_range(1.8..(zone.x * 0.55).max(1.81));
                let depth = r.random_range(2.0..(zone.y * 0.55).max(2.01));
                let center = zone_lo
                    + zone * Vec2::new(r.random_range(0.30..0.70), r.random_range(0.30..0.70));
                let lo = center - Vec2::new(width, depth) * 0.5;
                let hi = center + Vec2::new(width, depth) * 0.5;
                if !result.feature_clear(scene, lo, hi, 0.35) {
                    continue;
                }
                let steps = r.random_range(1..=3);
                let rise = r.random_range(0.12..0.17) * if r.random_bool(0.5) { 1. } else { -1. };
                let tread = r.random_range(0.28..0.36);
                result.floor_patches.push(FloorPatch {
                    min: lo,
                    max: hi - Vec2::Y * (steps as f32 * tread),
                    height: rise * steps as f32,
                });
                for i in 0..steps {
                    result.floor_patches.push(FloorPatch {
                        min: Vec2::new(lo.x, hi.y - (i + 1) as f32 * tread),
                        max: Vec2::new(hi.x, hi.y - i as f32 * tread),
                        height: rise * i as f32,
                    });
                }
                break;
            }
        }
        if size.y > 5.8 {
            let axis = usize::from(zone.y >= zone.x);
            let across = 1 - axis;
            if zone[axis] > 8.0 && zone[across] > 5.2 {
                for _ in 0..24 {
                    let height: f32 = r.random_range(2.65..3.05);
                    let steps = (height / 0.17).ceil() as u32;
                    let run = steps as f32 * r.random_range(0.28..0.31);
                    let mut lo = zone_lo + Vec2::splat(r.random_range(0.6..1.5));
                    let mut hi = zone_hi - Vec2::splat(0.5);
                    hi[axis] = lo[axis] + r.random_range(2.05..2.75);
                    hi[across] = lo[across] + (zone[across] * 0.62).clamp(3.0, 5.0);
                    let mut stair_lo = lo;
                    let mut stair_hi = hi;
                    stair_lo[axis] = hi[axis];
                    stair_hi[axis] = hi[axis] + run;
                    stair_lo[across] = lo[across] + 0.15;
                    stair_hi[across] = stair_lo[across] + 1.08;
                    // Alternate attachment side while preserving an open circulation aisle.
                    if r.random_bool(0.5) {
                        let shift = zone[across] - (hi[across] - lo[across]) - 1.;
                        lo[across] += shift;
                        hi[across] += shift;
                        stair_lo[across] += shift;
                        stair_hi[across] += shift;
                    }
                    if stair_hi[axis] < zone_hi[axis] - 0.4
                        && result.feature_clear(scene, lo, hi, 0.32)
                        && result.feature_clear(scene, stair_lo, stair_hi, 0.25)
                        && [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
                            .into_iter()
                            .all(|p| result.ceiling_height(size, p) > height + 2.3)
                        && result
                            .floor_patches
                            .iter()
                            .all(|p| !p.overlaps(lo, stair_hi))
                    {
                        result.mezzanine = Some(Mezzanine {
                            deck: FloorPatch {
                                min: lo,
                                max: hi,
                                height,
                            },
                            thickness: 0.20,
                            stair_min: stair_lo,
                            stair_max: stair_hi,
                            stair_axis: axis,
                            steps,
                            rail_height: 1.10,
                        });
                        break;
                    }
                }
            }
        }
        let count = (polygon::area(&result.footprint) / 65. * r.random_range(0.0..2.2))
            .round()
            .min(5.) as usize;
        for _ in 0..count {
            for _ in 0..24 {
                let center = Vec2::new(
                    r.random_range(-0.36..0.36) * size.x,
                    r.random_range(-0.36..0.32) * size.z,
                );
                let radius = r.random_range(0.14..0.34);
                let pad = Vec2::splat(radius + 0.35);
                if !result.feature_clear(scene, center - pad, center + pad, 0.4)
                    || result
                        .floor_patches
                        .iter()
                        .any(|p| p.overlaps(center - pad, center + pad))
                    || result.mezzanine.as_ref().is_some_and(|m| {
                        m.deck.overlaps(center - pad, center + pad)
                            || (FloorPatch {
                                min: m.stair_min,
                                max: m.stair_max,
                                height: 0.,
                            })
                            .overlaps(center - pad, center + pad)
                    })
                    || result
                        .pillars
                        .iter()
                        .any(|p| p.center.distance(center) < 1.5)
                {
                    continue;
                }
                result.pillars.push(Pillar {
                    center,
                    radius,
                    sides: if r.random_bool(0.65) { 24 } else { 4 },
                });
                break;
            }
        }
        let exposure = r.random_range(0.40..0.88);
        for (edge, (a, b)) in polygon::edges(&result.footprint).enumerate() {
            if result.shared_edge(edge, size) {
                continue;
            }
            let span = a.distance(b);
            let height = result
                .ceiling_height(size, a)
                .min(result.ceiling_height(size, b));
            let facade = if span > 1.5 && r.random_bool(exposure) {
                let mut sampled = ExteriorProgram::sample(
                    scene.seed.wrapping_add(edge as u64 * 7919),
                    Vec3::new(span, height, span),
                    0.08,
                );
                let mut f = sampled.facades.remove(0);
                f.side = FacadeSide::Rear;
                Some(f)
            } else {
                None
            };
            result.walls.push(EnvelopeWall { edge, facade });
        }
        result
    }

    fn feature_clear(&self, scene: &IndoorManifest, lo: Vec2, hi: Vec2, margin: f32) -> bool {
        polygon::box_inside(&self.footprint, lo, hi, margin)
            && scene.program.as_ref().is_none_or(|p| {
                p.portal_clear(Vec3::new(lo.x, 0., lo.y), Vec3::new(hi.x, 1., hi.y))
                    && p.partitions
                        .iter()
                        .flat_map(|p| p.obstacles(scene.room_size.y))
                        .all(|(a, b)| {
                            !lo.cmplt(b.xz() + Vec2::splat(margin)).all()
                                || !hi.cmpgt(a.xz() - Vec2::splat(margin)).all()
                        })
            })
            && !(hi.x > scene.door_x - 0.8 && hi.y > scene.room_size.z * 0.5 - 1.6)
    }
}
