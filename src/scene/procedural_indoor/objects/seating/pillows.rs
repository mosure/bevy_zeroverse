//! Loose cushions: independent poses on actual seat modules, rather than a
//! mirrored pair. Soft contact uses the seat profile, not a rotated-box bottom.
use super::*;
use crate::scene::procedural_indoor::{geometry::Geometry, placement::Footprint};

pub(super) fn build(o: &IndoorObject, p: &SofaProgram, widths: &[f32]) -> Vec<Geometry> {
    if !p.pillows || o.size.x <= 1.2 {
        return vec![];
    }
    let mut r = stream(o.seed, 3033);
    let target = r.random_range(1..=4.min(p.modules + 1));
    let mut accepted: Vec<Footprint> = Vec::with_capacity(target);
    let mut result = Vec::with_capacity(target);
    for _ in 0..target {
        for _ in 0..12 {
            let index = r.random_range(0..p.modules);
            let module_width = widths[index];
            let x = -o.size.x * 0.5
                + p.arm_width
                + widths[..index].iter().sum::<f32>()
                + module_width * 0.5;
            let is_return = p.chaise && index == if p.left_return { 0 } else { p.modules - 1 };
            let depth = if is_return {
                o.size.z
            } else {
                o.size.z.min(0.94)
            };
            let z = (o.size.z - depth) * 0.5;
            let seat_size = Vec3::new(module_width - p.cushion_gap_m, p.cushion, depth - 0.19);
            let seat = Vec3::new(x, p.seat_height - p.cushion * 0.5, z - 0.085);
            let min_width = (p.pillow_width_m * 0.75).max(0.19);
            let max_width = (p.pillow_width_m * 1.65).min(module_width * 0.80);
            if min_width >= max_width {
                continue;
            }
            let width = r.random_range(min_width..max_width);
            let size = Vec3::new(
                width,
                width * r.random_range(0.20..0.34),
                // A continuous lumbar-to-square throw-pillow aspect range.
                width * r.random_range(0.46..1.08),
            );
            let incline = r.random_range(0.04..1.48);
            let resting_on_back = incline > 0.55;
            let rotation = Quat::from_euler(
                EulerRot::YXZ,
                if resting_on_back {
                    r.random_range(-0.22..0.22)
                } else {
                    r.random_range(-PI..PI)
                },
                -incline,
                p.pillow_roll + r.random_range(-0.18..0.18),
            );
            let half = Mat3::from_quat(rotation).abs() * size * 0.5;
            let back_height = (o.size.y - p.seat_height + 0.015).max(0.18);
            let lo = seat.xz() - seat_size.xz() * 0.5 + half.xz() + Vec2::splat(0.012);
            let mut hi = seat.xz() + seat_size.xz() * 0.5 - half.xz() - Vec2::splat(0.012);
            // Keep the whole pose in front of the leaned back cushion.
            hi.y =
                hi.y.min(o.size.z * 0.5 - 0.205 - back_height * p.back_tilt.abs() - half.z);
            if lo.cmpge(hi).any() {
                continue;
            }
            let mut center = lo + (hi - lo) * Vec2::new(r.random(), r.random());
            if resting_on_back {
                center.y = hi.y;
            }
            let footprint = Footprint {
                center,
                half: half.xz(),
                yaw: 0.,
            };
            if accepted.iter().any(|q| footprint.overlaps(*q, 0.008)) {
                continue;
            }
            let mut g = Geometry::default();
            g.cushion(
                size,
                super::super::super::geometry::CushionProfile {
                    roundness: 3.,
                    crown: 2.8,
                    dish: 0.,
                },
                Transform::from_xyz(center.x, 0., center.y).with_rotation(rotation),
            );
            let shape = 4.0 + p.roundness * 20.0;
            let support_top = |v: Vec3| {
                let uv = (v.xz() - seat.xz()).abs() / (seat_size.xz() * 0.5);
                let radial = (uv.x.powf(shape) + uv.y.powf(shape)).powf(1. / shape);
                seat.y
                    + seat_size.y
                        * 0.5
                        * (1. - radial.powf(p.cushion_crown))
                            .max(0.)
                            .powf(1. / p.cushion_crown)
                    - p.cushion_dish_m * (1. - radial).powi(2)
            };
            let rise = g
                .positions
                .iter()
                .map(|v| {
                    let v = Vec3::from_array(*v);
                    support_top(v) - v.y
                })
                .fold(f32::NEG_INFINITY, f32::max);
            if g.positions.iter().any(|v| v[1] + rise > o.size.y - 0.003) {
                continue;
            }
            for v in &mut g.positions {
                v[1] += rise - 0.001;
            }
            accepted.push(footprint);
            result.push(g);
            break;
        }
    }
    result
}
