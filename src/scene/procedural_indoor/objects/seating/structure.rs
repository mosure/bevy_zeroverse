//! Upholstery supports and arm structures; positive clearances at every joint.
use super::*;
use crate::scene::procedural_indoor::geometry::CushionProfile;

/// Rounded corners retain positive bend radius even for a short leg and a long
/// sectional rail. A four-point sweep through sharp corners can fold its mesh.
pub(super) fn u_support(
    g: &mut Geometry,
    left: Vec3,
    right: Vec3,
    heights: [f32; 2],
    upper: bool,
    radius: f32,
) {
    let [floor, top] = heights;
    let bend = ((top - floor) * 0.4).min(0.028);
    let direction = (right - left).normalize();
    let left = left.with_y(floor);
    let right = right.with_y(floor);
    let mut path = vec![left.with_y(top), left + Vec3::Y * bend];
    for i in 1..=6 {
        let t = FRAC_PI_2 * i as f32 / 6.;
        path.push(left + direction * bend * (1. - t.cos()) + Vec3::Y * bend * (1. - t.sin()));
    }
    path.push(right - direction * bend);
    for i in 1..=6 {
        let t = FRAC_PI_2 * i as f32 / 6.;
        path.push(right - direction * bend * (1. - t.sin()) + Vec3::Y * bend * (1. - t.cos()));
    }
    path.push(right.with_y(top));
    if upper {
        for point in &mut path {
            point.y = top + floor - point.y;
        }
    }
    g.tube(&path, radius, 8);
}

pub(super) fn arms(a: &mut Assembly, o: &IndoorObject, p: &SofaProgram, base_depth: f32) {
    if p.arm_style == 5 {
        return;
    }
    let s = o.size;
    let label = o.kind.class_name();
    let frame = if p.exposed_frame {
        Surface::WoodEdge
    } else {
        p.upholstery
    };
    for side in [-1., 1.] {
        let mut arm = Assembly::default();
        let target = &mut arm;
        let is_return = p.chaise && (p.double_return || ((side < 0.) == p.left_return));
        let depth = if is_return { s.z } else { base_depth };
        let height = (p.seat_height + p.arm_rise_m + if p.arm_style == 4 { 0.12 } else { 0. })
            .min(s.y - 0.08);
        let x = side * (s.x - p.arm_width) * 0.5;
        let z = (s.z - depth) * 0.5;
        let pad = (p.cushion * 0.90).clamp(0.075, 0.13);
        let stem_top = height - pad * 0.60;
        // Connect the arms to the under-seat chassis, including open frames.
        for end in [-1., 1.] {
            let rail_y = (p.seat_height - p.cushion - 0.025).max(p.leg_height + 0.015);
            let zz = z + end * (depth * 0.5 - 0.04);
            target.part(frame, label).rod(
                Vec3::new(side * (s.x * 0.5 - p.arm_width - 0.035), rail_y, zz),
                Vec3::new(x, rail_y, zz),
                0.013,
            );
        }
        if p.arm_style == 3 {
            // Open bentwood arm: the two uprights and horizontal rail are one
            // continuous support, with upholstery fitted onto the rail.
            let reach = depth * 0.5 - 0.04;
            u_support(
                target.part(Surface::WoodEdge, label),
                Vec3::new(x, 0., z - reach),
                Vec3::new(x, 0., z + reach),
                [p.leg_height, height - pad],
                true,
                0.018,
            );
        } else if p.arm_style == 4 {
            // Rolled-in shell grows from a narrow heel into the padded arm.
            target.part(frame, label).housing(
                &[
                    (
                        p.leg_height,
                        Vec2::new(p.arm_width * 0.28, (depth - 0.04) * 0.5),
                    ),
                    (
                        p.seat_height,
                        Vec2::new(p.arm_width * 0.34, (depth - 0.04) * 0.5),
                    ),
                    (
                        stem_top,
                        Vec2::new(p.arm_width * 0.46, (depth - 0.02) * 0.5),
                    ),
                ],
                4.,
                Transform::from_xyz(x, 0., z),
            );
        } else {
            target.box_part(
                frame,
                label,
                Vec3::new(x, (p.leg_height + stem_top) * 0.5, z),
                Vec3::new(p.arm_width * 0.78, stem_top - p.leg_height, depth - 0.018),
                p.roundness * 0.55,
            );
        }
        target.part(p.upholstery, label).cushion(
            Vec3::new(
                p.arm_width,
                if p.arm_style == 1 { pad * 1.35 } else { pad },
                depth - 0.008,
            ),
            CushionProfile {
                roundness: if p.arm_style == 1 {
                    2.0
                } else {
                    p.arm_roundness
                },
                crown: if p.arm_style == 1 { 2.0 } else { p.arm_crown },
                dish: 0.,
            },
            Transform::from_xyz(x, height - pad * 0.675, z),
        );
        if p.arm_style == 2 {
            // The pad and side frame deform together, retaining their contact.
            let dy = height - p.leg_height;
            for g in arm.parts.values_mut() {
                for i in 0..g.positions.len() {
                    let v = Vec3::from_array(g.positions[i]);
                    let t = ((v.z - z) / (depth - 0.008) + 0.5).clamp(0., 1.);
                    let rise = ((v.y - p.leg_height) / dy).max(0.);
                    let sy = 1. - p.arm_sweep_m * (1. - t) / dy;
                    let sz = p.arm_sweep_m * rise / (depth - 0.008);
                    g.positions[i][1] -= p.arm_sweep_m * (1. - t) * rise;
                    let n = Vec3::from_array(g.normals[i]);
                    g.normals[i] = Vec3::new(n.x, n.y / sy, n.z - sz * n.y / sy)
                        .normalize()
                        .to_array();
                }
            }
        }
        for ((surface, label), g) in arm.parts {
            a.part(surface, &label).append(g);
        }
    }
}
