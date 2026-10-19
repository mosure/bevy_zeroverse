//! Additional support programs emit identical solids and rendering geometry.
use super::*;

pub(super) fn fit_center(mut center: Vec3, outline: &[Vec2], margin: f32) -> Vec3 {
    let pad = Vec2::splat(margin);
    for _ in 0..32 {
        if supports_outline(outline, center.xz() - pad, center.xz() + pad, 0.008) {
            break;
        }
        center *= 0.90;
    }
    center
}

pub(super) fn hairpin(
    a: &mut impl StructureSink,
    p: &TableProgram,
    anchor: Vec2,
    underside: f32,
    frame: Surface,
    label: &str,
) {
    for x in [-1., 1.] {
        for z in [-1., 1.] {
            let center = Vec3::new(x * anchor.x, 0., z * anchor.y);
            let spread = anchor.y.min(0.16) * 0.45;
            let radius = (p.leg_radius * 0.48).min(0.015).min(spread * 0.18);
            let bend = spread * 0.65;
            let foot = center + Vec3::new(x * p.leg_rake, radius + 0.004, z * p.leg_rake);
            // A resolved U bend, with curvature radius >3 tube radii. A sharp
            // three-point V self-intersects at the lower turn on small tables.
            let mut points = vec![center + Vec3::new(0., underside, -spread)];
            for i in 0..=10 {
                let angle = i as f32 * PI / 10.;
                points.push(foot + Vec3::new(0., bend * (1. - angle.sin()), -bend * angle.cos()));
            }
            points.push(center + Vec3::new(0., underside, spread));
            a.part(frame, label).tube(&points, radius, 8);
            a.box_part(
                frame,
                label,
                center.with_y(underside - 0.004),
                Vec3::new(0.055, 0.008, spread * 2. + 0.035),
                0.003,
            );
            a.part(Surface::Rubber, label).cylinder(
                radius + 0.006,
                0.014,
                Transform::from_translation(foot.with_y(0.007)),
            );
        }
    }
}

pub(super) fn drums(
    a: &mut impl StructureSink,
    o: &IndoorObject,
    p: &TableProgram,
    outline: &[Vec2],
    underside: f32,
    frame: Surface,
    label: &str,
) {
    let size = o.size.xz();
    let long_x = size.x > size.y;
    let count = if size.max_element() / size.min_element() > 1.8 {
        2
    } else {
        1
    };
    for i in 0..count {
        let offset = if count == 1 {
            0.
        } else {
            (i as f32 - 0.5) * size.max_element() * 0.48
        };
        let center = if long_x {
            Vec2::X * offset
        } else {
            Vec2::Y * offset
        };
        let mut radius = size.min_element() * p.pedestal_base_radius * 0.80;
        // The entire shoulder is supported, including asymmetric triangular tops.
        for _ in 0..24 {
            let pad = Vec2::splat(radius);
            if supports_outline(outline, center - pad, center + pad, 0.008) {
                break;
            }
            radius *= 0.92;
        }
        let profile = [
            (0., 0.),
            (radius * 0.94, 0.),
            (radius, 0.016),
            (radius * p.pedestal_waist, underside * p.pedestal_shoulder),
            (radius * 0.72, underside - 0.012),
            (radius * 0.75, underside),
            (0., underside),
        ];
        a.part(frame, label).lathe(
            &profile,
            p.pedestal_facets,
            Transform::from_xyz(center.x, 0., center.y),
        );
    }
}
