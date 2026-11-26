//! Archived queue products through generator 258.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let base_r = s.z.min(s.x * 0.45) * p.base_radius;
    let post_r = s.z * p.post_radius;
    let edge = s.x * 0.5 - base_r - 0.003;
    let h = s.y * p.rail_height;
    for x in [-edge, edge] {
        let base_h = (s.y * 0.05).min(base_r * 0.40);
        if p.base == 0 {
            a.box_part(
                p.frame,
                l,
                Vec3::new(x, base_h * 0.5, 0.),
                Vec3::new(base_r * 1.8, base_h, base_r * 1.8),
                0.006,
            );
        } else {
            a.part(p.frame, l).lathe(
                &[
                    (0., 0.),
                    (base_r, 0.),
                    (base_r, base_h * 0.35),
                    (base_r * if p.base == 1 { 0.55 } else { 0.92 }, base_h),
                    (post_r, base_h * 1.4),
                ],
                24,
                Transform::from_xyz(x, 0., 0.),
            );
        }
        a.part(p.frame, l).rod(
            Vec3::new(x, base_h, 0.),
            Vec3::new(x, s.y - post_r * 1.4 - 0.001, 0.),
            post_r,
        );
        a.part(p.frame, l).ellipsoid(
            Vec3::splat(post_r * 1.40),
            Transform::from_xyz(x, s.y - post_r * 1.4 - 0.001, 0.),
        );
    }
    let extent = edge - post_r * 0.5;
    if p.construction == 0 {
        // Flat webbing, unlike a cylindrical velvet rope.
        a.box_part(
            p.rail,
            l,
            Vec3::Y * h,
            Vec3::new(extent * 2., p.belt_width.min(s.y * 0.08), 0.003),
            0.0005,
        );
    } else if p.construction == 1 {
        let points: Vec<_> = (0..=20)
            .map(|i| {
                let t = i as f32 / 20.;
                Vec3::new((t * 2. - 1.) * extent, h - s.y * p.sag * (PI * t).sin(), 0.)
            })
            .collect();
        a.part(p.rail, l).tube(&points, post_r * 0.65, 8);
    } else {
        for y in [h, h * 0.32] {
            a.part(p.frame, l).rod(
                Vec3::new(-extent, y, 0.),
                Vec3::new(extent, y, 0.),
                post_r * 0.65,
            );
        }
        if p.construction == 3 {
            a.box_part(
                p.rail,
                l,
                Vec3::Y * h * 0.66,
                Vec3::new(extent * 1.88, h * 0.60, 0.012),
                0.003,
            );
        } else {
            let n = (extent * 2. / 0.20).round().clamp(2., 10.) as usize;
            for i in 1..n {
                let x = extent * (-1. + 2. * i as f32 / n as f32);
                a.part(p.frame, l).rod(
                    Vec3::new(x, h * 0.32, 0.),
                    Vec3::new(x, h, 0.),
                    post_r * 0.45,
                );
            }
        }
    }
}
