//! Baked portions sized to the flat serving surface of the plate.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) fn toast(a: &mut Assembly, flat: f32, floor: f32, height_limit: f32, r: &mut impl Rng) {
    let half = flat * r.random_range(0.68..0.78);
    let thick = (half * r.random_range(0.19..0.33)).min((height_limit - floor) * 0.74);
    let outline = [
        Vec2::new(-0.94, -0.78),
        Vec2::new(0.91, -0.78),
        Vec2::new(0.96, -0.62),
        Vec2::new(-0.65, 0.92),
        Vec2::new(-0.83, 0.88),
        Vec2::new(-0.98, 0.60),
    ]
    .map(|v| v * half);
    let tf = Transform::from_xyz(
        flat * r.random_range(-0.040..0.040),
        floor + thick * 0.5,
        0.,
    )
    .with_rotation(Quat::from_rotation_y(r.random_range(-PI..PI)));
    let mut crumb = Geometry::default();
    a.part(Surface::Terracotta, "food#finish103")
        .profile_slab_with_top(
            &outline,
            thick,
            thick * 0.13,
            tf,
            |_, poly, inset, y, tf| {
                let center = poly.iter().sum::<Vec2>() * (inset / poly.len() as f32);
                crumb.slab_cap(poly, inset, center, y, tf, |p| p);
            },
        );
    a.part(Surface::Paper, "food#finish104").append(crumb);
}

pub(crate) fn biscuit(
    a: &mut Assembly,
    flat: f32,
    floor: f32,
    height_limit: f32,
    r: &mut impl Rng,
) {
    let radius = flat * r.random_range(0.78..0.94);
    let h = (radius * r.random_range(0.22..0.41)).min((height_limit - floor) * 0.76);
    let tf = Transform::from_xyz(flat * r.random_range(-0.03..0.03), floor, 0.)
        .with_rotation(Quat::from_rotation_y(r.random_range(-PI..PI)))
        .with_scale(Vec3::new(1., 1., r.random_range(0.80..1.)));
    // Rounded baked shoulder, flat foot and a broad crown for inclusions.
    a.part(Surface::Paper, "food#finish104").lathe(
        &[
            (0., 0.),
            (radius * 0.88, 0.),
            (radius, h * 0.23),
            (radius * 0.97, h * 0.70),
            (radius * 0.83, h * 0.96),
            (radius * 0.73, h),
            (0., h),
        ],
        r.random_range(15..=23),
        tf,
    );
    let count = r.random_range(1..=7);
    let mut included: Vec<(Vec2, f32)> = Vec::with_capacity(count);
    for _ in 0..count {
        let bit = radius * r.random_range(0.055..0.13);
        let pos = (0..6).find_map(|_| {
            let p = Vec2::from_angle(r.random_range(-PI..PI))
                * radius
                * r.random::<f32>().sqrt()
                * 0.58;
            included
                .iter()
                .all(|(other, r)| p.distance_squared(*other) > (r + bit).powi(2))
                .then_some(p)
        });
        if let Some(p) = pos {
            included.push((p, bit));
            let rise = (bit * r.random_range(0.45..0.94)).min(h * 0.45);
            a.part(Surface::Terracotta, "food#finish106").lathe(
                &[
                    (bit, -rise * 0.42),
                    (bit * 0.91, 0.),
                    (bit * 0.57, rise * 0.74),
                    (0., rise),
                ],
                r.random_range(6..=9),
                tf.mul_transform(Transform::from_xyz(p.x, h, p.y)),
            );
        }
    }
}
