//! Tablet-arm and linked auditorium construction, preserving the adult seating datum.
use super::*;
pub fn is_folded(o: &IndoorObject) -> bool {
    o.variant == 11 && stream(o.seed, 8260).random_bool(0.22)
}
pub(super) fn clearance_boxes(o: &IndoorObject) -> Vec<(Vec3, Vec3)> {
    let half = o.size * 0.5;
    let mut boxes = vec![
        (
            Vec3::new(-half.x, 0., -half.z),
            Vec3::new(half.x, 0.50, half.z),
        ),
        (
            Vec3::new(-half.x, 0.48, 0.17),
            Vec3::new(half.x, o.size.y, half.z),
        ),
    ];
    if o.variant == 10 {
        boxes.push((
            Vec3::new(o.size.x * 0.10, 0.60, -o.size.z * 0.44),
            Vec3::new(half.x, 0.77, o.size.z * 0.15),
        ));
    }
    if is_folded(o) {
        boxes.push((
            Vec3::new(-half.x, 0.44, 0.),
            Vec3::new(half.x, 0.85, half.z),
        ));
    }
    boxes
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let mut r = stream(o.seed, 8261);
    let w = o.size.x * 0.66;
    let d = o.size.z * 0.67;
    let label = "chair";
    let audience = o.variant == 11;
    for x in [-w * 0.48, w * 0.48] {
        for z in [-d * 0.36, d * 0.36] {
            a.part(Surface::Metal, label).rod(
                Vec3::new(x, 0.018, z),
                Vec3::new(x * 0.91, 0.47, z * 0.8),
                0.018,
            );
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(x, 0.009, z),
                Vec3::new(0.045, 0.018, 0.045),
                0.005,
            );
        }
    }
    a.part(Surface::Metal, label).rod(
        Vec3::new(-o.size.x * 0.48, 0.30, 0.12),
        Vec3::new(o.size.x * 0.48, 0.30, 0.12),
        0.020,
    );
    let folded = is_folded(o);
    let seat_tf = if folded {
        Transform::from_xyz(0., 0.46 + d * 0.45, d * 0.38)
            .with_rotation(Quat::from_rotation_x(-1.42))
    } else {
        Transform::from_xyz(0., 0.46, 0.)
    };
    if audience {
        a.part(Surface::FabricAlt, label).cushion(
            Vec3::new(w, 0.078, d),
            super::super::super::geometry::CushionProfile {
                roundness: r.random_range(3.0..5.5),
                crown: r.random_range(2.8..4.5),
                dish: 0.006,
            },
            seat_tf,
        );
    } else {
        a.part(Surface::Wood, label)
            .cuboid(Vec3::new(w, 0.062, d), 0.025, seat_tf);
    }
    for x in [-w * 0.43, w * 0.43] {
        a.part(Surface::Metal, label).rod(
            Vec3::new(x, 0.43, d * 0.35),
            Vec3::new(x, o.size.y - 0.08, d * 0.47),
            0.014,
        );
    }
    let height = (o.size.y - 0.58).max(0.20);
    a.part(
        if audience {
            Surface::FabricAlt
        } else {
            Surface::Plastic
        },
        label,
    )
    .chair_back_profile(
        w,
        height,
        [
            r.random_range(0.025..0.055),
            r.random_range(-0.04..0.10),
            0.05,
            0.035,
        ],
        Transform::from_xyz(0., 0.58, d * 0.46),
    );
    if !audience {
        a.part(Surface::Metal, label).rod(
            Vec3::new(w * 0.48, 0.30, 0.10),
            Vec3::new(o.size.x * 0.29, 0.69, -0.10),
            0.014,
        );
        a.box_part(
            Surface::Wood,
            label,
            Vec3::new(o.size.x * 0.29, 0.71, -o.size.z * 0.15),
            Vec3::new(o.size.x * 0.34, 0.027, o.size.z * 0.48),
            0.018,
        );
    }
}
