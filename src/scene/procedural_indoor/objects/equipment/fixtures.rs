use super::*;

pub(super) fn shelf(a: &mut Assembly, o: &IndoorObject, p: &EquipmentProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let thickness = (s.y * 0.10).clamp(0.016, 0.032).min(s.y * 0.2);
    let surface = if p.housing == Surface::Metal {
        Surface::Metal
    } else {
        Surface::Wood
    };
    // Flat top is exactly size.y, so separate supported plants/books obey the
    // usual support contract. Brackets occupy the full declared vertical range.
    a.box_part(
        surface,
        label,
        Vec3::new(0., s.y - thickness * 0.5, 0.),
        Vec3::new(s.x, thickness, s.z),
        0.002,
    );
    for i in 0..p.shelf_brackets {
        let x = if p.shelf_brackets == 2 {
            [-0.32, 0.32][i]
        } else {
            (i as f32 - 1.) * 0.34
        } * s.x;
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(x, s.y * 0.5, -s.z * 0.5 + 0.008),
            Vec3::new(0.022, s.y, 0.016),
            0.002,
        );
        a.part(Surface::Metal, label).rod(
            Vec3::new(x, 0.021, -s.z * 0.5 + 0.017),
            Vec3::new(x, s.y - thickness - 0.013, s.z * 0.39),
            0.008,
        );
    }
}

pub(super) fn notice_board(a: &mut Assembly, o: &IndoorObject, p: &EquipmentProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let frame = s.x.min(s.y) * 0.025;
    a.box_part(
        if p.housing == Surface::Metal {
            Surface::Metal
        } else {
            Surface::WoodEdge
        },
        label,
        Vec3::Y * s.y * 0.5,
        s,
        0.004,
    );
    a.box_part(
        Surface::FabricAlt,
        label,
        Vec3::new(0., s.y * 0.5, s.z * 0.44),
        Vec3::new(s.x - frame * 2., s.y - frame * 2., s.z * 0.08),
        0.002,
    );
    let mut rng = stream(o.seed, 5730);
    // Notes occupy separately fitted cells; their rotated corners cannot overlap
    // the frame or each other. Density, dimensions and handwriting vary.
    let columns = if s.x > s.y { 4 } else { 3 };
    let rows = 4;
    let cell = Vec2::new(
        (s.x - frame * 4.) / columns as f32,
        (s.y - frame * 4.) / rows as f32,
    );
    let mut cells: Vec<_> = (0..columns * rows).collect();
    use rand::seq::SliceRandom;
    cells.shuffle(&mut rng);
    for &i in cells.iter().take(p.note_count.min(cells.len())) {
        let width = cell.x * rng.random_range(0.58..0.76);
        let height = cell.y * rng.random_range(0.55..0.74);
        let center = Vec3::new(
            (i % columns) as f32 * cell.x + cell.x * 0.5 - (s.x - frame * 4.) * 0.5,
            frame * 2. + (i / columns) as f32 * cell.y + cell.y * 0.5,
            s.z * 0.491,
        );
        let tf = Transform::from_translation(center)
            .with_rotation(Quat::from_rotation_z(rng.random_range(-0.10..0.10)));
        a.part(Surface::Paper, label)
            .cuboid(Vec3::new(width, height, s.z * 0.012), 0.0001, tf);
        let lines = rng.random_range(2..=5);
        for line in 0..lines {
            let length = width * rng.random_range(0.37..0.78);
            let start = Vec3::new(
                -width * 0.39,
                height * (0.28 - line as f32 * 0.105),
                s.z * 0.008,
            );
            a.part(Surface::Ink, label).rod(
                tf.transform_point(start),
                tf.transform_point(start + Vec3::X * length),
                0.00045,
            );
        }
        a.part(Surface::Chrome, label).ellipsoid(
            Vec3::new(0.002, 0.002, s.z * 0.010),
            Transform::from_translation(tf.transform_point(Vec3::new(
                0.,
                height * 0.41,
                s.z * 0.008,
            ))),
        );
    }
}

pub(super) fn projection_screen(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let label = o.kind.class_name();
    let case = (s.y * 0.045).clamp(0.036, 0.060).min(s.z * 0.8);
    a.box_part(
        Surface::Metal,
        label,
        Vec3::new(0., s.y - case * 0.5, 0.),
        Vec3::new(s.x, case, s.z * 0.82),
        case * 0.28,
    );
    for side in [-1., 1.] {
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(side * s.x * 0.38, s.y - case * 0.5, -s.z * 0.455),
            Vec3::new(0.030, case * 0.72, s.z * 0.09),
            0.001,
        );
    }
    a.box_part(
        Surface::Ink,
        label,
        Vec3::new(0., (s.y - case) * 0.5, s.z * 0.20),
        Vec3::new(s.x * 0.94, s.y - case, 0.004),
        0.001,
    );
    // A real unlit projection cloth. Content is not fabricated without a
    // calibrated active projector; matte white screens also occur powered off.
    a.box_part(
        Surface::Paper,
        label,
        Vec3::new(0., (s.y - case) * 0.5, s.z * 0.20 + 0.0028),
        Vec3::new(s.x * 0.90, (s.y - case) * 0.94, 0.0008),
        0.,
    );
    a.part(Surface::Metal, label).cylinder(
        0.009,
        s.x * 0.94,
        Transform::from_xyz(0., 0.009, s.z * 0.20).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    );
}

pub(super) fn light(a: &mut Assembly, o: &IndoorObject, p: &EquipmentProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    if o.kind == ObjectKind::CeilingPendant {
        let radius = s.x.min(s.z) * 0.46;
        let shade = s.y * p.shade_fraction;
        let thickness = 0.007_f32.min(radius * 0.08);
        // Conical shell with real inner wall and rim, suspended by a connected
        // stem. Geometry remains below the ceiling and above the seated zone.
        a.part(p.housing, label).lathe(
            &[
                (radius, s.y * 0.05),
                (radius * 0.48, shade),
                ((radius * 0.48 - thickness).max(0.003), shade),
                (radius - thickness, s.y * 0.05),
                (radius, s.y * 0.05),
            ],
            28,
            Transform::IDENTITY,
        );
        a.part(Surface::Rubber, label).rod(
            Vec3::new(0., shade, 0.),
            Vec3::new(0., s.y - 0.012, 0.),
            0.0035,
        );
        a.part(Surface::Metal, label).cylinder(
            radius * 0.22,
            0.024,
            Transform::from_xyz(0., s.y - 0.012, 0.),
        );
        a.part(Surface::Light, label).cylinder(
            radius * p.diffuser_fraction,
            0.003,
            Transform::from_xyz(0., s.y * 0.05 + 0.003, 0.),
        );
    } else {
        a.box_part(
            p.housing,
            label,
            Vec3::new(0., s.y * 0.5, -s.z * 0.43),
            Vec3::new(s.x * 0.49, s.y * 0.86, s.z * 0.14),
            0.006,
        );
        a.box_part(
            p.housing,
            label,
            Vec3::new(0., s.y * 0.5, s.z * 0.02),
            Vec3::new(s.x, s.y * 0.84, s.z * 0.80),
            0.010,
        );
        a.box_part(
            Surface::Light,
            label,
            Vec3::new(0., s.y * 0.079, s.z * 0.13),
            Vec3::new(s.x * p.diffuser_fraction, 0.002, s.z * 0.66),
            0.001,
        );
    }
}
