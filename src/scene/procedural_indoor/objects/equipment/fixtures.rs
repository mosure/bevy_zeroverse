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
            [-p.shelf_span, p.shelf_span][i]
        } else {
            (i as f32 - 1.) * p.shelf_span
        } * s.x;
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(x, s.y * 0.5, -s.z * 0.5 + 0.008),
            Vec3::new(0.022, s.y, 0.016),
            0.002,
        );
        if p.bracket_style != 1 {
            a.part(Surface::Metal, label).rod(
                Vec3::new(x, 0.021, -s.z * 0.5 + 0.017),
                Vec3::new(x, s.y - thickness - 0.013, s.z * 0.39),
                0.008,
            );
        } else {
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(x, s.y - thickness - 0.008, 0.),
                Vec3::new(0.022, 0.016, s.z * 0.93),
                0.002,
            );
        }
        if p.bracket_style == 2 {
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(x, 0.012, -s.z * 0.27),
                Vec3::new(0.024, 0.024, s.z * 0.44),
                0.003,
            );
        }
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
    super::super::presentation::screen(a, o);
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
        let bottom = s.y * (0.5 - p.sconce_height * 0.5);
        let width = s.x * p.sconce_width * 0.5;
        a.part(p.housing, label).housing(
            &[
                (bottom, Vec2::new(width, s.z * 0.4)),
                (s.y - bottom, Vec2::new(width * p.sconce_taper, s.z * 0.4)),
            ],
            5.,
            Transform::from_xyz(0., 0., s.z * 0.02),
        );
        a.box_part(
            Surface::Light,
            label,
            Vec3::new(0., bottom - 0.0011, s.z * 0.13),
            Vec3::new(
                s.x * p.sconce_width * p.diffuser_fraction,
                0.002,
                s.z * 0.50,
            ),
            0.001,
        );
    }
}
