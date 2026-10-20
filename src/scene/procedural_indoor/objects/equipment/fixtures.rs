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
    let frame = s.x.min(s.y) * p.board_frame_fraction;
    let frame_surface = if p.housing == Surface::Metal {
        Surface::Metal
    } else {
        Surface::WoodEdge
    };
    // A real backing and four rails leave the pinboard exposed. A full-depth
    // solid box here would occlude both the fabric and its attached notes.
    a.box_part(
        Surface::WoodEdge,
        label,
        Vec3::new(0., s.y * 0.5, -s.z * 0.30),
        Vec3::new(s.x, s.y, s.z * 0.40),
        0.001,
    );
    for sign in [-1., 1.] {
        a.box_part(
            frame_surface,
            label,
            Vec3::new(sign * (s.x - frame) * 0.5, s.y * 0.5, s.z * 0.15),
            Vec3::new(frame, s.y, s.z * 0.70),
            0.002,
        );
        a.box_part(
            frame_surface,
            label,
            Vec3::new(0., s.y * 0.5 + sign * (s.y - frame) * 0.5, s.z * 0.15),
            Vec3::new(s.x - frame * 2., frame, s.z * 0.70),
            0.002,
        );
    }
    a.box_part(
        Surface::FabricAlt,
        label,
        Vec3::new(0., s.y * 0.5, s.z * 0.10),
        Vec3::new(s.x - frame * 2., s.y - frame * 2., s.z * 0.40),
        0.001,
    );
    let divider = p
        .board_divider_fraction
        .map(|fraction| (fraction - 0.5) * (s.x - frame * 2.));
    if let Some(x) = divider {
        a.box_part(
            frame_surface,
            label,
            Vec3::new(x, s.y * 0.5, s.z * 0.35),
            Vec3::new(frame * 0.55, s.y - frame * 2., s.z * 0.30),
            0.001,
        );
    }
    for note in super::pinups::placements(o) {
        let mut rng = stream(note.seed, 5731);
        let width = note.size.x;
        let height = note.size.y;
        let center = Vec3::new(note.center.x, note.center.y, s.z * 0.309);
        let tf = Transform::from_translation(center).with_rotation(Quat::from_rotation_z(note.yaw));
        a.part(Surface::Paper, label)
            .cuboid(Vec3::new(width, height, s.z * 0.012), 0.0001, tf);
        let image = rng.random_bool(0.40);
        let left = rng.random_range(-0.44..-0.24);
        let right = rng.random_range(0.26..0.44);
        let top = rng.random_range(0.22..0.40);
        let lines = rng.random_range(2..=9);
        let pitch = rng.random_range(0.38..0.65) / lines as f32;
        if image {
            let slot = crate::scene::procedural_indoor::materials::variants::object_slot(
                Surface::Art,
                note.seed,
                None,
            );
            let mut picture = Assembly::default();
            super::super::presentation::face(
                &mut picture,
                Surface::Art,
                &format!("{label}#finish{slot}"),
                Vec3::new(left * width, -height * 0.35, s.z * 0.008),
                Vec3::new(right * width, height * 0.12, s.z * 0.008),
            );
            super::super::public::append_at(a, picture, tf);
        }
        for line in 0..lines {
            if image && line > 1 {
                break;
            }
            let length = width * rng.random_range((right - left) * 0.45..right - left);
            let start = Vec3::new(
                width * left,
                height * (top - line as f32 * pitch),
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
                width * rng.random_range(-0.22..0.22),
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
        // Conical shell with real inner wall and rim, suspended by a connected
        // stem. Geometry remains below the ceiling and above the seated zone.
        let style = super::super::luminaires::parameters(o.seed);
        let taper = 0.48 + style.shoulder * 0.46;
        super::super::luminaires::shade(
            a,
            label,
            &style,
            Vec3::Y * s.y * 0.05,
            radius,
            shade - s.y * 0.05,
            taper,
        );
        super::super::luminaires::bulb(
            a,
            label,
            &style,
            Vec3::Y * (s.y * 0.05 + (shade - s.y * 0.05) * 0.45),
            (radius * 0.19).min((shade - s.y * 0.05) * 0.17),
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
        let upper = radius
            * if matches!(style.shape, 1 | 3) {
                taper.clamp(0.88, 1.)
            } else {
                taper
            };
        for sign in [-1., 1.] {
            a.part(style.trim, label).rod(
                Vec3::Y * (shade - 0.002),
                Vec3::new(sign * (upper - 0.001), shade - 0.002, 0.),
                0.0015,
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // Frontmost surface at a point, independent of materials or normals.
    fn front(a: &Assembly, q: Vec2) -> Option<(f32, Surface)> {
        a.parts
            .iter()
            .flat_map(|((surface, _), g)| {
                g.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter_map(move |indices| {
                        let p = [indices[0], indices[1], indices[2]]
                            .map(|i| Vec3::from_array(g.positions[i as usize]));
                        let area = (p[1].xy() - p[0].xy()).perp_dot(p[2].xy() - p[0].xy());
                        if area.abs() < 1e-10 {
                            return None;
                        }
                        let weights = [
                            (p[1].xy() - q).perp_dot(p[2].xy() - q) / area,
                            (p[2].xy() - q).perp_dot(p[0].xy() - q) / area,
                            (p[0].xy() - q).perp_dot(p[1].xy() - q) / area,
                        ];
                        weights.iter().all(|&w| w >= -1e-5).then_some((
                            p[0].z * weights[0] + p[1].z * weights[1] + p[2].z * weights[2],
                            *surface,
                        ))
                    })
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
    }

    #[test]
    fn pinboards_have_exposed_fabric_and_notes_inside_their_placement_envelope() {
        for seed in 0..128 {
            let o = super::super::super::audit::fixture(ObjectKind::NoticeBoard, seed);
            let a = super::super::super::build_object(&o);
            super::super::super::audit::check(&o, &a).unwrap();
            let mut fabric = 0;
            let mut paper = 0;
            for y in 0..24 {
                for x in 0..32 {
                    let q = Vec2::new(
                        ((x as f32 + 0.5) / 32. - 0.5) * o.size.x,
                        (y as f32 + 0.5) / 24. * o.size.y,
                    );
                    match front(&a, q).map(|hit| hit.1) {
                        Some(Surface::FabricAlt) => fabric += 1,
                        Some(Surface::Paper) => paper += 1,
                        _ => {}
                    }
                }
            }
            assert!(
                fabric > 128 && paper > 0,
                "seed {seed}: fabric {fabric}, paper {paper}"
            );
        }
    }
}
