use super::*;

pub(super) fn notice_board(a: &mut Assembly, o: &IndoorObject, p: &EquipmentProgram, version: u32) {
    let s = o.size;
    let label = o.kind.class_name();
    let frame = s.x.min(s.y) * p.board_frame_fraction;
    let section = super::board_section::parameters(o);
    let frame_surface = section.frame_surface;
    let rear = -s.z * 0.5;
    let backing_front = rear + section.backing_m;
    let front = rear + section.depth_m;
    let bevel = frame * section.bevel_fraction;
    // A real backing and four rails leave the pinboard exposed. A full-depth
    // solid box here would occlude both the fabric and its attached notes.
    a.box_part(
        Surface::WoodEdge,
        label,
        Vec3::new(0., s.y * 0.5, rear + section.backing_m * 0.5),
        Vec3::new(s.x, s.y, section.backing_m),
        0.,
    );
    for sign in [-1., 1.] {
        a.box_part(
            frame_surface,
            label,
            Vec3::new(
                sign * (s.x - frame) * 0.5,
                s.y * 0.5,
                (backing_front + front) * 0.5,
            ),
            Vec3::new(frame, s.y, front - backing_front),
            bevel,
        );
        a.box_part(
            frame_surface,
            label,
            Vec3::new(
                0.,
                s.y * 0.5 + sign * (s.y - frame) * 0.5,
                (backing_front + front) * 0.5,
            ),
            Vec3::new(s.x - frame * 2., frame, front - backing_front),
            bevel,
        );
    }
    let facing = super::board_section::Facing::for_board(o, version);
    let cork_label = crate::scene::procedural_indoor::materials::cork::slot(facing)
        .map(|slot| format!("{label}#finish{slot}"));
    a.box_part(
        Surface::FabricAlt,
        cork_label.as_deref().unwrap_or(label),
        Vec3::new(0., s.y * 0.5, (backing_front + section.face_z_m) * 0.5),
        Vec3::new(
            s.x - frame * 2.,
            s.y - frame * 2.,
            section.face_z_m - backing_front,
        ),
        0.,
    );
    let divider = p
        .board_divider_fraction
        .map(|fraction| (fraction - 0.5) * (s.x - frame * 2.));
    if let Some(x) = divider {
        a.box_part(
            frame_surface,
            label,
            Vec3::new(x, s.y * 0.5, (section.face_z_m + front) * 0.5),
            Vec3::new(frame * 0.55, s.y - frame * 2., front - section.face_z_m),
            0.001,
        );
    }
    for note in super::pinups::placements_at(o, p, version) {
        super::pinups::sheet::build(a, o, &note, section.face_z_m, version);
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
    fn pinboards_expose_the_selected_stock_and_only_the_notes_actually_posted() {
        for seed in 0..128 {
            let mut o = super::super::super::audit::fixture(ObjectKind::NoticeBoard, seed);
            o.finish_seed = None;
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
            let version = bevy_zeroverse_capture::GENERATOR_VERSION;
            let facing = super::super::board_section::Facing::for_board(&o, version);
            if let Some(i) = crate::scene::procedural_indoor::materials::cork::slot(facing) {
                assert!(a.parts.keys().any(|(s, l)| *s == Surface::FabricAlt
                    && crate::scene::procedural_indoor::materials::finish_slot(*s, l) == Some(i)));
            }
            assert!(fabric > 128, "seed {seed}: exposed {fabric}");
            assert_eq!(paper == 0, super::super::pinups::placements(&o).is_empty());
        }
    }
}
