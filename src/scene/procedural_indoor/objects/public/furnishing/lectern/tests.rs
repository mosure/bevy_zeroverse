use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object_at_version};

#[test]
fn lectern_pages_contact_the_tilted_desk_and_stop_clear_of_the_microphone() {
    let mut mics = [0usize; 3];
    let mut sides = [0usize; 2];
    let mut sizes = std::collections::BTreeSet::new();
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::Lectern, seed);
        install(&mut o, 1.);
        let p = parameters(&o);
        let desk = p.reading_desk.as_ref().unwrap();
        let width = o.size.x * p.board_fraction.x;
        let depth = o.size.z * 0.88;
        let thickness = p.thickness.min(o.size.y * 0.035);
        let y = o.size.y * 0.97 - depth * p.desk_angle.sin() * 0.5 - thickness - desk.deck_drop_m;
        let pose = Transform::from_xyz(o.size.x * p.panel_offset * 0.2, y, 0.)
            .with_rotation(Quat::from_rotation_x(p.desk_angle));
        let local = pose.compute_affine().inverse();
        let mesh = build_object_at_version(&o, None, VERSION);
        audit::check(&o, &mesh).unwrap();
        let (lo, hi) = mesh.bounds();
        // A room-sized AABB cannot catch a crossbar breaking through a thin desktop.
        // The board's top vertices lie on its perimeter; the stop is at its downhill edge.
        for ((_, label), part) in &mesh.parts {
            if label == o.kind.class_name() {
                for &v in &part.positions {
                    let point = local.transform_point3(Vec3::from_array(v));
                    if point.x.abs() < width * 0.32 && point.z.abs() < depth * 0.32 {
                        assert!(
                            point.y <= thickness * 0.5 - 0.001,
                            "support through desk at seed {seed}"
                        );
                    }
                }
            }
        }
        assert!(lo
            .cmpge(Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5) - 0.001)
            .all());
        assert!(hi
            .cmple(Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5) + 0.001)
            .all());
        let mic = desk.microphone;
        mics[match mic.map(|m| m.style) {
            None => 0,
            Some(MicStyle::Boundary) => 1,
            Some(MicStyle::Gooseneck) => 2,
        }] += 1;
        if let Some(m) = mic {
            sides[usize::from(m.socket_xz_m.x > 0.)] += 1;
            assert!(mesh.parts.keys().any(|(_, label)| label == "microphone"));
            if m.style == MicStyle::Gooseneck {
                assert!(y >= 0.90 - 1e-5 && m.rise_m > 0.10 && m.reach_m > 0.06);
            }
        }
        let notes = desk.notes.unwrap();
        sizes.insert((notes.sheet_m.x * 1000.).round() as i32);
        let sheet = mesh
            .parts
            .iter()
            .find(|((s, _), _)| *s == Surface::PrintedPaper)
            .unwrap()
            .1;
        assert_eq!(sheet.positions.len(), 4);
        let pts: Vec<_> = sheet
            .positions
            .iter()
            .map(|&v| local.transform_point3(Vec3::from_array(v)))
            .collect();
        let downhill = pts.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max);
        assert!((downhill + desk.stop_depth_m * 0.5 - desk.stop_z_m).abs() < 0.0001);
        for point in pts {
            assert!((point.y - thickness * 0.5 - notes.stack_m).abs() < 0.0001);
            assert!(point.x.abs() < width * 0.5 && point.z.abs() < depth * 0.5);
            if let Some(m) = mic {
                assert!((point.x - m.socket_xz_m.x).abs() >= 0.0309);
            }
        }
        // Changing finish cannot move any part of the microphone or the notes.
        o.finish_seed = Some(seed.wrapping_add(17));
        assert_eq!(
            serde_json::to_value(desk).unwrap(),
            serde_json::to_value(parameters(&o).reading_desk.unwrap()).unwrap()
        );
        install(&mut o, 0.);
        assert!(parameters(&o).reading_desk.unwrap().notes.is_none());
    }
    assert!(mics.iter().all(|&n| n > 10) && sides.iter().all(|&n| n > 20));
    assert_eq!(sizes.len(), 4);
}

#[test]
fn older_lectern_meshes_ignore_installation_flags() {
    let mut o = audit::fixture(ObjectKind::Lectern, 19);
    o.variant = 0;
    let original = build_object_at_version(&o, None, VERSION - 1);
    assert!(parameters_at(&o, VERSION - 1).reading_desk.is_none());
    install(&mut o, 1.);
    let same = build_object_at_version(&o, None, VERSION - 1);
    assert_eq!(
        same.parts.keys().collect::<Vec<_>>(),
        original.parts.keys().collect::<Vec<_>>()
    );
    for (key, g) in same.parts {
        assert_eq!(g.positions, original.parts[&key].positions);
        assert_eq!(g.uvs, original.parts[&key].uvs);
        assert_eq!(g.indices, original.parts[&key].indices);
    }
}
