use super::*;

#[test]
fn speaker_drivers_fit_squat_and_tall_enclosures() {
    for seed in 0..256 {
        for size in [
            Vec3::new(0.16533229, 0.16184825, 0.105852365),
            Vec3::new(0.24, 0.10, 0.10),
            Vec3::new(0.10, 0.32, 0.16),
        ] {
            let mut object = fixture(ObjectKind::Speaker, seed);
            object.size = size;
            let assembly = build_object(&object);
            check(&object, &assembly).unwrap_or_else(|error| panic!("{error}"));
            let (lo, hi) = assembly.bounds();
            assert!(lo.y >= -1e-6 && hi.y <= size.y + 1e-6);
        }
    }
    let scene = crate::scene::procedural_indoor::layout::IndoorManifest::generate_with_humans(
        578,
        crate::scene::procedural_indoor::layout::IndoorLayout::Mixed,
        0.65,
        0,
        0.25,
    )
    .unwrap();
    crate::scene::procedural_indoor::validation::validate_geometry(&scene).unwrap();
}

#[test]
fn descriptor_measures_surfaces_without_counting_materials_or_tessellation() {
    let o = fixture(ObjectKind::StorageBox, 1);
    let mut a = Assembly::default();
    a.box_part(
        Surface::Metal,
        "other_prop",
        Vec3::Y * o.size.y * 0.5,
        o.size,
        0.,
    );
    let before = descriptor(&o, &a);
    let mut b = Assembly::default();
    for (_, mut g) in a.parts {
        // Duplicating triangles, material, UV and vertex order do not add shape.
        g.indices.extend(g.indices.clone());
        g.uvs.fill([123., -5.]);
        b.parts.insert((Surface::Fabric, "other_prop".into()), g);
    }
    assert_eq!(distance(&before, &descriptor(&o, &b)), 0.);
    for g in b.parts.values_mut() {
        for p in &mut g.positions {
            p[0] *= 0.7;
        }
    }
    assert!(distance(&before, &descriptor(&o, &b)) > 0.03);
    assert_eq!(representatives(&[before.clone(), before], 0.015), vec![0]);
}

#[test]
fn expanded_objects_replay_and_fit_metric_size_extremes() {
    use ObjectKind::*;
    for kind in [
        Keyboard,
        Mouse,
        Notebook,
        Books,
        FloorLamp,
        DeskLamp,
        CeilingPendant,
        StorageBox,
        Printer,
        CoatRack,
        Bag,
        Speaker,
        Headphones,
        Stapler,
        TapeDispenser,
        DocumentTray,
        DeskOrganizer,
        DeskFan,
        Notepad,
        Pencil,
        Phone,
        Microphone,
        WaterBottle,
        SodaCan,
        Whiteboard,
        ProjectionScreen,
        WallShelf,
        WallSconce,
        CoffeeMaker,
        DrawerUnit,
        Rug,
        FloorOutlet,
    ] {
        for seed in 0..32 {
            let mut o = fixture(kind, seed);
            o.size *= match seed % 3 {
                0 => Vec3::new(0.85, 0.85, 1.1),
                1 => Vec3::new(1.2, 1.2, 0.85),
                _ => Vec3::ONE,
            };
            let a = build_object(&o);
            check(&o, &a).unwrap_or_else(|e| panic!("{e}; size={:?}", o.size));
            let replay = build_object(&o);
            for (key, g) in a.parts {
                let h = &replay.parts[&key];
                assert_eq!(g.positions, h.positions, "{kind:?}/{seed}");
                assert_eq!(g.normals, h.normals);
                assert_eq!(g.uvs, h.uvs);
                assert_eq!(g.indices, h.indices);
            }
        }
    }
}

#[test]
fn shell_texture_coordinates_have_two_dimensional_area() {
    let mut g = Geometry::default();
    g.housing(
        &[(0., Vec2::splat(0.1)), (0.3, Vec2::new(0.12, 0.08))],
        5.,
        Transform::IDENTITY,
    );
    for tri in g.indices.as_chunks::<3>().0 {
        let uv = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
        assert!((uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-9);
    }
}

#[test]
fn floor_lamp_emitter_tracks_the_actual_bulb_after_placement() {
    for seed in 0..32 {
        let mut o = fixture(ObjectKind::FloorLamp, seed);
        o.position = Vec3::new(2., 0.1, -3.);
        o.yaw = 0.37;
        let a = build_object(&o);
        let light = a
            .parts
            .iter()
            .find(|((s, _), _)| *s == Surface::Light)
            .unwrap()
            .1;
        let (lo, hi) = light.positions.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), p| {
                let p = Vec3::from_array(*p);
                (lo.min(p), hi.max(p))
            },
        );
        let center = o.transform().transform_point((lo + hi) * 0.5);
        assert!(center.distance(standing_lamp::emitter(&o)) < 1e-5);
    }
}
