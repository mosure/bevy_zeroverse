use super::*;
use crate::scene::procedural_indoor::{
    layout::ObjectKind,
    objects::{self, audit},
};

fn policy() -> IndoorMotionConfig {
    IndoorMotionConfig {
        chair_fraction: 1.,
        appliance_fraction: 1.,
        cabinet_fraction: 1.,
        prop_fraction: 1.,
        cart_fraction: 1.,
        max_mechanisms: 128,
        ..default()
    }
}
fn fixture(kind: ObjectKind, seed: u64) -> super::super::super::layout::IndoorObject {
    let mut o = audit::fixture(kind, seed);
    o.id = 0;
    o.position = Vec3::ZERO;
    o.yaw = 0.;
    o
}
fn room() -> (IndoorManifest, SceneGeometry) {
    let (mut s, g) = super::tests::empty();
    s.envelope = None;
    s.ceiling_installation = None;
    s.room_size = Vec3::new(30., 5., 30.);
    s.indoor_motion = Some(IndoorMotionReport::new(policy()));
    (s, g)
}
fn add(
    scene: &mut IndoorManifest,
    g: &mut SceneGeometry,
    mut o: super::super::super::layout::IndoorObject,
) {
    o.id = scene.objects.len();
    g.objects
        .push(objects::build_object_with_motion(&o, Some(&policy())));
    scene.objects.push(o);
}
fn rest_vertices(a: &Assembly) -> Vec<[u32; 8]> {
    let mut values: Vec<_> = a
        .parts
        .values()
        .flat_map(|g| {
            g.positions
                .iter()
                .zip(&g.normals)
                .zip(&g.uvs)
                .map(|((p, n), uv)| {
                    [p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]].map(f32::to_bits)
                })
        })
        .collect();
    values.sort_unstable();
    values
}

#[test]
fn all_unoccupied_chair_and_stool_families_move_but_recliners_and_fixed_seats_do_not() {
    for family in 0..=10 {
        for seed in 0..8 {
            let (mut scene, mut g) = room();
            let mut o = fixture(ObjectKind::Chair, seed);
            o.variant = family;
            if family == 8 {
                o.size = Vec3::new(0.95, 1., 0.95);
            }
            add(&mut scene, &mut g, o);
            assert!(!g.objects[0].joints.is_empty(), "family {family}/{seed}");
            prepare(&mut scene, &mut g);
            let report = &scene.indoor_motion.as_ref().unwrap().mechanisms;
            assert!(
                report.iter().any(|m| m.rejection.is_none()),
                "{family}/{seed}: {report:?}"
            );
        }
    }
    let mut o = fixture(ObjectKind::Chair, 0);
    o.variant = 11;
    assert!(objects::build_object_with_motion(&o, Some(&policy()))
        .joints
        .is_empty());
    o.variant = 8;
    o.size = Vec3::new(0.95, 1.22, 1.5);
    o.seed = (0..100)
        .find(|&seed| {
            o.seed = seed;
            objects::seating::is_recliner(&o)
        })
        .unwrap();
    assert!(objects::build_object_with_motion(&o, Some(&policy()))
        .joints
        .is_empty());
}

#[test]
fn operated_appliances_and_laptops_keep_rest_meshes_and_admit_real_motion() {
    use ObjectKind::*;
    let mut stats = std::collections::BTreeMap::new();
    for kind in [
        Refrigerator,
        Oven,
        Range,
        Microwave,
        Dishwasher,
        DrawerUnit,
        Cabinet,
        KitchenCupboard,
        KitchenCounter,
        Pantry,
        Laptop,
    ] {
        let mut admitted = 0;
        let mut selected = 0;
        let mut reasons = std::collections::BTreeMap::<String, usize>::new();
        for seed in 0..12 {
            let (mut scene, mut g) = room();
            let o = fixture(kind, seed);
            let rest = objects::build_object(&o);
            add(&mut scene, &mut g, o);
            // Splitting articulated parts must not alter material coordinates.
            assert_eq!(
                rest_vertices(&rest),
                rest_vertices(&g.objects[0]),
                "{kind:?}/{seed}"
            );
            prepare(&mut scene, &mut g);
            for m in &scene.indoor_motion.as_ref().unwrap().mechanisms {
                selected += 1;
                if let Some(reason) = &m.rejection {
                    *reasons.entry(reason.clone()).or_default() += 1;
                } else {
                    admitted += 1;
                    assert!(m.angle_radians.abs() + m.translation_m.length() > 0.001);
                }
            }
        }
        stats.insert(
            format!("{kind:?}"),
            serde_json::json!({"selected":selected,"admitted":admitted,"rejections":reasons}),
        );
    }
    std::fs::create_dir_all("out/object_motion_review").unwrap();
    std::fs::write(
        "out/object_motion_review/appliance_admission.json",
        serde_json::to_vec_pretty(&stats).unwrap(),
    )
    .unwrap();
    assert!(
        stats.values().all(|s| s["admitted"].as_u64().unwrap() > 0
            && s["admitted"].as_u64().unwrap() * 2 >= s["selected"].as_u64().unwrap()),
        "{stats:#?}"
    );
}

#[test]
fn maximum_policy_moves_many_supported_props_without_overhang_or_peer_clipping() {
    use ObjectKind::*;
    let (mut scene, mut g) = room();
    let mut table = fixture(Cabinet, 0);
    table.size = Vec3::new(9., 0.75, 9.);
    // A rectangular test support with exact contact plane; no legs needed here.
    let mut top = Assembly::default();
    top.box_part(
        Surface::Wood,
        "table",
        Vec3::Y * 0.725,
        Vec3::new(9., 0.05, 9.),
        0.,
    );
    scene.objects.push(table);
    g.objects.push(top);
    let kinds = [
        Phone,
        Pencil,
        Mug,
        Mouse,
        Keyboard,
        Notepad,
        WaterBottle,
        Stapler,
        CoffeeCup,
        SodaCan,
        TissueBox,
        Headphones,
        SoapDispenser,
        Sponge,
        CuttingBoard,
        Bowl,
        Dishware,
        Bread,
        UtensilHolder,
        KnifeBlock,
        PenHolder,
        DocumentTray,
        TapeDispenser,
        DeskOrganizer,
        Notebook,
        Books,
        Kettle,
        DeskLamp,
        DeskFan,
        Speaker,
        DryingRack,
        PaperTowel,
        KitchenTowel,
        FruitBowl,
        Vase,
        Microphone,
    ];
    for (i, kind) in kinds.into_iter().enumerate() {
        let mut o = fixture(kind, i as u64 + 1);
        o.support = Some(0);
        o.position = Vec3::new(
            (i % 6) as f32 * 1.3 - 3.25,
            0.75,
            (i / 6) as f32 * 1.3 - 3.25,
        );
        add(&mut scene, &mut g, o);
    }
    prepare(&mut scene, &mut g);
    let report = scene.indoor_motion.as_ref().unwrap();
    assert_eq!(report.mechanisms.len(), kinds.len());
    assert!(
        report
            .mechanisms
            .iter()
            .filter(|m| m.rejection.is_none())
            .count()
            >= 32,
        "{report:#?}"
    );
    for m in &report.mechanisms {
        if m.rejection.is_some() {
            continue;
        }
        let id = m.object_id.unwrap();
        let o = &scene.objects[id];
        let a = &g.objects[id];
        let joint = &a.joints[0];
        assert!(support_clear(&scene, o, &sweeps(a, joint, o.transform())));
        for step in 0..=100 {
            let delta = m.rigid_transform_at(step as f32 / 100.).unwrap();
            for p in a.parts.values().flat_map(|g| &g.positions) {
                let p = delta.transform_point(o.transform().transform_point(Vec3::from_array(*p)));
                assert!(p.y >= 0.749 && p.x.abs() < 4.5 && p.z.abs() < 4.5);
            }
        }
    }
    let old = report.clone();
    prepare(&mut scene, &mut g);
    assert_eq!(&old, scene.indoor_motion.as_ref().unwrap());
}

#[test]
fn loaded_cart_carries_annotation_instances_and_rejects_connected_wires() {
    let (mut scene, mut g) = room();
    let cart = fixture(ObjectKind::MobileCart, 7);
    let height = cart.size.y;
    add(&mut scene, &mut g, cart);
    let mut phone = fixture(ObjectKind::Phone, 12);
    phone.support = Some(0);
    phone.position = Vec3::new(0.05, height, 0.05);
    add(&mut scene, &mut g, phone);
    prepare(&mut scene, &mut g);
    let report = scene.indoor_motion.clone().unwrap();
    assert_eq!(report.mechanisms.len(), 1);
    let m = &report.mechanisms[0];
    assert!(m.rejection.is_none(), "{m:?}");
    assert_eq!(m.carried_object_ids, [1]);
    assert!(m.semantic_labels.contains(&"phone".to_owned()));
    assert!(m.semantic_labels.contains(&"mobile_cart".to_owned()));
    let child = &g.objects[1].joints[0];
    for t in [0., 1., 0.23, 0.23, 0.] {
        let parent_delta = m.rigid_transform_at(t).unwrap();
        let child_delta = child.transform(child.amount(t));
        for p in g.objects[1].parts.values().flat_map(|g| &g.positions) {
            let p = Vec3::from_array(*p);
            let expected =
                parent_delta.transform_point(scene.objects[1].transform().transform_point(p));
            let actual = scene.objects[1]
                .transform()
                .transform_point(child_delta.transform_point(p));
            assert!(expected.distance(actual) < 1e-5);
        }
    }
    prepare(&mut scene, &mut g);
    assert_eq!(scene.indoor_motion.as_ref().unwrap(), &report);
    let mut cable = fixture(ObjectKind::Cable, 1);
    cable.id = 2;
    cable.cable_target = Some(1);
    scene.objects.push(cable);
    g.objects.push(Assembly::default());
    prepare(&mut scene, &mut g);
    assert_eq!(
        scene.indoor_motion.as_ref().unwrap().mechanisms[0]
            .rejection
            .as_deref(),
        Some("connected_cable")
    );
    assert!(
        g.objects[1].joints.is_empty(),
        "rejected cargo must not keep a stale animation"
    );
}

#[test]
fn props_cannot_slide_off_round_tables_even_with_clear_geometry() {
    let (mut scene, mut g) = room();
    let mut table = fixture(ObjectKind::Table, 5);
    table.size = Vec3::new(0.7, 0.75, 0.7);
    let mut phone = fixture(ObjectKind::Phone, 1);
    phone.id = 1;
    phone.support = Some(0);
    phone.position = Vec3::Y * 0.75;
    add(&mut scene, &mut g, table);
    add(&mut scene, &mut g, phone);
    g.objects[1].joints[0].translation = Vec3::X * 4.;
    g.objects[1].joints[0].angle = 0.;
    prepare(&mut scene, &mut g);
    let m = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
    assert!(m.rejection.is_none(), "{m:?}");
    assert!(m.translation_m.length() < 0.35);
    assert!(support_clear(
        &scene,
        &scene.objects[1],
        &sweeps(
            &g.objects[1],
            &g.objects[1].joints[0],
            scene.objects[1].transform()
        )
    ));
}

#[test]
fn moving_desk_lamp_keeps_direct_light_attached_during_reverse_playback() {
    let (mut scene, mut g) = room();
    let mut support = fixture(ObjectKind::Cabinet, 0);
    support.size = Vec3::new(2., 0.75, 2.);
    scene.objects.push(support);
    g.objects.push(Assembly::default());
    let mut lamp = fixture(ObjectKind::DeskLamp, 17);
    lamp.support = Some(0);
    lamp.position = Vec3::Y * 0.75;
    add(&mut scene, &mut g, lamp);
    prepare(&mut scene, &mut g);
    let m = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
    assert!(m.rejection.is_none(), "{m:?}");
    let mut app = App::new();
    app.init_resource::<Assets<Mesh>>()
        .init_resource::<crate::camera::Playback>()
        .add_plugins(IndoorMotionPlugin);
    let world = app.world_mut();
    let root = world.spawn_empty().id();
    super::super::super::architecture::spawn_lights(
        &scene,
        super::super::super::IndoorQuality::Auto,
        root,
        &mut world.commands(),
    );
    world.flush();
    let (entity, rest) = world
        .query_filtered::<(Entity, &Transform), With<AnimatedLight>>()
        .single(world)
        .map(|(e, t)| (e, *t))
        .unwrap();
    for progress in [0.3, 1., 0.3, 0.] {
        app.world_mut()
            .resource_mut::<crate::camera::Playback>()
            .progress = progress;
        app.update();
        let actual = *app.world().get::<Transform>(entity).unwrap();
        let expected = m.rigid_transform_at(progress).unwrap().mul_transform(rest);
        assert!(actual.translation.distance(expected.translation) < 1e-6);
        assert!(actual.rotation.dot(expected.rotation).abs() > 0.999999);
    }
}

#[test]
fn island_doors_and_drawer_boxes_operate_within_real_carcasses() {
    for seed in 0..8 {
        let (mut scene, mut g) = room();
        let mut o = fixture(ObjectKind::KitchenCounter, seed);
        objects::kitchen::configure_island(&mut o);
        let bays = objects::kitchen::island_parameters(&o).unwrap().bays as usize;
        let rest = objects::build_object(&o);
        audit::check(&o, &rest).unwrap();
        add(&mut scene, &mut g, o);
        assert_eq!(rest_vertices(&rest), rest_vertices(&g.objects[0]));
        assert!(g.objects[0].joints.len() >= bays * 2);
        prepare(&mut scene, &mut g);
        let report = scene.indoor_motion.as_ref().unwrap();
        let accepted = report
            .mechanisms
            .iter()
            .filter(|m| m.rejection.is_none())
            .count();
        assert!(accepted >= bays, "{seed}: {report:#?}");
    }
}

#[test]
fn motion_geometry_gallery_uses_the_actual_admitted_mesh_transforms() {
    use ObjectKind::*;
    let kinds = [
        Chair,
        MobileCart,
        Phone,
        Pencil,
        Laptop,
        Refrigerator,
        Oven,
        Dishwasher,
        KitchenCupboard,
    ];
    let n = 192;
    let mut gallery = image::RgbImage::new(n * 3, n * kinds.len() as u32);
    let mut metadata = vec![];
    for (row, kind) in kinds.into_iter().enumerate() {
        let (mut scene, mut g) = room();
        // Use a compact, uncluttered support so phone/pencil trajectories can
        // be reviewed without GPU activity alongside ongoing training.
        if matches!(kind, Phone | Pencil) {
            let mut support = fixture(Cabinet, 0);
            support.size = Vec3::new(0.7, 0.75, 0.7);
            scene.objects.push(support);
            g.objects.push(Assembly::default());
        }
        let owner = scene.objects.len();
        let seed = u64::from(kind == KitchenCupboard);
        let mut o = fixture(kind, seed);
        if kind == Chair {
            o.variant = 6;
        }
        if owner > 0 {
            o.support = Some(0);
            o.position.y = 0.75;
        }
        add(&mut scene, &mut g, o);
        prepare(&mut scene, &mut g);
        let a = &g.objects[owner];
        let posed = |t: f32| {
            let mut posed = Assembly::default();
            for ((surface, label), mesh) in &a.parts {
                let tf = tag(label)
                    .and_then(|i| a.joints.get(i))
                    .filter(|j| j.accepted)
                    .map_or(Transform::IDENTITY, |j| j.transform(j.amount(t)));
                let mut mesh = mesh.clone();
                for p in &mut mesh.positions {
                    *p = tf.transform_point(Vec3::from_array(*p)).to_array();
                }
                for n in &mut mesh.normals {
                    *n = (tf.rotation * Vec3::from_array(*n)).to_array();
                }
                posed.part(*surface, label).append(mesh);
            }
            posed
        };
        let b = (0..=20)
            .map(|i| posed(i as f32 / 20.).bounds())
            .reduce(union)
            .unwrap();
        for (column, t) in [0., 0.5, 1.].into_iter().enumerate() {
            let img = super::super::super::geometry::preview::orthographic(
                &posed(t),
                (b.0 + b.1) * 0.5,
                (b.1 - b.0).length() * 1.05,
                n,
                Vec3::new(0.7, 0.45, 1.),
            );
            image::imageops::replace(
                &mut gallery,
                &img,
                column as i64 * n as i64,
                row as i64 * n as i64,
            );
        }
        metadata.push(
            serde_json::json!({"row":row,"kind":format!("{kind:?}"),"seed":seed,
            "motion":scene.indoor_motion.as_ref().unwrap().mechanisms}),
        );
    }
    std::fs::create_dir_all("out/object_motion_review").unwrap();
    gallery
        .save("out/object_motion_review/mobility.png")
        .unwrap();
    std::fs::write(
        "out/object_motion_review/gallery.json",
        serde_json::to_vec_pretty(&serde_json::json!({"cpu_flat_shaded":true,
            "normalized_progress":[0.,0.5,1.],"rows":metadata}))
        .unwrap(),
    )
    .unwrap();
}
