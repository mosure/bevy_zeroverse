use super::*;
use objects::presentation::projection::{self, Deployment, Mechanism};

#[test]
fn projection_screen_stock_feeds_at_constant_scale_and_the_hem_tracks_the_report() {
    let (blank, mut g) = room();
    let mut counts = [0; 3];
    let mut states = [0; 3];
    let mut half_selected = 0;
    let mut directions = [0; 2];
    for seed in 0..64 {
        let mut scene = blank.clone();
        let mut o = fixture(ObjectKind::ProjectionScreen, seed);
        let (width, depth) = [(1.18, 0.057), (2.72, 0.118), (3.10, 0.066)][seed as usize % 3];
        o.size = projection::dimensions(seed, width, depth);
        o.position = Vec3::new(1.1, 0.63, -0.77);
        o.yaw = seed as f32 * 0.43;
        let p = projection::parameters(&o);
        let i = p.installation.as_ref().unwrap();
        counts[i.mechanism as usize] += 1;
        let rest = objects::build_manifest_object(&scene, &o);
        assert!(rest.joints.is_empty());
        let mut archived = scene.clone();
        archived.generator_version = PROJECTION_MOTION_VERSION - 1;
        archived
            .indoor_motion
            .as_mut()
            .unwrap()
            .policy
            .projection_screen_fraction = 1.;
        let old = objects::build_manifest_object(&archived, &o);
        assert!(old.joints.is_empty());
        assert_eq!(rest_vertices(&old), rest_vertices(&rest));
        let half = IndoorMotionConfig {
            projection_screen_fraction: 0.5,
            ..policy()
        };
        half_selected += usize::from(
            !objects::build_object_with_motion(&o, Some(&half))
                .joints
                .is_empty(),
        );
        scene
            .indoor_motion
            .as_mut()
            .unwrap()
            .policy
            .projection_screen_fraction = 1.;
        g.objects.clear();
        add(&mut scene, &mut g, o);
        if i.mechanism == Mechanism::FixedFrame {
            assert!(g.objects[0].joints.is_empty());
            assert_eq!(rest_vertices(&rest), rest_vertices(&g.objects[0]));
            continue;
        }
        states[i.deployment as usize] += 1;
        prepare(&mut scene, &mut g);
        let report = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
        assert!(report.rejection.is_none(), "seed {seed}: {report:?}");
        assert_eq!(report.semantic_labels, ["projection_screen"]);
        let travel = report.projection_screen_travel.as_ref().unwrap();
        assert_eq!(travel.end_drop_m, travel.requested_end_drop_m);
        let diff = travel.end_drop_m - travel.start_drop_m;
        assert!(diff.abs() > 0.05);
        directions[usize::from(diff > 0.)] += 1;
        let a = &g.objects[0];
        audit::check(&scene.objects[0], a).unwrap();
        let j = &a.joints[0];
        let law = j.cloth_motion.unwrap();
        let o = &scene.objects[0];
        for progress in [0., 0.19, 0.51, 0.8, 1., 0.51, 0.] {
            let t = j.amount(progress);
            let drop = report.projection_screen_drop_at(progress).unwrap();
            assert!((drop - (travel.start_drop_m - j.transform(t).translation.y)).abs() < 1e-5);
            for key in &j.parts {
                let start = &a.parts[key];
                if let Some(end) = j.morph.get(key) {
                    assert_eq!(start.uvs, end.uvs);
                    assert_eq!(start.normals, end.normals);
                    assert_eq!(start.indices, end.indices);
                    assert!(law.applies(&key.1));
                    for (k, (&s, &e)) in start
                        .positions
                        .iter()
                        .zip(&end.positions)
                        .enumerate()
                        .step_by(7)
                    {
                        let pos =
                            law.at(t)
                                .interpolate(Vec3::from_array(s), Vec3::from_array(e), t);
                        assert!(pos.y < p.sheet_top_m + 0.38 * p.case_height_m);
                        if key.0 == Surface::Paper {
                            let expected = -start.uvs[k][1] + travel.start_drop_m - drop;
                            if pos.y < p.sheet_top_m || expected < p.sheet_top_m {
                                assert!(
                                    (pos.y - expected).abs() < 0.00015,
                                    "seed {seed}, at {progress}: {pos:?} vs UV {expected}"
                                );
                            }
                        }
                        let world = o.transform().transform_point(pos);
                        assert!(report
                            .swept_bounds
                            .iter()
                            .any(|&(lo, hi)| world.cmpge(lo - 0.0001).all()
                                && world.cmple(hi + 0.0001).all()));
                    }
                } else {
                    assert!(!law.applies(&key.1));
                    let moved = j.transform(t).translation.y;
                    assert!(
                        (p.sheet_bottom_m + 2. * p.bar_radius_m + moved - (p.sheet_top_m - drop))
                            .abs()
                            < 1e-5
                    );
                }
            }
        }
    }
    eprintln!("64 installations: fixed/spring/motor {counts:?}, presented/partial/stowed retractables {states:?}, raising/lowering {directions:?}, {half_selected} selected at probability 0.5");
    assert!(
        counts.iter().all(|&n| n > 4)
            && states.iter().all(|&n| n > 2)
            && directions.iter().all(|&n| n > 5)
    );
    assert!((12..45).contains(&half_selected));
}

#[test]
fn projection_screen_reserves_the_whole_stroke_and_rechecks_after_obstructions_change() {
    let (mut scene, mut g) = room();
    scene
        .indoor_motion
        .as_mut()
        .unwrap()
        .policy
        .projection_screen_fraction = 1.;
    let o = (0..100)
        .find_map(|seed| {
            let mut o = fixture(ObjectKind::ProjectionScreen, seed);
            o.size = projection::dimensions(seed, 2.3, 0.084);
            let p = projection::parameters(&o);
            (p.installation.unwrap().deployment == Deployment::Stowed).then_some(o)
        })
        .unwrap();
    let p = projection::parameters(&o);
    add(&mut scene, &mut g, o);
    let report =
        |scene: &IndoorManifest| scene.indoor_motion.as_ref().unwrap().mechanisms[0].clone();
    prepare(&mut scene, &mut g);
    let initial = report(&scene);
    let end = initial.projection_screen_travel.as_ref().unwrap();
    assert!(initial.rejection.is_none());
    let blocker = |f: f32| {
        let center = Vec3::new(
            p.sheet_center_x_m + p.sheet_width_m * 0.28,
            p.sheet_bottom_m - (end.end_drop_m - end.start_drop_m) * f,
            p.sheet_front_m,
        );
        let mut a = Assembly::default();
        a.box_part(
            Surface::Wood,
            "shelf",
            center,
            Vec3::new(0.24, 0.06, 0.12),
            0.,
        );
        a
    };
    g.architecture = blocker(0.64);
    prepare(&mut scene, &mut g);
    let short = report(&scene);
    assert!(
        short.rejection.is_none() && short.adaptation.as_deref() == Some("shortened"),
        "{short:?}"
    );
    let span = short.projection_screen_travel.as_ref().unwrap();
    assert!(span.end_drop_m > span.start_drop_m && span.end_drop_m < end.end_drop_m);
    g.architecture = blocker(0.);
    prepare(&mut scene, &mut g);
    let blocked = report(&scene);
    assert_eq!(blocked.rejection.as_deref(), Some("static_geometry"));
    assert_eq!(
        blocked.projection_screen_drop_at(1.),
        Some(end.start_drop_m)
    );
    assert!(blocked.swept_bounds.is_empty());
    g.architecture = default();
    prepare(&mut scene, &mut g);
    assert_eq!(report(&scene), initial);
}

#[test]
#[ignore = "compares full and spatially scoped collision trees on furnished rooms"]
fn projection_screen_clearance_matches_full_geometry_in_furnished_rooms() {
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    let mut samples = vec![];
    for (index, seed) in [78703, 78709, 78719, 78720, 78742, 78766]
        .into_iter()
        .enumerate()
    {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.72, 2, 0.25).unwrap();
        let policy = IndoorMotionConfig {
            projection_screen_fraction: 1.,
            desk_fan_fraction: 1.,
            fan_fraction: 0.,
            chair_fraction: 0.,
            door_fraction: 0.,
            cabinet_fraction: 0.,
            covering_fraction: 0.,
            appliance_fraction: 0.,
            prop_fraction: 0.,
            cart_fraction: 0.,
            ..default()
        };
        scene.indoor_motion = Some(IndoorMotionReport::new(policy));
        let (mut geometry, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
        let screens = geometry
            .objects
            .iter()
            .flat_map(|a| &a.joints)
            .filter(|j| matches!(j.cloth_motion, Some(ClothMotion::ProjectionScreen { .. })))
            .count();
        if screens == 0 {
            continue;
        }
        let measure = |scoped| {
            let start = std::time::Instant::now();
            let tree = if scoped {
                Tree::for_admission(&scene, &geometry)
            } else {
                Tree::new(&scene, &geometry)
            };
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let mut hits = vec![];
            for (id, a) in geometry.objects.iter().enumerate() {
                for (index, j) in a.joints.iter().enumerate() {
                    for sweep in sweeps(a, j, scene.objects[id].transform()) {
                        hits.push(tree.hits(
                            sweep,
                            Some(id),
                            &[index],
                            j.attachment.as_deref(),
                            None,
                        ));
                    }
                }
            }
            (ms, tree.triangles.len(), hits)
        };
        let (full, scoped) = if index % 2 == 0 {
            (measure(false), measure(true))
        } else {
            let s = measure(true);
            (measure(false), s)
        };
        assert_eq!(full.2, scoped.2, "seed {seed}");
        assert!(!scoped.2.is_empty() && scoped.1 < full.1);
        prepare(&mut scene, &mut geometry);
        let screens: Vec<_> = scene
            .indoor_motion
            .as_ref()
            .unwrap()
            .mechanisms
            .iter()
            .filter(|m| m.projection_screen_travel.is_some())
            .collect();
        samples.push(serde_json::json!({"seed":seed,"screens":screens.len(),"accepted":screens.iter().filter(|m|m.rejection.is_none()).count(),
            "full_triangles":full.1,"scoped_triangles":scoped.1,"full_build_ms":full.0,"scoped_build_ms":scoped.0,"sweep_comparisons":scoped.2.len()}));
    }
    assert!(samples.len() >= 2);
    eprintln!("projection screen admission: {samples:?}");
    let dir = std::path::Path::new("out/indoor_screen_motion");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("planning.json"),
        serde_json::to_vec_pretty(&samples).unwrap(),
    )
    .unwrap();
}
