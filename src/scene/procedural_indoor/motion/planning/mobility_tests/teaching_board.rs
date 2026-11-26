use super::*;
use objects::public::furnishing;

// Compare the data actually bound by the renderer across the static and articulated batches.
fn writing_skins(board: &Assembly) -> Vec<(usize, [u32; 8])> {
    let mut vertices = Vec::new();
    for ((surface, label), mesh) in &board.parts {
        if *surface != Surface::Whiteboard {
            continue;
        }
        let slot =
            crate::scene::procedural_indoor::materials::finish_slot(*surface, label).unwrap();
        for ((p, n), uv) in mesh.positions.iter().zip(&mesh.normals).zip(&mesh.uvs) {
            vertices.push((
                slot,
                [p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]].map(f32::to_bits),
            ));
        }
    }
    vertices.sort_unstable();
    vertices
}

#[test]
fn teaching_board_wings_hold_return_and_keep_real_mounts_and_clearances() {
    let (blank, mut geometry) = room();
    let mut counts = [0usize; 3];
    let mut proposed = 0;
    let mut accepted = 0;
    let mut narrowed = 0;
    let mut angled = 0;
    let mut half = 0;
    for seed in 0..80 {
        let mut scene = blank.clone();
        let mut o = fixture(ObjectKind::TeachingBoard, seed);
        o.position = Vec3::new(1.24, 0.2, -2.6);
        o.yaw = seed as f32 * 0.37;
        o.size *= [
            Vec3::ONE,
            Vec3::new(0.83, 1.08, 0.8),
            Vec3::new(1.25, 0.92, 1.18),
        ][seed as usize % 3];
        let installation = furnishing::parameters(&o).writing_board.unwrap();
        counts[installation.arrangement as usize] += 1;
        let static_board = objects::build_manifest_object(&scene, &o);
        let config = IndoorMotionConfig {
            teaching_board_fraction: 1.,
            operation_return_fraction: 1.,
            ..policy()
        };
        let mut archive = scene.clone();
        archive.generator_version = OPERATION_CYCLE_VERSION - 1;
        archive.indoor_motion = Some(IndoorMotionReport::new(config.clone()));
        let old = objects::build_manifest_object(&archive, &o);
        assert!(old.joints.is_empty() && static_board.joints.is_empty());
        assert_eq!(rest_vertices(&old), rest_vertices(&static_board));
        let expected = installation
            .leaves
            .iter()
            .filter(|l| l.hinge.is_some_and(|h| h.stop_rad.abs() > 0.01))
            .count();
        half += objects::build_object_with_motion(
            &o,
            Some(&IndoorMotionConfig {
                teaching_board_fraction: 0.5,
                ..config.clone()
            }),
        )
        .joints
        .len();
        scene.indoor_motion = Some(IndoorMotionReport::new(config));
        geometry.objects.clear();
        add(&mut scene, &mut geometry, o);
        assert_eq!(geometry.objects[0].joints.len(), expected);
        if expected == 0 {
            assert_eq!(
                rest_vertices(&geometry.objects[0]),
                rest_vertices(&static_board)
            );
            continue;
        }
        proposed += expected;
        assert_eq!(
            writing_skins(&geometry.objects[0]),
            writing_skins(&static_board)
        );
        audit::check(&scene.objects[0], &geometry.objects[0]).unwrap();
        prepare(&mut scene, &mut geometry);
        let tf = scene.objects[0].transform();
        let a = &geometry.objects[0];
        for m in &scene.indoor_motion.as_ref().unwrap().mechanisms {
            if m.rejection.is_some() {
                eprintln!(
                    "blocked board seed {seed}, joint {}, {:?}",
                    m.joint, m.rejection
                );
                continue;
            }
            accepted += 1;
            narrowed += usize::from(m.adaptation.is_some());
            let cycle = m.operation_cycle.unwrap();
            assert_eq!(m.kind, MotionKind::TeachingBoard);
            assert_eq!(m.semantic_labels, ["whiteboard"]);
            let joint = &a.joints[m.joint];
            let leaf_index = joint
                .attachment
                .as_ref()
                .unwrap()
                .trim_start_matches("contact_board_pin")
                .parse::<usize>()
                .unwrap();
            let hinge = installation.leaves[leaf_index].hinge.unwrap();
            let target = hinge.angle_rad + joint.effective_angle();
            assert!(
                (hinge.stop_rad.min(0.) - 1e-5..=hinge.stop_rad.max(0.) + 1e-5).contains(&target)
            );
            assert!(joint.parts.iter().any(|(s, _)| *s == Surface::Whiteboard));
            assert!(a
                .parts
                .keys()
                .any(|(_, l)| tag(l).is_none() && l.contains(joint.attachment.as_ref().unwrap())));
            angled += usize::from(joint.axis.distance(Vec3::Y) > 0.001);
            for t in [
                0.,
                cycle.starts_at.lerp(cycle.reaches_stop_at, 0.36),
                cycle.reaches_stop_at,
                cycle.leaves_stop_at,
                cycle.leaves_stop_at.lerp(cycle.returns_at, 0.38),
                1.,
                cycle.reaches_stop_at,
            ] {
                let fraction = m.stroke_fraction_at(t);
                let delta = m.rigid_transform_at(t).unwrap();
                assert_eq!(fraction, joint.amount(t));
                assert!(delta.transform_point(m.pivot).distance(m.pivot) < 2e-5);
                if t == 0. || t == 1. {
                    assert_eq!(delta, Transform::IDENTITY);
                }
                for k in &joint.parts {
                    for &v in a.parts[k].positions.iter().step_by(19) {
                        let local = joint.transform(fraction).transform_point(Vec3::from(v));
                        let size = scene.objects[0].size;
                        assert!(
                            local.x.abs() <= size.x * 0.5 + 1e-5
                                && local.z.abs() <= size.z * 0.5 + 1e-5
                                && (0.0..=size.y).contains(&local.y)
                        );
                        let w = tf.transform_point(local);
                        assert!(
                            w.distance(delta.transform_point(tf.transform_point(Vec3::from(v))))
                                < 2e-5
                        );
                        assert!(m
                            .swept_bounds
                            .iter()
                            .any(|&(lo, hi)| w.cmpge(lo - 1e-5).all() && w.cmple(hi + 1e-5).all()));
                    }
                }
            }
        }
    }
    eprintln!("80 boards: single/fixed/folding {counts:?}, {accepted}/{proposed} wings admitted, {narrowed} shortened, {angled} tilted axes, {half} selected at 0.5");
    assert!(
        counts.iter().all(|&n| n > 5)
            && proposed > 25
            && accepted == proposed
            && angled > 1
            && half > proposed / 4
            && half < 3 * proposed / 4
    );
}

#[test]
fn teaching_board_blockers_check_the_entire_arc_and_clear_a_stale_admission() {
    let (mut scene, mut g) = room();
    scene
        .indoor_motion
        .as_mut()
        .unwrap()
        .policy
        .teaching_board_fraction = 1.;
    scene
        .indoor_motion
        .as_mut()
        .unwrap()
        .policy
        .operation_return_fraction = 1.;
    let o = (0..150)
        .map(|s| fixture(ObjectKind::TeachingBoard, s))
        .find(|o| {
            let a = objects::build_manifest_object(&scene, o);
            a.joints.len() == 1 && a.joints[0].angle.abs() > 0.26
        })
        .unwrap();
    add(&mut scene, &mut g, o);
    prepare(&mut scene, &mut g);
    let initial = scene.indoor_motion.clone().unwrap();
    let m = &initial.mechanisms[0];
    assert!(m.rejection.is_none(), "{m:?}");
    let j = &g.objects[0].joints[0];
    let surface = j
        .parts
        .iter()
        .find(|(s, _)| *s == Surface::Whiteboard)
        .unwrap();
    let face = &g.objects[0].parts[surface].positions;
    let center = face.iter().copied().map(Vec3::from).sum::<Vec3>() / face.len() as f32;
    let missing_angle = j.transform(0.75).transform_point(center);
    let blocker = |at| {
        let mut b = Assembly::default();
        b.box_part(Surface::Wood, "shelf", at, Vec3::splat(0.012), 0.);
        b
    };
    g.architecture = blocker(missing_angle);
    prepare(&mut scene, &mut g);
    let short = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
    assert_eq!(short.adaptation.as_deref(), Some("shortened"), "{short:?}");
    assert!(
        short.angle_radians.abs() < m.angle_radians.abs()
            && short.operation_cycle == m.operation_cycle
    );
    g.architecture = blocker(center);
    prepare(&mut scene, &mut g);
    let reject = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
    assert_eq!(reject.rejection.as_deref(), Some("static_geometry"));
    assert!(
        reject.operation_cycle.is_none()
            && reject.swept_bounds.is_empty()
            && reject.stroke_fraction_at(0.5) == 0.
    );
    g.architecture = default();
    prepare(&mut scene, &mut g);
    assert_eq!(scene.indoor_motion.unwrap(), initial);
}

#[test]
#[ignore = "compares full and local collision trees in furnished classrooms"]
fn teaching_board_clearance_matches_full_furnished_geometry() {
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    let mut results = Vec::new();
    for (index, seed) in [8204, 8206, 8323].into_iter().enumerate() {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Classroom, 0.65, 2, 0.25)
                .unwrap();
        scene.indoor_motion = Some(IndoorMotionReport::new(IndoorMotionConfig {
            teaching_board_fraction: 1.,
            fan_fraction: 0.,
            chair_fraction: 0.,
            door_fraction: 0.,
            cabinet_fraction: 0.,
            covering_fraction: 0.,
            appliance_fraction: 0.,
            prop_fraction: 0.,
            cart_fraction: 0.,
            ..default()
        }));
        let (mut geometry, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
        // Exercise a positive contact in a real room as well as clear wings.
        if index == 1 {
            let (o, board) = scene
                .objects
                .iter()
                .zip(&geometry.objects)
                .find(|(_, a)| !a.joints.is_empty())
                .unwrap();
            let wing = &board.joints[0];
            let face = wing
                .parts
                .iter()
                .find(|(s, _)| *s == Surface::Whiteboard)
                .unwrap();
            let pts = &board.parts[face].positions[..4];
            let center = pts.iter().copied().map(Vec3::from_array).sum::<Vec3>() * 0.25;
            let obstruction = o
                .transform()
                .transform_point(wing.transform(0.58).transform_point(center));
            geometry.architecture.box_part(
                Surface::Wood,
                "shelf",
                obstruction,
                Vec3::splat(0.025),
                0.,
            );
        }
        let measure = |local| {
            let start = std::time::Instant::now();
            let tree = if local {
                Tree::for_admission(&scene, &geometry)
            } else {
                Tree::new(&scene, &geometry)
            };
            let build_ms = start.elapsed().as_secs_f64() * 1000.;
            let mut hits = Vec::new();
            for (o, a) in scene.objects.iter().zip(&geometry.objects) {
                for (id, j) in a.joints.iter().enumerate() {
                    assert_eq!(j.kind, MotionKind::TeachingBoard);
                    let tf = o.transform();
                    let (lo, hi) = board::search_bounds(a, j, tf).unwrap();
                    let mut stroke = j.clone();
                    for fraction in [1., 0.37] {
                        stroke.adjustment.rotation = fraction;
                        for s in sweeps(a, &stroke, tf) {
                            let bb = s.world();
                            assert!(bb.0.cmpge(lo - 0.0001).all() && bb.1.cmple(hi + 0.0001).all());
                            hits.push(tree.hits(
                                s,
                                Some(o.id),
                                &[id],
                                j.attachment.as_deref(),
                                None,
                            ));
                        }
                    }
                }
            }
            (tree.triangles.len(), build_ms, hits)
        };
        let (full, local) = if index % 2 == 0 {
            (measure(false), measure(true))
        } else {
            let local = measure(true);
            (measure(false), local)
        };
        assert_eq!(full.2, local.2, "seed {seed}");
        assert!(!full.2.is_empty() && local.0 < full.0);
        if index == 1 {
            assert!(local.2.iter().any(|&contact| contact));
        }
        results.push(serde_json::json!({"seed":seed,"full_triangles":full.0,"local_triangles":local.0,"full_build_ms":full.1,"local_build_ms":local.1,"sweep_comparisons":full.2.len(),"contacts":full.2.iter().filter(|&&h|h).count()}));
    }
    let dir = std::path::Path::new("out/indoor_board_operations");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("planning.json"),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .unwrap();
    eprintln!("teaching board clearance: {results:?}");
}
