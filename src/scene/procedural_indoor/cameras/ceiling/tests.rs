mod rejections;
use super::*;
use crate::scene::procedural_indoor::{
    cameras::multiview::{MultiViewSettings, OverlapGraph},
    envelope::FloorPatch,
    layout::{IndoorLayout, IndoorManifest},
    validation::validate_layout,
};

fn policy() -> CameraSettings {
    CameraSettings {
        ceiling_mount: Some(default()),
        ..default()
    }
}

#[test]
fn mounting_policy_is_explicit_validated_and_archived() {
    assert!(CameraSettings::default().ceiling_mount.is_none());
    let p = CameraSettings::parse(r#"{"ceiling_mount":{}}"#).unwrap();
    assert_eq!(p, policy());
    assert_eq!(
        CameraSettings::parse(&serde_json::to_string(&p).unwrap()).unwrap(),
        p
    );
    for json in [
        r#"{"standoff_m":[0.09,0.2]}"#,
        r#"{"standoff_m":[0.3,0.2]}"#,
        r#"{"body_radius_m":0}"#,
        r#"{"wall_clearance_m":-1}"#,
        r#"{"lenses":{"fov_degrees":[200,190]}}"#,
    ] {
        assert!(CameraSettings::parse(&format!(r#"{{"ceiling_mount":{json}}}"#)).is_err());
    }
    for index in 0..12 {
        assert!(!p.camera_moves(123, index));
    }
}

#[test]
fn ceiling_rigs_are_safe_seeded_and_spread_across_one_to_twelve_views() {
    let started = std::time::Instant::now();
    let mut mounted = 0;
    let mut slopes = 0;
    let mut min_overlap = 1.0_f32;
    for seed in 0..32 {
        let mut scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.65, 0).unwrap();
        let before = scene.clone();
        for count in 1..=12 {
            let aspect = [1., 16. / 9., 3. / 4.][seed as usize % 3];
            scene
                .resample_cameras(count, policy(), aspect)
                .unwrap_or_else(|e| panic!("seed={seed} count={count}: {e}"));
            validate_layout(&scene).unwrap();
            assert_eq!(scene.objects, before.objects);
            assert_eq!(scene.humans, before.humans);
            for (index, camera) in scene.cameras.iter().enumerate() {
                let mount = camera.ceiling_mount.as_ref().unwrap();
                slopes += usize::from(mount.inward_normal.distance(Vec3::NEG_Y) > 0.01);
                assert_eq!(camera.fisheye, policy().lens_for(seed, index));
                assert!(camera.fisheye.is_some());
                assert!((mount.standoff_m - camera.start.distance(mount.surface)).abs() < 1e-5);
                assert!(scene.in_primary_room(camera.start, 0.35));
                assert!(camera.transform_at(0.).rotation.is_finite());
                assert_eq!(camera.path_length(), 0.);
                for i in 0..=32 {
                    assert_eq!(camera.transform_at(i as f32 / 32.), camera.transform_at(0.));
                }
                let calibration = camera
                    .fisheye
                    .unwrap()
                    .calibration([512, 512], [0.; 2])
                    .unwrap();
                for pixel in [[256., 256.], [160., 220.], [350., 320.]] {
                    let ray = Vec3::from_array(calibration.ray(pixel).unwrap());
                    let pose = camera.transform_at(0.);
                    let world = pose.transform_point(ray * 3.);
                    let local = pose.rotation.inverse() * (world - pose.translation);
                    let projected = calibration.project(local.to_array()).unwrap();
                    assert!(Vec2::from_array(projected).distance(Vec2::from_array(pixel)) < 0.001);
                }
                mounted += 1;
            }
            for pair in scene.camera_overlap() {
                for s in pair.samples {
                    min_overlap = min_overlap.min(s.reference_to_view.min(s.view_to_reference));
                }
            }
        }
        let cameras = scene.cameras.clone();
        scene
            .resample_cameras(12, policy(), scene.camera_aspect_ratio)
            .unwrap();
        assert_eq!(scene.cameras, cameras);
        let archive: IndoorManifest =
            serde_json::from_slice(&serde_json::to_vec(&scene).unwrap()).unwrap();
        assert_eq!(archive, scene);
    }
    assert!(slopes > 50, "sloped mounting coverage: {slopes}");
    eprintln!("ceiling qualification: 384 rigs, {mounted} cameras, {slopes} sloped, minimum directed overlap {min_overlap:.4}, {:.2}s", started.elapsed().as_secs_f32());
}

#[test]
fn ceiling_surface_normal_follows_slopes_and_soffits() {
    let mut scene = IndoorManifest::generate(3, IndoorLayout::Mixed, 0.65, 0).unwrap();
    scene.room_size.y = 4.;
    let e = scene.envelope.as_mut().unwrap();
    e.ceiling_drop = Vec2::new(0.8, -0.45);
    e.soffits = vec![FloorPatch {
        min: Vec2::splat(-1.),
        max: Vec2::splat(1.),
        height: 2.5,
    }];
    let (p, n) = placement::surface(&scene, Vec2::ZERO);
    assert_eq!(p.y, 2.5);
    assert_eq!(n, Vec3::NEG_Y);
    let (p, n) = placement::surface(&scene, Vec2::splat(1.2));
    for step in [Vec2::X * 0.01, Vec2::Y * 0.01] {
        let (q, _) = placement::surface(&scene, p.xz() + step);
        assert!(n.dot(q - p).abs() < 1e-6);
    }
    assert!(n.y < -0.9 && n != Vec3::NEG_Y);
}

#[test]
fn ceiling_baseline_graph_and_rectilinear_modes_preserve_constraints() {
    for baseline in [0., 0.5, 1.] {
        for graph in [OverlapGraph::Reference, OverlapGraph::Connected] {
            let mut scene = IndoorManifest::generate(16, IndoorLayout::Mixed, 0.65, 0).unwrap();
            let mut p = policy();
            p.multiview = Some(MultiViewSettings {
                graph,
                ..MultiViewSettings::from_baseline(baseline).unwrap()
            });
            let lens = &mut p.ceiling_mount.as_mut().unwrap().lenses;
            lens.probability = if baseline == 0. { 1. } else { 0.5 };
            lens.circular_probability = 1.;
            scene.resample_cameras(6, p, 1.).unwrap();
            validate_layout(&scene).unwrap();
            if baseline > 0. {
                assert!(scene.cameras.iter().any(|c| c.fisheye.is_none()));
            }
            assert!(scene.cameras.iter().any(|c| c.fisheye.is_some()));
        }
    }
}

#[test]
fn impossible_mounting_rolls_back_without_changing_lens_or_motion_roles() {
    let mut scene = IndoorManifest::generate(1, IndoorLayout::Mixed, 0.65, 2).unwrap();
    let previous = scene.clone();
    // A reservation completely fills the ceiling volume.
    let mut p = policy();
    p.ceiling_mount.as_mut().unwrap().wall_clearance_m = 2.;
    scene.room_size.x = 3.;
    let before = scene.clone();
    assert!(scene.resample_cameras(2, p, 1.).is_err());
    assert_eq!(scene, before);
    assert_eq!(scene.cameras, previous.cameras);
}

#[test]
fn mounting_points_hit_constructed_ceiling_triangles_and_reject_occupied_mounts() {
    use crate::scene::procedural_indoor::{
        architecture::ceiling::{CeilingInstallation, CeilingModule, CeilingSystem},
        envelope::construction,
    };
    for seed in [0, 3, 7, 11] {
        let mut scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.65, 0).unwrap();
        scene.resample_cameras(3, policy(), 1.).unwrap();
        let shell = construction::build(&scene);
        for camera in &scene.cameras {
            let mount = camera.ceiling_mount.as_ref().unwrap();
            let origin = camera.start;
            let ray = -mount.inward_normal;
            let nearest = shell
                .parts
                .values()
                .flat_map(|g| {
                    g.indices.as_chunks::<3>().0.iter().filter_map(|ids| {
                        let [a, b, c] =
                            std::array::from_fn(|i| Vec3::from_array(g.positions[ids[i] as usize]));
                        let e1 = b - a;
                        let e2 = c - a;
                        let h = ray.cross(e2);
                        let det = e1.dot(h);
                        if det.abs() < 1e-8 {
                            return None;
                        }
                        let s = origin - a;
                        let u = s.dot(h) / det;
                        let q = s.cross(e1);
                        let v = ray.dot(q) / det;
                        let t = e2.dot(q) / det;
                        (u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001 && t > 0.).then_some(t)
                    })
                })
                .fold(f32::INFINITY, f32::min);
            assert!(
                (nearest - mount.standoff_m).abs() < 1e-4,
                "seed {seed}: mesh hit {nearest} vs mount {}",
                mount.standoff_m
            );
        }
        let camera = scene.cameras[0].clone();
        scene.ceiling_installation = Some(CeilingInstallation {
            framing: None,
            exposed_slab: true,
            modules: vec![CeilingModule {
                suspension: None,
                system: CeilingSystem::RectangularDuct,
                center: camera.start,
                size: Vec3::splat(0.4),
                support_top_m: scene.ceiling_height(camera.start.xz()),
                pitch_m: 0.5,
                thickness_m: 0.02,
                seed,
                detail: None,
            }],
        });
        assert!(
            !scene.camera_curve_clear(&camera),
            "occupied mount must be rejected"
        );
        let mut invalid = camera.clone();
        invalid.ceiling_mount.as_mut().unwrap().rotation = Quat::IDENTITY;
        assert!(!scene.camera_curve_clear(&invalid));
    }
}

#[test]
fn mixed_roles_are_independent_seeded_and_count_stable() {
    let mut p = policy();
    p.motion_fraction = 0.7;
    p.lenses = Some(LensSampling {
        probability: 0.5,
        ..default()
    });
    let ceiling = p.ceiling_mount.as_mut().unwrap();
    ceiling.probability = 0.4;
    ceiling.motion_fraction = 0.6;
    ceiling.lenses.probability = 0.5;
    let mut counts = [0usize; 8];
    for seed in 0..2048 {
        for index in 0..12 {
            let mounted = p.ceiling_mounted(seed, index);
            let moving = p.camera_moves(seed, index);
            let fisheye = p.lens_for(seed, index).is_some();
            counts[usize::from(mounted) * 4 + usize::from(moving) * 2 + usize::from(fisheye)] += 1;
            let mut changed = p.clone();
            changed.ceiling_mount.as_mut().unwrap().probability = 0.8;
            assert!(!mounted || changed.ceiling_mounted(seed, index));
            assert_eq!(
                p.ceiling_mounted(seed, index),
                p.clone().ceiling_mounted(seed, index)
            );
        }
    }
    let total = counts.iter().sum::<usize>() as f32;
    assert!((counts[4..].iter().sum::<usize>() as f32 / total - 0.4).abs() < 0.015);
    assert!(
        counts.iter().all(|n| *n > 1200),
        "all fixed/moving/lens/mount combinations: {counts:?}"
    );
    for invalid in [
        r#"{"probability":1.1}"#,
        r#"{"motion_fraction":-0.1}"#,
        r#"{"travel_m":[0,1]}"#,
        r#"{"travel_m":[2,1]}"#,
    ] {
        assert!(CameraSettings::parse(&format!(r#"{{"ceiling_mount":{invalid}}}"#)).is_err());
    }
}

#[test]
fn mixed_ceiling_room_rails_qualify_across_rig_sizes() {
    let started = std::time::Instant::now();
    let mut combinations = [0usize; 8];
    for seed in 0..8 {
        let mut scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.65, 0).unwrap();
        let mut p = policy();
        p.motion_fraction = 0.65;
        p.lenses = Some(LensSampling {
            probability: 0.5,
            ..default()
        });
        let c = p.ceiling_mount.as_mut().unwrap();
        c.probability = 0.5;
        c.motion_fraction = 0.65;
        c.lenses.probability = 0.5;
        for count in 1..=12 {
            scene
                .resample_cameras(count, p.clone(), [1., 16. / 9.][seed as usize % 2])
                .unwrap_or_else(|e| panic!("seed={seed} count={count}: {e}"));
            validate_layout(&scene).unwrap();
            for (index, camera) in scene.cameras.iter().enumerate() {
                assert_eq!(camera.is_ceiling_mounted(), p.ceiling_mounted(seed, index));
                assert_eq!(camera.fisheye, p.lens_for(seed, index));
                assert!(p.accepts_motion(seed, index, camera));
                combinations[usize::from(camera.is_ceiling_mounted()) * 4
                    + usize::from(p.camera_moves(seed, index)) * 2
                    + usize::from(camera.fisheye.is_some())] += 1;
                if let Some(mount) = &camera.ceiling_mount {
                    for step in 0..=64 {
                        let pose = camera.transform_at(step as f32 / 64.);
                        let support = pose.translation - mount.inward_normal * mount.standoff_m;
                        let (point, normal) = placement::surface(&scene, support.xz());
                        assert!(point.distance(support) < 1e-4);
                        assert!((pose.rotation * Vec3::NEG_Z).distance(normal) < 1e-5);
                    }
                }
            }
        }
        let saved = scene.clone();
        scene
            .resample_cameras(12, p.clone(), scene.camera_aspect_ratio)
            .unwrap();
        assert_eq!(scene.cameras, saved.cameras);
        let restored: IndoorManifest =
            serde_json::from_slice(&serde_json::to_vec(&scene).unwrap()).unwrap();
        assert_eq!(scene, restored);
    }
    assert!(
        combinations.iter().all(|n| *n > 0),
        "missing roles: {combinations:?}"
    );
    eprintln!(
        "mixed rig qualification: 96 rigs, combinations={combinations:?}, {:.3}s",
        started.elapsed().as_secs_f32()
    );
}

#[test]
fn zero_ceiling_probability_preserves_room_rig_and_impossible_rails_roll_back() {
    let mut scene = IndoorManifest::generate(4, IndoorLayout::Mixed, 0.65, 3).unwrap();
    let original = scene.cameras.clone();
    let mut p = scene.camera_settings.clone();
    p.ceiling_mount = Some(CeilingCameraSettings {
        probability: 0.,
        ..default()
    });
    scene
        .resample_cameras(3, p, scene.camera_aspect_ratio)
        .unwrap();
    assert_eq!(original, scene.cameras);
    let before = scene.clone();
    let mut p = policy();
    p.ceiling_mount.as_mut().unwrap().motion_fraction = 1.;
    p.ceiling_mount.as_mut().unwrap().travel_m = [99., 100.];
    assert!(scene.resample_cameras(1, p, 1.).is_err());
    assert_eq!(scene, before);
}

#[test]
fn rails_reject_thin_between_endpoint_obstructions_and_soffit_steps() {
    use crate::scene::procedural_indoor::architecture::ceiling::{
        CeilingInstallation, CeilingModule, CeilingSystem,
    };
    let mut scene = IndoorManifest::generate(3, IndoorLayout::Mixed, 0.65, 0).unwrap();
    let mut p = policy();
    p.ceiling_mount.as_mut().unwrap().motion_fraction = 1.;
    p.ceiling_mount.as_mut().unwrap().travel_m = [0.8, 1.2];
    scene.resample_cameras(1, p, 1.).unwrap();
    let camera = scene.cameras[0].clone();
    assert!(scene.camera_curve_clear(&camera));
    let halfway = camera.transform_at(0.501).translation;
    let mut blocked = scene.clone();
    blocked.ceiling_installation = Some(CeilingInstallation {
        framing: None,
        exposed_slab: true,
        modules: vec![CeilingModule {
            suspension: None,
            system: CeilingSystem::RectangularDuct,
            center: halfway,
            size: Vec3::splat(0.005),
            support_top_m: blocked.ceiling_height(halfway.xz()),
            pitch_m: 0.5,
            thickness_m: 0.002,
            seed: 0,
            detail: None,
        }],
    });
    assert!(
        !blocked.camera_curve_clear(&camera),
        "thin fixture between time samples"
    );
    let mut step = scene.clone();
    let height = step.ceiling_height(halfway.xz()) - 0.08;
    step.envelope.as_mut().unwrap().soffits.push(FloorPatch {
        min: halfway.xz() - Vec2::splat(0.001),
        max: halfway.xz() + Vec2::splat(0.001),
        height,
    });
    assert!(
        !step.camera_curve_clear(&camera),
        "rail cannot detach from a thin ceiling step"
    );
    let mut bad = camera.clone();
    bad.motion.as_mut().unwrap().control[0].x = f32::NAN;
    assert!(!scene.camera_curve_clear(&bad));
    let mut crossing = camera.clone();
    std::mem::swap(&mut crossing.start, &mut crossing.end);
    crossing.ceiling_mount.as_mut().unwrap().surface += crossing.start - camera.start;
    assert!(bodies_overlap(&camera, &crossing));
}
