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
            exposed_slab: true,
            modules: vec![CeilingModule {
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
