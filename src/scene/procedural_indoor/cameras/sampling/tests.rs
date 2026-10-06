//! Copied original sampler is independent of production fallback selection.
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{coverage::Coverage, CameraSettings},
    layout::{stream, IndoorLayout},
    validation::validate_layout,
};

impl IndoorManifest {
    fn original_sample_independent_cameras(
        &mut self,
        count: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
        coverage: &super::super::coverage::Coverage,
    ) -> Result<(), String> {
        let half = self.room_size * 0.5;
        self.camera_settings.validate()?;
        let (primary_lo, primary_hi) = self.primary_room_bounds();
        let people: Vec<_> = self
            .humans
            .iter()
            .filter(|h| {
                !h.neighbor
                    && (!self.camera_settings.primary_room || self.in_primary_room(h.position, 0.0))
            })
            .collect();
        for index in 0..count {
            // The former fixed 0.85 m wall rail dominated accepted cameras.
            // Mix free interior proposals with continuously offset perimeter views.
            let perimeter = rng.random_bool(0.4);
            let long_path = rng.random_bool(self.camera_settings.long_path_fraction as f64)
                && self.camera_settings.handheld.is_none();
            let edge = rng.random_range(0..4);
            let require_person = index == 0 && !people.is_empty();
            let mut found = None;
            let mut rejected = [0usize; 5];
            for attempt in 0..2048 {
                // Mix eye-level, seated, low and elevated viewpoints; stratify room edges.
                let height = match self.seed.wrapping_add(index as u64) % 5 {
                    0 | 1 => rng.random_range(1.45..1.80),
                    2 => rng.random_range(1.05..1.35),
                    3 => rng.random_range(0.78..1.02),
                    _ => rng.random_range(1.85..3.25),
                };
                let upper = (self.room_size.y
                    - self.program.as_ref().map_or(0.48, |p| p.light_drop)
                    - 0.04
                    - CAMERA_CLEARANCE
                    - 0.04)
                    .min(3.25);
                // A low camera can have no view of a seated person above an
                // intervening desk. After the preferred height band is tried,
                // search the full safe height range without weakening visibility.
                let height = if height > upper || (require_person && attempt >= 256) {
                    rng.random_range(0.78..upper)
                } else {
                    height
                };
                let mut p = Vec3::new(
                    rng.random_range(-half.x + 0.65..half.x - 0.65),
                    height,
                    rng.random_range(-half.z + 0.65..half.z - 0.65),
                );
                if self.camera_settings.primary_room {
                    p.x = rng.random_range(primary_lo.x + 0.50..primary_hi.x - 0.50);
                    p.z = rng.random_range(primary_lo.y + 0.50..primary_hi.y - 0.50);
                }
                if perimeter && attempt < 256 && !self.camera_settings.primary_room {
                    let inset = rng.random_range(0.65..1.8);
                    match edge {
                        0 => p.z = half.z - inset,
                        1 => p.x = -half.x + inset,
                        2 => p.z = -half.z + inset,
                        _ => p.x = half.x - inset,
                    }
                }
                if !self.camera_clear(p) {
                    rejected[0] += 1;
                    continue;
                }
                // Aim at actual content in this room zone as well as broad views.
                // A central target shared by every camera over-samples glass walls.
                let zone = self.program.as_ref().and_then(|program| {
                    program
                        .zones
                        .iter()
                        .find(|z| p.x > z.min.x && p.x < z.max.x && p.z > z.min.y && p.z < z.max.y)
                });
                let candidates: Vec<_> = self
                    .objects
                    .iter()
                    .filter(|o| {
                        o.solid
                            && !o.neighbor
                            && zone.is_none_or(|z| {
                                o.position.x > z.min.x
                                    && o.position.x < z.max.x
                                    && o.position.z > z.min.y
                                    && o.position.z < z.max.y
                            })
                    })
                    .collect();
                let target = if require_person {
                    let human = people[rng.random_range(0..people.len())];
                    human.transform().transform_point(human.joints[2])
                } else if !candidates.is_empty() && rng.random_bool(0.72) {
                    let object = candidates[rng.random_range(0..candidates.len())];
                    object.position
                        + Vec3::new(
                            rng.random_range(-0.18..0.18),
                            (object.size.y * rng.random_range(0.7..1.15)).clamp(0.75, 1.55),
                            rng.random_range(-0.18..0.18),
                        )
                } else {
                    Vec3::new(
                        rng.random_range(-half.x * 0.28..half.x * 0.28),
                        rng.random_range(0.85..1.4),
                        rng.random_range(-half.z * 0.32..half.z * 0.25),
                    )
                };
                if !self.camera_view_clear(p, target) {
                    rejected[1] += 1;
                    continue;
                }
                // Metric baselines span short stereo captures through walking
                // motion. Every family remains subject to full-path rejection.
                let forward = (target - p).with_y(0.0).normalize();
                let right = forward.cross(Vec3::Y);
                let angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                let direction = forward * angle.cos() + right * angle.sin();
                let min_length = self.camera_settings.path_length_min.max(0.001);
                let max_length = self.camera_settings.path_length_max.max(min_length);
                let desired_min = if long_path && attempt < 128 {
                    min_length.max(1.8).min(max_length)
                } else {
                    min_length
                };
                let distance = if self.camera_settings.path_length_max == 0.0
                    || self.camera_settings.handheld.is_some()
                {
                    0.0
                } else {
                    rng.random_range(desired_min.ln()..=max_length.ln()).exp()
                };
                let end = p
                    + direction * distance
                    + Vec3::Y
                        * if long_path || distance == 0.0 {
                            0.0
                        } else {
                            rng.random_range(-0.25..0.25)
                        };
                let route = if long_path && attempt < 256 {
                    let Some(path) = super::super::navigation::route(self, p, end) else {
                        rejected[2] += 1;
                        continue;
                    };
                    path
                } else {
                    Vec::new()
                };
                let bend = right * rng.random_range(-0.60..0.60) * distance;
                let mut camera = IndoorCamera {
                    start: p,
                    end,
                    target,
                    // Log-uniform focal length gives useful wide through normal views.
                    fov_degrees: (0.5 / rng.random_range(0.37_f32.ln()..2.0_f32.ln()).exp())
                        .atan()
                        .to_degrees()
                        * 2.0,
                    motion: (distance > 0.0).then_some(super::super::CameraMotion {
                        orientations: None,
                        route,
                        control: [p.lerp(end, 0.33) + bend, p.lerp(end, 0.67) + bend],
                        target_end: target
                            + Vec3::new(
                                rng.random_range(-0.25..0.25),
                                rng.random_range(-0.12..0.12),
                                rng.random_range(-0.25..0.25),
                            ),
                        roll: [rng.random_range(-0.09..0.09), rng.random_range(-0.09..0.09)],
                    }),
                };
                if let Some(handheld) = &self.camera_settings.handheld {
                    handheld.apply(&mut camera, rng);
                }
                if camera.path_length() + 1e-4 < self.camera_settings.path_length_min
                    || camera.path_length() > self.camera_settings.path_length_max + 1e-4
                    || !self.camera_curve_clear(&camera)
                {
                    rejected[3] += 1;
                    continue;
                }
                if !coverage.suitable(&camera, require_person) {
                    rejected[4] += 1;
                    continue;
                }
                if self
                    .cameras
                    .iter()
                    .any(|c| c.start.distance(p) < 0.18 && c.target.distance(target) < 0.5)
                {
                    continue;
                }
                found = Some(camera);
                break;
            }
            self.cameras
                .push(found.ok_or_else(|| {
                    format!("seed {}: unable to place camera {index}; rejections clearance/view/route/curve/coverage={rejected:?}; primary={primary_lo:?}..{primary_hi:?}; people={}", self.seed, people.len())
                })?);
        }
        Ok(())
    }
}

fn people(scene: &IndoorManifest) -> Vec<&crate::scene::procedural_indoor::humans::IndoorHuman> {
    scene
        .humans
        .iter()
        .filter(|human| {
            !human.neighbor
                && (!scene.camera_settings.primary_room
                    || scene.in_primary_room(human.position, 0.0))
        })
        .collect()
}

fn assert_accepted(scene: &IndoorManifest, camera: &IndoorCamera, coverage: &Coverage) {
    assert!(scene.camera_clear(camera.start));
    assert!(scene.camera_view_clear(camera.start, camera.target));
    assert!(scene.camera_curve_clear(camera));
    assert!(coverage.suitable(camera, true));
    assert!(camera.path_length() + 1e-4 >= scene.camera_settings.path_length_min);
    assert!(camera.path_length() <= scene.camera_settings.path_length_max + 1e-4);
    let fov_min = (0.5_f32 / 2.0).atan().to_degrees() * 2.0;
    let fov_max = (0.5_f32 / 0.37).atan().to_degrees() * 2.0;
    assert!((fov_min..=fov_max).contains(&camera.fov_degrees));
    for step in 0..=64 {
        let p = camera.transform_at(step as f32 / 64.0).translation;
        assert!(scene.in_primary_room(p, CAMERA_CLEARANCE));
        assert!(scene.camera_clear(p));
    }
}

#[test]
fn successful_independent_proposals_and_rng_match_copied_original() {
    for seed in [0, 7, 13, 42, 100, 200, 207, 239] {
        for density in [0.0, 0.65] {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, density, 0, 0.25)
                    .unwrap();
            scene.camera_settings = CameraSettings::independent();
            let mut original = scene.clone();
            let coverage = Coverage::new(&scene);
            let mut rng = stream(seed, 3);
            let mut original_rng = rng.clone();
            original
                .original_sample_independent_cameras(4, &mut original_rng, &coverage)
                .unwrap();
            scene
                .sample_independent_cameras(4, &mut rng, &coverage)
                .unwrap();
            assert_eq!(
                serde_json::to_vec(&scene.cameras).unwrap(),
                serde_json::to_vec(&original.cameras).unwrap(),
                "seed={seed} density={density}"
            );
            assert_eq!(
                rng.random::<u64>(),
                original_rng.random::<u64>(),
                "RNG continuation seed={seed} density={density}"
            );
            validate_layout(&scene).unwrap();
        }
    }
}

#[test]
fn sparse_failed_seed_anchors_keep_primary_room_coverage_and_deterministic_motion() {
    for seed in [881, 2342, 3779, 5608] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.0, 0, 0.25).unwrap();
        let mut original = scene.clone();
        let coverage = Coverage::new(&scene);
        let mut original_rng = stream(seed, 3);
        let original_result =
            original.original_sample_independent_cameras(1, &mut original_rng, &coverage);
        let mut replay_rng = original_rng.clone();
        let fallback = scene
            .person_camera_fallback(&people(&scene), &mut original_rng, &coverage)
            .unwrap();
        assert_accepted(&scene, &fallback, &coverage);
        let fallback_replay = scene
            .person_camera_fallback(&people(&scene), &mut replay_rng, &coverage)
            .unwrap();
        assert_eq!(fallback, fallback_replay);
        assert_eq!(original_rng.random::<u64>(), replay_rng.random::<u64>());
        // Full placement below uses exactly the original production stream.
        scene
            .resample_cameras(1, CameraSettings::default(), 1.0)
            .unwrap();
        let cameras = scene.cameras.clone();
        scene
            .resample_cameras(1, CameraSettings::default(), 1.0)
            .unwrap();
        assert_eq!(scene.cameras, cameras, "seed={seed}");
        assert_eq!(scene.objects, original.objects);
        assert_eq!(scene.humans, original.humans);
        assert_eq!(scene.cameras.len(), 1);
        validate_layout(&scene).unwrap();
        assert_accepted(&scene, &scene.cameras[0], &coverage);
        println!("sparse seed={seed}: original first-anchor result={original_result:?}; bounded fallback and deterministic production placement retain all predicates");
    }
}

#[test]
fn person_fallback_preserves_requested_motion_and_rejects_impossible_paths() {
    let mut scene =
        IndoorManifest::generate_with_humans(881, IndoorLayout::Mixed, 0.0, 0, 0.25).unwrap();
    scene.camera_settings = CameraSettings::independent();
    let coverage = Coverage::new(&scene);
    for (min, max) in [(0.0, 0.0), (0.03, 0.25), (0.3, 0.5), (0.5, 0.5)] {
        scene.camera_settings.path_length_min = min;
        scene.camera_settings.path_length_max = max;
        let camera = scene
            .person_camera_fallback(&people(&scene), &mut stream(881, 3), &coverage)
            .unwrap();
        assert_accepted(&scene, &camera, &coverage);
        if max == 0.0 {
            assert_eq!(camera.start, camera.end);
            assert!(camera.motion.is_none());
        } else {
            assert!(camera.path_length() > 0.0);
        }
    }
    scene.camera_settings.path_length_min = 100.0;
    scene.camera_settings.path_length_max = 100.0;
    assert!(scene
        .person_camera_fallback(&people(&scene), &mut stream(881, 3), &coverage)
        .is_none());
    assert!(scene
        .person_camera_fallback(&[], &mut stream(881, 3), &coverage)
        .is_none());
}

#[test]
fn person_fallback_keeps_authored_handheld_orientations_reversal_and_duration() {
    use crate::scene::procedural_indoor::cameras::handheld::HandheldSettings;
    let mut scene =
        IndoorManifest::generate_with_humans(881, IndoorLayout::Mixed, 0.0, 0, 0.25).unwrap();
    scene.camera_settings = CameraSettings {
        duration_seconds: Some(1.2),
        path_length_min: 0.01,
        path_length_max: 0.25,
        long_path_fraction: 0.0,
        handheld: Some(HandheldSettings {
            translation_m: [[0.0, 0.0], [0.0, 0.0], [0.08, 0.08]],
            rotation_degrees: [[4.0, 4.0], [1.0, 1.0], [0.0, 0.0]],
            reverse_probability: 1.0,
        }),
        ..CameraSettings::independent()
    };
    let settings = scene.camera_settings.clone();
    let coverage = Coverage::new(&scene);
    let camera = scene
        .person_camera_fallback(&people(&scene), &mut stream(881, 3), &coverage)
        .unwrap();
    assert_accepted(&scene, &camera, &coverage);
    assert_eq!(scene.camera_settings, settings);
    assert!((camera.path_length() - 0.08).abs() < 1e-5);
    let orientation = camera.motion.as_ref().unwrap().orientations.unwrap();
    let base = Transform::from_translation(camera.end)
        .looking_at(camera.target, Vec3::Y)
        .rotation;
    assert!(
        orientation[1].angle_between(base) < 1e-5,
        "reverse preserves original starting orientation at the endpoint"
    );
    assert!(
        orientation[0].angle_between(base) > 0.01,
        "authored endpoint rotation must survive reversal"
    );
    assert_eq!(scene.camera_settings.duration_seconds, Some(1.2));
}

#[test]
fn person_fallback_accepts_authored_rotation_only_handheld_with_zero_minimum_path() {
    use crate::scene::procedural_indoor::cameras::handheld::HandheldSettings;
    let mut scene =
        IndoorManifest::generate_with_humans(881, IndoorLayout::Mixed, 0.0, 0, 0.25).unwrap();
    scene.camera_settings = CameraSettings {
        duration_seconds: Some(1.2),
        path_length_min: 0.0,
        path_length_max: 0.25,
        long_path_fraction: 0.0,
        handheld: Some(HandheldSettings {
            translation_m: [[0.0, 0.0]; 3],
            rotation_degrees: [[4.0, 4.0], [1.0, 1.0], [0.0, 0.0]],
            reverse_probability: 1.0,
        }),
        ..CameraSettings::independent()
    };
    scene.camera_settings.validate().unwrap();
    let settings = scene.camera_settings.clone();
    let coverage = Coverage::new(&scene);
    let mut rng = stream(881, 3);
    let mut replay_rng = rng.clone();
    let camera = scene
        .person_camera_fallback(&people(&scene), &mut rng, &coverage)
        .unwrap();
    let replay = scene
        .person_camera_fallback(&people(&scene), &mut replay_rng, &coverage)
        .unwrap();
    assert_eq!(camera, replay);
    assert_eq!(rng.random::<u64>(), replay_rng.random::<u64>());
    assert_accepted(&scene, &camera, &coverage);
    assert_eq!(scene.camera_settings, settings);
    assert_eq!(camera.start, camera.end);
    assert!(camera.path_length() < 1e-4);
    let orientations = camera.motion.as_ref().unwrap().orientations.unwrap();
    let base = Transform::from_translation(camera.start)
        .looking_at(camera.target, Vec3::Y)
        .rotation;
    assert!(orientations[1].angle_between(base) < 1e-5);
    assert!(orientations[0].angle_between(base) > 0.01);
    for step in 0..=64 {
        assert!(
            camera
                .transform_at(step as f32 / 64.0)
                .translation
                .distance(camera.start)
                < 1e-5
        );
    }
    assert_eq!(scene.camera_settings.duration_seconds, Some(1.2));
}
