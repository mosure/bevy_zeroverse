//! The seed-local boundary applies only to freshly generated CPU scene data.
//! External settings, loaded manifests, renderer/IPC errors and retirement
//! failures never pass through this conversion. No diagnostic string matching.
use super::*;
use crate::sample::failure::SceneRejectionKind;

pub(super) struct Inputs {
    pub clutter: crate::scene::procedural_indoor::items::distribution::ClutterSettings,
    pub appearance: Option<appearance::AppearanceSettings>,
    pub camera: cameras::CameraSettings,
    pub motion: Option<crate::human_motion::HumanMotionConfig>,
    pub indoor_motion: Option<crate::scene::procedural_indoor::motion::IndoorMotionConfig>,
}
impl Inputs {
    pub fn parse(request: &Request) -> Result<Self, CaptureError> {
        if request.quality.diffuse_gi() && request.gi_enabled {
            request.gi.validate_capture()?;
        }
        for bits in [request.density, request.humans] {
            if !(0.0..=1.0).contains(&f32::from_bits(bits)) {
                return Err("indoor densities must be finite and in [0,1]".into());
            }
        }
        let aspect = f32::from_bits(request.aspect);
        if request.cameras == 0
            || request.cameras > cameras::MAX_INDOOR_CAMERAS
            || !aspect.is_finite()
            || aspect <= 0.
        {
            return Err("indoor capture needs 1..=256 cameras and a finite positive aspect".into());
        }
        Ok(Self {
            clutter: crate::scene::procedural_indoor::items::distribution::ClutterSettings::parse(
                request.clutter.as_deref().unwrap_or("{}"),
            )?,
            appearance: request
                .appearance
                .as_deref()
                .map(appearance::AppearanceSettings::parse)
                .transpose()?,
            camera: request
                .camera
                .as_deref()
                .map(cameras::CameraSettings::parse)
                .transpose()?
                .unwrap_or_default(),
            motion: request
                .motion
                .as_deref()
                .map(crate::human_motion::HumanMotionConfig::parse)
                .transpose()?,
            indoor_motion: request
                .indoor_motion
                .as_deref()
                .map(crate::scene::procedural_indoor::motion::IndoorMotionConfig::parse)
                .transpose()?,
        })
    }
}

pub(super) fn failure(seed: u64, stage: &str, error: CaptureError) -> CaptureError {
    match error {
        // Keep existing construction categories and diagnostics unchanged.
        CaptureError::Rejected { .. } => error,
        CaptureError::Failed(reason) => CaptureError::rejected(
            seed,
            SceneRejectionKind::PlacementClearance,
            format!("generated scene {stage} rejected for seed {seed}: {reason}"),
        ),
    }
}

pub(super) fn validate(scene: &IndoorManifest) -> Result<(), CaptureError> {
    validation::validate_prepared_layout(scene)
        .map_err(|error| failure(scene.seed, "validation", error))
}

/// The worker and local paths call the identical seed-local oracle. Only owned
/// manifest data crosses the browser boundary. GPU and Bevy handles stay local.
#[cfg_attr(
    any(test, target_arch = "wasm32"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub(super) struct Plan {
    pub scene: IndoorManifest,
    pub moving_humans: Vec<usize>,
    pub layout_seconds: f64,
    pub cameras_seconds: f64,
    pub validation_seconds: f64,
    pub motion_planning_seconds: f64,
}
pub(super) fn plan(request: &Request) -> Result<Plan, CaptureError> {
    // Configuration is external to the sampled room. Validate it before the
    // seed-local boundary; invalid configuration must not consume every seed.
    let inputs = Inputs::parse(request)?;
    let started = bevy::platform::time::Instant::now();
    let mut scene = IndoorManifest::generate_with_clutter(
        request.seed,
        request.layout,
        f32::from_bits(request.density),
        0,
        f32::from_bits(request.humans),
        inputs.clutter,
    )
    .map_err(|error| failure(request.seed, "layout", error.into()))?;
    if request.quality.diffuse_gi() && request.gi_enabled {
        request.gi.validate_room(&scene)?;
    }
    if let Some(appearance) = inputs.appearance {
        scene
            .apply_appearance(appearance)
            .map_err(|error| failure(request.seed, "appearance", error.into()))?;
    }
    let layout_seconds = started.elapsed().as_secs_f64();
    let started = bevy::platform::time::Instant::now();
    let policy = inputs.camera;
    if let Err(error) =
        scene.resample_cameras_typed(request.cameras, policy, f32::from_bits(request.aspect))
    {
        if matches!(error, CaptureError::Rejected { .. }) {
            // Prefer the generated layout's diagnostic over a downstream
            // failed search. Only rejection incurs this additional check.
            validate(&scene)?;
        }
        return Err(failure(request.seed, "camera placement", error));
    }
    let cameras_seconds = started.elapsed().as_secs_f64();
    let started = bevy::platform::time::Instant::now();
    validate(&scene)?;
    let validation_seconds = started.elapsed().as_secs_f64();
    let started = bevy::platform::time::Instant::now();
    if request.rotation {
        scene.world_yaw = layout::stream(request.seed, 11).random_range(0.0..std::f32::consts::TAU);
    }
    let moving_humans = if let Some(config) = inputs.motion {
        crate::human_motion::planning::prepare_scene(&mut scene, &config)
            .map_err(|error| failure(request.seed, "motion preparation", error.into()))?;
        crate::human_motion::planning::plan(&scene, &config)
            .map_err(|error| failure(request.seed, "motion planning", error.into()))?
            .0
            .into_iter()
            .map(|p| p.actor_id)
            .collect()
    } else {
        Vec::new()
    };
    if let Some(config) = inputs.indoor_motion {
        scene.indoor_motion =
            Some(crate::scene::procedural_indoor::motion::IndoorMotionReport::new(config));
    }
    let motion_planning_seconds = started.elapsed().as_secs_f64();
    Ok(Plan {
        scene,
        moving_humans,
        layout_seconds,
        cameras_seconds,
        validation_seconds,
        motion_planning_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ZeroverseSceneSettings {
        ZeroverseSceneSettings {
            num_cameras: 3,
            ..default()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn invalid_active_irradiance_settings_fail_before_scene_generation() {
        let mut request = Request::new(614440199, &default(), &settings(), default());
        request.gi_gpu = true;
        for (spacing, rays, bounces) in [
            (0.0, 256, 3),
            (f32::NAN, 256, 3),
            (0.85, 15, 3),
            (0.85, 65_537, 3),
            (0.85, 256, 33),
        ] {
            request.gi.spacing = spacing;
            request.gi.rays_per_probe = rays;
            request.gi.diffuse_bounces = bounces;
            assert!(matches!(plan(&request), Err(CaptureError::Failed(_))));
        }
        request.gi_gpu = false;
        request.gi.rays_per_probe = u32::MAX;
        request.gi.diffuse_bounces = 3;
        assert!(matches!(plan(&request), Err(CaptureError::Failed(_))));
        request.gi_enabled = false;
        assert!(Inputs::parse(&request).is_ok());
        request.gi_enabled = true;
        request.quality = IndoorQuality::Portable;
        assert!(Inputs::parse(&request).is_ok());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn oversized_gym_irradiance_grid_returns_a_configuration_error() {
        let mut request = Request::new(400, &default(), &settings(), default());
        request.layout = layout::IndoorLayout::Gymnasium;
        request.gi.spacing = 0.2;
        let Err(CaptureError::Failed(error)) = plan(&request) else {
            panic!("an oversized irradiance volume must fail the request");
        };
        assert!(error.contains("100000-probe budget"), "{error}");
    }

    #[test]
    fn studio_worker_wire_preserves_full_seed_camera_poses_and_typed_errors() {
        let mut key = Request::new((1u64 << 60) + 9, &default(), &settings(), default());
        key.humans = 0.5f32.to_bits();
        key.indoor_motion = Some("{}".into());
        let bytes = rmp_serde::to_vec_named(&key).unwrap();
        let received: Request = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(received, key);
        let original = plan(&received).unwrap();
        let bytes = rmp_serde::to_vec_named(&Ok::<_, CaptureError>(&original)).unwrap();
        let restored: Result<Plan, CaptureError> = rmp_serde::from_slice(&bytes).unwrap();
        let restored = restored.unwrap();
        assert_eq!(original.scene, restored.scene);
        assert_eq!(original.moving_humans, restored.moving_humans);
        let error = CaptureError::rejected(
            key.seed,
            SceneRejectionKind::CameraPlacement,
            "camera".into(),
        );
        let bytes = rmp_serde::to_vec_named(&Err::<Plan, _>(&error)).unwrap();
        let decoded: Result<Plan, CaptureError> = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.err(), Some(error));
    }
    #[test]
    fn generated_validation_is_seed_local_but_direct_manifest_validation_is_not() {
        // This historical failure now validates after the upstream scene update.
        let scene = IndoorManifest::generate_with_humans(
            66242909,
            layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        validation::validate_prepared_layout(&scene).unwrap();
        validate(&scene).unwrap();
        // Deterministically invalid generated data must still be rejected, while
        // direct validation of loaded manifests remains a fatal integrity error.
        let mut invalid = scene.clone();
        invalid.objects[0].position.x = f32::NAN;
        assert!(matches!(
            validation::validate_prepared_layout(&invalid),
            Err(CaptureError::Failed(_))
        ));
        match validate(&invalid).unwrap_err() {
            CaptureError::Rejected {
                rejection,
                retired: false,
            } => {
                assert_eq!(rejection.seed, 66242909);
                assert_eq!(rejection.kind, SceneRejectionKind::PlacementClearance);
            }
            other => panic!("wrong recovery boundary: {other:?}"),
        }
        // Future geometric validation messages are not a growing string allowlist.
        let mut scene = IndoorManifest::generate_with_humans(
            65538610,
            layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        scene.objects[0].position.x = f32::NAN;
        assert!(validate(&scene).is_err()); // never accept an invalid scene
        assert!(matches!(
            validate(&scene),
            Err(CaptureError::Rejected { .. })
        ));
        assert!(matches!(
            validation::validate_prepared_layout(&scene),
            Err(CaptureError::Failed(_))
        ));
    }

    #[test]
    fn malformed_external_configuration_is_not_a_seed_rejection() {
        let base = Request::new(66242909, &default(), &settings(), default());
        Inputs::parse(&base).unwrap();
        for field in 0..9 {
            let mut request = base.clone();
            match field {
                0 => request.density = f32::NAN.to_bits(),
                1 => request.humans = (-1_f32).to_bits(),
                2 => request.aspect = 0_f32.to_bits(),
                3 => request.cameras = 257,
                4 => request.camera = Some("not json".into()),
                5 => request.appearance = Some("not json".into()),
                6 => request.motion = Some("not json".into()),
                7 => request.clutter = Some("not json".into()),
                _ => request.indoor_motion = Some("not json".into()),
            }
            assert!(matches!(
                Inputs::parse(&request),
                Err(CaptureError::Failed(_))
            ));
        }
    }

    fn assert_fatal(request: Request) {
        // Exercise the real entry point, not only the parser. An invalid setting
        // must fail before generating geometry, even with a valid scene seed.
        let error = bevy::tasks::block_on(request.prepare(
            StagedAssets::new(&Assets::<Image>::default()),
            StagedAssets::new(&Assets::<StandardMaterial>::default()),
            StagedAssets::new(&Assets::<Mesh>::default()),
        ))
        .err()
        .expect("invalid configuration was accepted");
        assert!(matches!(error, CaptureError::Failed(_)), "{error:?}");
    }

    #[test]
    fn invalid_external_ranges_fail_before_generating_a_room() {
        let base = Request::new(66242909, &default(), &settings(), default());
        Inputs::parse(&base).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            for humans in [false, true] {
                let mut request = base.clone();
                if humans {
                    request.humans = value.to_bits();
                } else {
                    request.density = value.to_bits();
                }
                assert_fatal(request);
            }
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1., 0.] {
            assert_fatal(Request {
                aspect: value.to_bits(),
                ..base.clone()
            });
        }
        for count in [0, 257, usize::MAX] {
            let settings = ZeroverseSceneSettings {
                num_cameras: count,
                ..default()
            };
            let args = BevyZeroverseConfig {
                headless: true,
                ..default()
            };
            assert_fatal(Request::new(base.seed, &args, &settings, default()));
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0., -1.] {
            for width in [false, true] {
                let mut args = BevyZeroverseConfig::default();
                if width {
                    args.width = value;
                } else {
                    args.height = value;
                }
                assert_fatal(Request::new(base.seed, &args, &settings(), default()));
            }
        }
        for field in 0..5 {
            for json in [
                "not json",
                r#"{"unknown_setting":1}"#,
                match field {
                    0 => r#"{"probability":1.1}"#,
                    1 => r#"{"illumination_scale":0.0}"#,
                    2 => r#"{"motion_fraction":-0.1}"#,
                    3 => r#"{"diffusion_steps":0}"#,
                    _ => r#"{"chair_fraction":1.1}"#,
                },
            ] {
                let mut request = base.clone();
                let slot = match field {
                    0 => &mut request.clutter,
                    1 => &mut request.appearance,
                    2 => &mut request.camera,
                    3 => &mut request.motion,
                    _ => &mut request.indoor_motion,
                };
                *slot = Some(json.into());
                assert_fatal(request);
            }
        }
    }

    #[test]
    fn editor_without_capture_views_keeps_its_reference_camera() {
        let request = Request::new(66242909, &default(), &default(), default());
        assert_eq!(request.cameras, 1);
        Inputs::parse(&request).unwrap();
    }

    #[test]
    fn existing_rejection_categories_and_retirement_are_preserved() {
        for kind in [
            SceneRejectionKind::CameraPlacement,
            SceneRejectionKind::PlacementClearance,
            SceneRejectionKind::MissingProgramAnchor,
            SceneRejectionKind::InsufficientProgramCapacity,
        ] {
            for retired in [false, true] {
                let mut error = CaptureError::rejected(17, kind, "original diagnostic".into());
                if let CaptureError::Rejected { retired: state, .. } = &mut error {
                    *state = retired;
                }
                assert_eq!(failure(99, "validation", error.clone()), error);
            }
        }
    }

    #[test]
    fn generated_failures_keep_seed_stage_and_full_diagnostic() {
        for stage in [
            "layout",
            "appearance",
            "camera placement",
            "validation",
            "motion preparation",
            "motion planning",
        ] {
            let reason = "new geometric invariant: object 19 intersects object 7";
            let raw = CaptureError::from(reason);
            assert!(matches!(raw, CaptureError::Failed(_)));
            let error = failure(42, stage, raw);
            assert!(
                matches!(&error, CaptureError::Rejected { rejection, retired: false }
                if rejection.seed == 42 && rejection.kind == SceneRejectionKind::PlacementClearance
                    && rejection.reason.contains(stage) && rejection.reason.contains(reason))
            );
            let wire = serde_json::to_string(&error).unwrap();
            assert_eq!(error, serde_json::from_str(&wire).unwrap());
        }
    }

    #[test]
    fn scene_specific_motion_failure_is_rejected_after_valid_external_settings() {
        let args = BevyZeroverseConfig {
            indoor_human_density: 0.,
            human_motion: Some(r#"{"trajectories":[{"actor_id":999,"prompt":"walking","waypoints":[{"frame":0,"position":[0,0,0]},{"frame":40,"position":[0,0,1]}]}]}"#.into()),
            ..default()
        };
        let request = Request::new(66242909, &args, &settings(), default());
        Inputs::parse(&request).unwrap();
        let error = bevy::tasks::block_on(request.prepare(
            StagedAssets::new(&Assets::<Image>::default()),
            StagedAssets::new(&Assets::<StandardMaterial>::default()),
            StagedAssets::new(&Assets::<Mesh>::default()),
        ))
        .err()
        .expect("a missing motion actor must not be accepted");
        assert!(
            matches!(&error, CaptureError::Rejected { rejection, retired: false }
            if rejection.seed == 66242909 && rejection.kind == SceneRejectionKind::PlacementClearance
                && rejection.reason.contains("motion planning")
                && rejection.reason.contains("actor 999 does not exist")),
            "{error:?}"
        );
    }
}
