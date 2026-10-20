use super::*;
use crate::scene::procedural_indoor::{cameras::CameraSettings, layout::IndoorLayout};
use bevy::ecs::system::RunSystemOnce;

fn decode_query(query: &str) -> impl Iterator<Item = (String, String)> + '_ {
    fn decode(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = vec![];
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' {
                out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap());
                i += 3;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap()
    }
    query.split('&').map(|p| {
        let (k, v) = p.split_once('=').unwrap();
        (decode(k), decode(v))
    })
}

#[test]
fn supersampling_defaults_to_auto_and_shares_without_automatic_regeneration() {
    let config = BevyZeroverseConfig::default();
    assert_eq!(config.ssaa_factor, 0);
    let mut old = serde_json::to_value(&config).unwrap();
    old.as_object_mut().unwrap().remove("ssaa_factor");
    assert_eq!(
        serde_json::from_value::<BevyZeroverseConfig>(old)
            .unwrap()
            .ssaa_factor,
        0
    );
    let mut state = EditorState::new(&config);
    let applied = state.applied.clone();
    edit(&mut state, "/ssaa_factor", json!(2)).unwrap();
    let config = state.config().unwrap();
    assert_eq!(config.ssaa_factor, 2);
    assert_eq!(state.applied, applied);
    let query = share::query(&config, &state.view);
    let parsed =
        super::super::config_with_query(BevyZeroverseConfig::default(), decode_query(&query))
            .unwrap();
    assert_eq!(parsed.ssaa_factor, 2);
    state.draft["ssaa_factor"] = json!(5);
    assert!(state.config().is_err());
}

#[test]
fn capacity_evidence_controls_do_not_change_or_share_scene_configuration() {
    let config = BevyZeroverseConfig::default();
    let mut state = EditorState::new(&config);
    let draft = state.draft.clone();
    let query = share::query(&config, &state.view);
    assert!(state.capacity_status().contains("No imported"));
    assert!(edit(
        &mut state,
        "@capacity_receipt",
        json!({"schema_version":999})
    )
    .is_err());
    assert!(state.capacity_receipt.is_none());
    edit(&mut state, "@capacity_receipt", Value::Null).unwrap();
    assert_eq!(state.draft, draft);
    assert!(!state.pending());
    assert_eq!(share::query(&state.config().unwrap(), &state.view), query);
    assert!(!query.contains("capacity_receipt"));
}

#[test]
fn camera_motion_percentage_preserves_travel_animation_and_shared_urls() {
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_camera: Some(
            r#"{"handheld":{},"path_length_min":0.01,"path_length_max":0.6}"#.into(),
        ),
        indoor_motion: Some("{}".into()),
        ..default()
    };
    let mut state = EditorState::new(&config);
    let applied = state.applied.clone();
    let travel = state.draft["indoor_camera"].clone();
    assert_eq!(value(&state, "@camera_motion_percent"), json!(100.));
    for fraction in [0., 37., 100.] {
        assert!(
            !edit(&mut state, "@camera_motion_percent", json!(fraction)).unwrap(),
            "slider must retain its entity while dragging"
        );
        let config = state.config().unwrap();
        let settings = CameraSettings::parse(config.indoor_camera.as_deref().unwrap()).unwrap();
        assert!((settings.motion_fraction - fraction as f32 / 100.).abs() < 1e-6);
        assert_eq!(state.applied, applied);
        assert_eq!(state.draft["indoor_camera"]["handheld"], travel["handheld"]);
        assert_eq!(
            state.draft["indoor_camera"]["path_length_max"],
            travel["path_length_max"]
        );
        assert!(config.indoor_motion.is_some());
        let query = share::query(&config, &state.view);
        let pairs = decode_query(&query);
        let roundtrip =
            super::super::config_with_query(BevyZeroverseConfig::default(), pairs).unwrap();
        assert_eq!(
            CameraSettings::parse(roundtrip.indoor_camera.as_deref().unwrap()).unwrap(),
            settings
        );
    }
    assert!(edit(&mut state, "@camera_motion_percent", json!(-1.)).is_err());
    assert!(edit(&mut state, "@camera_motion_percent", json!(101.)).is_err());
}

#[test]
fn lens_controls_migrate_explicit_old_distribution_only_when_edited() {
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_camera: Some(r#"{"intrinsics":{"ultra_wide_probability":0.7}}"#.into()),
        ..default()
    };
    let mut state = EditorState::new(&config);
    assert_eq!(
        state.draft["indoor_camera"]["intrinsics"]["ultra_wide_probability"],
        json!(0.7f32)
    );
    let baseline = state.draft["indoor_camera"]["multiview"].clone();
    assert!(!edit(&mut state, "@lens_spread", json!(0.4)).unwrap());
    assert!(!edit(&mut state, "@lens_bias", json!(-0.35)).unwrap());
    let settings =
        CameraSettings::parse(state.config().unwrap().indoor_camera.as_deref().unwrap()).unwrap();
    assert_eq!(settings.intrinsics.spread, 0.4);
    assert_eq!(settings.intrinsics.bias, -0.35);
    assert_eq!(settings.intrinsics.ultra_wide_probability, None);
    assert_eq!(state.draft["indoor_camera"]["multiview"], baseline);
    assert!(state.pending());
}

#[test]
fn optical_center_and_graph_controls_roundtrip_and_do_not_regenerate() {
    let mut state = EditorState::new(&BevyZeroverseConfig::default());
    let applied = state.applied.clone();
    edit(&mut state, "@lens_x", json!(0.12)).unwrap();
    edit(&mut state, "@lens_y", json!(0.08)).unwrap();
    edit(&mut state, "@overlap_graph", json!("connected")).unwrap();
    let config = model::validate(&state.draft).unwrap();
    let restored = EditorState::new(&config);
    assert_eq!(value(&restored, "@overlap_graph"), json!("connected"));
    assert!((value(&restored, "@lens_x").as_f64().unwrap() - 0.12).abs() < 1e-6);
    assert_eq!(state.applied, applied);
    edit(&mut state, "@mixture", json!(true)).unwrap();
    assert!(model::validate(&state.draft).is_err());
}

#[test]
fn ultra_wide_controls_are_drafts_and_roundtrip_as_typed_camera_policy() {
    let mut state = EditorState::new(&BevyZeroverseConfig::default());
    let original = state.applied.clone();
    edit(
        &mut state,
        "/indoor_camera/intrinsics/fov_y_min_degrees",
        json!(120.),
    )
    .unwrap();
    edit(
        &mut state,
        "/indoor_camera/intrinsics/fov_y_max_degrees",
        json!(150.),
    )
    .unwrap();
    edit(&mut state, "@lens_bias", json!(0.7)).unwrap();
    let config = state.config().unwrap();
    let parsed = CameraSettings::parse(config.indoor_camera.as_deref().unwrap()).unwrap();
    assert_eq!(parsed.intrinsics.fov_y_min_degrees, 120.);
    assert_eq!(parsed.intrinsics.fov_y_max_degrees, 150.);
    assert_eq!(state.applied, original);
    assert!(state.pending());
    assert!(share::query(&config, &state.view).contains("fov_y_max_degrees"));
    edit(
        &mut state,
        "/indoor_camera/intrinsics/fov_y_min_degrees",
        json!(151.),
    )
    .unwrap();
    assert!(state.config().is_err());
}

#[test]
fn camera_shorthand_is_expanded_and_every_catalog_choice_is_valid() {
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_camera: Some(r#"{"baseline":0.8}"#.into()),
        indoor_gpu_materials: false,
        indoor_appearance: Some(r#"{"exposure_ev100_offset":1.25}"#.into()),
        ..default()
    };
    let mut state = EditorState::new(&config);
    assert!((value(&state, "@baseline").as_f64().unwrap() - 0.8).abs() < 1e-5);
    for page in Page::ALL {
        state.view.page = page;
        for item in fields::items(&state) {
            if let fields::Item::Field(f) = item {
                if let fields::Kind::Choice(options) = f.kind {
                    if f.path.starts_with('@') {
                        continue;
                    }
                    for (_, option) in options {
                        let mut draft = state.draft.clone();
                        *draft.pointer_mut(&f.path).unwrap() = option;
                        // O-voxel's single-timestep constraint is intentional.
                        if f.path == "/ovoxel_mode" {
                            draft["playback_steps"] = json!(1);
                        }
                        model::validate(&draft).unwrap_or_else(|e| panic!("{}: {e}", f.path));
                    }
                }
            }
        }
    }
    state.view.page = Page::Scene;
    let choices = fields::items(&state)
        .into_iter()
        .find_map(|i| match i {
            fields::Item::Field(fields::Field {
                path,
                kind: fields::Kind::Choice(c),
                ..
            }) if path == "/indoor_layout" => Some(c),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        choices.len(),
        <IndoorLayout as clap::ValueEnum>::value_variants().len()
    );
}
#[test]
fn drafts_preserve_full_width_seeds_and_reject_invalid_constraints() {
    let mut state = EditorState::new(&BevyZeroverseConfig::default());
    edit(&mut state, "/indoor_seed", json!(u64::MAX)).unwrap();
    assert_eq!(state.config().unwrap().indoor_seed, Some(u64::MAX));
    edit(&mut state, "/num_cameras", json!(4.0)).unwrap();
    assert_eq!(state.config().unwrap().num_cameras, 4);
    edit(&mut state, "@baseline", json!(0.9)).unwrap();
    let camera =
        CameraSettings::parse(state.config().unwrap().indoor_camera.as_deref().unwrap()).unwrap();
    assert!((camera.multiview.unwrap().baseline().unwrap() - 0.9).abs() < 1e-5);
    edit(&mut state, "/indoor_camera/path_length_min", json!(4.)).unwrap();
    edit(&mut state, "/indoor_camera/path_length_max", json!(2.)).unwrap();
    assert!(state.config().is_err());
}
#[test]
fn edits_do_not_generate_and_apply_and_next_have_distinct_seed_semantics() {
    let mut app = App::new();
    app.insert_resource(BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(46839170),
        ..default()
    })
    .add_message::<RegenerateSceneEvent>();
    let mut state = EditorState::new(app.world().resource::<BevyZeroverseConfig>());
    state.active_seed = Some(46839170);
    for density in [0.2, 0.4, 0.8] {
        edit(&mut state, "/indoor_density", json!(density)).unwrap();
    }
    assert!(state.pending());
    assert!(app
        .world()
        .resource::<Messages<RegenerateSceneEvent>>()
        .is_empty());
    apply(app.world_mut(), &mut state, false).unwrap();
    assert_eq!(state.config().unwrap().indoor_seed, Some(46839170));
    assert!(!state.pending());
    apply(app.world_mut(), &mut state, true).unwrap();
    assert_eq!(state.config().unwrap().indoor_seed, Some(46839171));
    assert_eq!(
        app.world()
            .resource::<Messages<RegenerateSceneEvent>>()
            .len(),
        2
    );
    edit(&mut state, "/width", json!(-1.)).unwrap();
    assert!(apply(app.world_mut(), &mut state, false).is_err());
    assert_eq!(
        app.world()
            .resource::<Messages<RegenerateSceneEvent>>()
            .len(),
        2
    );
}
#[test]
fn live_controls_do_not_apply_pending_geometry() {
    let mut app = App::new();
    app.init_resource::<BevyZeroverseConfig>();
    app.init_resource::<Playback>();
    let mut state = EditorState::new(app.world().resource::<BevyZeroverseConfig>());
    edit(&mut state, "/indoor_density", json!(0.9)).unwrap();
    edit(&mut state, "/render_mode", json!("Semantic")).unwrap();
    live_update(app.world_mut(), &mut state, "/render_mode").unwrap();
    assert_eq!(
        app.world().resource::<BevyZeroverseConfig>().render_mode,
        RenderMode::Semantic
    );
    assert_eq!(
        app.world().resource::<BevyZeroverseConfig>().indoor_density,
        0.65
    );
    assert!(state.pending());
}

#[test]
fn depth_visualization_is_live_and_shared_without_changing_defaults() {
    use crate::render::depth::DepthFormat;
    let mut app = App::new();
    app.init_resource::<BevyZeroverseConfig>();
    let mut state = EditorState::new(app.world().resource::<BevyZeroverseConfig>());
    assert_eq!(
        state.config().unwrap().depth_format,
        DepthFormat::Normalized
    );
    edit(&mut state, "/indoor_density", json!(0.9)).unwrap();
    edit(&mut state, "/depth_format", json!("Turbo")).unwrap();
    live_update(app.world_mut(), &mut state, "/depth_format").unwrap();
    let config = app.world().resource::<BevyZeroverseConfig>();
    assert_eq!(config.depth_format, DepthFormat::Turbo);
    assert_eq!(config.indoor_density, 0.65);
    assert!(state.pending());
    assert!(share::query(config, &state.view).contains("depth_format=Turbo"));
    let parsed = super::super::config_with_query(
        BevyZeroverseConfig::default(),
        [("depth_format".into(), "Turbo".into())],
    )
    .unwrap();
    assert_eq!(parsed.depth_format, DepthFormat::Turbo);
    assert!(model::live("/depth_format"));
}
#[test]
fn camera_presets_retain_valid_continuous_policies() {
    let mut state = EditorState::new(&BevyZeroverseConfig::default());
    for p in ["handheld", "explore"] {
        edit(&mut state, "@path_preset", json!(p)).unwrap();
        state.config().unwrap();
        assert_eq!(value(&state, "@path_preset"), json!(p));
    }
    edit(&mut state, "@mixture", json!(true)).unwrap();
    state.config().unwrap();
    edit(&mut state, "@multiview", json!(false)).unwrap();
    assert!(state.draft["indoor_camera"]["overlap_mixture"].is_null());
    state.config().unwrap();
}
#[test]
fn share_query_roundtrips_nested_policies_seed_and_view_state() {
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(u64::MAX),
        num_cameras: 4,
        camera_grid: false,
        room_schematic: true,
        render_mode: RenderMode::CoVisibility,
        indoor_camera: Some(r#"{"baseline":0.8}"#.into()),
        ..default()
    };
    let view = model::ViewState {
        page: Page::Cameras,
        progress: 0.37,
        ..default()
    };
    let query = share::query(&config, &view);
    let pairs = decode_query(&query);
    assert!(query.contains("depth_format=Normalized"));
    let parsed =
        super::super::config_with_query(super::super::web_viewer_defaults(), pairs).unwrap();
    assert_eq!(model::expand(&config), model::expand(&parsed));
    let restored = model::ViewState::parse(parsed.viewer_state.as_deref().unwrap()).unwrap();
    assert_eq!(restored.page, Page::Cameras);
    assert_eq!(restored.progress, 0.37);
    assert_eq!(parsed.indoor_seed, Some(u64::MAX));
}
#[test]
fn invalid_view_states_fail_closed() {
    for value in [
        r#"{"version":2}"#,
        r#"{"progress":2}"#,
        r#"{"flow_scale":0}"#,
    ] {
        assert!(model::ViewState::parse(value).is_err());
    }
}
#[test]
fn typing_r_does_not_request_a_new_room() {
    let mut app = App::new();
    app.init_resource::<BevyZeroverseConfig>()
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<Time>()
        .init_resource::<Actions>()
        .insert_resource(EditorInputCapture { keyboard: true })
        .add_message::<RegenerateSceneEvent>();
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::KeyR);
    app.world_mut()
        .run_system_once(super::super::regenerate_scene_system)
        .unwrap();
    assert!(app.world().resource::<Actions>().0.is_empty());
    assert!(app
        .world()
        .resource::<Messages<RegenerateSceneEvent>>()
        .is_empty());
}

#[test]
fn every_available_control_has_a_defined_value_including_optional_camera_fields() {
    let mut state = EditorState::new(&BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..default()
    });
    edit(&mut state, "@motion", json!(true)).unwrap();
    edit(&mut state, "@handheld", json!(true)).unwrap();
    edit(&mut state, "@mixture", json!(true)).unwrap();
    for page in Page::ALL {
        state.view.page = page;
        for item in fields::items(&state) {
            if let fields::Item::Field(f) = item {
                if !f.path.starts_with('@') {
                    assert!(
                        state.draft.pointer(&f.path).is_some(),
                        "undefined control {}",
                        f.path
                    );
                }
            }
        }
    }
}

#[test]
fn shared_links_preserve_active_scene_and_pending_edits_separately() {
    let config = BevyZeroverseConfig {
        indoor_seed: Some(46839170),
        ..default()
    };
    let mut state = EditorState::new(&config);
    edit(&mut state, "/indoor_density", json!(0.9)).unwrap();
    edit(&mut state, "/indoor_seed", json!(46839171)).unwrap();
    let (mut active, view) = share::snapshot(&state).unwrap();
    assert_eq!(active.indoor_seed, Some(46839170));
    assert_eq!(active.indoor_density, config.indoor_density);
    active.viewer_state = Some(serde_json::to_string(&view).unwrap());
    let restored = EditorState::new(&active);
    assert!(restored.pending());
    assert_eq!(restored.draft, state.draft);
    assert_eq!(restored.applied, state.applied);
    assert!(restored.error.is_none());
}
#[test]
fn startup_only_changes_cannot_silently_misconfigure_the_running_viewer() {
    let mut app = App::new();
    app.init_resource::<BevyZeroverseConfig>()
        .add_message::<RegenerateSceneEvent>();
    let mut state = EditorState::new(app.world().resource::<BevyZeroverseConfig>());
    edit(&mut state, "/headless", json!(true)).unwrap();
    assert!(apply(app.world_mut(), &mut state, false)
        .unwrap_err()
        .contains("startup option"));
    assert!(app
        .world()
        .resource::<Messages<RegenerateSceneEvent>>()
        .is_empty());
}

#[test]
fn editor_pose_restores_when_grid_startup_creates_the_editor_camera_later() {
    let saved = model::ViewState {
        camera: Some(model::CameraPose {
            focus: [1., 2., 3.],
            yaw: 0.8,
            pitch: 0.2,
            radius: 4.,
            fov: 1.1,
        }),
        ..default()
    };
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(46839170),
        num_cameras: 4,
        camera_grid: true,
        viewer_state: Some(serde_json::to_string(&saved).unwrap()),
        ..default()
    };
    let mut state = EditorState::new(&config);
    state.active_seed = config.indoor_seed;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(config)
        .insert_resource(state)
        .init_resource::<Playback>()
        .add_systems(Last, share::synchronize);
    app.update();
    let entity = app
        .world_mut()
        .spawn((
            EditorCameraMarker::default(),
            PanOrbitCamera::default(),
            Transform::default(),
            Projection::default(),
        ))
        .id();
    app.update();
    assert_eq!(
        {
            let t = app.world().get::<Transform>(entity).unwrap();
            t.translation
                + t.forward().as_vec3()
                    * app
                        .world()
                        .get::<PanOrbitCamera>(entity)
                        .unwrap()
                        .last_anchor_depth()
                        .abs() as f32
        },
        Vec3::new(1., 2., 3.)
    );
    let Projection::Perspective(p) = app.world().get::<Projection>(entity).unwrap() else {
        panic!("perspective lens")
    };
    assert!((p.fov - 1.1).abs() < 1e-6);
}

#[test]
fn small_editor_viewports_stay_inside_the_window() {
    for (width, height) in [(1, 1), (300, 60), (380, 96), (1360, 900)] {
        let window = Window {
            resolution: bevy::window::WindowResolution::new(width, height),
            ..default()
        };
        for left in [0., 380.] {
            let v = shell::bounded_viewport(&window, left);
            assert!(v.physical_size.min_element() >= 1);
            assert!(v.physical_position.x + v.physical_size.x <= window.physical_width());
            assert!(v.physical_position.y + v.physical_size.y <= window.physical_height());
        }
    }
}

#[test]
fn viewport_selection_is_live_exclusive_and_shared() {
    let mut app = App::new();
    app.init_resource::<BevyZeroverseConfig>();
    let mut state = EditorState::new(app.world().resource::<BevyZeroverseConfig>());
    edit(&mut state, "/indoor_density", json!(0.9)).unwrap();
    for mode in ["schematic", "grid", "editor"] {
        edit(&mut state, "@viewport", json!(mode)).unwrap();
        live_update(app.world_mut(), &mut state, "@viewport").unwrap();
        let config = app.world().resource::<BevyZeroverseConfig>();
        assert_eq!(config.room_schematic, mode == "schematic");
        assert_eq!(config.camera_grid, mode == "grid");
        assert_eq!(config.indoor_density, 0.65);
        assert_eq!(value(&state, "@viewport"), json!(mode));
        assert!(state.pending());
    }
}

#[test]
fn first_pointer_frame_over_controls_cannot_start_orbit() {
    let state = EditorState::new(&BevyZeroverseConfig::default());
    let mut window = Window {
        focused: true,
        ..default()
    };
    let capture = EditorInputCapture::default();
    window.set_cursor_position(Some(Vec2::new(20., 200.)));
    assert!(!shell::orbit_enabled(&state, &window, &capture));
    window.set_cursor_position(Some(Vec2::new(600., 200.)));
    assert!(shell::orbit_enabled(&state, &window, &capture));
    assert!(!shell::orbit_enabled(
        &state,
        &window,
        &EditorInputCapture { keyboard: true }
    ));
}

#[test]
fn room_motion_controls_roundtrip_without_enabling_human_models() {
    let mut state = EditorState::new(&BevyZeroverseConfig::default());
    edit(&mut state, "@indoor_motion", json!(true)).unwrap();
    edit(&mut state, "/indoor_motion/chair_fraction", json!(0.8)).unwrap();
    edit(&mut state, "/indoor_motion/fan_fraction", json!(1.0)).unwrap();
    edit(&mut state, "/indoor_motion/fan_turns/0", json!(2.0)).unwrap();
    edit(&mut state, "/indoor_motion/fan_turns/1", json!(8.0)).unwrap();
    let config = state.config().unwrap();
    assert!(config.human_motion.is_none());
    assert_eq!(
        crate::scene::procedural_indoor::motion::IndoorMotionConfig::parse(
            config.indoor_motion.as_deref().unwrap()
        )
        .unwrap()
        .chair_fraction,
        0.8
    );
    let policy = crate::scene::procedural_indoor::motion::IndoorMotionConfig::parse(
        config.indoor_motion.as_deref().unwrap(),
    )
    .unwrap();
    assert_eq!(policy.fan_fraction, 1.);
    assert_eq!(policy.fan_turns, [2., 8.]);
    edit(&mut state, "@indoor_motion", json!(false)).unwrap();
    assert!(state.config().unwrap().indoor_motion.is_none());
}
