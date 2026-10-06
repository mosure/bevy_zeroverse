//! Run explicitly on a machine with a native graphics adapter.
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    headless::{create_app, setup_globals},
    io::channels,
    render::{color::ColorEncoding, depth::DepthFormat, RenderMode},
    sample::{Sample, SamplerState},
    scene::{
        procedural_indoor::validation::validate_annotations_with_precision, RegenerateSceneEvent,
        ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};

#[derive(Default)]
struct PersistentAssets {
    materials: std::collections::HashSet<AssetId<StandardMaterial>>,
    images: std::collections::HashSet<AssetId<Image>>,
}

impl PersistentAssets {
    fn from_world(world: &World) -> Self {
        Self {
            materials: world.resource::<Assets<StandardMaterial>>().ids().collect(),
            images: world.resource::<Assets<Image>>().ids().collect(),
        }
    }

    /// With lookahead disabled, only bootstrap assets and this room's live
    /// references may remain. Geometry/finish diversity determines the budget.
    fn assert_current_room(&self, world: &mut World) {
        use bevy::{
            asset::VisitAssetDependencies,
            camera::RenderTarget,
            light::{EnvironmentMapLight, IrradianceVolume},
        };
        use bevy_zeroverse::render::{ground_truth::GroundTruthCamera, DisabledPbrMaterial};
        assert_eq!(
            world
                .resource::<bevy_zeroverse::scene::procedural_indoor::preparation::IndoorPrefetch>()
                .depth,
            0,
            "this ownership audit excludes speculative future rooms"
        );
        assert_eq!(
            world
                .query_filtered::<(), With<bevy_zeroverse::scene::ZeroverseSceneRoot>>()
                .iter(world)
                .count(),
            1,
            "regeneration must retire the preceding room root"
        );
        let mut material_ids = self.materials.clone();
        material_ids.extend(
            world
                .query::<&MeshMaterial3d<StandardMaterial>>()
                .iter(world)
                .map(|material| material.0.id()),
        );
        material_ids.extend(
            world
                .query::<&DisabledPbrMaterial>()
                .iter(world)
                .map(|material| material.material.id()),
        );
        material_ids.extend(
            world
                .resource::<bevy_zeroverse::material::ZeroverseMaterials>()
                .materials
                .iter()
                .map(Handle::id),
        );
        let materials = world.resource::<Assets<StandardMaterial>>();
        let unreferenced_materials: Vec<_> = materials
            .ids()
            .filter(|id| !material_ids.contains(id))
            .collect();
        assert!(
            unreferenced_materials.is_empty(),
            "materials retained without current-room or bootstrap ownership: {unreferenced_materials:?}"
        );
        let mut image_ids = self.images.clone();
        for (_, material) in materials.iter() {
            // Include every StandardMaterial map, including optional coat,
            // anisotropy and transmission maps as the PBR vocabulary evolves.
            material.visit_dependencies(&mut |id| {
                if let Ok(image) = id.try_typed::<Image>() {
                    image_ids.insert(image);
                }
            });
        }
        for target in world.query::<&RenderTarget>().iter(world) {
            if let RenderTarget::Image(target) = target {
                image_ids.insert(target.handle.id());
            }
        }
        for environment in world.query::<&EnvironmentMapLight>().iter(world) {
            image_ids.extend([environment.diffuse_map.id(), environment.specular_map.id()]);
        }
        for volume in world.query::<&IrradianceVolume>().iter(world) {
            image_ids.insert(volume.voxels.id());
        }
        let mut depth_targets = 0;
        for target in world.query::<&GroundTruthCamera>().iter(world) {
            image_ids.extend([target.world_depth.id(), target.normal_semantic.id()]);
            if let Some(flow) = &target.flow {
                image_ids.insert(flow.id());
            }
            depth_targets += 1;
        }
        // The private depth attachment has exactly one slot per live geometric
        // camera. Its image label identifies that fixed storage, not a room map.
        let mut private_depth_images = 0;
        let mut unreferenced_images = Vec::new();
        for (id, image) in world.resource::<Assets<Image>>().iter() {
            if image_ids.contains(&id) {
                continue;
            }
            if image.texture_descriptor.label == Some("ground_truth_depth_test_f32") {
                private_depth_images += 1;
            } else {
                unreferenced_images.push(id);
            }
        }
        assert!(
            unreferenced_images.is_empty(),
            "images retained without current-room or bootstrap ownership: {unreferenced_images:?}"
        );
        assert!(private_depth_images <= depth_targets);
    }
}

// Capture channels and asset-root discovery are process globals. Keep fixtures
// isolated even when the ignored GPU tests are launched with default threading.
static RENDER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn capture(app: &mut App, modes: Vec<RenderMode>) -> Sample {
    capture_with_state(
        app,
        SamplerState {
            enabled: true,
            regenerate_scene: false,
            render_modes: modes,
            frames: 8,
            warmup_frames: 16,
            timesteps: Vec::new(),
            ..default()
        },
        false,
    )
}

fn capture_with_state(app: &mut App, state: SamplerState, production_poll: bool) -> Sample {
    app.insert_resource(state);
    let start = Instant::now();
    loop {
        if production_poll {
            bevy_zeroverse::headless::update_capture(app);
        } else {
            app.update();
        }
        assert!(
            app.should_exit().is_none(),
            "render app exited during capture"
        );
        assert!(
            app.world()
                .resource::<bevy_zeroverse::sample::CaptureFailure>()
                .0
                .is_none(),
            "capture failed: {:?}",
            app.world()
                .resource::<bevy_zeroverse::sample::CaptureFailure>()
                .0
        );
        if let Ok(sample) = channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            return sample;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "capture stalled: scene={:?}, ovoxel={:?}, readiness={:?}, sampler={:?}, assets={:?}, pipeline={:?}",
            app.world().resource::<BevyZeroverseConfig>().scene_type,
            app.world().resource::<BevyZeroverseConfig>().ovoxel_mode,
            app.world().resource::<bevy_zeroverse::sample::CaptureReadiness>(),
            app.world().resource::<SamplerState>(),
            app.world().resource::<bevy_zeroverse::asset::WaitForAssets>(),
            app.world().get_resource::<bevy_zeroverse::io::image_copy::CapturePipelineReadiness>().map(|p| (p.ready(), p.pipeline_count(), p.missing_assets(), p.failure())),
        );
    }
}

#[test]
#[ignore = "requires native GPU; render-fenced multi-step indoor flow and co-visibility"]
fn render_fenced_indoor_capture_preserves_flow_and_shared_visibility() {
    use bevy::diagnostic::{DiagnosticPath, DiagnosticsStore};
    use bevy_zeroverse::{
        render::{co_visibility, ground_truth::GroundTruthCamera},
        sample::{qualification::rendered_overlap, CaptureProgress},
    };

    let _guard = RENDER_TEST_LOCK.lock().unwrap();
    std::env::set_var(
        "BEVY_ASSET_ROOT",
        format!("{}/assets", env!("CARGO_MANIFEST_DIR")),
    );
    setup_globals(Some(format!("{}/assets", env!("CARGO_MANIFEST_DIR"))));
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Semantic,
        RenderMode::OpticalFlow,
        RenderMode::MotionVectors,
        RenderMode::CoVisibility,
    ];
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(200),
        indoor_density: 0.65,
        indoor_human_density: 0.,
        human_motion: None,
        indoor_camera: Some(r#"{"duration_seconds":4.0,"path_length_min":0.1,"path_length_max":2.0,"long_path_fraction":0.0}"#.into()),
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        num_cameras: 3,
        width: 321.,
        height: 239.,
        playback_mode: PlaybackMode::Still,
        playback_steps: 3,
        playback_step: 0.25,
        depth_format: DepthFormat::Linear,
        render_modes: modes,
        ..default()
    };
    let mut state = SamplerState::from_config(&config);
    state.regenerate_scene = false;
    let mut app = create_app(None, Some(config), false);
    app.finish();
    app.cleanup();
    let sample = capture_with_state(&mut app, state, true);

    assert_eq!(sample.view_dim, 3);
    assert_eq!(sample.views.len(), 9);
    assert_eq!(sample.indoor.as_ref().unwrap().seed, 200);
    assert!(sample.indoor.as_ref().unwrap().humans.is_empty());
    assert!(sample.ovoxel.is_none());
    assert_eq!(
        app.world().resource::<CaptureProgress>().completed_requests,
        3
    );
    let hits = app
        .world()
        .resource::<DiagnosticsStore>()
        .get(&DiagnosticPath::const_new(
            "capture/settling_fence/hit_count",
        ))
        .and_then(|diagnostic| diagnostic.value())
        .unwrap_or(0.);
    assert!(
        hits >= 3.,
        "each changed camera timestep needs its own render fence: {hits}"
    );
    let mut cameras = app.world_mut().query::<&GroundTruthCamera>();
    let stamps: Vec<_> = cameras
        .iter(app.world())
        .map(|camera| {
            (
                camera.flow_sequence,
                camera.frame_id,
                camera.rendered_frame(),
            )
        })
        .collect();
    assert_eq!(stamps.len(), 3);
    assert!(stamps
        .iter()
        .all(|stamp| stamp.0 != 0 && stamp.1 == 3 && stamp.2 == Some(3)));
    assert!(stamps.iter().all(|stamp| *stamp == stamps[0]));
    co_visibility::validate_metadata(sample.co_visibility_metadata.as_ref().unwrap(), 3).unwrap();

    let floats = |bytes: &[u8]| -> Vec<[f32; 4]> {
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|pixel| bytemuck::pod_read_unaligned(pixel))
            .collect()
    };
    for (step, frame) in sample.views.as_chunks::<3>().0.iter().enumerate() {
        for (camera, view) in frame.iter().enumerate() {
            assert_eq!(view.time, step as f32 * 0.25);
            assert_eq!(view.trajectory_progress, Some(view.time));
            assert_eq!(view.time_seconds, Some(step as f32));
            let alignment = validate_annotations_with_precision(
                view,
                sample.aabb,
                321,
                239,
                sample.annotation_precision,
            )
            .unwrap();
            assert!(alignment.checked_pixels > 1_000);
            assert!(alignment.reprojection_max_pixels < 0.005, "{alignment:?}");
            co_visibility::validate_plane(&view.co_visibility, 321 * 239, 3, camera).unwrap();
            let optical = floats(&view.optical_flow);
            let motion = floats(&view.motion_vectors);
            assert_eq!(optical.len(), 321 * 239);
            assert_eq!(motion.len(), optical.len());
            for (flow, vector) in optical.iter().zip(motion) {
                assert!(flow.iter().chain(&vector).all(|value| value.is_finite()));
                assert_eq!(vector, [flow[0] / 321., flow[1] / 239., flow[2], flow[3]]);
                assert!([0., 1.].contains(&flow[2]) && [0., 1.].contains(&flow[3]));
            }
            if step == 2 {
                assert!(
                    optical.iter().all(|flow| *flow == [0.; 4]),
                    "terminal flow needs zero vectors and masks"
                );
                continue;
            }
            // All geometry is static. Independently project source world hits
            // into the next captured camera; extra warmup/render frames must
            // not become flow endpoints or advance temporal history.
            let next = &sample.views[(step + 1) * 3 + camera];
            assert_ne!(view.world_from_view, next.world_from_view);
            let next_from_world = Mat4::from_cols_array_2d(&next.world_from_view)
                .as_dmat4()
                .inverse();
            let k = next.calibration.as_ref().unwrap().k;
            let low = Vec3::from_array(sample.aabb[0]).as_dvec3();
            let range = Vec3::from_array(sample.aabb[1]).as_dvec3() - low;
            let positions = floats(&view.position);
            let mut checked = 0;
            let mut maximum = 0f64;
            for (pixel, (position, flow)) in positions.iter().zip(optical).enumerate() {
                if position[3] == 0. || flow[2] != 1. || flow[3] != 1. {
                    continue;
                }
                let world =
                    low + Vec3::new(position[0], position[1], position[2]).as_dvec3() * range;
                let p = next_from_world.transform_point3(world);
                let endpoint = bevy::math::DVec2::new(
                    (k[0][0] as f64 * p.x - k[0][1] as f64 * p.y) / -p.z + k[0][2] as f64,
                    -k[1][1] as f64 * p.y / -p.z + k[1][2] as f64,
                );
                let source =
                    bevy::math::DVec2::new((pixel % 321) as f64 + 0.5, (pixel / 321) as f64 + 0.5);
                let actual = bevy::math::DVec2::new(flow[0] as f64, flow[1] as f64);
                maximum = maximum.max((endpoint - source).distance(actual));
                checked += 1;
            }
            assert!(
                checked > 1_000,
                "too few visible temporal correspondences: {checked}"
            );
            assert!(
                maximum < 0.005,
                "step {step} camera {camera}: analytic flow error {maximum}px"
            );
            println!("step {step} camera {camera}: {checked} analytic flow endpoints, max {maximum:.7}px");
        }
    }
    let overlap = rendered_overlap(&sample.views, 3, sample.aabb);
    assert_eq!(overlap.len(), 18);
    assert!(overlap.iter().all(|pair| pair.valid_source_pixels > 1_000));
    assert!(overlap.iter().any(|pair| pair.shared_pixels > 0));
    println!(
        "three capture epochs, {hits} settling-fence hits, and {} directed same-time overlap pairs",
        overlap.len()
    );
}

#[test]
#[ignore = "requires native GPU; full-pixel qualification of thin-triangle regression seeds"]
fn thin_triangle_seeds_keep_every_pixel_on_its_calibrated_ray() {
    let _guard = RENDER_TEST_LOCK.lock().unwrap();
    setup_globals(Some(format!("{}/assets", env!("CARGO_MANIFEST_DIR"))));
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Semantic,
        RenderMode::CoVisibility,
    ];
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(42_430_575),
        indoor_density: 0.65,
        indoor_human_density: 0.,
        indoor_camera: Some(r#"{"primary_room":false,"path_length_min":0.0,"path_length_max":0.0,"long_path_fraction":0.0,"multiview":{"max_baseline":6.0,"min_baseline":0.4,"min_overlap":0.15,"min_reference_baseline":0.8,"min_spread":0.25,"trajectory_variation":0.3}}"#.into()),
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        num_cameras: 5,
        width: 512.,
        height: 512.,
        playback_mode: PlaybackMode::Still,
        playback_steps: 1,
        playback_step: 0.,
        depth_format: DepthFormat::Linear,
        render_modes: modes.clone(),
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.finish();
    app.cleanup();
    for seed in [42_430_575, 43_084_584] {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_seed = Some(seed);
        app.world_mut().write_message(RegenerateSceneEvent);
        app.update();
        let sample = capture(&mut app, modes.clone());
        assert_eq!(sample.indoor.as_ref().unwrap().seed, seed);
        assert_eq!(sample.views.len(), 5);
        for (index, view) in sample.views.iter().enumerate() {
            let report = validate_annotations_with_precision(
                view,
                sample.aabb,
                512,
                512,
                sample.annotation_precision,
            )
            .unwrap();
            assert!(report.checked_pixels > 100_000);
            assert!(
                report.reprojection_max_pixels < 0.005,
                "seed {seed} view {index}: {report:?}"
            );
            println!("seed {seed} view {index}: {report:?}");
        }
    }
}

#[test]
#[ignore = "requires native GPU and Anny assets; real mode switches, R and CPU/GPU voxels"]
fn indoor_annotations_voxels_and_keyboard_after_object_scene() {
    let _guard = RENDER_TEST_LOCK.lock().unwrap();
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    use bevy::input::{
        keyboard::{Key, KeyboardInput},
        ButtonState,
    };
    use bevy_zeroverse::{
        app::OvoxelMode,
        ovoxel::{OvoxelCache, OvoxelVolume},
        scene::ZeroverseSceneSettings,
    };
    setup_globals(None);
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::Object,
            indoor_seed: Some(115),
            indoor_human_density: 0.5,
            headless: true,
            editor: false,
            gizmos: false,
            image_copiers: true,
            keybinds: true,
            press_esc_close: false,
            initialize_scene: false,
            num_cameras: 1,
            width: 321.0,
            height: 239.0,
            ovoxel_resolution: 48,
            playback_mode: PlaybackMode::Still,
            playback_steps: 1,
            depth_format: DepthFormat::Linear,
            ..default()
        }),
        false,
    );
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    app.world_mut().write_message(RegenerateSceneEvent);
    let object = capture(&mut app, vec![RenderMode::Color]);
    assert!(object.indoor.is_none());
    app.world_mut().resource_mut::<SamplerState>().enabled = false;
    // Exercise the old inspector resource, which previously disagreed with asset demand.
    app.world_mut()
        .resource_mut::<ZeroverseSceneSettings>()
        .scene_type = ZeroverseSceneType::ProceduralIndoor;
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .ovoxel_mode = OvoxelMode::CpuAsync;
    // Property edits only select configuration; the viewer's explicit R/apply
    // request commits the scene change without interfering with UI sliders.
    app.world_mut().write_message(RegenerateSceneEvent);
    app.update();
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Semantic,
        RenderMode::Position,
    ];
    let indoor = capture(&mut app, modes.clone());
    assert_eq!(indoor.indoor.as_ref().unwrap().seed, 115);
    assert!(!indoor.indoor.as_ref().unwrap().humans.is_empty());
    for view in &indoor.views {
        validate_annotations_with_precision(
            view,
            indoor.aabb,
            321,
            239,
            indoor.annotation_precision,
        )
        .unwrap();
    }
    let cpu = indoor.ovoxel.unwrap();
    assert!(cpu.coords.len() > 1000);
    for label in ["wall", "floor", "chair", "person", "window"] {
        let id = cpu
            .semantic_labels
            .iter()
            .position(|v| v == label)
            .expect(label) as u16;
        assert!(cpu.semantics.contains(&id), "missing voxels for {label}");
    }
    // Recompute the same geometry with the GPU implementation while in annotation mode.
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .ovoxel_mode = OvoxelMode::GpuCompute;
    let roots: Vec<_> = app
        .world_mut()
        .query_filtered::<Entity, With<OvoxelVolume>>()
        .iter(app.world())
        .collect();
    for root in roots {
        app.world_mut()
            .entity_mut(root)
            .remove::<(OvoxelVolume, OvoxelCache)>();
    }
    let gpu = capture(&mut app, modes.clone()).ovoxel.unwrap();
    assert!(
        cpu.coords == gpu.coords,
        "CPU/GPU indoor occupancy mismatch: {} vs {} cells; first differing pair {:?}",
        cpu.coords.len(),
        gpu.coords.len(),
        cpu.coords.iter().zip(&gpu.coords).find(|(a, b)| a != b)
    );
    assert_eq!(cpu.semantic_labels, gpu.semantic_labels);
    assert!(
        cpu.semantics == gpu.semantics,
        "CPU/GPU semantic mismatch in {} / {} cells",
        cpu.semantics
            .iter()
            .zip(&gpu.semantics)
            .filter(|(a, b)| a != b)
            .count(),
        cpu.coords.len()
    );
    app.world_mut().resource_mut::<SamplerState>().enabled = false;
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .ovoxel_mode = OvoxelMode::Disabled;
    app.world_mut().write_message(KeyboardInput {
        key_code: KeyCode::KeyR,
        logical_key: Key::Character("r".into()),
        state: ButtonState::Pressed,
        text: Some("r".into()),
        repeat: false,
        window: Entity::PLACEHOLDER,
    });
    app.update();
    app.update();
    let regenerated = capture(&mut app, modes);
    assert_eq!(
        regenerated.indoor.as_ref().unwrap().seed,
        116,
        "R must regenerate after Object -> Indoor"
    );
    println!("Object -> Indoor; Anny people; depth/normal/semantic/position; CPU/GPU semantic voxels; R -> seed 116 passed");
}

#[test]
#[ignore = "requires a native GPU; exercises real shaders and readback"]
fn empty_assets_rotated_odd_size_capture_and_legacy_scene_switches() {
    let _guard = RENDER_TEST_LOCK.lock().unwrap();
    let assets = tempfile::tempdir().unwrap();
    std::env::set_var("BEVY_ASSET_ROOT", assets.path());
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Semantic,
        RenderMode::Position,
    ];
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(3),
        indoor_human_density: 0.0,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        initialize_scene: false,
        num_cameras: 2,
        width: 321.0,
        height: 239.0,
        render_modes: modes.clone(),
        playback_mode: PlaybackMode::Still,
        playback_steps: 1,
        rotation_augmentation: true,
        depth_format: DepthFormat::Linear,
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    let persistent_assets = PersistentAssets::from_world(app.world());
    for seed in 3..7 {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_seed = Some(seed);
        app.update();
        app.world_mut().write_message(RegenerateSceneEvent);
        *app.world_mut().resource_mut::<RenderMode>() = RenderMode::Color;
        for _ in 0..16 {
            app.update();
        }
        let sample = capture(&mut app, modes.clone());
        assert_eq!(sample.indoor.as_ref().unwrap().seed, seed);
        assert_eq!(sample.color_encoding, ColorEncoding::TonemappedLinear);
        assert_eq!(sample.views.len(), 2);
        assert!(sample.object_obbs.len() >= sample.indoor.as_ref().unwrap().objects.len());
        for view in &sample.views {
            assert_eq!(view.color.len(), 321 * 239 * 16);
            assert_eq!(view.semantic.len(), view.color.len());
            validate_annotations_with_precision(
                view,
                sample.aabb,
                321,
                239,
                sample.annotation_precision,
            )
            .unwrap();
        }
        // Asset tracking and pipelined render extraction get two bounded
        // retirement updates, with capture and automatic regeneration disabled.
        assert!(!app.world().resource::<SamplerState>().enabled);
        for _ in 0..2 {
            app.update();
        }
        persistent_assets.assert_current_room(app.world_mut());
    }
    // Regenerate while an annotation material is active. This catches extraction
    // before Bevy's EntitySpecializationTicks bookkeeping (observed on WebGPU).
    for (index, mode) in [
        RenderMode::Semantic,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Depth,
    ]
    .into_iter()
    .enumerate()
    {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .render_modes = vec![mode.clone()];
        *app.world_mut().resource_mut::<RenderMode>() = mode.clone();
        for _ in 0..12 {
            app.update();
        }
        let seed = 700 + index as u64;
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_seed = Some(seed);
        app.update();
        app.world_mut().write_message(RegenerateSceneEvent);
        for _ in 0..16 {
            app.update();
        }
        let sample = capture(&mut app, vec![mode.clone()]);
        assert_eq!(sample.indoor.as_ref().unwrap().seed, seed);
        for view in sample.views {
            let bytes = match mode {
                RenderMode::Semantic => view.semantic,
                RenderMode::Normal => view.normal,
                RenderMode::Position => view.position,
                RenderMode::Depth => view.depth,
                _ => unreachable!(),
            };
            let values: &[f32] = bytemuck::cast_slice(&bytes);
            assert_eq!(values.len(), 321 * 239 * 4);
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().copied().fold(0.0_f32, f32::max) > 0.0);
        }
        println!("annotation {mode:?} regeneration passed");
    }
    for scene in [ZeroverseSceneType::CornellCube, ZeroverseSceneType::Room] {
        {
            let mut args = app.world_mut().resource_mut::<BevyZeroverseConfig>();
            args.scene_type = scene.clone();
            args.render_mode = RenderMode::Color;
            args.render_modes = vec![RenderMode::Color];
        }
        // Allow deferred legacy catalog loading before requesting its scene.
        for _ in 0..4 {
            app.update();
        }
        app.world_mut().write_message(RegenerateSceneEvent);
        for _ in 0..16 {
            app.update();
        }
        let sample = capture(&mut app, vec![RenderMode::Color]);
        assert!(sample.indoor.is_none());
        assert_eq!(sample.color_encoding, ColorEncoding::Legacy);
        for view in sample.views {
            let values: &[f32] = bytemuck::cast_slice(&view.color);
            assert_eq!(values.len(), 321 * 239 * 4);
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().copied().fold(0.0_f32, f32::max) > 0.0);
        }
        println!("legacy {scene:?} capture passed");
    }
}

#[test]
#[ignore = "requires native GPU; nonrectangular envelope CPU/GPU voxel and annotation contract"]
fn nonrectangular_envelope_voxels_exclude_context_and_keep_floor_levels() {
    let _guard = RENDER_TEST_LOCK.lock().unwrap();
    use bevy_zeroverse::{
        app::OvoxelMode,
        ovoxel::{OvoxelCache, OvoxelExcluded, OvoxelVolume},
    };
    let assets = tempfile::tempdir().unwrap();
    std::env::set_var("BEVY_ASSET_ROOT", assets.path());
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Semantic,
        RenderMode::Position,
    ];
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(229),
            indoor_human_density: 0.0,
            headless: true,
            editor: false,
            gizmos: false,
            image_copiers: true,
            keybinds: false,
            press_esc_close: false,
            initialize_scene: false,
            num_cameras: 2,
            width: 320.,
            height: 200.,
            ovoxel_resolution: 48,
            ovoxel_mode: OvoxelMode::CpuAsync,
            playback_mode: PlaybackMode::Still,
            playback_steps: 1,
            rotation_augmentation: false,
            depth_format: DepthFormat::Linear,
            ..default()
        }),
        false,
    );
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    app.world_mut().write_message(RegenerateSceneEvent);
    let sample = capture(&mut app, modes.clone());
    let scene = sample.indoor.as_ref().unwrap();
    let e = scene.envelope.as_ref().unwrap();
    assert!(e.mezzanine.is_some() && e.minimum_floor() < -0.1 && e.footprint.len() > 6);
    assert!(
        app.world_mut()
            .query_filtered::<Entity, (With<Mesh3d>, With<OvoxelExcluded>)>()
            .iter(app.world())
            .count()
            >= 2,
        "context meshes need explicit exclusion even inside the rectangular crop"
    );
    for v in &sample.views {
        validate_annotations_with_precision(v, sample.aabb, 320, 200, sample.annotation_precision)
            .unwrap();
    }
    let cpu = sample.ovoxel.unwrap();
    assert_eq!(cpu.aabb, sample.aabb);
    assert!((cpu.aabb[0][1] - (e.minimum_floor() - 0.2)).abs() < 1e-5);
    assert!(cpu.coords.len() > 1000);
    let min = Vec3::from(cpu.aabb[0]);
    let voxel = (Vec3::from(cpu.aabb[1]) - min) / cpu.resolution as f32;
    let clearance = voxel.length() + 0.20;
    for c in &cpu.coords {
        let p = (min + (UVec3::from(*c).as_vec3() + Vec3::splat(0.5)) * voxel).xz();
        if bevy_zeroverse::scene::procedural_indoor::envelope::polygon::contains(
            &e.footprint,
            p,
            0.,
        ) {
            continue;
        }
        let distance =
            bevy_zeroverse::scene::procedural_indoor::envelope::polygon::edges(&e.footprint)
                .map(|(a, b)| {
                    p.distance(
                        a + (b - a) * ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0., 1.),
                    )
                })
                .fold(f32::INFINITY, f32::min);
        assert!(
            distance <= clearance,
            "context/courtyard voxel outside shell at {p:?}"
        );
    }
    app.world_mut().resource_mut::<SamplerState>().enabled = false;
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .ovoxel_mode = OvoxelMode::GpuCompute;
    let roots: Vec<_> = app
        .world_mut()
        .query_filtered::<Entity, With<OvoxelVolume>>()
        .iter(app.world())
        .collect();
    for root in roots {
        app.world_mut()
            .entity_mut(root)
            .remove::<(OvoxelVolume, OvoxelCache)>();
    }
    let gpu = capture(&mut app, modes).ovoxel.unwrap();
    assert!(
        cpu.coords == gpu.coords,
        "CPU/GPU envelope occupancy mismatch"
    );
    assert_eq!(cpu.semantic_labels, gpu.semantic_labels);
    assert!(
        cpu.semantics == gpu.semantics,
        "CPU/GPU envelope semantic mismatch"
    );
    println!("seed 229: {} CPU/GPU matching occupied cells; courtyard excluded; depressed floor and mezzanine retained",cpu.coords.len());
}
