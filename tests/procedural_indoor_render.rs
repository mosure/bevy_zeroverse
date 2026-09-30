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

// Capture channels and asset-root discovery are process globals. Keep fixtures
// isolated even when the ignored GPU tests are launched with default threading.
static RENDER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn capture(app: &mut App, modes: Vec<RenderMode>) -> Sample {
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: modes,
        frames: 8,
        warmup_frames: 16,
        timesteps: Vec::new(),
        ..default()
    });
    let start = Instant::now();
    loop {
        app.update();
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
            "capture stalled"
        );
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
        assert!(app.world().resource::<Assets<StandardMaterial>>().len() <= 80);
        assert!(app.world().resource::<Assets<Image>>().len() < 80);
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
