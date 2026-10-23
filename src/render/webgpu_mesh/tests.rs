use super::*;

/// Compare the actual float render attachments against Bevy's unmodified draw
/// commands, in the same world/device with the same assets and camera poses.
#[test]
#[ignore = "requires a native GPU; full PBR and shadow command parity"]
fn ordinary_bindings_preserve_rendered_fisheye_and_pinhole_pixels() {
    for probability in [0., 1.] {
        qualify(probability, 400, 1, 128.);
    }
}

#[test]
#[ignore = "requires a native GPU; multi-view shadow batching regression"]
fn batching_preserves_seed401_with_independent_bevy_draw_commands() {
    qualify(1., 401, 3, 512.);
}

fn qualify(probability: f32, seed: u64, cameras: usize, size: f32) {
    use crate::{
        app::BevyZeroverseConfig,
        camera::{lens::LensSampling, Playback, PlaybackMode},
        headless::{create_app, setup_globals},
        scene::{procedural_indoor::cameras::CameraSettings, ZeroverseSceneType},
    };
    setup_globals(None);
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(seed),
        indoor_human_density: 0.,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: false,
        num_cameras: cameras,
        width: size,
        height: size,
        ssaa_factor: 2,
        playback_mode: PlaybackMode::Still,
        indoor_camera: Some(
            serde_json::to_string(&CameraSettings {
                lenses: Some(LensSampling {
                    probability,
                    ..default()
                }),
                ..default()
            })
            .unwrap(),
        ),
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.add_plugins(crate::io::image_copy::ImageCopyPlugin);
    app.finish();
    // Native installs the ordinary-mesh fast path too. Keep Bevy's commands as
    // the independent control instead of accidentally comparing fast vs fast.
    restore_bevy_commands(&mut app);
    super::super::preview_shadows::force_for_test(&mut app, false);
    // This same-world oracle changes draw functions between captures. Retain
    // direct access to RenderApp instead of moving it into the render worker.
    // The end-to-end benchmark separately exercises the pipelined production path.
    app.remove_sub_app(bevy::render::pipelined_rendering::RenderExtractApp);
    app.cleanup();
    let device = app
        .world()
        .resource::<bevy::render::renderer::RenderDevice>()
        .clone();
    let start = std::time::Instant::now();
    for frame in 0..6000 {
        app.world_mut().resource_mut::<Playback>().progress = 0.5;
        app.update();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(
            start.elapsed().as_secs() < 60,
            "scene did not become renderable"
        );
        if frame >= 80
            && app
                .world()
                .resource::<crate::sample::CaptureReadiness>()
                .scene_ready()
            && app
                .world()
                .resource::<crate::io::image_copy::CapturePipelineReadiness>()
                .ready()
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(app
        .world()
        .resource::<crate::io::image_copy::CapturePipelineReadiness>()
        .ready());
    assert!(app
        .world()
        .resource::<crate::sample::CaptureReadiness>()
        .scene_ready());
    let before = capture(&mut app);
    let original_grids: Vec<_> = app
        .world_mut()
        .query_filtered::<(Entity, Option<&bevy::light::cluster::ClusterConfig>), With<Camera3d>>()
        .iter(app.world())
        .map(|(e, c)| (e, c.copied()))
        .collect();
    let enabled = app
        .world_mut()
        .query::<&SpotLight>()
        .iter(app.world())
        .filter(|l| l.shadow_maps_enabled)
        .count();
    println!("enabled spot shadows: {enabled}");
    assert!(enabled > 0);
    install(&mut app);
    let fast_only = capture(&mut app);
    let fast_error = maximum_error(&before, &fast_only);
    println!("seed {seed}: fast bindings alone max error {fast_error}");
    assert!(fast_error <= 1e-6, "ordinary bindings changed pixels");
    super::super::preview_shadows::force_for_test(&mut app, true);
    // Exercise the densest light grid together with exact shadow batching.
    {
        let cameras: Vec<_> = app
            .world_mut()
            .query_filtered::<Entity, With<Camera3d>>()
            .iter(app.world())
            .collect();
        for entity in cameras {
            app.world_mut()
                .entity_mut(entity)
                .insert(crate::camera::lens::clusters::grid(4));
        }
    }
    let after = capture(&mut app);
    assert_proxies_absent_from_camera_queues(&mut app, cameras);
    println!(
        "seed {seed}: batching max error {}",
        maximum_error(&before, &after)
    );
    {
        let batches = super::super::preview_shadows::active_batches(app.world_mut());
        println!("active compatible shadow batches: {batches}");
        assert!(
            batches > 0,
            "the test must exercise the batched rendering path"
        );
        super::super::preview_shadows::force_for_test(&mut app, false);
        for (entity, original) in original_grids {
            if let Some(config) = original {
                app.world_mut().entity_mut(entity).insert(config);
            } else {
                app.world_mut()
                    .entity_mut(entity)
                    .remove::<bevy::light::cluster::ClusterConfig>();
            }
        }
        let restored = capture(&mut app);
        assert_eq!(
            super::super::preview_shadows::active_batches(app.world_mut()),
            0
        );
        assert_eq!(
            before, restored,
            "capture must restore the original render path"
        );
        // This oracle is meaningful only if shadows actually affect the image.
        for mut light in app
            .world_mut()
            .query::<&mut DirectionalLight>()
            .iter_mut(app.world_mut())
        {
            light.shadow_maps_enabled = false;
        }
        for mut light in app
            .world_mut()
            .query::<&mut PointLight>()
            .iter_mut(app.world_mut())
        {
            light.shadow_maps_enabled = false;
        }
        for mut light in app
            .world_mut()
            .query::<&mut SpotLight>()
            .iter_mut(app.world_mut())
        {
            light.shadow_maps_enabled = false;
        }
        let no_shadows = capture(&mut app);
        let shadow_effect = before
            .iter()
            .flatten()
            .zip(no_shadows.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("maximum shadow contribution: {shadow_effect}");
        assert!(shadow_effect > 1e-4, "the fixture must see shadows");
    }
    assert_eq!(before.len(), cameras);
    for (index, (before, after)) in before.iter().zip(&after).enumerate() {
        let (low, high) = before
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| pixel[..3].iter().copied())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), v| {
                (low.min(v), high.max(v))
            });
        println!("lens probability {probability}, view {index}: RGB range {low}..{high}");
        assert!(
            high - low > 1e-4 && high > 1e-3,
            "blank or constant RGB view {index}"
        );
        assert_eq!(before.len(), after.len());
        let error = before
            .iter()
            .zip(after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("view {index}: max linear RGBA error {error}");
        assert!(error <= 1e-6, "view {index} changed by {error}");
    }
}

fn maximum_error(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0., |error, (a, b)| {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(error, |error, (a, b)| {
            assert!(a.is_finite() && b.is_finite());
            error.max((a - b).abs())
        })
    })
}

fn assert_proxies_absent_from_camera_queues(app: &mut App, camera_count: usize) {
    use bevy::render::{
        camera::ExtractedCamera, sync_world::MainEntity, view::RenderVisibleEntities,
    };
    let proxies: std::collections::HashSet<_> = app
        .world_mut()
        .query::<(Entity, &Name)>()
        .iter(app.world())
        .filter(|(_, name)| name.as_str() == "procedural_shadow_batch")
        .map(|(entity, _)| MainEntity::from(entity))
        .collect();
    assert!(!proxies.is_empty());
    let world = app.sub_app_mut(RenderApp).world_mut();
    let mut views = 0;
    let mut leaked = 0;
    for visible in world
        .query_filtered::<&RenderVisibleEntities, With<ExtractedCamera>>()
        .iter(world)
    {
        views += 1;
        if let Some(meshes) = visible.get::<Mesh3d>() {
            leaked += meshes
                .iter_visible()
                .filter(|(_, main)| proxies.contains(main))
                .count();
        }
    }
    println!(
        "actual render-world camera queues: {views}, proxies: {}, leaked: {leaked}",
        proxies.len()
    );
    assert!(views >= camera_count);
    assert_eq!(leaked, 0, "shadow proxies entered camera render queues");
}

fn capture(app: &mut App) -> Vec<Vec<f32>> {
    use crate::{
        camera::{lens::LensCamera, CaptureCameraIndex, ZeroverseCamera},
        io::image_copy::ImageCopier,
        render::ssaa::SsaaCamera,
    };
    use bevy::camera::RenderTarget;
    let device = app
        .world()
        .resource::<bevy::render::renderer::RenderDevice>()
        .clone();
    let started = std::time::Instant::now();
    for frame in 0..500 {
        app.update();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        if frame >= 20
            && app
                .world()
                .resource::<crate::io::image_copy::CapturePipelineReadiness>()
                .ready()
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(
            started.elapsed().as_secs() < 30,
            "changed view failed readiness"
        );
    }
    assert!(app
        .world()
        .resource::<crate::io::image_copy::CapturePipelineReadiness>()
        .ready());
    let mut targets: Vec<_> = app
        .world_mut()
        .query_filtered::<(
            &CaptureCameraIndex,
            &RenderTarget,
            Option<&LensCamera>,
            Option<&SsaaCamera>,
        ), With<ZeroverseCamera>>()
        .iter(app.world())
        .map(|(index, target, lens, ssaa)| {
            let RenderTarget::Image(target) = target else {
                panic!("offscreen view")
            };
            (
                index.0,
                ssaa.map(|s| s.output.clone())
                    .or_else(|| lens.map(|l| l.output.clone()))
                    .unwrap_or(target.handle.clone()),
            )
        })
        .collect();
    targets.sort_by_key(|(index, _)| *index);
    targets
        .into_iter()
        .map(|(_, target)| {
            let descriptor = app
                .world()
                .resource::<Assets<Image>>()
                .get(&target)
                .unwrap()
                .texture_descriptor
                .clone();
            assert_eq!(
                descriptor.format,
                bevy::render::render_resource::TextureFormat::Rgba32Float
            );
            let copier =
                ImageCopier::for_targets(vec![target], descriptor.size, descriptor.format, &device);
            let entity = app.world_mut().spawn(copier.clone()).id();
            copier.request(1);
            let start = std::time::Instant::now();
            let bytes = loop {
                app.update();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                assert!(copier.failure().is_none(), "{:?}", copier.failure());
                if let Some(frame) = copier.take(1) {
                    break frame.planes.into_iter().next().unwrap();
                }
                assert!(start.elapsed().as_secs() < 20, "readback timed out");
            };
            app.world_mut().despawn(entity);
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .inspect(|v| assert!(v.is_finite()))
                .collect()
        })
        .collect()
}
