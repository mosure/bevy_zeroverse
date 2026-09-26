#![recursion_limit = "256"]
//! Matched native RGB ablation of actual baked indirect illumination.
use bevy::{light::IrradianceVolume, prelude::*};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    headless::{create_app, setup_globals},
    io::channels,
    render::{color::linear_to_srgb, RenderMode},
    sample::{Sample, SamplerState},
    scene::{procedural_indoor::gi::BakeStatistics, RegenerateSceneEvent, ZeroverseSceneType},
};
use std::time::{Duration, Instant};

fn capture(app: &mut App) -> Sample {
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![RenderMode::Color],
        frames: 12,
        warmup_frames: 24,
        timesteps: Vec::new(),
        ..default()
    });
    let started = Instant::now();
    loop {
        app.update();
        assert!(
            app.should_exit().is_none(),
            "renderer exited during GI capture"
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
            started.elapsed() < Duration::from_secs(180),
            "GI capture stalled"
        );
    }
}

#[test]
#[ignore = "requires native GPU; exports fixed-scene indirect-light ablations"]
fn baked_indirect_light_changes_rgb_without_changing_geometry_or_direct_lights() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(6),
        indoor_human_density: 0.0,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        initialize_scene: false,
        num_cameras: 2,
        width: 640.0,
        height: 480.0,
        playback_steps: 1,
        playback_mode: bevy_zeroverse::camera::PlaybackMode::Still,
        playback_speed: 0.0,
        render_modes: vec![RenderMode::Color],
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    if let Ok(rays) = std::env::var("INDOOR_GI_TEST_RAYS") {
        app.world_mut()
            .resource_mut::<bevy_zeroverse::scene::procedural_indoor::gi::IndoorGiSettings>()
            .bake
            .rays_per_probe = rays.parse().unwrap();
    }
    install_readback(&mut app);
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    app.world_mut().write_message(RegenerateSceneEvent);
    let baseline = capture(&mut app);
    let statistics = app.world().resource::<BakeStatistics>().clone();
    validate_gpu_volume(&mut app);
    // Interactive rooms become visible before their CPU irradiance task finishes.
    // Capturing that same path must still wait for the complete volume.
    app.world_mut()
        .resource_mut::<bevy_zeroverse::scene::procedural_indoor::gi::IndoorGiSettings>()
        .gpu = false;
    bevy_zeroverse::scene::procedural_indoor::reset_indoor_sequence(app.world_mut(), 6);
    app.world_mut().write_message(RegenerateSceneEvent);
    let interactive = capture(&mut app);
    let interactive_statistics = app.world().resource::<BakeStatistics>().clone();
    assert_eq!(app.world().resource::<BakeStatistics>().backend, "cpu_bvh");
    assert!(!bevy_zeroverse::scene::procedural_indoor::indoor_generation_pending(app.world()));
    let mut count = 0;
    for mut volume in app
        .world_mut()
        .query::<&mut IrradianceVolume>()
        .iter_mut(app.world_mut())
    {
        volume.intensity = 0.0;
        count += 1;
    }
    assert_eq!(count, 1);
    let no_gi = capture(&mut app);
    let output = output_directory();
    std::fs::create_dir_all(&output).unwrap();
    let mut metrics = Vec::new();
    for (index, view) in interactive.views.iter().enumerate() {
        assert_eq!(
            view.world_from_view, no_gi.views[index].world_from_view,
            "GI ablation cameras must remain fixed"
        );
        let a: &[f32] = bytemuck::cast_slice(&view.color);
        let b: &[f32] = bytemuck::cast_slice(&no_gi.views[index].color);
        let luma = |p: &[f32]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        let mut delta = 0.0;
        let mut changed = 0;
        for (x, y) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()) {
            let difference = (luma(x) - luma(y)).abs();
            delta += difference;
            changed += usize::from(difference > 0.005);
        }
        let pixels = (a.len() / 4) as f32;
        eprintln!(
            "GI view {index}: mean delta {}, changed {}",
            delta / pixels,
            changed as f32 / pixels
        );
        for (label, data) in [("gi", a), ("no_gi", b)] {
            let bytes: Vec<u8> = data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| {
                    pixel[..3]
                        .iter()
                        .map(|v| (linear_to_srgb(*v) * 255.0).round().clamp(0.0, 255.0) as u8)
                })
                .collect();
            image::save_buffer(
                output.join(format!("camera_{index}_{label}.png")),
                &bytes,
                640,
                480,
                image::ColorType::Rgb8,
            )
            .unwrap();
        }
        assert!(
            delta / pixels > 0.0005,
            "GI must contribute a measurable amount of light"
        );
        metrics.push(
            serde_json::json!({"camera":index,"mean_absolute_linear_luma_delta":delta/pixels,
            "fraction_changed_above_0_005":changed as f32/pixels}),
        );
    }
    std::fs::write(output.join("render_ablation.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "seed":6,"width":640,"height":480,"statistics":interactive_statistics,"gpu_statistics":statistics,"views":metrics,
        "gpu_views":baseline.views.len(), "interactive_capture_waited_for_cpu_gi":true,
        "control":"Only IrradianceVolume.intensity changed from 1 to 0; geometry, lights, shadows, exposure unchanged."})).unwrap()).unwrap();
}

fn output_directory() -> std::path::PathBuf {
    std::env::var_os("INDOOR_GI_TEST_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "out/indoor_gi".into())
}

type VolumeReadback = (bevy::render::render_resource::Extent3d, u32, Vec<u8>);
#[derive(Resource, Clone, Default)]
struct GiReadback(std::sync::Arc<std::sync::Mutex<Option<VolumeReadback>>>);

fn install_readback(app: &mut App) {
    use bevy::render::{Render, RenderApp, RenderSystems};
    let readback = GiReadback::default();
    app.insert_resource(readback.clone());
    app.sub_app_mut(RenderApp)
        .insert_resource(readback)
        .add_systems(Render, readback_volume.in_set(RenderSystems::Cleanup));
}

fn readback_volume(
    request: Option<Res<bevy_zeroverse::scene::procedural_indoor::gi::gpu::GpuBakeRequest>>,
    images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    device: Res<bevy::render::renderer::RenderDevice>,
    queue: Res<bevy::render::renderer::RenderQueue>,
    output: Res<GiReadback>,
) {
    use bevy::render::render_resource::*;
    let Some(request) = request else {
        return;
    };
    if !request.readiness.ready() || output.0.lock().unwrap().is_some() {
        return;
    }
    let gpu_image = images.get(&request.image).unwrap();
    let extent = gpu_image.texture.size();
    let row_pitch = (extent.width * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("GI validation readback"),
        size: (row_pitch * extent.height * extent.depth_or_array_layers) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("GI validation copy"),
    });
    encoder.copy_texture_to_buffer(
        gpu_image.texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_pitch),
                rows_per_image: Some(extent.height),
            },
        },
        extent,
    );
    queue.submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(MapMode::Read, move |r| send.send(r).unwrap());
    // Explicit validation-only readback; production GI never maps/polls a buffer.
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receive
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    let bytes = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    *output.0.lock().unwrap() = Some((extent, row_pitch, bytes));
}

fn validate_gpu_volume(app: &mut App) {
    use bevy_zeroverse::scene::procedural_indoor::{
        gi::{gpu::GpuBakeRequest, BakeScene},
        layout::IndoorManifest,
        materials::IndoorMaterials,
    };
    let request = app
        .world()
        .get_resource::<GpuBakeRequest>()
        .expect("native Auto must dispatch GPU GI")
        .clone();
    assert!(request.readiness.ready());
    assert!(request.readiness.failure().is_none());
    let (extent, row_pitch, bytes) = app
        .world()
        .resource::<GiReadback>()
        .0
        .lock()
        .unwrap()
        .clone()
        .expect("GI validation readback completed");
    let n = UVec3::new(
        extent.width,
        extent.height / 2,
        extent.depth_or_array_layers / 3,
    );
    let read = |x: u32, y: u32, z: u32| {
        let offset = ((z * extent.height + y) * row_pitch + x * 8) as usize;
        Vec3::from_array(std::array::from_fn(|i| {
            half::f16::from_bits(u16::from_le_bytes([
                bytes[offset + i * 2],
                bytes[offset + i * 2 + 1],
            ]))
            .to_f32()
        }))
    };
    let mut total = Vec3::ZERO;
    let mut nonzero = 0;
    for z in 0..extent.depth_or_array_layers {
        for y in 0..extent.height {
            for x in 0..extent.width {
                let v = read(x, y, z);
                assert!(
                    v.is_finite() && v.min_element() >= 0.0,
                    "invalid GPU radiance"
                );
                total += v;
                nonzero += usize::from(v.max_element() > 0.001);
            }
        }
    }
    assert!(nonzero > (extent.width * extent.height * extent.depth_or_array_layers) as usize / 2);
    let scene = app.world().resource::<IndoorManifest>();
    let mut images = Assets::default();
    let mut materials = Assets::default();
    let set = IndoorMaterials::build(scene, &mut images, &mut materials);
    let oracle = BakeScene::from_manifest(scene, &set, &materials, &images);
    let mut controls = Vec::new();
    let mut absolute = 0.0;
    let mut reference_total = 0.0;
    for z in [n.z / 4, n.z / 2, n.z * 3 / 4] {
        for x in [n.x / 4, n.x / 2, n.x * 3 / 4] {
            let y = n.y / 2;
            let point = oracle.bounds_min
                + (Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) / n.as_vec3()
                    * (oracle.bounds_max - oracle.bounds_min);
            let reference = oracle.integrate(point, 4096, 3, 92741);
            let gpu: [Vec3; 6] = std::array::from_fn(|face| {
                read(x, y + (face as u32 % 2) * n.y, z + (face as u32 / 2) * n.z)
            });
            for (a, b) in gpu.iter().zip(reference) {
                absolute += (*a - b).abs().element_sum();
                reference_total += b.element_sum();
            }
            controls.push(serde_json::json!({"position":point.to_array(),"gpu":gpu.map(|v|v.to_array()),"cpu_reference":reference.map(|v|v.to_array())}));
        }
    }
    let relative = absolute / reference_total.max(0.001);
    let settings = app
        .world()
        .resource::<bevy_zeroverse::scene::procedural_indoor::gi::IndoorGiSettings>();
    let output = output_directory();
    std::fs::create_dir_all(&output).unwrap();
    std::fs::write(output.join("gpu_reference.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "seed":scene.seed,"gpu_rays":settings.bake.rays_per_probe,"cpu_rays":4096,"diffuse_bounces":settings.bake.diffuse_bounces,
        "relative_mean_absolute_error":relative,"mean_gpu_radiance":(total/(extent.width*extent.height*extent.depth_or_array_layers) as f32).to_array(),
        "nonzero_lobes":nonzero,"points":controls})).unwrap()).unwrap();
    eprintln!("GPU GI vs CPU reference relative MAE: {relative}");
    assert!(
        relative < 0.35,
        "GPU transport must agree with CPU high-sample reference within stochastic budget"
    );
}
