use std::{
    error::Error,
    fmt::Write as FmtWrite,
    fs,
    path::{Path, PathBuf},
    sync::Once,
    thread,
    time::{Duration, Instant},
};

use bevy::prelude::App;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    render::{depth::DepthMaterial, RenderMode},
    scene::{RegenerateSceneEvent, ZeroverseSceneType},
};
use bevy_zeroverse_ffi::{create_app, setup_globals, Sample, SamplerState, SAMPLE_RECEIVER};
use bytemuck::cast_slice;
use image::{ImageBuffer, Rgba};

static INIT_GLOBALS: Once = Once::new();

fn init_globals() {
    INIT_GLOBALS.call_once(|| {
        let asset_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("ffi crate has a parent directory")
            .to_string_lossy()
            .into_owned();
        setup_globals(Some(asset_root));

        if let Some(receiver) = SAMPLE_RECEIVER.get() {
            if let Ok(receiver) = receiver.lock() {
                while receiver.try_recv().is_ok() {}
            }
        }
    });
}

#[derive(Debug)]
struct BufferStats {
    min: f32,
    max: f32,
    nan_count: usize,
    finite_count: usize,
}

impl BufferStats {
    fn from_values(values: &[f32]) -> Self {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut nan_count = 0;
        let mut finite_count = 0;

        for &v in values {
            if v.is_nan() {
                nan_count += 1;
            } else {
                min = min.min(v);
                max = max.max(v);
                finite_count += 1;
            }
        }

        Self {
            min: if finite_count == 0 { 0.0 } else { min },
            max: if finite_count == 0 { 0.0 } else { max },
            nan_count,
            finite_count,
        }
    }

    fn assert_usable(&self, mode: &RenderMode) {
        assert!(
            self.finite_count > 0,
            "{mode:?} buffer is empty or entirely non-finite"
        );
        assert_eq!(self.nan_count, 0, "{mode:?} buffer contained NaN values");
        if self.max <= self.min {
            println!(
                "    [warn] {mode:?} buffer appears constant (min={:.4}, max={:.4})",
                self.min, self.max
            );
        }
    }
}

#[derive(Clone)]
struct ScenarioConfig {
    scene: ZeroverseSceneType,
    render_modes: Vec<RenderMode>,
    playback_mode: PlaybackMode,
    label: &'static str,
}

fn base_config(
    scene: ZeroverseSceneType,
    modes: &[RenderMode],
    playback_mode: PlaybackMode,
) -> BevyZeroverseConfig {
    let mut cfg = BevyZeroverseConfig::default();
    cfg.editor = false;
    cfg.gizmos = false;
    cfg.headless = true;
    cfg.image_copiers = true;
    cfg.press_esc_close = false;
    cfg.keybinds = false;
    cfg.width = 320.0;
    cfg.height = 240.0;
    cfg.num_cameras = 2;
    cfg.camera_grid = false;
    cfg.render_mode = modes[0].clone();
    cfg.render_modes = modes.to_vec();
    cfg.scene_type = scene;
    cfg.playback_mode = playback_mode;
    cfg.playback_speed = 0.6;
    cfg.playback_step = 0.1;
    cfg.playback_steps = 1;
    cfg.depth_format = bevy_zeroverse::render::depth::DepthFormat::Colorized;
    cfg.z_depth = true;
    cfg
}

fn scene_name(scene: &ZeroverseSceneType) -> &'static str {
    match scene {
        ZeroverseSceneType::CornellCube => "cornell_cube",
        ZeroverseSceneType::Custom => "custom",
        ZeroverseSceneType::Human => "human",
        ZeroverseSceneType::Object => "object",
        ZeroverseSceneType::SemanticRoom => "semantic_room",
        ZeroverseSceneType::Room => "room",
    }
}

fn mode_name(mode: &RenderMode) -> &'static str {
    match mode {
        RenderMode::Color => "color",
        RenderMode::Depth => "depth",
        RenderMode::MotionVectors => "motion_vectors",
        RenderMode::Normal => "normal",
        RenderMode::OpticalFlow => "optical_flow",
        RenderMode::Position => "position",
        RenderMode::Semantic => "semantic",
    }
}

fn sampler_state_for_config(cfg: &BevyZeroverseConfig) -> SamplerState {
    let mut render_modes = cfg.render_modes.clone();
    if render_modes.is_empty() {
        render_modes.push(cfg.render_mode.clone());
    }

    let timesteps = (1..cfg.playback_steps)
        .map(|i| {
            let x = i as f32 * cfg.playback_step;
            assert!(
                (0.0..=1.0).contains(&x),
                "timestep value {x} has range [0.0, 1.0]"
            );
            x
        })
        .collect();

    let mut state = SamplerState::default();
    state.render_modes = render_modes;
    state.timesteps = timesteps;
    state.frames = 4;
    state.warmup_frames = 4;
    state
}

fn apply_config(app: &mut App, cfg: &BevyZeroverseConfig) {
    {
        let mut args = app.world_mut().resource_mut::<BevyZeroverseConfig>();
        *args = cfg.clone();
    }

    {
        let mut render_mode_res = app.world_mut().resource_mut::<RenderMode>();
        *render_mode_res = cfg
            .render_modes
            .first()
            .cloned()
            .unwrap_or(cfg.render_mode.clone());
    }

    app.world_mut().send_event(RegenerateSceneEvent);
}

fn collect_sample(app: &mut App, cfg: &BevyZeroverseConfig) -> Result<Sample, Box<dyn Error>> {
    if let Some(receiver) = SAMPLE_RECEIVER.get() {
        if let Ok(receiver) = receiver.lock() {
            while receiver.try_recv().is_ok() {}
        }
    }

    let mut sampler_state = sampler_state_for_config(cfg);
    if sampler_state.render_modes.is_empty() {
        sampler_state.render_modes.push(cfg.render_mode.clone());
    }
    let active_modes = sampler_state.render_modes.clone();

    app.insert_resource(sampler_state);
    {
        let mut render_mode_res = app.world_mut().resource_mut::<RenderMode>();
        *render_mode_res = active_modes[0].clone();
    }

    let expected_views = cfg.num_cameras.max(1) * cfg.playback_steps as usize;
    let receiver = SAMPLE_RECEIVER
        .get()
        .expect("SAMPLE_RECEIVER was not initialized")
        .clone();

    let start = Instant::now();
    loop {
        app.update();

        if let Ok(sample) = receiver.lock().unwrap().try_recv() {
            let (depth_material_count, depth_tag_count) = {
                let world = app.world_mut();
                let mut material_query = world.query::<&bevy::pbr::MeshMaterial3d<DepthMaterial>>();
                let mut tag_query = world.query::<&bevy_zeroverse::render::depth::Depth>();

                let mats = material_query.iter(&world).count();
                let tags = tag_query.iter(&world).count();
                (mats, tags)
            };
            let normal_tag_count = {
                let world = app.world_mut();
                let mut normal_query = world.query::<&bevy_zeroverse::render::normal::Normal>();
                normal_query.iter(&world).count()
            };
            let position_tag_count = {
                let world = app.world_mut();
                let mut position_query =
                    world.query::<&bevy_zeroverse::render::position::Position>();
                position_query.iter(&world).count()
            };
            let optical_flow_tag_count = {
                let world = app.world_mut();
                let mut flow_query =
                    world.query::<&bevy_zeroverse::render::optical_flow::OpticalFlow>();
                flow_query.iter(&world).count()
            };

            println!(
                "Captured sample with {} views (depth tags/materials: {}/{}, normal tags: {}, position tags: {}, optical flow tags: {})",
                sample.views.len(),
                depth_tag_count,
                depth_material_count,
                normal_tag_count,
                position_tag_count,
                optical_flow_tag_count
            );
            assert_eq!(
                sample.views.len(),
                expected_views,
                "unexpected number of views captured"
            );
            assert_eq!(
                sample.view_dim as usize,
                cfg.num_cameras.max(1),
                "view_dim should match the number of cameras"
            );
            return Ok(sample);
        }

        if start.elapsed() > Duration::from_secs(90) {
            panic!("timed out waiting for headless sample");
        }

        thread::sleep(Duration::from_millis(10));
    }
}

fn clamp_to_byte(v: f32) -> u8 {
    let v = if v.is_finite() { v } else { 0.0 };
    (v.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8
}

fn save_color_image(floats: &[f32], width: u32, height: u32, output: &Path) -> BufferStats {
    let mut values = Vec::with_capacity(floats.len());
    for chunk in floats.chunks_exact(4) {
        values.extend_from_slice(&chunk[0..3]);
    }

    let stats = BufferStats::from_values(&values);
    let bytes: Vec<u8> = values
        .chunks_exact(3)
        .flat_map(|px| {
            [
                clamp_to_byte(px[0]),
                clamp_to_byte(px[1]),
                clamp_to_byte(px[2]),
                255,
            ]
        })
        .collect();

    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bytes).expect("invalid color buffer");
    image.save(output).unwrap();

    stats
}

fn save_depth_like_image(floats: &[f32], width: u32, height: u32, output: &Path) -> BufferStats {
    let mut depth_values = Vec::with_capacity(floats.len() / 4);
    for chunk in floats.chunks_exact(4) {
        depth_values.push(chunk[0]);
    }

    let stats = BufferStats::from_values(&depth_values);
    let span = (stats.max - stats.min).max(1e-6);

    let bytes: Vec<u8> = depth_values
        .into_iter()
        .flat_map(|v| {
            let normalized = if v.is_finite() {
                (v - stats.min) / span
            } else {
                0.0
            };
            let byte = clamp_to_byte(normalized);
            [byte, byte, byte, 255]
        })
        .collect();

    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bytes).expect("invalid depth buffer");
    image.save(output).unwrap();

    stats
}

fn save_normal_image(floats: &[f32], width: u32, height: u32, output: &Path) -> BufferStats {
    let mut values = Vec::with_capacity(floats.len());
    for chunk in floats.chunks_exact(4) {
        values.extend_from_slice(&chunk[0..3]);
    }

    let stats = BufferStats::from_values(&values);
    let bytes: Vec<u8> = values
        .chunks_exact(3)
        .flat_map(|px| {
            let map = |v: f32| clamp_to_byte(v * 0.5 + 0.5);
            [map(px[0]), map(px[1]), map(px[2]), 255]
        })
        .collect();

    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bytes).expect("invalid normal buffer");
    image.save(output).unwrap();

    stats
}

fn save_position_like_image(floats: &[f32], width: u32, height: u32, output: &Path) -> BufferStats {
    let mut values = Vec::with_capacity(floats.len());
    for chunk in floats.chunks_exact(4) {
        values.extend_from_slice(&chunk[0..3]);
    }

    let stats = BufferStats::from_values(&values);
    let span = (stats.max - stats.min).max(1e-6);

    let bytes: Vec<u8> = values
        .chunks_exact(3)
        .flat_map(|px| {
            let map = |v: f32| clamp_to_byte(((v - stats.min) / span).clamp(0.0, 1.0));
            [map(px[0]), map(px[1]), map(px[2]), 255]
        })
        .collect();

    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, bytes)
        .expect("invalid position buffer");
    image.save(output).unwrap();

    stats
}

fn select_buffer<'a>(view: &'a bevy_zeroverse_ffi::View, mode: &RenderMode) -> &'a [u8] {
    match mode {
        RenderMode::Color => &view.color,
        RenderMode::Depth => &view.depth,
        RenderMode::Normal => &view.normal,
        RenderMode::OpticalFlow => &view.optical_flow,
        RenderMode::Position => &view.position,
        unsupported => panic!("{unsupported:?} is not captured by the sampler tests"),
    }
}

fn write_sample_outputs(
    sample: &Sample,
    modes: &[RenderMode],
    cfg: &BevyZeroverseConfig,
    output_dir: &Path,
) -> Result<Vec<String>, Box<dyn Error>> {
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;

    let width = cfg.width as u32;
    let height = cfg.height as u32;
    let cameras = cfg.num_cameras.max(1);

    let mut summaries = Vec::new();

    for (view_idx, view) in sample.views.iter().enumerate() {
        let camera_idx = view_idx % cameras;
        let step_idx = view_idx / cameras;

        for mode in modes {
            let buffer = select_buffer(view, mode);
            let floats: &[f32] = cast_slice(buffer);
            let expected_len = (width * height * 4) as usize;
            assert_eq!(
                floats.len(),
                expected_len,
                "{:?} buffer len {} did not match expected {}",
                mode,
                floats.len(),
                expected_len
            );

            let file_name = format!("{}_cam{}_step{}.png", mode_name(mode), camera_idx, step_idx);
            let path = output_dir.join(file_name);

            let stats = match mode {
                RenderMode::Color => save_color_image(floats, width, height, &path),
                RenderMode::Depth => save_depth_like_image(floats, width, height, &path),
                RenderMode::Normal => save_normal_image(floats, width, height, &path),
                RenderMode::OpticalFlow => save_position_like_image(floats, width, height, &path),
                RenderMode::Position => save_position_like_image(floats, width, height, &path),
                other => panic!("{other:?} not supported in headless tests"),
            };

            let summary = format!(
                "{:>12} | cam {:>2} step {:>2} | near {:>6.3} far {:>6.3} t {:>5.2} | min {:>8.4} max {:>8.4}",
                mode_name(mode),
                camera_idx,
                step_idx,
                view.near,
                view.far,
                view.time,
                stats.min,
                stats.max
            );
            println!("    {summary}");
            summaries.push(summary);

            stats.assert_usable(mode);
        }
    }

    let mut summary_txt = String::new();
    for line in &summaries {
        writeln!(&mut summary_txt, "{line}")?;
    }
    fs::write(output_dir.join("stats.txt"), summary_txt)?;

    Ok(summaries)
}

#[test]
fn headless_sampling_exports_pngs() -> Result<(), Box<dyn Error>> {
    init_globals();

    let output_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffi crate has a parent")
        .join("data")
        .join("integration_test");

    fs::create_dir_all(&output_root)?;

    let scenarios = vec![
        ScenarioConfig {
            scene: ZeroverseSceneType::Object,
            render_modes: vec![RenderMode::Color, RenderMode::Depth],
            playback_mode: PlaybackMode::Still,
            label: "color_depth",
        },
        ScenarioConfig {
            scene: ZeroverseSceneType::SemanticRoom,
            render_modes: vec![RenderMode::Color, RenderMode::Normal, RenderMode::Position],
            playback_mode: PlaybackMode::Still,
            label: "color_normal_position",
        },
        ScenarioConfig {
            scene: ZeroverseSceneType::CornellCube,
            render_modes: vec![RenderMode::Color, RenderMode::OpticalFlow],
            playback_mode: PlaybackMode::PingPong,
            label: "color_optical_flow",
        },
    ];

    let first_cfg = base_config(
        scenarios[0].scene.clone(),
        &scenarios[0].render_modes,
        scenarios[0].playback_mode,
    );
    let mut app = create_app(None, Some(first_cfg.clone()), false);
    app.finish();
    app.cleanup();

    for scenario in scenarios {
        let cfg = base_config(
            scenario.scene.clone(),
            &scenario.render_modes,
            scenario.playback_mode,
        );

        apply_config(&mut app, &cfg);

        // allow scene regeneration + asset loading to settle
        for _ in 0..6 {
            app.update();
        }

        let sample = collect_sample(&mut app, &cfg)?;

        let scenario_dir = output_root
            .join(scene_name(&scenario.scene))
            .join(scenario.label);

        let summaries = write_sample_outputs(&sample, &scenario.render_modes, &cfg, &scenario_dir)?;

        println!(
            "Captured {} views for {} ({}) -> {}",
            sample.views.len(),
            scene_name(&scenario.scene),
            scenario.label,
            scenario_dir.display()
        );
        for line in summaries {
            println!("    {line}");
        }
    }

    Ok(())
}
