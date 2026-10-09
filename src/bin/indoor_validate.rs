//! Reproducible CPU audit and real-render qualification/export tool.
use anyhow::{ensure, Context as ContextExt, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    headless::{create_app, setup_globals},
    io::channels,
    render::{depth::DepthFormat, RenderMode},
    sample::{Sample, SamplerState},
    scene::{
        procedural_indoor::{
            cameras::CameraSettings,
            layout::{IndoorLayout, IndoorManifest},
            metrics::{export_metrics_with_factors, select_strata},
            validation::{audit_layout_with_factors, validate_geometry},
            GlassFilter, IndoorQuality,
        },
        RegenerateSceneEvent, ZeroverseSceneType,
    },
};
use clap::Parser;
use serde::Serialize;
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

#[path = "indoor_validate/co_visibility.rs"]
mod co_visibility;

#[derive(Parser)]
#[command(
    about = "Audit seeded indoor layouts; optionally capture real RGB and aligned annotations"
)]
struct Args {
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1024)]
    audit_seeds: usize,
    /// Check actual mesh topology/normals for every audited seed (default: first 16).
    #[arg(long)]
    audit_geometry: bool,
    #[arg(long, default_value_t = 0)]
    renders: usize,
    #[arg(long, default_value_t = 4)]
    cameras: usize,
    /// Camera policy JSON including optional multiview shared-surface constraints.
    #[arg(long)]
    indoor_camera: Option<String>,
    /// Material/light factors applied after geometry generation.
    #[arg(long)]
    indoor_appearance: Option<String>,
    #[arg(long, default_value_t = 800)]
    width: u32,
    #[arg(long, default_value_t = 600)]
    height: u32,
    #[arg(long, value_enum, default_value_t = IndoorLayout::Mixed)]
    layout: IndoorLayout,
    #[arg(long, default_value_t = 0.65)]
    density: f32,
    #[arg(long, value_enum, default_value_t = IndoorQuality::Auto)]
    quality: IndoorQuality,
    #[arg(long, default_value_t = 0.25)]
    human_density: f32,
    /// Keep native lighting/shadows but disable baked indirect lighting for ablations.
    #[arg(long)]
    no_gi: bool,
    /// Controlled ablation of screen-space ambient occlusion.
    #[arg(long)]
    no_ssao: bool,
    /// Controlled ablation of shadow maps; retains the same direct-light intensities.
    #[arg(long)]
    no_shadows: bool,
    /// Export exposed scene-linear HDR RGB for transport comparisons, with display effects disabled.
    #[arg(long)]
    linear_rgb: bool,
    /// Offline glass sampling ablation; references integrate 2048 samples per pixel.
    #[arg(long, value_enum, default_value_t = GlassFilter::Default)]
    glass_filter: GlassFilter,
    /// Shared geometric annotation surface policy, independent of RGB glass.
    #[arg(long, value_enum, default_value = "surface")]
    annotation_glass: bevy_zeroverse::render::glass::AnnotationGlass,
    #[arg(long, default_value_t = bevy_zeroverse::scene::procedural_indoor::gi::BakeSettings::default().rays_per_probe)]
    gi_rays: u32,
    /// Diffuse transport depth for controlled reference comparisons.
    #[arg(long, default_value_t = bevy_zeroverse::scene::procedural_indoor::gi::BakeSettings::default().diffuse_bounces, value_parser = clap::value_parser!(u32).range(1..=12))]
    gi_bounces: u32,
    #[arg(long)]
    labels: bool,
    /// Capture exact same-time camera membership, validity masks and additive previews.
    #[arg(long, requires = "labels")]
    co_visibility: bool,
    /// Export PNG previews and metrics without large RGBA32F files.
    #[arg(long)]
    no_raw: bool,
    /// Export live geometry, PBR maps and sampled cameras for independent Cycles comparisons.
    #[arg(long)]
    export_reference: bool,
    /// Sample layout, lighting, floor, furniture and architecture strata from the audit.
    #[arg(long)]
    stratified: bool,
    /// Capture uniform trajectory samples including both endpoints (one means start only).
    #[arg(long, default_value_t = 1)]
    playback_steps: u32,
    /// Asset root; an empty directory is supported with --human-density 0.
    #[arg(long)]
    asset_root: Option<PathBuf>,
    #[arg(long)]
    rotation_augmentation: bool,
    #[arg(long, default_value = "out/procedural_indoor")]
    output: PathBuf,
}

#[derive(Serialize)]
struct CaptureReport {
    build_provenance: serde_json::Value,
    camera_qualification: Option<serde_json::Value>,
    run_id: String,
    seed: u64,
    elapsed_seconds: f64,
    image_size: [u32; 2],
    capabilities: serde_json::Value,
    renderer: Option<String>,
    aabb: [[f32; 3]; 2],
    color_encoding: &'static str,
    label_policy: &'static str,
    views: Vec<ViewReport>,
    mesh_assets: usize,
    material_assets: usize,
    image_assets: usize,
    annotation_precision: bevy_zeroverse::sample::AnnotationPrecision,
    diffuse_gi: Option<bevy_zeroverse::scene::procedural_indoor::gi::BakeStatistics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    co_visibility_metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ViewReport {
    calibration: Option<bevy_zeroverse::calibration::CameraCalibration>,
    time_seconds: Option<f32>,
    trajectory_progress: Option<f32>,
    camera_index: usize,
    step_index: usize,
    time: f32,
    pose_max_absolute_error: f32,
    fx_pixels: f32,
    fy_pixels: f32,
    world_from_view: [[f32; 4]; 4],
    fov_y: f32,
    near: f32,
    far: f32,
    mean_luminance: f32,
    luminance_std: f32,
    dark_fraction: f32,
    clipped_fraction: f32,
    semantic_colors: usize,
    semantic_pixel_counts: std::collections::BTreeMap<String, usize>,
    annotation_alignment:
        Option<bevy_zeroverse::scene::procedural_indoor::validation::AnnotationAlignment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    co_visibility: Option<co_visibility::VisibilityReport>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    ensure!(args.audit_seeds > 0, "audit-seeds must be positive");
    ensure!(
        args.cameras > 0 && args.cameras <= 256,
        "cameras must be between 1 and 256"
    );
    if args.co_visibility {
        bevy_zeroverse::render::co_visibility::validate_config(
            &[RenderMode::CoVisibility],
            args.cameras,
        )
        .map_err(anyhow::Error::msg)?;
    }
    ensure!(
        args.width >= 64 && args.height >= 64 && args.width <= 4096 && args.height <= 4096,
        "image dimensions must be in [64, 4096]"
    );
    ensure!(
        args.density.is_finite() && (0.0..=1.0).contains(&args.density),
        "density must be in [0, 1]"
    );
    ensure!(
        (1..=64).contains(&args.playback_steps),
        "playback-steps must be in [1, 64]"
    );
    ensure!(
        (64..=16384).contains(&args.gi_rays),
        "GI rays must be in [64, 16384]"
    );
    setup_globals(Some(
        args.asset_root
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").to_owned()),
    ));
    fs::create_dir_all(&args.output)?;
    let identity = bevy_zeroverse_capture::GeneratorIdentity {
        schema_version: bevy_zeroverse_capture::CAPTURE_SCHEMA_VERSION,
        crate_version: bevy_zeroverse::provenance::capture_provenance()["crate_version"]
            .as_str()
            .expect("compiled generator version")
            .into(),
        generator_version: bevy_zeroverse_capture::GENERATOR_VERSION,
        source_sha256: bevy_zeroverse::provenance::capture_provenance()["source_sha256"]
            .as_str()
            .expect("compiled generator provenance")
            .into(),
    };
    fs::write(
        args.output.join("generator_identity.json"),
        serde_json::to_vec_pretty(&identity)?,
    )?;
    let camera_settings = args
        .indoor_camera
        .as_deref()
        .map(CameraSettings::parse)
        .transpose()
        .map_err(anyhow::Error::msg)?
        .unwrap_or_default();
    let appearance = args
        .indoor_appearance
        .as_deref()
        .map(bevy_zeroverse::scene::procedural_indoor::appearance::AppearanceSettings::parse)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let audit_start = Instant::now();
    let report = audit_layout_with_factors(
        args.seed,
        args.audit_seeds,
        args.cameras,
        args.density,
        args.layout,
        args.human_density,
        &camera_settings,
        args.width as f32 / args.height as f32,
        appearance.as_ref(),
    );
    fs::write(
        args.output.join("distribution.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "audited {} seeds in {:.2}s; layouts={:?}; invalid={}",
        args.audit_seeds,
        audit_start.elapsed().as_secs_f64(),
        report.layout_counts,
        report.invalid_seeds.len()
    );
    ensure!(
        report.invalid_seeds.is_empty(),
        "layout validation failed; see distribution.json"
    );
    let metrics = export_metrics_with_factors(
        args.seed,
        args.audit_seeds,
        args.cameras,
        args.density,
        args.layout,
        args.width,
        args.height,
        &args.output,
        args.human_density,
        &camera_settings,
        appearance.as_ref(),
    )
    .map_err(anyhow::Error::msg)?;
    let selected: Vec<u64> = if args.stratified {
        ensure!(
            args.renders == 0 || args.renders <= metrics.stratified_seeds.len(),
            "requested more renders than observed strata"
        );
        select_strata(
            &metrics.stratified_seeds,
            if args.renders == 0 {
                usize::MAX
            } else {
                args.renders
            },
        )
    } else {
        (0..args.renders)
            .map(|i| args.seed.wrapping_add(i as u64))
            .collect()
    };
    fs::write(
        args.output.join("render_selection.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "run_id": run_id, "identity": identity, "quality": args.quality, "capture_engine": bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,
            "policy": if args.stratified { "first observed seed per layout/lighting/floor/furniture/architecture cell; category-balanced order, no image quality filtering" } else { "consecutive seeds" },
            "observed_strata": metrics.stratified_seeds, "selected_seeds": selected,
            "playback_steps": args.playback_steps, "density": args.density, "indoor_camera": camera_settings, "indoor_appearance": appearance,
            "human_density": args.human_density, "diffuse_gi_enabled": !args.no_gi && args.quality.diffuse_gi(), "gi_rays": args.gi_rays, "gi_bounces": args.gi_bounces,
            "co_visibility": args.co_visibility,
        }))?,
    )?;
    // Check actual mesh construction independently of the cheaper distribution pass.
    let geometry_seeds = if args.audit_geometry {
        args.audit_seeds
    } else {
        args.audit_seeds.min(16)
    };
    for i in 0..geometry_seeds {
        let mut manifest = IndoorManifest::generate_with_humans(
            args.seed.wrapping_add(i as u64),
            args.layout,
            args.density,
            0,
            args.human_density,
        )
        .map_err(anyhow::Error::msg)?;
        manifest
            .resample_cameras(
                args.cameras,
                camera_settings.clone(),
                args.width as f32 / args.height as f32,
            )
            .map_err(anyhow::Error::msg)?;
        if let Some(settings) = &appearance {
            manifest
                .apply_appearance(settings.clone())
                .map_err(anyhow::Error::msg)?;
        }
        let stats = validate_geometry(&manifest).map_err(anyhow::Error::msg)?;
        fs::write(
            args.output.join(format!("geometry_{}.json", manifest.seed)),
            serde_json::to_vec_pretty(&stats)?,
        )?;
    }
    if selected.is_empty() {
        return Ok(());
    }
    let mut modes = if args.labels {
        vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Semantic,
            RenderMode::Position,
        ]
    } else {
        vec![RenderMode::Color]
    };
    if args.co_visibility {
        modes.push(RenderMode::CoVisibility);
    }
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(args.seed),
        indoor_camera: args.indoor_camera.clone(),
        indoor_appearance: args.indoor_appearance.clone(),
        indoor_layout: args.layout,
        indoor_density: args.density,
        indoor_human_density: args.human_density,
        indoor_quality: args.quality,
        annotation_glass: args.annotation_glass,
        indoor_gi_rays: args.gi_rays,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        initialize_scene: false,
        num_cameras: args.cameras,
        width: args.width as f32,
        height: args.height as f32,
        playback_mode: PlaybackMode::Still,
        playback_steps: args.playback_steps,
        render_modes: modes.clone(),
        depth_format: DepthFormat::Linear,
        rotation_augmentation: args.rotation_augmentation,
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.insert_resource(args.glass_filter);
    if args.linear_rgb {
        app.add_systems(
            Last,
            |mut commands: Commands, cameras: Query<Entity, With<Camera3d>>| {
                for entity in &cameras {
                    commands
                        .entity(entity)
                        .insert((
                            bevy::core_pipeline::tonemapping::Tonemapping::None,
                            bevy::core_pipeline::tonemapping::DebandDither::Disabled,
                        ))
                        .remove::<(
                            bevy::post_process::bloom::Bloom,
                            bevy::anti_alias::fxaa::Fxaa,
                        )>();
                }
            },
        );
    }
    if args.no_ssao {
        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(
            Last,
            |mut commands: Commands,
             cameras: Query<Entity, With<bevy::pbr::ScreenSpaceAmbientOcclusion>>| {
                for entity in &cameras {
                    commands
                        .entity(entity)
                        .remove::<bevy::pbr::ScreenSpaceAmbientOcclusion>();
                }
            },
        );
    }
    if args.no_shadows {
        app.add_systems(
            Last,
            |mut points: Query<&mut PointLight>,
             mut spots: Query<&mut SpotLight>,
             mut suns: Query<&mut DirectionalLight>| {
                for mut light in &mut points {
                    light.shadow_maps_enabled = false;
                }
                for mut light in &mut spots {
                    light.shadow_maps_enabled = false;
                }
                for mut light in &mut suns {
                    light.shadow_maps_enabled = false;
                }
            },
        );
    }
    app.insert_resource(
        bevy_zeroverse::scene::procedural_indoor::gi::IndoorGiSettings {
            enabled: !args.no_gi,
            bake: bevy_zeroverse::scene::procedural_indoor::gi::BakeSettings {
                rays_per_probe: args.gi_rays,
                diffuse_bounces: args.gi_bounces,
                ..default()
            },
            ..default()
        },
    );
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    for &seed in &selected {
        let start = Instant::now();
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_seed = Some(seed);
        app.update();
        app.world_mut().write_message(RegenerateSceneEvent);
        // Build and compile RGB pipelines before enabling the sampler; mode transitions
        // get their own settling frames so readback cannot contain the preceding mode.
        *app.world_mut().resource_mut::<RenderMode>() = RenderMode::Color;
        loop {
            app.update();
            ensure!(
                app.world()
                    .resource::<bevy_zeroverse::sample::CaptureFailure>()
                    .0
                    .is_none(),
                "scene preparation failed: {:?}",
                app.world()
                    .resource::<bevy_zeroverse::sample::CaptureFailure>()
                    .0
            );
            if !bevy_zeroverse::scene::procedural_indoor::indoor_generation_pending(app.world())
                && app
                    .world()
                    .get_resource::<IndoorManifest>()
                    .is_some_and(|scene| scene.seed == seed)
            {
                break;
            }
            ensure!(
                start.elapsed() < Duration::from_secs(60),
                "scene preparation timeout for seed {seed}"
            );
        }
        let manifest = app
            .world()
            .get_resource::<IndoorManifest>()
            .context("scene generation failed")?
            .clone();
        ensure!(manifest.seed == seed, "captured stale seed");
        app.insert_resource(SamplerState {
            enabled: true,
            regenerate_scene: false,
            frames: 1,
            warmup_frames: 3,
            render_modes: modes.clone(),
            timesteps: (1..args.playback_steps)
                .map(|i| i as f32 / (args.playback_steps - 1) as f32)
                .collect(),
            ..default()
        });
        let sample = loop {
            app.update();
            ensure!(
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
                break sample;
            }
            ensure!(
                start.elapsed() < Duration::from_secs(180),
                "capture timeout for seed {seed}"
            );
        };
        ensure!(
            sample.views.len() == args.cameras * args.playback_steps as usize,
            "wrong number of views"
        );
        ensure!(
            sample.indoor.as_ref() == Some(&manifest),
            "sampler lost scene manifest"
        );
        let directory = args.output.join(format!("seed_{seed:06}"));
        fs::create_dir_all(&directory)?;
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        if args.export_reference {
            #[cfg(not(target_arch = "wasm32"))]
            bevy_zeroverse::scene::procedural_indoor::reference::export(
                app.world_mut(),
                &sample,
                [args.width, args.height],
                &directory.join("reference"),
            )?;
            #[cfg(target_arch = "wasm32")]
            anyhow::bail!("reference export requires the native validator");
        }
        let views = save_sample(&sample, &args, &directory)?;
        let report = CaptureReport {
            build_provenance: bevy_zeroverse::provenance::capture_provenance(),
            camera_qualification: sample.indoor_render_metadata.as_ref().and_then(|m| m.get("camera_qualification")).cloned(),
            run_id: run_id.clone(),
            seed,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            image_size: [args.width, args.height],
            renderer: app.world().get_resource::<bevy::render::renderer::RenderAdapterInfo>().map(|adapter| format!("{:?}", **adapter)),
            capabilities: serde_json::json!({ "quality": args.quality, "glass_filter": args.glass_filter, "shadows": args.quality.shadows() && !args.no_shadows,
                "ssao": args.quality.ssao() && !args.no_ssao, "bloom": args.quality.bloom() && !args.linear_rgb, "specular_transmission": args.quality.specular_transmission(),
                "shadow_map_size": args.quality.shadow_map_size(), "annotation_hdr_format": if sample.annotation_precision == bevy_zeroverse::sample::AnnotationPrecision::Float32Geometry { "RGBA32Float_direct" } else { "RGBA16Float" } }),
            aabb: sample.aabb,
            color_encoding: if args.linear_rgb {
                "raw: exposed scene-linear little-endian RGBA32F (RGBA16F rendering intermediate); PNG: clipped sRGB preview; no tone map, bloom, FXAA or dither"
            } else {
                "PNG: sRGB OETF of Bevy tone-mapped linear RGB; raw: little-endian RGBA32F"
            },
            label_policy:
                "first geometric surface; glass is an opaque window in geometry annotations; native indoor uses direct float32 MRT",
            views,
            mesh_assets: app.world().resource::<Assets<Mesh>>().len(),
            material_assets: app.world().resource::<Assets<StandardMaterial>>().len(),
            image_assets: app.world().resource::<Assets<Image>>().len(),
            annotation_precision: sample.annotation_precision,
            diffuse_gi: app.world().get_resource::<bevy_zeroverse::scene::procedural_indoor::gi::BakeStatistics>().cloned(),
            co_visibility_metadata: sample.co_visibility_metadata.clone(),
        };
        fs::write(
            directory.join("capture.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!(
            "captured seed {seed} in {:.2}s, {} views, {} meshes -> {}",
            report.elapsed_seconds,
            report.views.len(),
            report.mesh_assets,
            directory.display()
        );
    }
    fs::write(
        args.output.join("run_complete.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "run_id": run_id, "identity": identity, "captured_scenes": selected.len(), "selected_seeds": selected,
        }))?,
    )?;
    Ok(())
}

fn save_sample(
    sample: &Sample,
    args: &Args,
    directory: &std::path::Path,
) -> Result<Vec<ViewReport>> {
    let mut reports = Vec::new();
    for (i, view) in sample.views.iter().enumerate() {
        let camera_index = i % args.cameras;
        let step_index = i / args.cameras;
        let time = if args.playback_steps == 1 {
            0.0
        } else {
            step_index as f32 / (args.playback_steps - 1) as f32
        };
        let scene = sample
            .indoor
            .as_ref()
            .context("missing indoor calibration manifest")?;
        let camera = &scene.cameras[camera_index];
        let expected =
            Mat4::from_rotation_y(scene.world_yaw) * camera.transform_at(time).to_matrix();
        let pose_error = expected
            .to_cols_array()
            .into_iter()
            .zip(Mat4::from_cols_array_2d(&view.world_from_view).to_cols_array())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        ensure!(pose_error < 0.0002 && (view.time-time).abs() < 0.00001 &&
            (view.fovy-camera.fov_degrees.to_radians()).abs() < 0.00001 &&
            (view.near-0.1).abs() < 0.00001 && (view.far-50.0).abs() < 0.00001,
            "camera calibration/trajectory mismatch at camera {camera_index} step {step_index}: pose error {pose_error}, time {} expected {time}", view.time);
        let mut semantic_colors = 0;
        let mut semantic_pixel_counts = std::collections::BTreeMap::new();
        let semantic_palette = semantic_palette();
        for (name, bytes) in [
            ("color", &view.color),
            ("depth", &view.depth),
            ("normal", &view.normal),
            ("semantic", &view.semantic),
            ("position", &view.position),
        ] {
            if bytes.is_empty() {
                ensure!(!args.labels && name != "color", "missing {name} buffer");
                continue;
            }
            ensure!(
                bytes.len() == (args.width * args.height * 16) as usize,
                "incorrect {name} readback length"
            );
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_ne_bytes(*v))
                .collect();
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "non-finite {name} output: {} channels; first index {:?}",
                values.iter().filter(|x| !x.is_finite()).count(),
                values.iter().position(|x| !x.is_finite())
            );
            if !args.no_raw {
                let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                fs::write(directory.join(format!("view_{i:02}_{name}.rgba32f")), raw)?;
            }
            let mut rgb = Vec::new();
            let mut colors = std::collections::BTreeSet::new();
            for p in values.as_chunks::<4>().0.iter() {
                let c = match name {
                    "color" | "semantic" => [
                        linear_to_srgb(p[0]),
                        linear_to_srgb(p[1]),
                        linear_to_srgb(p[2]),
                    ],
                    "depth" => [p[0] / 15.0; 3],
                    _ => [p[0], p[1], p[2]], // normal shader already encodes [-1, 1] as [0, 1]
                };
                let c = c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
                if name == "semantic" {
                    colors.insert(c);
                    let label = semantic_palette
                        .get(&c)
                        .with_context(|| format!("unrecognized semantic palette color: {c:?}"))?;
                    *semantic_pixel_counts.entry(label.clone()).or_default() += 1;
                }
                rgb.extend(c);
            }
            if name == "semantic" {
                semantic_colors = colors.len();
                // Class richness is a dataset review metric, not annotation
                // validity. Close views and opaque annotation glass can contain
                // only one or two classes; every pixel was palette-checked above.
                ensure!(
                    (1..=41).contains(&semantic_colors),
                    "semantic image has an invalid palette size: {semantic_colors}"
                );
            }
            image::RgbImage::from_raw(args.width, args.height, rgb)
                .unwrap()
                .save(directory.join(format!("view_{i:02}_{name}.png")))?;
        }
        let color: Vec<f32> = view
            .color
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_ne_bytes(*v))
            .collect();
        let luminance: Vec<_> = color
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .collect();
        let n = luminance.len() as f32;
        let mean = luminance.iter().sum::<f32>() / n;
        let std = (luminance.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / n).sqrt();
        let dark = luminance.iter().filter(|&&l| l < 0.002).count() as f32 / n;
        let clipped = luminance.iter().filter(|&&l| l > 0.99).count() as f32 / n;
        // Low-light and exposure-tail samples belong to the domain. Reject lost
        // signal, not a dark_fraction chosen for normally lit offices; retain all
        // brightness/clipping statistics so dataset curation can be explicit.
        ensure!(
            std > 0.0001 && mean > 0.000001 && clipped < 0.995,
            "degenerate RGB image: mean={mean} std={std} dark={dark} clipped={clipped}"
        );
        let focal_length = args.height as f32 / (2.0 * (view.fovy * 0.5).tan());
        let co_visibility = if args.co_visibility {
            Some(co_visibility::save(
                view,
                directory,
                i,
                camera_index,
                args.cameras,
                [args.width, args.height],
                !args.no_raw,
            )?)
        } else {
            None
        };
        reports.push(ViewReport {
            calibration: view.calibration.clone(),
            time_seconds: view.time_seconds,
            trajectory_progress: view.trajectory_progress,
            camera_index,
            step_index,
            time,
            pose_max_absolute_error: pose_error,
            fx_pixels: focal_length,
            fy_pixels: focal_length,
            world_from_view: view.world_from_view,
            fov_y: view.fovy,
            near: view.near,
            far: view.far,
            mean_luminance: mean,
            luminance_std: std,
            dark_fraction: dark,
            clipped_fraction: clipped,
            semantic_colors,
            semantic_pixel_counts,
            co_visibility,
            annotation_alignment: if args.labels {
                Some(
                    bevy_zeroverse::scene::procedural_indoor::validation::validate_annotations_with_precision(
                        view,
                        sample.aabb,
                        args.width,
                        args.height,
                        sample.annotation_precision,
                    )
                    .map_err(anyhow::Error::msg)?,
                )
            } else {
                None
            },
        });
    }
    Ok(reports)
}

fn linear_to_srgb(v: f32) -> f32 {
    bevy_zeroverse::render::color::linear_to_srgb(v)
}

fn semantic_palette() -> std::collections::BTreeMap<[u8; 3], String> {
    let mut palette = std::collections::BTreeMap::from([([0, 0, 0], "background".into())]);
    for name in [
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "blinds",
        "desk",
        "shelves",
        "curtain",
        "dresser",
        "pillow",
        "mirror",
        "floormat",
        "clothes",
        "ceiling",
        "books",
        "refrigerator",
        "television",
        "paper",
        "towel",
        "shower_curtain",
        "box",
        "whiteboard",
        "person",
        "nightstand",
        "toilet",
        "sink",
        "lamp",
        "bathtub",
        "bag",
        "other_structure",
        "other_furniture",
        "other_prop",
    ] {
        let color = bevy_zeroverse::render::semantic::SemanticLabel::from_label(name)
            .unwrap()
            .color()
            .to_srgba();
        palette.insert(
            [color.red, color.green, color.blue].map(|c| (c * 255.0).round() as u8),
            name.into(),
        );
    }
    palette
}
