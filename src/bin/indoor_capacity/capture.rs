//! Opt-in completed render qualification using the production sampler/readiness
//! and readback polling. One app lives across the cohort; no per-room device init.
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    headless::{create_app, setup_globals, update_capture},
    io::channels,
    render::{depth::DepthFormat, RenderMode},
    sample::{CaptureFailure, Sample, SamplerState},
    scene::{
        procedural_indoor::{
            self, capacity::*, layout::IndoorManifest, validation::validate_geometry,
        },
        RegenerateSceneEvent,
    },
};
use clap::ValueEnum;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};

pub fn run(plan: &CapacityPlan, output: &Path) -> Result<()> {
    validate_plan(plan).map_err(anyhow::Error::msg)?;
    ensure!(
        plan.request.budget.rendered_rooms == plan.request.budget.manifests,
        "render qualification currently captures every declared draw"
    );
    ensure!(
        plan.request.budget.rendered_rooms > 0 && plan.request.budget.rendered_rooms <= 2048,
        "render budget must be 1..2048"
    );
    ensure!(!output.exists(), "render audit exists");
    ensure!(
        plan.population.generator.source_sha256 == env!("ZEROVERSE_SOURCE_SHA256"),
        "plan uses a different source build"
    );
    let config: serde_json::Value = serde_json::from_str(&plan.population.resolved_config_json)?;
    let mut args: BevyZeroverseConfig = serde_json::from_value(config["runtime"].clone())?;
    ensure!(
        args.headless,
        "capture capacity plans must use headless settings"
    );
    let modes = plan
        .population
        .capture
        .modes
        .iter()
        .map(|s| RenderMode::from_str(&s.replace('_', "-"), true).map_err(anyhow::Error::msg))
        .collect::<Result<Vec<_>>>()?;
    args.initialize_scene = false;
    args.editor = false;
    args.gizmos = false;
    args.image_copiers = true;
    args.keybinds = false;
    args.press_esc_close = false;
    args.render_modes = modes.clone();
    args.playback_mode = PlaybackMode::Still;
    args.depth_format = DepthFormat::Linear;
    args.playback_steps = plan.population.capture.progress.len() as u32;
    args.regenerate_ms = 0;
    let check_alignment = args.z_depth && modes.contains(&RenderMode::Depth);
    setup_globals(Some(env!("CARGO_MANIFEST_DIR").into()));
    fs::create_dir_all(output)?;
    fs::write(output.join("plan.json"), serde_json::to_vec_pretty(plan)?)?;
    let started = Instant::now();
    let mut app = create_app(None, Some(args), false);
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    let startup = started.elapsed().as_secs_f64();
    let mut records = BufWriter::new(fs::File::create(output.join("records.jsonl.partial"))?);
    let mut timings = Vec::new();
    let mut failures = 0;
    for draw in 0..plan.request.budget.manifests {
        let seed = seed_at(&plan.request.seed_policy, draw);
        let start = Instant::now();
        app.world_mut().resource_mut::<CaptureFailure>().0 = None;
        app.world_mut().resource_mut::<SamplerState>().enabled = false;
        // Indexed cohorts may intentionally repeat a root seed. Merely writing
        // the config preserves the sequence cursor and would generate seed + 1.
        procedural_indoor::reset_indoor_sequence(app.world_mut(), seed);
        app.world_mut().write_message(RegenerateSceneEvent);
        *app.world_mut().resource_mut::<RenderMode>() = RenderMode::Color;
        let result = (|| -> Result<(Sample, f64)> {
            loop {
                update_capture(&mut app);
                ensure!(
                    app.world().resource::<CaptureFailure>().0.is_none(),
                    "preparation failed: {:?}",
                    app.world().resource::<CaptureFailure>().0
                );
                if !procedural_indoor::indoor_generation_pending(app.world())
                    && app
                        .world()
                        .get_resource::<IndoorManifest>()
                        .is_some_and(|s| s.seed == seed)
                {
                    break;
                }
                ensure!(
                    start.elapsed() < Duration::from_secs(120),
                    "preparation timeout"
                );
            }
            app.insert_resource(SamplerState {
                enabled: true,
                regenerate_scene: false,
                frames: 1,
                warmup_frames: 3,
                start_progress: plan.population.capture.progress[0],
                render_modes: modes.clone(),
                timesteps: plan.population.capture.progress[1..].to_vec(),
                ..default()
            });
            loop {
                update_capture(&mut app);
                ensure!(
                    app.world().resource::<CaptureFailure>().0.is_none(),
                    "capture failed: {:?}",
                    app.world().resource::<CaptureFailure>().0
                );
                if let Ok(sample) = channels::sample_receiver()
                    .ok_or_else(|| anyhow::anyhow!("sample receiver unavailable"))?
                    .lock()
                    .unwrap()
                    .try_recv()
                {
                    return Ok((sample, start.elapsed().as_secs_f64()));
                }
                ensure!(
                    start.elapsed() < Duration::from_secs(240),
                    "capture timeout"
                );
            }
        })();
        let mesh_checked = draw < plan.request.budget.mesh_checks;
        let result = result.and_then(|(sample, completed_seconds)| {
            let audit = Instant::now();
            if mesh_checked {
                validate_geometry(
                    sample
                        .indoor
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("missing manifest"))?,
                )
                .map_err(anyhow::Error::msg)?;
            }
            let alignment = if check_alignment {
                sample
                    .views
                    .iter()
                    .map(|view| {
                        procedural_indoor::validation::validate_annotations_with_precision(
                            view,
                            sample.aabb,
                            plan.population.capture.image_size[0],
                            plan.population.capture.image_size[1],
                            sample.annotation_precision,
                        )
                        .map_err(anyhow::Error::msg)
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            let record =
                observe_rendered(&sample, plan, draw, mesh_checked).map_err(anyhow::Error::msg)?;
            timings.push(serde_json::json!({
                "draw": draw, "seed": seed, "completed_capture_seconds": completed_seconds,
                "audit_seconds": audit.elapsed().as_secs_f64()
            }));
            save_preview(
                &sample,
                plan,
                &record,
                &output.join(format!("draw_{draw:04}")),
                &alignment,
            )?;
            Ok(record)
        });
        let record = result.unwrap_or_else(|error| {
            failures += 1;
            CapacityRecord {
                draw,
                seed,
                mesh_checked: false,
                outcome: SampleOutcome::Failed {
                    stage: "completed render qualification".into(),
                    reason: error.to_string(),
                },
            }
        });
        serde_json::to_writer(&mut records, &record)?;
        writeln!(records)?;
        records.flush()?;
        eprintln!(
            "{}/{} completed draws; {failures} failures retained",
            draw + 1,
            plan.request.budget.manifests
        );
    }
    records.get_ref().sync_all()?;
    fs::rename(
        output.join("records.jsonl.partial"),
        output.join("records.jsonl"),
    )?;
    fs::write(
        output.join("execution.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"startup_seconds":startup,"elapsed_seconds":started.elapsed().as_secs_f64(),"timings":timings,"failures":failures,"quality":"all production lighting/shadow/material settings retained","scope":"completed captures exclude explicit audit/preview export; no concurrent consumer is launched or modified"}),
        )?,
    )?;
    ensure!(failures == 0, "{failures} failed draws retained");
    Ok(())
}

fn save_preview(
    sample: &Sample,
    plan: &CapacityPlan,
    record: &CapacityRecord,
    folder: &Path,
    alignment: &[procedural_indoor::validation::AnnotationAlignment],
) -> Result<()> {
    fs::create_dir_all(folder)?;
    fs::write(
        folder.join("manifest.json"),
        serde_json::to_vec_pretty(&sample.indoor)?,
    )?;
    fs::write(
        folder.join("render_metadata.json"),
        serde_json::to_vec_pretty(&sample.indoor_render_metadata)?,
    )?;
    fs::write(
        folder.join("annotation_alignment.json"),
        serde_json::to_vec_pretty(alignment)?,
    )?;
    let mut images = Vec::new();
    for (i, view) in sample.views.iter().enumerate() {
        let pixels: Vec<u8> = view
            .color
            .as_chunks::<16>()
            .0
            .iter()
            .flat_map(|p| {
                (0..3).map(move |c| {
                    let v = f32::from_ne_bytes(p[c * 4..c * 4 + 4].try_into().unwrap());
                    (bevy_zeroverse::render::color::linear_to_srgb(v) * 255.).round() as u8
                })
            })
            .collect();
        let filename = format!("view_{i:02}.png");
        let path = folder.join(&filename);
        image::save_buffer(
            &path,
            &pixels,
            plan.population.capture.image_size[0],
            plan.population.capture.image_size[1],
            image::ColorType::Rgb8,
        )?;
        images.push(serde_json::json!({
            "path": filename,
            "sha256": bevy_zeroverse_capture::provenance::sha(&fs::read(path)?),
            "camera": i % sample.view_dim as usize,
            "time": view.time,
        }));
    }
    fs::write(
        folder.join("previews.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "record_sha256": bevy_zeroverse_capture::provenance::sha(&serde_json::to_vec(record)?),
            "images": images,
            "encoding": "tonemapped linear RGB -> sRGB8 PNG; whole image; no geometry masking",
        }))?,
    )?;
    Ok(())
}
