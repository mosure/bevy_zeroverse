#![cfg(not(target_arch = "wasm32"))]
#![recursion_limit = "256"]
//! Run explicitly on a native graphics adapter; verifies cache eviction and pixels.
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    headless::{create_app, setup_globals},
    io::channels,
    render::{
        depth::DepthFormat,
        residency::{RenderResidencyDiagnostics, RenderResidencyPolicy, RenderResidencySnapshot},
        RenderMode,
    },
    sample::{CaptureFailure, Sample, SamplerState},
    scene::{
        procedural_indoor::{
            gi::IndoorGiSettings, reset_indoor_sequence,
            validation::validate_annotations_with_precision,
        },
        RegenerateSceneEvent, ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};

fn capture(app: &mut App, seed: Option<u64>) -> Sample {
    if let Some(seed) = seed {
        reset_indoor_sequence(app.world_mut(), seed);
        app.world_mut().write_message(RegenerateSceneEvent);
        app.update();
    }
    let mut sampler = SamplerState::from_config(app.world().resource::<BevyZeroverseConfig>());
    sampler.regenerate_scene = false;
    app.insert_resource(sampler);
    let start = Instant::now();
    loop {
        app.update();
        assert!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "capture failed: {:?}",
            app.world().resource::<CaptureFailure>().0
        );
        if let Ok(sample) = channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            for view in &sample.views {
                assert!(view.color.iter().any(|byte| *byte != 0));
                validate_annotations_with_precision(
                    view,
                    sample.aabb,
                    161,
                    119,
                    sample.annotation_precision,
                )
                .unwrap();
            }
            return sample;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "capture timed out"
        );
    }
}

fn assert_equal_views(before: &Sample, after: &Sample) {
    assert_eq!(before.views.len(), after.views.len());
    for (a, b) in before.views.iter().zip(&after.views) {
        assert_eq!(a.world_from_view, b.world_from_view);
        if a.color != b.color {
            let deltas: Vec<_> = a
                .color
                .as_chunks::<4>()
                .0
                .iter()
                .zip(b.color.as_chunks::<4>().0)
                .map(|(a, b)| (f32::from_le_bytes(*a) - f32::from_le_bytes(*b)).abs())
                .collect();
            panic!(
                "RGB mismatch: {} of {} channels; max={}, mean={}",
                deltas.iter().filter(|d| **d != 0.0).count(),
                deltas.len(),
                deltas.iter().copied().fold(0.0_f32, f32::max),
                deltas.iter().sum::<f32>() / deltas.len() as f32
            );
        }
        assert_eq!(a.depth, b.depth, "cache cleanup changed depth");
        assert_eq!(a.position, b.position, "cache cleanup changed position");
        assert_eq!(a.normal, b.normal, "cache cleanup changed normals");
        assert_eq!(a.semantic, b.semantic, "cache cleanup changed semantics");
    }
}

fn assert_bounded(app: &App) -> RenderResidencySnapshot {
    let stats = app
        .world()
        .resource::<RenderResidencyDiagnostics>()
        .snapshot();
    assert!(stats.pruning_enabled);
    assert!(stats.main_world_entities >= stats.live_meshes);
    for (name, cache) in &stats.caches {
        // Bin-unpacking keys include the phase (opaque/masked/deferred/shadow),
        // so one live view can own several entries.
        let bound = stats.live_views * if name == "bin_unpacking" { 8 } else { 1 };
        assert!(
            cache.after <= bound,
            "{name} retains {} entries for {bound} live keys",
            cache.after
        );
        assert!(
            cache.capacity <= 2048,
            "{name} capacity grows beyond this bounded scene cohort"
        );
    }
    stats
}

#[test]
#[ignore = "requires native GPU; 64 regenerated scenes plus fixed-scene cache/pixel controls"]
fn retired_view_keys_are_evicted_without_changing_captured_pixels() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        initialize_scene: false,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        num_cameras: 2,
        width: 161.0,
        height: 119.0,
        playback_steps: 1,
        playback_mode: PlaybackMode::Still,
        playback_speed: 0.0,
        depth_format: DepthFormat::Linear,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Position,
            RenderMode::Normal,
            RenderMode::Semantic,
        ],
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    // Exercise the indirect bin-unpacking cache as well as the default direct path.
    app.insert_resource(bevy_zeroverse::camera::CaptureDrawPolicy { indirect: true });
    // Keep native shadows/material specialization while avoiding unrelated GI cost.
    app.world_mut().resource_mut::<IndoorGiSettings>().enabled = false;
    app.world_mut()
        .resource_mut::<RenderResidencyPolicy>()
        .prune = false;
    app.finish();
    let render_instance = app
        .sub_app(bevy::render::RenderApp)
        .world()
        .resource::<bevy::render::renderer::RenderInstance>()
        .clone();
    app.cleanup();
    assert!(
        app.world()
            .resource::<bevy::light::cluster::GlobalClusterSettings>()
            .gpu_clustering
            .is_none(),
        "native dataset capture requires deterministic light clustering"
    );
    for _ in 0..4 {
        app.update();
    }
    let before = capture(&mut app, Some(6));
    assert_eq!(
        before.indoor_render_metadata.as_ref().unwrap()["light_clustering"],
        "cpu_deterministic"
    );
    let control = capture(&mut app, None);
    assert_equal_views(&before, &control);
    println!("Same-state RGB and labels are bit-exact before enabling cleanup");
    app.world_mut()
        .resource_mut::<RenderResidencyPolicy>()
        .prune = true;
    let after = capture(&mut app, None);
    assert_equal_views(&before, &after);
    let mut records = vec![assert_bounded(&app)];
    let mut last = None;
    for seed in 0..64 {
        let sample = capture(&mut app, Some(seed));
        assert_eq!(sample.indoor.as_ref().unwrap().seed, seed);
        records.push(assert_bounded(&app));
        last = Some(sample);
    }
    let regenerated = records.last().unwrap().clone();
    // Bevy 0.19 already prunes light keys and removed all tick maps. The
    // Camera/prepass keys and bin-unpacking bind groups need our cleanup.
    for name in ["view_keys", "prepass_keys", "bin_unpacking"] {
        assert!(
            regenerated.caches[name].evicted_total > 0,
            "{name} eviction was not exercised"
        );
    }
    for _ in 0..8 {
        let fixed = capture(&mut app, None);
        assert_equal_views(last.as_ref().unwrap(), &fixed);
        records.push(assert_bounded(&app));
    }
    let replay = capture(&mut app, Some(6));
    assert_equal_views(&before, &replay);
    app.world_mut()
        .resource_mut::<bevy_zeroverse::camera::CaptureDrawPolicy>()
        .indirect = false;
    let direct = capture(&mut app, Some(6));
    let mut draw_differences = Vec::new();
    for (view_index, (a, b)) in before.views.iter().zip(&direct.views).enumerate() {
        assert_eq!(a.world_from_view, b.world_from_view);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.position, b.position);
        assert_eq!(a.normal, b.normal);
        assert_eq!(a.semantic, b.semantic);
        let differences: Vec<_> = a
            .color
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.color.as_chunks::<4>().0)
            .enumerate()
            .filter(|(index, _)| index % 4 != 3)
            .map(|(_, (a, b))| (f32::from_le_bytes(*a) - f32::from_le_bytes(*b)).abs())
            .collect();
        let maximum = differences.iter().copied().fold(0.0_f32, f32::max);
        let mean = differences.iter().sum::<f32>() / differences.len() as f32;
        if maximum > 0.002 || mean > 0.000001 {
            let directory = "out/indoor_draw_divergence";
            std::fs::create_dir_all(directory).unwrap();
            for (label, bytes) in [
                ("indirect", &a.color),
                ("direct", &b.color),
                ("semantic", &a.semantic),
            ] {
                std::fs::write(
                    format!("{directory}/view_{view_index}_{label}.rgba32f"),
                    bytes,
                )
                .unwrap();
            }
        }
        // Different batching orders may change edge pixels in the FP16 color
        // intermediate. Enforce a small absolute bound and a much tighter mean;
        // geometry annotations must remain exactly equal.
        assert!(
            maximum <= 0.002 && mean <= 0.000001,
            "direct/indirect RGB divergence: maximum={maximum}, mean={mean}"
        );
        draw_differences.push(
            serde_json::json!({"maximum_absolute_rgb_difference": maximum,
            "mean_absolute_rgb_difference": mean,
            "different_channels": differences.iter().filter(|d| **d != 0.0).count()}),
        );
    }
    assert_eq!(
        direct.indoor_render_metadata.as_ref().unwrap()["draw_submission"],
        "direct_gpu_preprocessing"
    );
    records.push(assert_bounded(&app));
    let registry = render_instance.generate_report().unwrap();
    let live_bind_groups = registry.hub.bind_groups.num_kept_from_user;
    assert!(
        live_bind_groups < 2048,
        "GPU bind groups accumulated across regenerated scenes: {live_bind_groups}"
    );
    std::fs::create_dir_all("out/indoor_render_residency_bevy019").unwrap();
    std::fs::write("out/indoor_render_residency_bevy019/report.json", serde_json::to_vec_pretty(&serde_json::json!({
        "bevy_version":"0.19.1","regenerated_scenes":64,"fixed_scene_captures":8,"pixels_bit_exact_with_cleanup_disabled_and_enabled":true,
        "same_seed_replay_bit_exact":true,"direct_and_indirect_annotations_bit_exact":true,"draw_submission_rgb_differences":draw_differences,"resolution":[161,119],"cameras":2,"gi_enabled":false,"native_shadows":true,"live_gpu_bind_groups":live_bind_groups,"records":records,
    })).unwrap()).unwrap();
    println!(
        "64 regenerated scenes and 8 fixed-scene captures passed; RGB plus all four labels bit-exact across cleanup toggle/replay; view caches and GPU bin-unpacking bindings evicted obsolete keys; all four observed caches remained bounded"
    );
}
