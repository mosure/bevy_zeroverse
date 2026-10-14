//! Diagnostic detail views of complete production rooms (not population samples).
//! cargo run --example review_room_context -- out/context_details 0,1,2,3
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::{glass::AnnotationGlass, RenderMode},
    sample::{CaptureFailure, SamplerState},
    scene::{
        procedural_indoor::{
            indoor_generation_pending,
            layout::{IndoorManifest, ObjectKind},
        },
        RegenerateSceneEvent, ZeroverseSceneType,
    },
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

fn eye(scene: &IndoorManifest, target: Vec3, height: f32) -> Vec3 {
    for radius in [1.8, 2.4, 3.1, 4.] {
        for i in 0..48 {
            let angle = i as f32 * std::f32::consts::TAU / 48.;
            let p = target + Vec3::new(angle.cos(), 0., angle.sin()) * radius;
            let p = p.with_y(scene.floor_height(p.xz()) + height);
            if scene.in_primary_room(p, 0.3)
                && scene.camera_clear(p)
                && scene.camera_view_clear(p, target)
            {
                return p;
            }
        }
    }
    scene.cameras[0].start
}
fn targets(scene: &IndoorManifest) -> [(Vec3, Vec3); 3] {
    let fallback = scene.cameras[0].target;
    let ceiling = scene
        .ceiling_installation
        .as_ref()
        .and_then(|p| {
            p.modules
                .iter()
                .filter(|m| scene.in_primary_room(m.center, 0.1))
                .max_by(|a, b| (a.size.x * a.size.z).total_cmp(&(b.size.x * b.size.z)))
        })
        .map_or(
            fallback.with_y(scene.ceiling_height(fallback.xz()) - 0.1),
            |m| m.center,
        );
    let floor = scene
        .objects
        .iter()
        .filter(|o| scene.in_primary_room(o.position, 0.1))
        .find(|o| {
            o.kind == ObjectKind::FloorOutlet
                && scene.objects.iter().any(|c| c.cable_target == Some(o.id))
        })
        .or_else(|| {
            scene.objects.iter().find(|o| {
                o.kind == ObjectKind::FloorOutlet && scene.in_primary_room(o.position, 0.1)
            })
        })
        .or_else(|| scene.objects.iter().find(|o| o.kind == ObjectKind::Rug))
        .map_or(fallback.with_y(scene.floor_height(fallback.xz())), |o| {
            o.position + Vec3::Y * o.size.y * 0.5
        });
    let mut window = fallback;
    if let Some(e) = &scene.envelope {
        'walls: for wall in &e.walls {
            if let Some(f) = &wall.facade {
                for w in &f.openings {
                    let tf = e.wall_transform(wall.edge);
                    let p = tf.transform_point(Vec3::new(
                        (w.min.x + w.max.x) * 0.5,
                        1.55_f32.clamp(w.min.y + 0.10, w.max.y - 0.10),
                        0.,
                    ));
                    if scene.in_primary_room(p + tf.rotation * Vec3::Z * 0.35, 0.02) {
                        window = p;
                        break 'walls;
                    }
                }
            }
        }
    }
    [
        (eye(scene, ceiling, 1.4), ceiling),
        (eye(scene, window, 1.5), window),
        (eye(scene, floor, 1.25), floor),
    ]
}

fn main() -> Result<()> {
    let output = PathBuf::from(std::env::args().nth(1).expect("output directory"));
    ensure!(!output.exists(), "use a new directory");
    std::fs::create_dir_all(&output)?;
    let seeds = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0,1,2,3".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<u64>, _>>()?;
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let modes = vec![RenderMode::Color, RenderMode::Depth, RenderMode::Semantic];
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(seeds[0]),
            initialize_scene: false,
            headless: true,
            editor: false,
            gizmos: false,
            keybinds: false,
            image_copiers: true,
            num_cameras: 3,
            width: 640.,
            height: 640.,
            playback_steps: 1,
            indoor_human_density: 0.,
            annotation_glass: AnnotationGlass::Through,
            render_modes: modes.clone(),
            ..default()
        }),
        false,
    );
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    for seed in seeds {
        let start = Instant::now();
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_seed = Some(seed);
        app.update();
        app.world_mut().write_message(RegenerateSceneEvent);
        loop {
            app.update();
            ensure!(start.elapsed().as_secs() < 120, "scene load timeout");
            ensure!(
                app.world().resource::<CaptureFailure>().0.is_none(),
                "scene failure"
            );
            if !indoor_generation_pending(app.world())
                && app
                    .world()
                    .get_resource::<IndoorManifest>()
                    .is_some_and(|s| s.seed == seed)
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let scene = app.world().resource::<IndoorManifest>().clone();
        let views = targets(&scene);
        for (index, mut c, projection) in app
            .world_mut()
            .query::<(
                &CaptureCameraIndex,
                &mut ZeroverseCamera,
                Option<&mut Projection>,
            )>()
            .iter_mut(app.world_mut())
        {
            let (p, target) = views[index.0];
            c.perspective_sampler = PerspectiveSampler::exact(if index.0 == 2 { 42. } else { 68. });
            if let Some(mut projection) = projection {
                if let Projection::Perspective(p) = &mut *projection {
                    p.fov = if index.0 == 2 { 42_f32 } else { 68_f32 }.to_radians();
                }
            }
            c.trajectory = TrajectorySampler::Static {
                start: ExtrinsicsSampler {
                    position: ExtrinsicsSamplerType::Transform(Transform::from_translation(p)),
                    looking_at: LookingAtSampler::Exact(target),
                    ..default()
                },
            };
        }
        app.insert_resource(SamplerState {
            enabled: true,
            regenerate_scene: false,
            frames: 1,
            timesteps: Vec::new(),
            warmup_frames: 12,
            render_modes: modes.clone(),
            ..default()
        });
        let sample = loop {
            app.update();
            ensure!(start.elapsed().as_secs() < 240, "detail capture timeout");
            ensure!(
                app.world().resource::<CaptureFailure>().0.is_none(),
                "detail capture failure"
            );
            if let Ok(s) = bevy_zeroverse::io::channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                break s;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        ensure!(sample.views.len() == 3, "incomplete detail capture");
        for (i, v) in sample.views.iter().enumerate() {
            let rgba: &[f32] = bytemuck::cast_slice(&v.color);
            ensure!(rgba.iter().all(|v| v.is_finite()), "nonfinite RGB");
            let rgb = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    p[..3].iter().map(|v| {
                        (bevy_zeroverse::render::color::linear_to_srgb(*v).clamp(0., 1.) * 255.)
                            .round() as u8
                    })
                })
                .collect();
            image::RgbImage::from_raw(640, 640, rgb)
                .unwrap()
                .save(output.join(format!("seed_{seed}_{i}.png")))?;
            std::fs::write(
                output.join(format!("seed_{seed}_{i}.semantic.rgba32f")),
                &v.semantic,
            )?;
            std::fs::write(
                output.join(format!("seed_{seed}_{i}.depth.rgba32f")),
                &v.depth,
            )?;
        }
        std::fs::write(
            output.join(format!("seed_{seed}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"scene":scene,"detail_camera_poses":views,"annotation_glass":"through","image_size":[640,640],"provenance":bevy_zeroverse::provenance::capture_provenance(),"protocol":"Full production room and lighting; diagnostic ceiling, exterior-window and floor-detail views replace population cameras. Not distribution qualification."}),
            )?,
        )?;
    }
    Ok(())
}
