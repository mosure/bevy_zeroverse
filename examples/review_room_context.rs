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
fn targets(
    scene: &IndoorManifest,
    lighting: bool,
    conference: bool,
    services: bool,
) -> [(Vec3, Vec3); 3] {
    if services {
        let priority = if std::env::args().any(|v| v == "--signals") {
            [
                ObjectKind::StringLights,
                ObjectKind::FireAlarm,
                ObjectKind::FirePullStation,
            ]
        } else {
            [
                ObjectKind::StringLights,
                ObjectKind::Speaker,
                ObjectKind::Rug,
            ]
        };
        return std::array::from_fn(|view| {
            let object = scene
                .objects
                .iter()
                .filter(|o| !o.neighbor && o.kind == priority[view])
                .max_by(|a, b| a.size.max_element().total_cmp(&b.size.max_element()));
            let Some(o) = object else {
                return (scene.cameras[view].start, scene.cameras[view].target);
            };
            let target = o.position + Vec3::Y * (o.size.y * 0.5);
            if o.kind == ObjectKind::StringLights {
                // Inspect the full supported span from normal eye height. An
                // eye beside the ceiling is often invalid; its generic fallback
                // can end up looking along the strand and hide its sag/anchors.
                let normal = o.transform().rotation * Vec3::Z;
                for radius in [o.size.x * 0.82 + 0.2, 4., 3., 2.2] {
                    for side in [1., -1.] {
                        let p = target + normal * (side * radius);
                        let p = p.with_y(scene.floor_height(p.xz()) + 1.65);
                        if scene.camera_clear(p) && scene.camera_view_clear(p, target) {
                            return (p, target);
                        }
                    }
                }
            }
            if o.kind == ObjectKind::Rug {
                // A furniture-bearing rug's center may be below a chair/table.
                // Requiring that floor point to be unoccluded rejects every
                // useful overview. Keep the camera and architectural sightline
                // clear; ordinary foreground furniture remains in the render.
                // Diagnostic details may visit secondary furnished zones; the
                // dataset's primary-room camera policy is retained in manifest.
                let mut inspection = scene.clone();
                inspection.camera_settings.primary_room = false;
                let mut structure =
                    bevy_zeroverse::scene::procedural_indoor::floorplan::obstacles(scene);
                if let Some(e) = &scene.envelope {
                    structure.extend(e.structural_boxes(scene.room_size));
                }
                if let Some(program) = &scene.program {
                    for partition in &program.partitions {
                        structure.extend(partition.portal_obstacles(scene.room_size.y));
                    }
                }
                for radius in [1.8, 2.4, 3.1, 4.] {
                    for height in [2.2, 1.6, 1.2] {
                        for i in 0..48 {
                            let angle = i as f32 * std::f32::consts::TAU / 48.;
                            let p = (target + Vec3::new(angle.cos(), 0., angle.sin()) * radius)
                                .with_y(o.position.y + height);
                            if inspection.camera_clear(p)
                                && !structure.iter().any(|&(a,b)| {
                                    bevy_zeroverse::scene::procedural_indoor::layout::segment_hits_box(p, target, a, b)
                                })
                                && scene.envelope.as_ref().is_none_or(|e| {
                                    bevy_zeroverse::scene::procedural_indoor::envelope::polygon::segment_inside(
                                        &e.footprint, p.xz(), target.xz(), 0.05,
                                    )
                                })
                            {
                                return (p, target);
                            }
                        }
                    }
                }
                return (eye(scene, target, 2.1), target);
            }
            let normal = o.transform().rotation * Vec3::Z;
            for radius in [o.size.x.max(o.size.y) * 1.2 + 0.65, 1.5, 2.0, 2.5] {
                let p = target + normal * radius - Vec3::Y * 0.25;
                if scene.camera_clear(p) && scene.camera_view_clear(p, target) {
                    return (p, target);
                }
            }
            (eye(scene, target, 1.65), target)
        });
    }
    if conference {
        let table = scene
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.kind == ObjectKind::Table)
            .max_by(|a, b| (a.size.x * a.size.z).total_cmp(&(b.size.x * b.size.z)))
            .expect("conference diagnostic seed must contain a table");
        let target = table.position + Vec3::Y * table.size.y;
        return std::array::from_fn(|view| {
            for radius in [table.size.x.max(table.size.z) * 0.75 + 1., 4., 3., 2.4] {
                for i in 0..48 {
                    let angle = view as f32 * std::f32::consts::TAU / 3.
                        + i as f32 * std::f32::consts::TAU / 48.;
                    let p = target + Vec3::new(angle.cos(), 0., angle.sin()) * radius;
                    let p = p.with_y(scene.floor_height(p.xz()) + 1.7 + view as f32 * 0.13);
                    if scene.in_primary_room(p, 0.3)
                        && scene.camera_clear(p)
                        && scene.camera_view_clear(p, target)
                    {
                        return (p, target);
                    }
                }
            }
            (scene.cameras[view].start, scene.cameras[view].target)
        });
    }
    if lighting {
        use bevy_zeroverse::scene::procedural_indoor::{architecture, photometry};
        let fallback = scene.cameras[0].target;
        let fixtures = architecture::fixture_positions(scene);
        let ceiling = fixtures
            .iter()
            .find(|p| scene.in_primary_room(**p, 0.2))
            .copied()
            .unwrap_or(fallback);
        let lamp = scene
            .objects
            .iter()
            .find(|o| photometry::is_lamp(o.kind) && scene.in_primary_room(o.position, 0.1))
            .map_or(fallback, |o| o.position + Vec3::Y * o.size.y * 0.7);
        let screen = scene
            .objects
            .iter()
            .find(|o| {
                matches!(
                    o.kind,
                    ObjectKind::Laptop | ObjectKind::Monitor | ObjectKind::Display
                ) && scene.in_primary_room(o.position, 0.1)
            })
            .map_or(fallback, |o| o.position + Vec3::Y * o.size.y * 0.65);
        return [
            (eye(scene, ceiling, 1.4), ceiling),
            (eye(scene, lamp, 1.3), lamp),
            (eye(scene, screen, 1.3), screen),
        ];
    }
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
    let lighting = std::env::args().any(|a| a == "--lighting");
    let conference = std::env::args().any(|a| a == "--conference");
    let signals = std::env::args().any(|a| a == "--signals");
    let clutter = std::env::args().any(|a| a == "--clutter");
    let services = signals || std::env::args().any(|a| a == "--services");
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Semantic,
    ];
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(seeds[0]),
            indoor_layout: if conference {
                bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout::Conference
            } else {
                default()
            },
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
            indoor_density: if signals || clutter {
                0.7
            } else {
                BevyZeroverseConfig::default().indoor_density
            },
            indoor_human_density: if signals || clutter { 0.25 } else { 0. },
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
        let views = targets(&scene, lighting, conference, services);
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
            let fov = if services {
                68.
            } else if conference {
                78_f32
            } else if index.0 == 2 {
                42.
            } else {
                68.
            };
            c.perspective_sampler = PerspectiveSampler::exact(fov);
            if let Some(mut projection) = projection {
                if let Projection::Perspective(p) = &mut *projection {
                    p.fov = fov.to_radians();
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
            for attachment in [&v.color, &v.depth, &v.normal, &v.semantic] {
                let values: &[f32] = bytemuck::cast_slice(attachment);
                ensure!(
                    values.len() == 640 * 640 * 4 && values.iter().all(|v| v.is_finite()),
                    "invalid attachment"
                );
            }
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
                output.join(format!("seed_{seed}_{i}.normal.rgba32f")),
                &v.normal,
            )?;
            std::fs::write(
                output.join(format!("seed_{seed}_{i}.depth.rgba32f")),
                &v.depth,
            )?;
        }
        std::fs::write(
            output.join(format!("seed_{seed}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"scene":scene,"detail_camera_poses":views,"annotation_glass":"through","image_size":[640,640],"provenance":bevy_zeroverse::provenance::capture_provenance(),"lighting_details":lighting,"conference_details":conference,"services_details":services,"protocol":"Full production room and lighting; diagnostic detail views replace population cameras. Not distribution qualification."}),
            )?,
        )?;
    }
    Ok(())
}
