//! Matched interactive-material benchmark (no capture readback or model loading).
#![recursion_limit = "256"]
use bevy::{prelude::*, render::renderer::RenderDevice};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler, PerspectiveSampler,
        TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::{semantic::SemanticLabel, RenderMode},
    scene::{SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType},
};
use std::time::Instant;

fn main() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::Custom,
            initialize_scene: false,
            headless: true,
            editor: false,
            gizmos: false,
            keybinds: false,
            image_copiers: false,
            num_cameras: 2,
            width: 640.0,
            height: 360.0,
            ..default()
        }),
        false,
    );
    app.add_systems(
        Startup,
        |mut commands: Commands,
         mut meshes: ResMut<Assets<Mesh>>,
         mut materials: ResMut<Assets<StandardMaterial>>| {
            let root = commands
                .spawn((
                    ZeroverseSceneRoot,
                    SceneAabbNode,
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            let material = materials.add(StandardMaterial {
                base_color: Color::srgb(0.55, 0.38, 0.2),
                ..default()
            });
            // Equal geometry and PBR materials: annotation modes must retain batching.
            let mesh = meshes.add(Cuboid::new(0.5, 0.7, 0.5));
            for z in 0..24 {
                for x in 0..24 {
                    commands.spawn((
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(material.clone()),
                        SemanticLabel::Chair,
                        Transform::from_xyz(x as f32 * 0.65 - 7.5, 0.35, z as f32 * 0.65 - 7.5),
                        ChildOf(root),
                    ));
                }
            }
            commands.spawn((
                DirectionalLight {
                    illuminance: 8000.0,
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_xyz(-4.0, 9.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
            ));
            for side in [-1.0, 1.0] {
                commands.spawn((
                    ZeroverseCamera {
                        perspective_sampler: PerspectiveSampler::exact(65.0),
                        trajectory: TrajectorySampler::Static {
                            start: ExtrinsicsSampler {
                                position: ExtrinsicsSamplerType::Transform(Transform::from_xyz(
                                    side * 10.0,
                                    7.0,
                                    12.0,
                                )),
                                looking_at: LookingAtSampler::Exact(Vec3::ZERO),
                                ..default()
                            },
                        },
                        ..default()
                    },
                    ChildOf(root),
                ));
            }
        },
    );
    app.finish();
    app.cleanup();
    let device = app.world().resource::<RenderDevice>().clone();
    let mut rows = Vec::new();
    for mode in [
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Semantic,
        RenderMode::OpticalFlow,
        RenderMode::Color,
    ] {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .render_mode = mode.clone();
        let mut times = Vec::new();
        for frame in 0..180 {
            let start = Instant::now();
            app.update();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            if frame >= 100 {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        times.sort_by(f64::total_cmp);
        rows.push(serde_json::json!({"mode":format!("{mode:?}"), "frames":times.len(), "median_ms":times[40], "p95_ms":times[76]}));
    }
    let output = std::env::args()
        .nth(1)
        .unwrap_or("out/annotation_preview.json".into());
    std::fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
        "policy":"Two 640x360 cameras, 576 identical meshes, 100 warmup and 80 measured updates per mode. Device-synchronized wall time; not GPU occupancy.", "rows":rows
    })).unwrap()).unwrap();
}
