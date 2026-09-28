#![cfg(not(target_arch = "wasm32"))]
#![recursion_limit = "256"]
//! GPU regressions for editor-only overlays and frame-rate-independent previews.
use bevy::{
    camera::{ImageRenderTarget, RenderTarget},
    prelude::*,
    render::{render_resource::*, renderer::RenderDevice},
    time::TimeUpdateStrategy,
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{EditorCameraMarker, PlaybackMode},
    headless::{create_app, setup_globals},
    io::image_copy::ImageCopier,
    render::{semantic::SemanticLabel, RenderMode},
    scene::{procedural_indoor::humans::IndoorHumanInstance, ZeroverseSceneType},
};
use std::time::{Duration, Instant};

#[derive(Component)]
struct Moving;

fn settle(app: &mut App) {
    let start = Instant::now();
    for _ in 0..10 {
        app.update();
    }
    loop {
        app.update();
        let ready = app
            .world()
            .resource::<bevy_zeroverse::io::image_copy::CapturePipelineReadiness>();
        assert!(ready.failure().is_none(), "{:?}", ready.failure());
        if ready.ready() && start.elapsed() >= Duration::from_millis(150) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "pipelines not ready"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn capture(app: &mut App, copier: &ImageCopier, stamp: u64) -> Vec<u8> {
    settle(app);
    copier.request(stamp);
    let start = Instant::now();
    while !copier.ready(stamp) {
        app.update();
        assert!(copier.failure().is_none(), "{:?}", copier.failure());
        assert!(start.elapsed().as_secs() < 30, "readback timed out");
    }
    copier.take(stamp).unwrap().planes.remove(0)
}

fn box_mesh(app: &mut App, size: Vec3, pos: Vec3, label: SemanticLabel) -> Entity {
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::from_size(size));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            unlit: true,
            base_color: Color::srgb(0.3, 0.3, 0.3),
            ..default()
        });
    app.world_mut()
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(pos),
            label,
        ))
        .id()
}

#[test]
#[ignore = "requires native GPU; pose occlusion, teardown and 30/60/120 Hz flow"]
fn pose_overlay_respects_environment_and_flow_preview_respects_time() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::Custom,
            initialize_scene: false,
            headless: true,
            editor: false,
            camera_grid: false,
            image_copiers: true,
            draw_pose_gizmos: false,
            gizmos: false,
            gizmos_alpha: 1.0,
            keybinds: false,
            num_cameras: 0,
            yaw_speed: 0.0,
            playback_speed: 0.0,
            playback_mode: PlaybackMode::Still,
            ..default()
        }),
        false,
    );
    app.finish();
    app.cleanup();
    let mut target = Image::new_target_texture(256, 256, TextureFormat::Rgba8UnormSrgb, None);
    target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(target);
    let copier = ImageCopier::for_targets(
        vec![target.clone()],
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba8UnormSrgb,
        app.world().resource::<RenderDevice>(),
    );
    app.world_mut().spawn((
        EditorCameraMarker {
            transform: Some(Transform::from_xyz(0., 0., 4.).looking_at(Vec3::ZERO, Vec3::Y)),
        },
        RenderTarget::Image(ImageRenderTarget::from(target)),
        copier.clone(),
    ));
    let human = box_mesh(
        &mut app,
        Vec3::new(1.6, 1.8, 0.6),
        Vec3::ZERO,
        SemanticLabel::Person,
    );
    // These joints are buried inside the opaque "person". The right half is
    // additionally hidden by a foreground wall, testing both occluder classes.
    let joints: Vec<_> = (0..21)
        .map(|i| {
            Vec3::new(
                if i % 2 == 0 { -0.4 } else { 0.4 },
                (i as f32 / 20.0 - 0.5) * 1.2,
                0.0,
            )
        })
        .collect();
    app.world_mut()
        .entity_mut(human)
        .insert(IndoorHumanInstance {
            id: 0,
            local_joints: joints,
        });
    let wall = box_mesh(
        &mut app,
        Vec3::new(2.0, 3.0, 0.1),
        Vec3::new(1.0, 0., 1.5),
        SemanticLabel::Wall,
    );
    for _ in 0..40 {
        app.update();
    }
    let baseline = capture(&mut app, &copier, 1);
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .draw_pose_gizmos = true;
    for _ in 0..40 {
        app.update();
    }
    let overlay = capture(&mut app, &copier, 2);
    let poses = app
        .world_mut()
        .query::<&bevy_zeroverse::annotation::pose::HumanPose>()
        .iter(app.world())
        .count();
    let cams = app
        .world_mut()
        .query::<(&Camera, Option<&Name>)>()
        .iter(app.world())
        .map(|(c, n)| (c.order, c.is_active, n.map(ToString::to_string)))
        .collect::<Vec<_>>();
    println!("poses {poses}; cameras {cams:?}");
    let mut left = 0;
    let mut right = 0;
    for (i, (a, b)) in baseline
        .as_chunks::<4>()
        .0
        .iter()
        .zip(overlay.as_chunks::<4>().0.iter())
        .enumerate()
    {
        if a[..3].iter().zip(&b[..3]).any(|(&a, &b)| a.abs_diff(b) > 4) {
            if i % 256 < 125 {
                left += 1;
            }
            if i % 256 > 130 {
                right += 1;
            }
        }
    }
    let output = std::path::Path::new("out/viewer_annotation_review");
    std::fs::create_dir_all(output).unwrap();
    for (name, data) in [("pose_off", &baseline), ("pose_on", &overlay)] {
        image::RgbaImage::from_raw(256, 256, data.to_vec())
            .unwrap()
            .save(output.join(format!("{name}.png")))
            .unwrap();
    }
    assert!(
        left > 100,
        "joints hidden inside human: changed {left} pixels"
    );
    assert!(
        right < 5,
        "joints leaked through wall: changed {right} pixels"
    );
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .draw_pose_gizmos = false;
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(
        capture(&mut app, &copier, 3),
        baseline,
        "pose layer not cleaned up"
    );
    app.world_mut().entity_mut(human).despawn();
    app.world_mut().entity_mut(wall).despawn();
    let plane = box_mesh(
        &mut app,
        Vec3::new(100., 100., 0.1),
        Vec3::ZERO,
        SemanticLabel::Wall,
    );
    app.world_mut().entity_mut(plane).insert(Moving);
    app.add_systems(
        Update,
        |time: Res<Time>, mut moving: Query<&mut Transform, With<Moving>>| {
            for mut transform in &mut moving {
                transform.translation.x += 0.3 * time.delta_secs();
            }
        },
    );
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .render_mode = RenderMode::OpticalFlow;
    let mut samples = Vec::new();
    for hz in [30, 60, 120] {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            1.0 / hz as f64,
        )));
        for _ in 0..40 {
            app.update();
        }
        let bytes = capture(&mut app, &copier, hz as u64);
        let center: [u8; 3] = bytes[(128 * 256 + 128) * 4..][..3].try_into().unwrap();
        assert!(
            center[0] > center[1] + 5,
            "expected nonzero rightward flow: {center:?}"
        );
        samples.push(serde_json::json!({"hz":hz,"rgb":center}));
        image::RgbaImage::from_raw(256, 256, bytes)
            .unwrap()
            .save(output.join(format!("flow_{hz}hz.png")))
            .unwrap();
    }
    for channel in 0..3 {
        let values: Vec<_> = samples
            .iter()
            .map(|s| s["rgb"][channel].as_i64().unwrap())
            .collect();
        assert!(
            values.iter().max().unwrap() - values.iter().min().unwrap() <= 2,
            "FPS-dependent color: {samples:?}"
        );
    }
    let report = serde_json::json!({"pose_pixels_through_person":left,"pose_pixels_through_wall":right,"flow_reference_interval_seconds":0.05,"flow":samples});
    println!("{report}");
    std::fs::write(
        output.join("metrics.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
