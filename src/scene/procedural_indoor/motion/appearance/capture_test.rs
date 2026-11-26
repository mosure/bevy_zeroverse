//! Integrated RGB and geometry replay on the ordinary native dataset sampler.
use super::super::capture_tests;
use super::*;
use crate::{
    app::BevyZeroverseConfig,
    render::RenderMode,
    scene::{
        procedural_indoor::{
            layout::{IndoorLayout, SupportPlane},
            objects,
        },
        ZeroverseSceneType,
    },
};

#[test]
#[ignore = "requires a native graphics adapter, including software Vulkan"]
fn actual_room_pages_dimmers_pickups_and_geometric_flow_replay() {
    let _session = capture_tests::SAMPLER.lock().unwrap();
    let seed = 78710;
    let width = 160;
    let policy = IndoorMotionConfig {
        chair_fraction: 0.,
        door_fraction: 0.,
        cabinet_fraction: 0.,
        covering_fraction: 0.,
        appliance_fraction: 0.,
        cart_fraction: 0.,
        fan_fraction: 0.,
        prop_fraction: 1.,
        prop_travel_m: 0.,
        prop_rotation_degrees: 0.,
        prop_pickup_fraction: 1.,
        prop_lift_m: 0.31,
        screen_change_fraction: 1.,
        light_change_fraction: 1.,
        max_mechanisms: 32,
        ..default()
    };
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Semantic,
        RenderMode::OpticalFlow,
    ];
    let cfg = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(seed),
        indoor_layout: IndoorLayout::OpenOffice,
        initialize_scene: false,
        headless: true,
        editor: false,
        gizmos: false,
        keybinds: false,
        image_copiers: true,
        indoor_camera: Some(
            r#"{"motion_fraction":0,"lenses":{"probability":0},"ceiling_mount":{"probability":0}}"#
                .into(),
        ),
        indoor_motion: Some(serde_json::to_string(&policy).unwrap()),
        num_cameras: 3,
        width: width as f32,
        height: width as f32,
        playback_steps: 4,
        playback_step: 0.25,
        indoor_density: 0.72,
        indoor_human_density: 0.,
        render_modes: modes.clone(),
        ..default()
    };
    let (mut app, scene) = capture_tests::load(cfg);
    let report = scene.indoor_motion.as_ref().unwrap();
    let screen = report
        .appearance
        .screens
        .iter()
        .map(|track| &scene.objects[track.object_id])
        .find(|o| !o.neighbor && o.kind == ObjectKind::Monitor)
        .expect("active monitor");
    let pickup = report
        .mechanisms
        .iter()
        .filter(|m| m.rejection.is_none() && m.lift_m > 0.02)
        .find(|m| {
            let o = &scene.objects[m.object_id.unwrap()];
            !o.neighbor && scene.support_plane(o) == Some(SupportPlane::Top)
        })
        .expect("admitted tabletop pickup");
    assert!(!report.appearance.lights.is_empty());
    let item = &scene.objects[pickup.object_id.unwrap()];
    let geometry = objects::build_manifest_object(&scene, screen);
    let (_, face) = geometry
        .parts
        .iter()
        .find(|((surface, _), _)| *surface == Surface::Screen)
        .unwrap();
    let centroid = face
        .positions
        .iter()
        .map(|p| Vec3::from_array(*p))
        .sum::<Vec3>()
        / face.positions.len() as f32;
    let outward = face
        .normals
        .iter()
        .map(|n| Vec3::from_array(*n))
        .sum::<Vec3>()
        .normalize();
    let panel = screen.transform().transform_point(centroid);
    let front = screen.transform().rotation * outward;
    let raised = item.position + Vec3::Y * (item.size.y * 0.5 + pickup.lift_m * 0.45);
    let shots = [
        (panel + front * (screen.size.x * 1.02).max(0.45), panel),
        (raised + Vec3::new(0.42, 0.46, 0.52), raised),
    ];
    capture_tests::aim(&mut app, &shots, 58.);
    let peak = (pickup.interval[0] + pickup.interval[1]) * 0.5;
    let sample = capture_tests::capture(&mut app, vec![peak, 1., 0.], modes);
    assert_eq!(sample.views.len(), 12);
    assert_eq!(
        sample.indoor.as_ref().unwrap().indoor_motion.as_ref(),
        Some(report)
    );
    let out = std::path::Path::new("out/indoor_temporal_programs");
    std::fs::create_dir_all(out).unwrap();
    let mut gallery = image::RgbImage::new(width * 4, width * 3);
    let mut stats = vec![];
    for camera in 0..3 {
        let frame = |step: usize| &sample.views[step * 3 + camera];
        let rest = frame(0);
        let mut photometric = Vec::new();
        let mut depths = Vec::new();
        let mut motion = Vec::new();
        for (step, time) in [0., peak, 1., 0.].into_iter().enumerate() {
            let view = frame(step);
            assert!((view.trajectory_progress.unwrap() - time).abs() < 1e-6);
            assert_eq!(rest.world_from_view, view.world_from_view);
            assert_eq!(rest.calibration, view.calibration);
            for map in [
                &view.color,
                &view.depth,
                &view.normal,
                &view.semantic,
                &view.optical_flow,
            ] {
                let f: &[f32] = bytemuck::cast_slice(map);
                assert_eq!(f.len(), (width * width * 4) as usize);
                assert!(f.iter().all(|v| v.is_finite()));
            }
            if step == 3 {
                assert!(
                    view.color == rest.color,
                    "rewinding must reproduce the original lossless RGB"
                );
            }
            if step >= 2 {
                assert!(
                    rest.depth == view.depth
                        && rest.normal == view.normal
                        && rest.semantic == view.semantic,
                    "geometry at end of a pickup or on rewind"
                );
            }
            let f: &[f32] = bytemuck::cast_slice(&view.color);
            let first: &[f32] = bytemuck::cast_slice(&rest.color);
            let changed = first
                .as_chunks::<4>()
                .0
                .iter()
                .zip(f.as_chunks::<4>().0.iter())
                .filter(|(a, b)| (0..3).any(|i| (a[i] - b[i]).abs() > 0.02))
                .count();
            if camera == 0 && step == 2 {
                let labels: &[f32] = bytemuck::cast_slice(&rest.semantic);
                let monitor = crate::render::semantic::SemanticLabel::Monitor
                    .color()
                    .to_linear()
                    .to_f32_array();
                let changed_panel = labels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(first.as_chunks::<4>().0)
                    .zip(f.as_chunks::<4>().0)
                    .filter(|((label, a), b)| {
                        **label == monitor && (0..3).any(|k| (a[k] - b[k]).abs() > 0.04)
                    })
                    .count();
                eprintln!("{changed_panel} labeled monitor pixels changed at the page switch");
                assert!(
                    changed_panel > 600,
                    "{changed_panel} changed labeled monitor pixels"
                );
            }
            photometric.push(changed);
            depths.push(
                rest.depth
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .zip(view.depth.as_chunks::<16>().0.iter())
                    .filter(|(a, b)| a != b)
                    .count(),
            );
            let flow: &[f32] = bytemuck::cast_slice(&view.optical_flow);
            motion.push(
                flow.as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|v| v[2] > 0.5 && v[0].hypot(v[1]) > 0.25)
                    .count(),
            );
            for (i, p) in f.as_chunks::<4>().0.iter().enumerate() {
                let encode = |v: f32| {
                    (crate::render::color::linear_to_srgb(v).clamp(0., 1.) * 255.).round() as u8
                };
                gallery.put_pixel(
                    width * step as u32 + i as u32 % width,
                    width * camera as u32 + i as u32 / width,
                    image::Rgb([encode(p[0]), encode(p[1]), encode(p[2])]),
                );
            }
        }
        eprintln!("camera {camera}: RGB {photometric:?}, depth {depths:?}, flow {motion:?}");
        stats.push(serde_json::json!({"camera":camera,"rgb_changes_above_0_02":photometric,"depth_changes":depths,"flow_above_quarter_pixel":motion}));
    }
    gallery.save(out.join("replay.png")).unwrap();
    std::fs::write(out.join("capture.json"),serde_json::to_vec_pretty(&serde_json::json!({"seed":seed,"progress":[0.,peak,1.,0.],"screen":screen.id,"pickup":item.id,"lift_m":pickup.lift_m,"appearance":report.appearance,"views":stats})).unwrap()).unwrap();
    assert!(
        stats[0]["rgb_changes_above_0_02"][2].as_u64().unwrap() > 500,
        "screen page must visibly change"
    );
    assert!(
        stats[1]["depth_changes"][1].as_u64().unwrap() > 12,
        "lifted item must change depth"
    );
    assert!(
        stats[1]["flow_above_quarter_pixel"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t.as_u64().unwrap() > 12),
        "moving surfaces must carry flow"
    );
    for s in &stats {
        assert_eq!(
            s["flow_above_quarter_pixel"][2], 0,
            "rest-to-rest correspondence despite RGB change"
        );
    }
}
