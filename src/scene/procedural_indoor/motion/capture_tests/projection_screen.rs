use super::*;
use crate::{
    render::semantic::SemanticLabel,
    scene::{
        procedural_indoor::{
            layout::IndoorLayout, motion::IndoorMotionConfig, objects::presentation::projection,
        },
        ZeroverseSceneType,
    },
};

#[test]
#[ignore = "requires a native graphics adapter, including software Vulkan"]
fn actual_room_projection_screen_replays_geometry_flow_and_rgb() {
    let _session = SAMPLER.lock().unwrap();
    let seed = 78703;
    let policy = IndoorMotionConfig {
        projection_screen_fraction: 1.,
        chair_fraction: 0.,
        door_fraction: 0.,
        cabinet_fraction: 0.,
        covering_fraction: 0.,
        fan_fraction: 0.,
        appliance_fraction: 0.,
        prop_fraction: 0.,
        cart_fraction: 0.,
        ..default()
    };
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Semantic,
        RenderMode::OpticalFlow,
        RenderMode::CoVisibility,
    ];
    let width = 192u32;
    let (mut app, scene) = load(BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_layout: IndoorLayout::Mixed,
        indoor_seed: Some(seed),
        indoor_density: 0.72,
        indoor_human_density: 0.25,
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
        num_cameras: 2,
        width: width as f32,
        height: width as f32,
        playback_steps: 4,
        playback_step: 0.25,
        render_modes: modes.clone(),
        ..default()
    });
    assert_eq!(scene.seed, seed);
    let report = scene.indoor_motion.as_ref().unwrap();
    let m = report
        .mechanisms
        .iter()
        .find(|m| {
            m.rejection.is_none()
                && m.projection_screen_travel.as_ref().is_some_and(|v| {
                    v.start_drop_m > 0.2 && (v.start_drop_m - v.end_drop_m).abs() > 0.12
                })
                && m.object_id.is_some_and(|id| !scene.objects[id].neighbor)
        })
        .unwrap_or_else(|| panic!("seed {seed}: {:?}", report.mechanisms));
    let o = &scene.objects[m.object_id.unwrap()];
    let p = projection::parameters(o);
    let travel = m.projection_screen_travel.as_ref().unwrap();
    let tf = o.transform();
    let inverse = tf.compute_affine().inverse();
    let target = tf.transform_point(Vec3::new(
        p.sheet_center_x_m,
        p.sheet_top_m - 0.38 * travel.start_drop_m.max(travel.end_drop_m),
        p.sheet_front_m,
    ));
    let views = [Vec3::new(0., 0., 1.), Vec3::new(0.34, 0.07, 0.94)]
        .map(|dir| (target + tf.rotation * dir * p.sheet_width_m * 1.05, target));
    aim(&mut app, &views, 56.);
    let middle = m.interval[0].lerp(m.interval[1], 0.38);
    let later = m.interval[0].lerp(m.interval[1], 0.68);
    let times = [0., middle, later, 0.];
    let sample = capture(&mut app, times[1..].to_vec(), modes);
    assert_eq!(sample.views.len(), 8);
    assert_eq!(
        sample.indoor.as_ref().unwrap().indoor_motion.as_ref(),
        Some(report)
    );
    let label = SemanticLabel::from_label(o.kind.class_name())
        .unwrap()
        .color()
        .to_linear()
        .to_f32_array();
    let [lo, hi] = sample.aabb.map(Vec3::from_array);
    let mut gallery = image::RgbImage::new(4 * width, 2 * width);
    let mut stats = vec![];
    for camera in 0..2 {
        let at = |t: usize| &sample.views[t * 2 + camera];
        let start = at(0);
        let current = at(1);
        let restored = at(3);
        for (a, b) in [
            (&start.color, &restored.color),
            (&start.depth, &restored.depth),
            (&start.normal, &restored.normal),
            (&start.position, &restored.position),
            (&start.semantic, &restored.semantic),
            (&start.co_visibility, &restored.co_visibility),
        ] {
            assert!(a == b, "view {camera} did not rewind");
        }
        let view_from_world = Mat4::from_cols_array_2d(&current.world_from_view).inverse();
        let project = |w: Vec3| {
            Vec2::from_array(
                current
                    .calibration
                    .as_ref()
                    .unwrap()
                    .project(view_from_world.transform_point3(w).to_array())
                    .unwrap(),
            )
        };
        let rgba = bytemuck::cast_slice::<u8, [f32; 4]>;
        let correspondences = |frame: usize| {
            let current = at(frame);
            let delta = m.projection_screen_drop_at(times[frame]).unwrap()
                - m.projection_screen_drop_at(times[frame + 1]).unwrap();
            let (mut moving, mut fixed, mut other, mut shared) = (0, 0, 0, 0);
            let mut worst = 0.0_f32;
            for (((flow, class), pos), co) in rgba(&current.optical_flow)
                .iter()
                .zip(rgba(&current.semantic))
                .zip(rgba(&current.position))
                .zip(rgba(&current.co_visibility))
            {
                let mask = co[0] as u32;
                assert_eq!(mask & !(1 << (1 - camera)), 0);
                assert_eq!(co[1], mask.count_ones() as f32);
                if flow[2] != 1. || pos[3] == 0. {
                    continue;
                }
                let measured = Vec2::new(flow[0], flow[1]);
                if *class != label {
                    other += usize::from(measured.length() > 0.5);
                    continue;
                }
                let w = lo + (hi - lo) * Vec3::new(pos[0], pos[1], pos[2]);
                let local = inverse.transform_point3(w);
                if local.y > p.sheet_top_m + 0.002
                    && local.y < p.sheet_top_m + p.case_height_m + 0.002
                {
                    assert!(measured.length() < 0.02, "moving cassette: {measured:?}");
                    fixed += 1;
                } else if local.y.max(local.y + delta) < p.sheet_top_m - 0.015
                    && (local.z - p.sheet_front_m).abs() < 0.0015
                    && (local.x - p.sheet_center_x_m).abs() < 0.47 * p.sheet_width_m
                {
                    let expected = project(w + tf.rotation * Vec3::Y * delta) - project(w);
                    worst = worst.max(expected.distance(measured));
                    moving += 1;
                    shared += usize::from(co[2] == 1. && mask != 0);
                }
            }
            eprintln!("seed {seed}, view {camera}, t{} -> t{}: {moving} cloth, {fixed} stationary cassette, {other} off-class motion, {shared} shared cloth pixels, max {worst:.5} px",frame,frame+1);
            assert!(moving > 400 && fixed > 20 && other == 0 && shared > 300 && worst < 0.04);
            serde_json::json!({"from":frame,"to":frame+1,"moving_correspondences":moving,"stationary_cassette_hits":fixed,"off_class_flow":other,"shared_cloth_hits":shared,"maximum_flow_error_px":worst})
        };
        let forward = correspondences(1);
        let rewind = correspondences(2);
        let count_changed = |a: &[u8], b: &[u8]| {
            a.as_chunks::<16>()
                .0
                .iter()
                .zip(b.as_chunks::<16>().0)
                .filter(|(a, b)| a != b)
                .count()
        };
        let rgb = count_changed(&start.color, &at(2).color);
        let depth = count_changed(&start.depth, &at(2).depth);
        let semantic = count_changed(&start.semantic, &at(2).semantic);
        assert!(rgb > 100 && depth > 100 && semantic > 100);
        for (frame, &progress) in times.iter().enumerate() {
            let view = at(frame);
            assert_eq!(view.world_from_view, start.world_from_view);
            assert_eq!(view.calibration, start.calibration);
            assert!((view.trajectory_progress.unwrap() - progress).abs() < 1e-6);
            for (n, v) in rgba(&view.color).iter().enumerate() {
                let c = std::array::from_fn(|k| {
                    (crate::render::color::linear_to_srgb(v[k]).clamp(0., 1.) * 255.).round() as u8
                });
                gallery.put_pixel(
                    frame as u32 * width + n as u32 % width,
                    camera as u32 * width + n as u32 / width,
                    image::Rgb(c),
                );
            }
        }
        stats.push(serde_json::json!({
            "camera": camera, "forward": forward, "rewind": rewind,
            "changed_rgb": rgb, "changed_depth": depth, "changed_semantic": semantic
        }));
    }
    let dir = std::path::Path::new("out/indoor_screen_motion");
    std::fs::create_dir_all(dir).unwrap();
    gallery.save(dir.join("screen-replay.png")).unwrap();
    std::fs::write(dir.join("capture.json"),serde_json::to_vec_pretty(&serde_json::json!({"generator":scene.generator_version,"seed":seed,"object":o.id,"times":times,"travel":travel,"width":width,"views":stats})).unwrap()).unwrap();
}
