use super::*;
use crate::{
    render::semantic::SemanticLabel,
    scene::{
        procedural_indoor::{
            layout::{IndoorLayout, ObjectKind},
            motion::{IndoorMotionConfig, MotionKind},
            objects::{self, accessories::fan},
        },
        ZeroverseSceneType,
    },
};

#[test]
#[ignore = "requires a native graphics adapter, including software Vulkan"]
fn actual_room_desk_fan_has_localized_forward_flow_and_exact_rewind() {
    let _session = SAMPLER.lock().unwrap();
    let seed = 78700;
    let width = 192_u32;
    let policy = IndoorMotionConfig {
        desk_fan_fraction: 1.,
        fan_fraction: 0.,
        fan_turns: [1.5; 2],
        chair_fraction: 0.,
        door_fraction: 0.,
        cabinet_fraction: 0.,
        covering_fraction: 0.,
        appliance_fraction: 0.,
        cart_fraction: 0.,
        prop_fraction: 0.,
        ..default()
    };
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Semantic,
        RenderMode::OpticalFlow,
    ];
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
        playback_steps: 3,
        playback_step: 0.25,
        render_modes: modes.clone(),
        ..default()
    });
    assert_eq!(scene.seed, seed);
    assert_eq!(
        scene.generator_version,
        bevy_zeroverse_capture::GENERATOR_VERSION
    );
    let report = scene.indoor_motion.as_ref().unwrap();
    let m = report
        .mechanisms
        .iter()
        .filter(|m| m.kind == MotionKind::Fan && m.rejection.is_none())
        .find(|m| {
            m.object_id.is_some_and(|id| {
                !scene.objects[id].neighbor && scene.objects[id].kind == ObjectKind::DeskFan
            })
        })
        .expect("spinning desk fan in the primary room");
    let o = &scene.objects[m.object_id.unwrap()];
    let p = fan::parameters(o);
    let head = o.transform().mul_transform(p.head_transform());
    // In the annulus outside the hub the blade stock and the grille are
    // separated axially. Classify first hits geometrically, independently of
    // the flow image, so a moving grille or backward flow cannot pass.
    let stock = objects::build_manifest_object(&scene, o);
    let local_head = p.head_transform().compute_affine().inverse();
    let (z_min, z_max) = stock.joints[0]
        .parts
        .iter()
        .flat_map(|key| &stock.parts[key].positions)
        .map(|xyz| local_head.transform_point3(Vec3::from_array(*xyz)).z)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), z| {
            (lo.min(z), hi.max(z))
        });
    let from_world = head.compute_affine().inverse();
    let half_blade = std::f32::consts::PI / (p.blades as f32 * m.angle_radians.abs());
    let target = head.translation - Vec3::Y * (p.radius * 0.25);
    let views = [Vec3::new(0.04, 0.04, 1.), Vec3::new(0.5, 0.17, 0.88)]
        .map(|v| (target + head.rotation * v * (p.radius * 3.6), target));
    aim(&mut app, &views, 49.);
    let sample = capture(&mut app, vec![half_blade, 0.], modes);
    assert_eq!(sample.views.len(), 6);
    assert_eq!(
        sample.indoor.as_ref().unwrap().indoor_motion.as_ref(),
        Some(report)
    );
    let label = SemanticLabel::from_label(o.kind.class_name())
        .unwrap()
        .color()
        .to_linear()
        .to_f32_array();
    fn rgba(bytes: &[u8]) -> &[[f32; 4]] {
        bytemuck::cast_slice(bytes)
    }
    let mut gallery = image::RgbImage::new(3 * width, 2 * width);
    let mut stats = vec![];
    let [lo, hi] = sample.aabb.map(Vec3::from_array);
    for camera in 0..2 {
        let start = &sample.views[camera];
        let turn = &sample.views[2 + camera];
        let back = &sample.views[4 + camera];
        let labels = rgba(&start.semantic);
        let flow = rgba(&start.optical_flow);
        let view_from_world = Mat4::from_cols_array_2d(&start.world_from_view).inverse();
        let calibration = start.calibration.as_ref().unwrap();
        let project = |point| {
            Vec2::from_array(
                calibration
                    .project(view_from_world.transform_point3(point).to_array())
                    .unwrap(),
            )
        };
        let delta = m.rigid_transform_at(half_blade).unwrap();
        let changed = |a: &[u8], b: &[u8]| {
            a.as_chunks::<16>()
                .0
                .iter()
                .zip(b.as_chunks::<16>().0)
                .filter(|(x, y)| x != y)
                .count()
        };
        let changed_depth = changed(&start.depth, &turn.depth);
        let changed_normals = changed(&start.normal, &turn.normal);
        let changed_positions = changed(&start.position, &turn.position);
        let changed_rgb = changed(&start.color, &turn.color);
        let mut moving = 0;
        let mut off_object = 0;
        let (mut blades, mut guard) = (0, 0);
        let mut worst_error = 0.0_f32;
        for ((v, class), hit) in flow.iter().zip(labels).zip(rgba(&start.position)) {
            if v[2] > 0.5 && v[0].hypot(v[1]) > 0.5 {
                moving += 1;
                off_object += usize::from(*class != label);
            }
            if *class != label || v[2] != 1. || hit[3] == 0. {
                continue;
            }
            let world = lo + (hi - lo) * Vec3::new(hit[0], hit[1], hit[2]);
            let point = from_world.transform_point3(world);
            if !(0.39..0.68).contains(&(point.xy().length() / p.radius))
                || point.z.abs() > p.depth * 0.56
            {
                continue;
            }
            let measured = Vec2::new(v[0], v[1]);
            if (z_min - 0.0003..=z_max + 0.0003).contains(&point.z) {
                let expected = project(delta.transform_point(world)) - project(world);
                worst_error = worst_error.max(measured.distance(expected));
                blades += 1;
            } else if point.z < z_min - 0.002 || point.z > z_max + 0.002 {
                assert!(measured.length() < 0.01, "guard displacement {measured:?}");
                guard += 1;
            }
        }
        eprintln!("view {camera}, {blades} blade correspondences, {guard} fixed guard hits, max {worst_error:.5} px");
        assert!(blades > 100 && guard > 50 && worst_error < 0.02);
        assert!(
            moving > 100 && off_object == 0,
            "view {camera}: {moving} moving, {off_object} outside fan class"
        );
        assert!(
            changed_rgb > 100
                && changed_depth > 100
                && changed_normals > 100
                && changed_positions > 100
        );
        for (a, b) in [
            (&start.color, &back.color),
            (&start.depth, &back.depth),
            (&start.normal, &back.normal),
            (&start.position, &back.position),
            (&start.semantic, &back.semantic),
        ] {
            assert!(a == b, "rewind must restore every attachment");
        }
        for (i, (view, t)) in [(start, 0.), (turn, half_blade), (back, 0.)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(start.calibration, view.calibration);
            assert_eq!(start.world_from_view, view.world_from_view);
            assert!((view.trajectory_progress.unwrap() - t).abs() < 1e-6);
            assert!([
                &view.color,
                &view.depth,
                &view.normal,
                &view.position,
                &view.semantic,
                &view.optical_flow
            ]
            .into_iter()
            .all(|map| map.len() == (width * width * 16) as usize
                && bytemuck::cast_slice::<_, f32>(map)
                    .iter()
                    .all(|v| v.is_finite())));
            for (pixel, value) in rgba(&view.color).iter().enumerate() {
                let rgb = std::array::from_fn(|k| {
                    (crate::render::color::linear_to_srgb(value[k]).clamp(0., 1.) * 255.).round()
                        as u8
                });
                gallery.put_pixel(
                    i as u32 * width + pixel as u32 % width,
                    camera as u32 * width + pixel as u32 / width,
                    image::Rgb(rgb),
                );
            }
        }
        stats.push(serde_json::json!({"view":camera,"rotor_flow_pixels_above_half_pixel":moving,"flow_pixels_outside_appliance":off_object,"reprojected_blade_hits":blades,"stationary_grille_hits":guard,"max_blade_flow_error_px":worst_error,"changed_rgb":changed_rgb,"changed_depth":changed_depth,"changed_normal":changed_normals,"changed_position":changed_positions}));
    }
    let dir = std::path::Path::new("out/indoor_rotating_objects");
    std::fs::create_dir_all(dir).unwrap();
    gallery.save(dir.join("fan-replay.png")).unwrap();
    std::fs::write(dir.join("capture.json"), serde_json::to_vec_pretty(&serde_json::json!({"generator":scene.generator_version,"seed":seed,"object":o.id,"blades":p.blades,"axis":m.axis,"progress":[0.,half_blade,0.],"width":width,"views":stats})).unwrap()).unwrap();
}
