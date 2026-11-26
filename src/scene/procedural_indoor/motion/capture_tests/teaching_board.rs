use super::*;
use crate::{
    render::semantic::SemanticLabel,
    scene::{
        procedural_indoor::{
            layout::IndoorLayout,
            materials::Surface,
            motion::{self, IndoorMotionConfig, MotionKind},
            objects,
        },
        ZeroverseSceneType,
    },
};

// Independent oracle on the interior of the real registered, rectangular writing skins.
struct Sheet {
    moving: bool,
    corner: Vec3,
    u: Vec3,
    v: Vec3,
    extent: Vec2,
}

impl Sheet {
    fn contains(&self, world: Vec3) -> bool {
        let p = world - self.corner;
        let uv = Vec2::new(p.dot(self.u), p.dot(self.v)) / self.extent;
        p.dot(self.u.cross(self.v)).abs() < 0.0008
            && uv.cmpgt(Vec2::splat(0.035)).all()
            && uv.cmplt(Vec2::splat(0.965)).all()
    }
}

#[test]
#[ignore = "requires native graphics, including software Vulkan"]
fn actual_room_teaching_board_holds_and_returns_with_calibrated_flow() {
    let _session = SAMPLER.lock().unwrap();
    let policy = IndoorMotionConfig {
        teaching_board_fraction: 1.,
        operation_return_fraction: 1.,
        chair_fraction: 0.,
        door_fraction: 0.,
        cabinet_fraction: 0.,
        covering_fraction: 0.,
        fan_fraction: 0.,
        appliance_fraction: 0.,
        prop_fraction: 0.,
        cart_fraction: 0.,
        max_mechanisms: 1,
        ..default()
    };
    let seed = 8323;
    let width = 192u32;
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Semantic,
        RenderMode::OpticalFlow,
        RenderMode::CoVisibility,
    ];
    let (mut app, scene) = load(BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_layout: IndoorLayout::Classroom,
        indoor_seed: Some(seed),
        indoor_density: 0.65,
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
    let report = scene.indoor_motion.as_ref().unwrap();
    let m = report
        .mechanisms
        .iter()
        .find(|m| {
            m.kind == MotionKind::TeachingBoard
                && m.operation_cycle.is_some()
                && m.object_id.is_some_and(|id| !scene.objects[id].neighbor)
        })
        .unwrap_or_else(|| panic!("seed {seed} mechanisms: {:?}", report.mechanisms));
    let cycle = m.operation_cycle.unwrap();
    let o = &scene.objects[m.object_id.unwrap()];
    let tf = o.transform();
    let assembly = objects::build_manifest_object(&scene, o);
    let mut sheets = Vec::new();
    for ((surface, label), mesh) in &assembly.parts {
        if *surface != Surface::Whiteboard {
            continue;
        }
        let moving = motion::tag(label) == Some(m.joint);
        let (quads, rest) = mesh.positions.as_chunks::<4>();
        assert!(rest.is_empty());
        for quad in quads {
            let [a, b, _, d] = quad.map(|p| tf.transform_point(Vec3::from(p)));
            let u = b - a;
            let v = d - a;
            sheets.push(Sheet {
                moving,
                corner: a,
                u: u.normalize(),
                v: v.normalize(),
                extent: Vec2::new(u.length(), v.length()),
            });
        }
    }
    assert!(sheets.iter().any(|s| s.moving) && sheets.iter().any(|s| !s.moving));
    let target = tf.transform_point(Vec3::Y * (o.size.y * 0.65));
    let views = [Vec3::new(0.05, 0.05, 1.), Vec3::new(-0.39, 0.09, 0.97)]
        .map(|v| (target + tf.rotation * v * o.size.x * 1.02, target));
    aim(&mut app, &views, 55.);
    let times = [0., cycle.reaches_stop_at, cycle.leaves_stop_at, 1.];
    let sample = capture(&mut app, times[1..].to_vec(), modes);
    assert_eq!(sample.views.len(), 8);
    assert_eq!(
        sample.indoor.as_ref().unwrap().indoor_motion.as_ref(),
        Some(report)
    );
    let rgba = bytemuck::cast_slice::<u8, [f32; 4]>;
    let label = SemanticLabel::from_label("whiteboard")
        .unwrap()
        .color()
        .to_linear()
        .to_f32_array();
    let [lo, hi] = sample.aabb.map(Vec3::from_array);
    let mut gallery = image::RgbImage::new(4 * width, 2 * width);
    let mut metrics = Vec::new();
    for camera in 0..2 {
        let at = |t: usize| &sample.views[t * 2 + camera];
        let start = at(0);
        for (t0, t1) in [(0, 3), (1, 2)] {
            let (a, b) = (at(t0), at(t1));
            for (name, l, r) in [
                ("rgb", &a.color, &b.color),
                ("depth", &a.depth, &b.depth),
                ("normal", &a.normal, &b.normal),
                ("position", &a.position, &b.position),
                ("semantic", &a.semantic, &b.semantic),
                ("co-visibility", &a.co_visibility, &b.co_visibility),
            ] {
                assert!(l == r, "{name}: camera {camera}, frames {t0}/{t1} differ");
            }
        }
        let view_from_world = Mat4::from_cols_array_2d(&start.world_from_view).inverse();
        let project = |w: Vec3| {
            Vec2::from_array(
                start
                    .calibration
                    .as_ref()
                    .unwrap()
                    .project(view_from_world.transform_point3(w).to_array())
                    .unwrap(),
            )
        };
        for (frame, &t) in times.iter().enumerate() {
            let view = at(frame);
            assert_eq!(view.calibration, start.calibration);
            assert_eq!(view.world_from_view, start.world_from_view);
            assert!((view.trajectory_progress.unwrap() - t).abs() < 1e-6);
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
            let Some(next) = times.get(frame + 1) else {
                continue;
            };
            let undo = m.rigid_transform_at(t).unwrap().compute_affine().inverse();
            let dest = m.rigid_transform_at(*next).unwrap();
            let (mut moving, mut fixed, mut displaced, mut shared, mut off_class) = (0, 0, 0, 0, 0);
            let mut max_error = 0f32;
            for (((f, p), s), co) in rgba(&view.optical_flow)
                .iter()
                .zip(rgba(&view.position))
                .zip(rgba(&view.semantic))
                .zip(rgba(&view.co_visibility))
            {
                let mask = co[0] as u32;
                assert_eq!(mask & !(1 << (1 - camera)), 0);
                assert_eq!(co[1], mask.count_ones() as f32);
                if f[2] != 1. || p[3] == 0. {
                    continue;
                }
                let measured = Vec2::new(f[0], f[1]);
                if frame == 1 {
                    assert_eq!(measured, Vec2::ZERO, "motion during dwell");
                }
                if *s != label {
                    off_class += usize::from(measured.length() > 0.5);
                    continue;
                }
                let w = lo + (hi - lo) * Vec3::new(p[0], p[1], p[2]);
                for sheet in &sheets {
                    let rest = if sheet.moving {
                        undo.transform_point3(w)
                    } else {
                        w
                    };
                    if !sheet.contains(rest) {
                        continue;
                    }
                    let expected = if sheet.moving {
                        moving += 1;
                        shared += usize::from(co[2] == 1. && mask != 0);
                        let flow = project(dest.transform_point(rest)) - project(w);
                        displaced += usize::from(flow.length() > 0.3);
                        flow
                    } else {
                        fixed += 1;
                        Vec2::ZERO
                    };
                    max_error = max_error.max(expected.distance(measured));
                    break;
                }
            }
            eprintln!("board {seed}, camera {camera}, {frame}: {moving} wing, {fixed} stationary, {displaced} displaced, {shared} shared, {off_class} off-class, {max_error:.6} px");
            assert!(
                moving > 300 && fixed > 300 && shared > 100 && off_class == 0 && max_error < 0.04
            );
            if frame != 1 {
                assert!(displaced > 250);
            }
            metrics.push(serde_json::json!({"camera":camera,"from_frame":frame,"wing_pixels":moving,"stationary_pixels":fixed,"moving_over_0_3_px":displaced,"shared_wing_pixels":shared,"off_class_motion":off_class,"max_flow_error_px":max_error}));
        }
        assert!(
            start.depth != at(1).depth
                && start.color != at(1).color
                && start.semantic != at(1).semantic
        );
    }
    let dir = std::path::Path::new("out/indoor_board_operations");
    std::fs::create_dir_all(dir).unwrap();
    gallery.save(dir.join("board-replay.png")).unwrap();
    std::fs::write(dir.join("capture.json"),serde_json::to_vec_pretty(&serde_json::json!({"generator":scene.generator_version,"seed":seed,"object":o.id,"times":times,"angle_deg":m.angle_radians.to_degrees(),"cycle":cycle,"width":width,"metrics":metrics})).unwrap()).unwrap();
}
