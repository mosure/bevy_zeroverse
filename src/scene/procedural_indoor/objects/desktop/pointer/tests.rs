use super::*;
use crate::scene::procedural_indoor::objects::audit;

#[test]
fn operating_side_is_in_the_keyboards_frame_and_belongs_to_this_workstation() {
    let mut key = audit::fixture(ObjectKind::Keyboard, 0);
    key.support = Some(7);
    key.interaction_target = Some(2);
    let mut mouse = audit::fixture(ObjectKind::Mouse, 2);
    mouse.support = key.support;
    mouse.interaction_target = key.interaction_target;
    for yaw in [-2.4, 0., 1.8] {
        key.yaw = yaw;
        key.position = Vec3::new(2., 0.8, -1.4);
        for (side, bit) in [(-1., LEFT), (1., RIGHT)] {
            mouse.variant = 3;
            mouse.yaw = yaw + 0.17;
            mouse.position =
                key.position + Quat::from_rotation_y(yaw) * Vec3::new(side * 0.28, 0., 0.015);
            let mut opposite = key.clone();
            opposite.yaw += PI;
            let mut other = key.clone();
            other.support = Some(8);
            let mut neighbor = key.clone();
            neighbor.interaction_target = Some(5);
            fit_operator_side(&mut mouse, &[other, opposite, neighbor]);
            assert_eq!(mouse.variant, 3);
            fit_operator_side(&mut mouse, std::slice::from_ref(&key));
            assert_eq!(mouse.variant, 3 | bit);
        }
    }
}

fn hit(g: &Geometry, origin: Vec3, ray: Vec3) -> Option<f32> {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|tri| {
            let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let edge = b - a;
            let other = c - a;
            let h = ray.cross(other);
            let det = edge.dot(h);
            if det.abs() < 1e-12 {
                return None;
            }
            let from = origin - a;
            let u = from.dot(h) / det;
            let q = from.cross(edge);
            let v = ray.dot(q) / det;
            let d = other.dot(q) / det;
            (u >= -1e-5 && v >= -1e-5 && u + v <= 1. + 1e-5 && d >= 0.).then_some(d)
        })
        .min_by(f32::total_cmp)
}

#[test]
fn sockets_are_open_controls_visible_and_shells_sit_on_both_skates() {
    let mut families = [0; 3];
    let mut finishes = [false; 4];
    for seed in 0..320 {
        let mut o = audit::fixture(ObjectKind::Mouse, seed);
        // Includes the short/wide and tall/narrow desk-product envelopes.
        o.size = match seed % 3 {
            0 => Vec3::new(0.068, 0.030, 0.096),
            1 => Vec3::new(0.059, 0.045, 0.126),
            _ => o.size,
        };
        o.variant |= if seed % 2 == 0 { LEFT } else { RIGHT };
        let input = parameters(&o);
        let p = PointerProgram::sample(&o, &input);
        assert_eq!(p, PointerProgram::sample(&o, &input));
        let socket = p.socket(o.size);
        let up = socket.tf.rotation * Vec3::Y;
        let mut a = Assembly::default();
        build(&mut a, &o, &input);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let (lo, hi) = a.bounds();
        assert!(
            lo.y.abs() < 1e-6 && hi.y <= o.size.y + 1e-5,
            "{seed} {lo:?} {hi:?}"
        );
        let skates = &a.parts[&(Surface::Plastic, Housing::for_object(&o).trim_label(&o))];
        for side in [-1., 1.] {
            assert!(skates
                .positions
                .iter()
                .any(|v| v[1].abs() < 1e-7 && v[2] * side > o.size.z * 0.15));
        }
        let origin = socket.tf.translation + up * 0.1;
        let front = |g: &Geometry| hit(g, origin, -up);
        let top = a
            .parts
            .values()
            .filter_map(front)
            .min_by(f32::total_cmp)
            .unwrap();
        assert!(
            (0.1 - top - p.radius_m * p.exposure_fraction).abs() < 0.00075,
            "{seed} {:?} center ray covered by housing: {top}",
            p.control
        );
        let past_edge = if p.control.ball().is_some() {
            socket.half.x - p.clearance_m * 0.48
        } else {
            p.control_width_m * 0.5 + p.clearance_m * 0.65
        };
        let inside_gap = origin + socket.tf.rotation * Vec3::X * past_edge;
        let gap = a
            .parts
            .values()
            .filter_map(|g| hit(g, inside_gap, -up))
            .min_by(f32::total_cmp)
            .unwrap();
        assert!(gap > 0.1001, "{seed} blocked clearance ray {gap}");
        match p.control {
            Control::ScrollWheel { .. } => families[0] += 1,
            Control::ThumbTrackball { .. } => {
                families[1] += 1;
                assert!(p.control_position_m.x * p.grip.side() < 0.);
                let mut mirrored = o.clone();
                mirrored.variant ^= LEFT | RIGHT;
                let q = PointerProgram::sample(&mirrored, &input);
                assert_eq!(p.control_position_m.x, -q.control_position_m.x);
                assert_eq!(p.radius_m, q.radius_m);
            }
            Control::FingerTrackball { .. } => {
                families[2] += 1;
                assert_eq!(p.control_position_m.x, 0.);
            }
        }
        if let Some(finish) = p.control.ball() {
            finishes[finish as usize] = true;
            let ball = &a.parts[&(Surface::Plastic, finish.label().into())];
            assert!((front(ball).unwrap() - top).abs() < 1e-6);
            let center = socket.tf.translation - up * socket.buried;
            for (pos, n) in ball.positions.iter().zip(&ball.normals) {
                let v = Vec3::from_array(*pos) - center;
                assert!((v.length() - p.radius_m).abs() < 1e-6);
                assert!(v.normalize().dot(Vec3::from_array(*n)) > 0.9999);
            }
        }
    }
    assert!(families.iter().all(|&v| v > 12));
    assert_eq!(finishes, [true; 4]);
}
