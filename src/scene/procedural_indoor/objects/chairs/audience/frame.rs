//! Free-standing tablet frames and floor-mounted tip-up standards have different load paths.
use super::*;

pub(super) fn build(a: &mut Assembly, p: &TeachingSeat) {
    let s = p.frame_stock;
    let z = p.hinge.z;
    let x = p.seat_width * 0.5 + 0.022;
    let y = p.hinge.y;
    let beam_y = if !p.floor_mounted {
        y - p.pad * 0.5
    } else {
        y * 0.53
    };
    for side in [-1., 1.] {
        let post = Vec3::new(side * x, y, z);
        if p.floor_mounted {
            let foot = post.with_y(0.008);
            a.box_part(s, "chair", foot, Vec3::new(0.067, 0.016, 0.170), 0.006);
            for offset in [-0.061, 0.061] {
                a.part(Surface::Metal, "chair").cylinder(
                    0.008,
                    0.005,
                    Transform::from_translation(foot + Vec3::new(0., 0.010, offset)),
                );
            }
            a.box_part(
                s,
                "chair",
                post.with_y(y * 0.5 + 0.007),
                Vec3::new(0.034, y - 0.010, 0.044),
                0.005,
            );
            // The axle penetrates the seat insert and both fixed bearings.
            a.part(s, "chair").rod(
                post.with_x(side * p.seat_width * 0.39),
                post.with_x(side * (x + 0.018)),
                0.021,
            );
        } else {
            let run = p.seat_depth;
            let front = Vec3::new(side * x, 0.015, -run * 0.42);
            let rear = Vec3::new(side * x, 0.015, run * 0.46);
            let local = |z: f32, h: f32| Vec3::new(side * x, h, z);
            a.part(s, "chair").tube(
                &[
                    front,
                    local(-run * 0.33, beam_y - 0.034),
                    local(-run * 0.28, beam_y),
                    local(z, beam_y),
                    local(z + 0.018, beam_y - 0.027),
                    rear,
                ],
                0.013,
                10,
            );
            for foot in [front, rear] {
                a.part(Surface::Rubber, "chair").cylinder(
                    0.016,
                    0.019,
                    Transform::from_translation(foot.with_y(0.0095)),
                );
            }
        }
        // Outside the seat sweep until the stay is behind the back, where its
        // two ends are captured by the upright and the moulded back insert.
        let (root, bend_z, end) = if !p.floor_mounted {
            let (plate, size, end) = p.back.stay_plate(side);
            a.part(s, "chair").cuboid(size, 0.003, plate);
            (post.with_y(beam_y), end.z + 0.004, end)
        } else {
            (post, p.back.z + 0.012, p.back.mount(side))
        };
        a.part(s, "chair")
            .tube(&[root, Vec3::new(side * x, 0.491, bend_z), end], 0.012, 10);
    }
    a.part(s, "chair")
        .rod(Vec3::new(-x, beam_y, z), Vec3::new(x, beam_y, z), 0.014);
    let front = -p.seat_depth * 0.28;
    if !p.floor_mounted {
        // The forward cross-tube bears against the underside of the seat.
        a.part(s, "chair").rod(
            Vec3::new(-x, beam_y, front),
            Vec3::new(x, beam_y, front),
            0.014,
        );
    }
    if let Some(t) = p.tablet {
        let side = t.side();
        let (plate, size) = t.mounting_plate();
        let bend = Vec3::new(side * x, 0.49, plate.translation.z + 0.025);
        a.part(s, "chair").tube(
            &[Vec3::new(side * x, beam_y, front), bend, plate.translation],
            0.013,
            10,
        );
        a.part(s, "chair").cuboid(size, 0.003, plate);
    }
}
