//! Stationary stops and jamb wings, moving leaf wings and a common hinge pin.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
use swing::HangingFit;

pub(super) fn stops(
    g: &mut Geometry,
    d: &DoorInstallation,
    h: SwingHanging,
    cuts: &mut TimberCuts,
) {
    let p = &d.program;
    let half = (d.width - FRAME_WIDTH_M) * 0.5;
    let head = d.height - FRAME_WIDTH_M * 0.5;
    let projection = h.stop_projection_m;
    let back = h.swing_face(d) - p.thickness;
    // Even the thinnest lining keeps the stop inside the finished reveal.
    let depth = (1.9 * projection).min((back + (d.wall_depth + 0.020) * 0.5) * 0.90);
    let z = p.swing_side * (back - 0.0008 - depth * 0.5);
    let tf = d.transform();
    let upright = head - projection;
    // Screw-fixed stock seats into the lining. Butt the uprights below the
    // head stop with a fine joint, leaving one exterior surface at each corner.
    let bed = 0.0005;
    for (at, size, grain) in [
        (
            Vec3::new(0., head - (projection - bed) * 0.5, z),
            Vec3::new(half * 2. + 2. * bed, projection + bed, depth),
            GrainAxis::X,
        ),
        (
            Vec3::new(-half + (projection - bed) * 0.5, upright * 0.5, z),
            Vec3::new(projection + bed, upright - 0.0006, depth),
            GrainAxis::Y,
        ),
        (
            Vec3::new(half - (projection - bed) * 0.5, upright * 0.5, z),
            Vec3::new(projection + bed, upright - 0.0006, depth),
            GrainAxis::Y,
        ),
    ] {
        cuts.cuboid(
            g,
            p.frame_finish,
            size,
            0.,
            tf.mul_transform(Transform::from_translation(at)),
            grain,
        );
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Wing {
    Jamb,
    Leaf,
}

pub(super) fn wing(
    g: &mut Geometry,
    d: &DoorInstallation,
    h: SwingHanging,
    fit: &HangingFit,
    pin_frame: Transform,
    wing: Wing,
) {
    let fixed = wing == Wing::Jamb;
    let sign = if fixed { fit.side } else { -fit.side };
    let toward = d.program.swing_side;
    let r = h.barrel_radius_m;
    let gauge = h.gauge();
    let x = sign * (h.half_gap() + 0.0003 - gauge * 0.5);
    let toe = -1.32 * r;
    let rear = -(r + 0.0008 + h.reveal_m + d.program.thickness * 0.83);
    let pitch = h.hinge_length_m / f32::from(h.knuckles);
    let at = |x, y, z| pin_frame.mul_transform(Transform::from_xyz(x, y, toward * z));
    for mid in h.heights(fit.size) {
        let bottom = mid - h.hinge_length_m * 0.5;
        g.cuboid(
            Vec3::new(gauge, h.hinge_length_m, toe - rear),
            0.,
            at(x, mid, (rear + toe) * 0.5),
        );
        for i in 0..h.knuckles {
            if i.is_multiple_of(2) != fixed {
                continue;
            }
            let y = bottom + (f32::from(i) + 0.5) * pitch;
            let half = (pitch - 0.001) * 0.5;
            // A folded tongue runs from the plate into its own knuckle. Its
            // side is inset from the plate so the two visible faces never fight.
            let start = toe - 0.0005;
            let end = -r * 0.56;
            g.cuboid(
                Vec3::new(gauge * 0.90, half * 2. - 0.0004, end - start),
                0.,
                at(x, y, (start + end) * 0.5),
            );
            g.lathe(
                &[(0., -half), (r, -half), (r, half), (0., half)],
                12,
                at(0., y, 0.),
            );
        }
        if fixed {
            let half = h.hinge_length_m * 0.5 + 0.0012;
            let pin = r * 0.58;
            g.lathe(
                &[(0., -half), (pin, -half), (pin, half), (0., half)],
                12,
                at(0., mid, 0.),
            );
        }
        // Low-profile screw heads on the exposed edge plate, behind the roll.
        let face = x - sign * gauge * 0.5;
        let screw = r * 0.61;
        for dy in [-0.29, 0.29] {
            let tf = at(face, mid + dy * h.hinge_length_m, (rear + toe) * 0.5).mul_transform(
                Transform::from_rotation(Quat::from_rotation_z(sign * FRAC_PI_2)),
            );
            g.lathe(
                &[(screw, -0.0001), (screw * 0.82, 0.0004), (0., 0.0004)],
                8,
                tf,
            );
        }
    }
}
