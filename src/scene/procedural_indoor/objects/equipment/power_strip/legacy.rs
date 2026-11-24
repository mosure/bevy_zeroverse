//! Raised-face stock retained for manifests with the earlier product variants.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &StripProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let h = s.y * p.height_fraction;
    if p.end_caps {
        // End caps share boundaries with the channel; no stacked coplanar lids.
        let cap = s.x * 0.055;
        a.box_part(
            p.housing,
            l,
            Vec3::Y * h * 0.5,
            Vec3::new(s.x - 2. * cap, h, s.z),
            h * p.bevel_fraction,
        );
        for sign in [-1., 1.] {
            a.box_part(
                p.plate,
                l,
                Vec3::new(sign * (s.x - cap) * 0.5, h * 0.5, 0.),
                Vec3::new(cap, h, s.z),
                h * p.bevel_fraction * 0.45,
            );
        }
    } else {
        a.box_part(
            p.housing,
            l,
            Vec3::Y * h * 0.5,
            s.with_y(h),
            h * p.bevel_fraction,
        );
    }
    let width = p.plate_width(s);
    for i in 0..p.socket_count {
        let center = p.socket_center(s, i);
        a.part(p.plate, l).cylinder(
            width * 0.5,
            0.001,
            Transform::from_xyz(center.x, h + 0.0005, center.y),
        );
        let turn = Quat::from_rotation_y(p.socket_yaw);
        for z in [-1., 1.] {
            let position = Vec3::new(center.x, h + 0.0012, center.y)
                + turn * Vec3::new(0., 0., z * width * 0.22);
            if o.variant.is_multiple_of(4) {
                a.part(Surface::Ink, l).cylinder(
                    width * 0.065,
                    0.0004,
                    Transform::from_translation(position),
                );
            } else {
                let size = Vec3::new(
                    width * if o.variant % 4 == 2 { 0.26 } else { 0.19 },
                    0.0004,
                    width * 0.072,
                );
                a.part(Surface::Ink, l).cuboid(
                    size,
                    0.0001,
                    Transform::from_translation(position).with_rotation(turn),
                );
            }
        }
        if o.variant % 4 >= 2 {
            let position =
                Vec3::new(center.x, h + 0.0012, center.y) + turn * Vec3::new(width * 0.22, 0., 0.);
            a.part(Surface::Ink, l).cuboid(
                Vec3::new(width * 0.08, 0.0004, width * 0.10),
                0.0001,
                Transform::from_translation(position).with_rotation(turn),
            );
        }
    }
}
