//! Base loads enter the lower crossrail. Hinges join adjacent stiles.
use super::*;

pub(super) fn feet(a: &mut Assembly, o: &IndoorObject, p: &DividerProgram) {
    let label = o.kind.class_name();
    let h = 0.024_f32.min(o.size.y * 0.04);
    for side in [-1., 1.] {
        let (pos, bearing) = p.plan.support(
            side * o.size.x * p.stand_span,
            p.frame_depth(),
            p.border * 1.1,
        );
        let tf = Transform::from_xyz(pos.x, 0., pos.y)
            .with_rotation(Quat::from_rotation_y(bearing * 0.5));
        let half_width = if p.base == 1 {
            (o.size.z * 0.22).min(o.size.x * 0.069)
        } else {
            h * 1.5
        };
        // Clip both ends separately so a folded bay can stand off centre.
        let dir = (tf.rotation * Vec3::Z).xz();
        let limit = o.size.xz() * 0.497 - Vec2::new(dir.y.abs(), dir.x.abs()) * half_width;
        let reach = |v: Vec2| {
            let gap = limit - pos * v.signum();
            let allowance = gap / v.abs().max(Vec2::splat(1e-6));
            allowance.min_element().min(o.size.z * 0.49)
        };
        let (back, front) = (reach(-dir), reach(dir));
        let center_z = (front - back) * 0.5;
        let base_tf = tf * Transform::from_xyz(0., 0., center_z);
        let from = match p.base {
            0 => {
                let base_tf = base_tf * Transform::from_xyz(0., h * 0.5, 0.);
                a.part(p.frame, label).cuboid(
                    Vec3::new(2. * half_width, h, front + back),
                    0.003,
                    base_tf,
                );
                h
            }
            1 => {
                let r = Vec2::new(half_width, (front + back) * 0.5);
                a.part(p.frame, label).housing(
                    &[(0., r), (h, r * Vec2::new(0.86, 0.88))],
                    2.4,
                    base_tf,
                );
                h
            }
            _ => {
                let junction = p.bottom * 0.67;
                for z in [-back + h * 0.6, front - h * 0.6] {
                    let pad = tf * Transform::from_xyz(0., h * 0.25, z);
                    a.part(Surface::Rubber, label)
                        .cuboid(Vec3::new(h, h * 0.5, h), 0.002, pad);
                    a.part(p.frame, label).rod(
                        tf.transform_point(Vec3::new(0., h * 0.47, z)),
                        tf.transform_point(Vec3::Y * junction),
                        h * 0.37,
                    );
                }
                junction
            }
        };
        a.part(p.frame, label).rod(
            tf.transform_point(Vec3::Y * from),
            tf.transform_point(Vec3::Y * (p.bottom + p.border * 0.48)),
            p.border * 0.43,
        );
    }
}

pub(super) fn hinges(a: &mut Assembly, label: &str, p: &DividerProgram, i: usize) {
    let joint = p.plan.hinge_centres_xz[i];
    let depth = p.frame_depth();
    let left = p.plan.bay(i - 1, depth);
    let right = p.plan.bay(i, depth);
    let anchors = [(left, 1.), (right, -1.)].map(|(bay, side)| {
        bay.frame
            .transform_point(Vec3::X * (side * (bay.width * 0.5 - p.border * 0.35)))
            .xz()
    });
    let radius = (p.plan.joint_gap_m * 0.42).min(p.border * 0.24);
    let h = (p.border * 2.0).min(0.05);
    let metal = a.part(Surface::Metal, label);
    for fraction in [0.18, 0.80] {
        let y = p.bottom + (p.top - p.bottom) * fraction;
        // One captive pin, with a narrow relief between its two knuckles.
        let tf = Transform::from_xyz(joint.x, y, joint.y);
        metal.lathe(
            &[
                (0., -h * 0.5),
                (radius, -h * 0.5),
                (radius, -0.001),
                (radius * 0.60, -0.001),
                (radius * 0.60, 0.001),
                (radius, 0.001),
                (radius, h * 0.5),
                (0., h * 0.5),
            ],
            8,
            tf,
        );
        for (side, anchor) in anchors.into_iter().enumerate() {
            let d = anchor - joint;
            let mid = (anchor + joint) * 0.5;
            let tf = Transform::from_xyz(mid.x, y + (side as f32 - 0.5) * h * 0.50, mid.y)
                .with_rotation(Quat::from_rotation_y(-d.y.atan2(d.x)));
            metal.cuboid(Vec3::new(d.length(), h * 0.40, radius), 0., tf);
        }
    }
}
