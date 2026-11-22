//! Hardware starts on a load-bearing face. Projection is measured from that face.
use super::*;

pub(super) fn build(
    a: &mut Assembly,
    p: &DoorProgram,
    leaf: Leaf,
    index: usize,
    fit: &Fit,
    j: &Joinery,
    label: &str,
) {
    let tf = leaf.transform;
    let s = leaf.size;
    let finish = p.hardware_finish;
    for side in [-1., 1.] {
        let at = fit.mount.extend(side * s.z * 0.5);
        let into_room = Vec3::Z * side;
        match Fit::handle(p, index, side) {
            DoorHandle::Lever | DoorHandle::Knob => {
                let r = j.rose.min(fit.stile * 0.40);
                let boss = at + into_room * 0.0026;
                if j.round_rose {
                    a.part(finish, label).cylinder(
                        r,
                        0.0052,
                        tf.mul_transform(
                            Transform::from_translation(boss)
                                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                        ),
                    );
                } else {
                    box_at(
                        a,
                        finish,
                        label,
                        tf,
                        boss,
                        Vec3::new(r * 1.7, r * 4.5, 0.0052),
                    );
                }
                let root = at + into_room * 0.004;
                let grip = at + into_room * j.projection;
                if p.handle == DoorHandle::Knob {
                    let radius = (r * 1.1).clamp(0.018, 0.027);
                    pipe(a, finish, label, tf, &[root, grip], 0.010);
                    a.part(finish, label).ellipsoid(
                        Vec3::new(radius, radius, radius * 0.85),
                        tf.mul_transform(Transform::from_translation(grip)),
                    );
                } else {
                    let tip = grip + Vec3::X * fit.side * j.lever;
                    pipe(
                        a,
                        finish,
                        label,
                        tf,
                        &[root, grip, tip, tip - into_room * 0.020],
                        0.0085,
                    );
                }
            }
            DoorHandle::Pull => {
                let mounts = [at - Vec3::Y * j.pull, at + Vec3::Y * j.pull];
                for point in mounts {
                    a.part(finish, label).cylinder(
                        0.014_f32.min(fit.stile * 0.36),
                        0.004,
                        tf.mul_transform(
                            Transform::from_translation(point + into_room * 0.002)
                                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                        ),
                    );
                }
                pipe(
                    a,
                    finish,
                    label,
                    tf,
                    &[
                        mounts[0] + into_room * 0.002,
                        mounts[0] + into_room * j.projection,
                        mounts[1] + into_room * j.projection,
                        mounts[1] + into_room * 0.002,
                    ],
                    0.009,
                );
            }
            DoorHandle::PanicBar => {
                // Bridge the two stiles. A bar centred on a glass/louver opening
                // needs both mounting shoes to land on the surrounding frame.
                let [left, right] = fit.bar_anchors;
                let shoe = j.rose.min(fit.stile * 0.40);
                for x in [left, right] {
                    box_at(
                        a,
                        finish,
                        label,
                        tf,
                        Vec3::new(x, at.y, at.z) + into_room * (j.projection * 0.5),
                        Vec3::new(shoe * 1.66, shoe * 3.6, j.projection),
                    );
                }
                box_at(
                    a,
                    finish,
                    label,
                    tf,
                    Vec3::new((left + right) * 0.5, at.y, at.z) + into_room * j.projection,
                    Vec3::new(right - left + shoe, shoe * 2.25, 0.018),
                );
            }
            DoorHandle::Recessed => panel::finger_cup(a.part(finish, label), leaf, fit, side),
        }
    }
    if matches!(p.mechanism, DoorMechanism::Hinged | DoorMechanism::Paired) {
        for h in [0.22, s.y * 0.5, s.y - 0.22] {
            // The analytic hinge is 2 mm beyond the trimmed leaf edge.
            let pos = Vec3::new(fit.side * (s.x * 0.5 + 0.002), h - s.y * 0.5, 0.);
            a.part(finish, label).cylinder(
                0.008,
                0.09,
                tf.mul_transform(Transform::from_translation(pos)),
            );
        }
    }
}
