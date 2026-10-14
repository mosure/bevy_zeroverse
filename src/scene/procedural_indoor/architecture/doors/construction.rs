use super::*;

fn box_at(
    a: &mut Assembly,
    surface: Surface,
    label: &str,
    tf: Transform,
    center: Vec3,
    size: Vec3,
) {
    a.part(surface, label).cuboid(
        size,
        0.0015_f32.min(size.min_element() * 0.15),
        Transform::from_translation(tf.transform_point(center)).with_rotation(tf.rotation),
    );
}
fn pipe(a: &mut Assembly, s: Surface, label: &str, tf: Transform, points: &[Vec3], radius: f32) {
    // Fillet mechanical bends before sweeping: a tube over a sharp 90-degree
    // polyline can invert its inner corner at short handle standoffs.
    let mut rounded = vec![points[0]];
    for i in 1..points.len() - 1 {
        let before = points[i - 1] - points[i];
        let after = points[i + 1] - points[i];
        let cut = (radius * 2.8)
            .min(before.length() * if i == 1 { 0.85 } else { 0.45 })
            .min(after.length() * if i + 2 == points.len() { 0.85 } else { 0.45 });
        let a0 = points[i] + before.normalize() * cut;
        let b0 = points[i] + after.normalize() * cut;
        for j in 0..=6 {
            let t = j as f32 / 6.;
            rounded.push(a0 * (1. - t).powi(2) + points[i] * (2. * t * (1. - t)) + b0 * t * t);
        }
    }
    rounded.push(*points.last().unwrap());
    let points: Vec<_> = rounded.into_iter().map(|p| tf.transform_point(p)).collect();
    a.part(s, label).tube(&points, radius, 12);
}
fn panel(a: &mut Assembly, d: &DoorInstallation, leaf: Leaf) {
    let p = &d.program;
    let label = format!("door#portal{}#finish{}", d.id, p.finish_variant);
    let s = leaf.size;
    let tf = leaf.transform;
    let edge = p.stile.min(s.x * 0.18);
    if p.panel == DoorPanel::Flush {
        box_at(a, p.finish, &label, tf, Vec3::ZERO, s);
    } else {
        for side in [-1., 1.] {
            box_at(
                a,
                p.finish,
                &label,
                tf,
                Vec3::X * side * (s.x - edge) * 0.5,
                Vec3::new(edge, s.y, s.z),
            );
            box_at(
                a,
                p.finish,
                &label,
                tf,
                Vec3::Y * side * (s.y - edge) * 0.5,
                Vec3::new(s.x - 2. * edge, edge, s.z),
            );
        }
        let inner = Vec2::new(s.x - 2. * edge - 0.002, s.y - 2. * edge - 0.002);
        match p.panel {
            DoorPanel::Glazed => box_at(
                a,
                Surface::GlassInterior,
                &label,
                tf,
                Vec3::ZERO,
                Vec3::new(inner.x, inner.y, 0.008),
            ),
            DoorPanel::Recessed => {
                let count = if s.x > 0.6 { 3 } else { 2 };
                let pitch = (s.y - 2. * edge) / count as f32;
                let base = -(s.y - 2. * edge) * 0.5;
                for i in 0..count {
                    let lo =
                        base + i as f32 * pitch + if i == 0 { 0.001 } else { edge * 0.5 + 0.001 };
                    let hi = base + (i + 1) as f32 * pitch
                        - if i + 1 == count {
                            0.001
                        } else {
                            edge * 0.5 + 0.001
                        };
                    box_at(
                        a,
                        p.finish,
                        &label,
                        tf,
                        Vec3::Y * ((lo + hi) * 0.5),
                        Vec3::new(inner.x, hi - lo, s.z * 0.42),
                    );
                    if i > 0 {
                        box_at(
                            a,
                            p.finish,
                            &label,
                            tf,
                            Vec3::Y * (base + i as f32 * pitch),
                            Vec3::new(inner.x, edge, s.z),
                        );
                    }
                }
            }
            DoorPanel::VisionLite => {
                let glass_h = inner.y * 0.42;
                let split = inner.y * 0.5 - glass_h;
                box_at(
                    a,
                    p.finish,
                    &label,
                    tf,
                    Vec3::Y * (-glass_h * 0.5),
                    Vec3::new(inner.x, inner.y - glass_h - 0.003, s.z),
                );
                box_at(
                    a,
                    Surface::GlassInterior,
                    &label,
                    tf,
                    Vec3::Y * (split + glass_h * 0.5 + 0.001),
                    Vec3::new(inner.x, glass_h - 0.005, 0.008),
                );
            }
            DoorPanel::Louvered => {
                let count = (inner.y / 0.065).ceil() as usize;
                let pitch = inner.y / count as f32;
                for i in 0..count {
                    let slat = Transform::from_translation(
                        tf.transform_point(Vec3::Y * ((i as f32 + 0.5) * pitch - inner.y * 0.5)),
                    )
                    .with_rotation(tf.rotation * Quat::from_rotation_x(0.35));
                    box_at(
                        a,
                        p.finish,
                        &label,
                        slat,
                        Vec3::ZERO,
                        Vec3::new(inner.x, pitch * 0.70, 0.009),
                    );
                }
            }
            DoorPanel::Flush => unreachable!(),
        }
    }
    hardware(a, d, leaf);
}
fn hardware(a: &mut Assembly, d: &DoorInstallation, leaf: Leaf) {
    let p = &d.program;
    let label = format!("door_hardware#portal{}#finish{}", d.id, p.finish_variant);
    let tf = leaf.transform;
    let s = leaf.size;
    let handedness = if p.mechanism == DoorMechanism::Paired {
        d.transform()
            .to_matrix()
            .inverse()
            .transform_point3(tf.translation)
            .x
            .signum()
    } else {
        p.handedness
    };
    let x = if p.mechanism == DoorMechanism::Revolving {
        0.
    } else {
        -handedness * (s.x * 0.5 - 0.09).max(0.06)
    };
    let y = p.handle_height - s.y * 0.5 - 0.012;
    for side in [-1., 1.] {
        let z = side * (s.z * 0.5 + 0.004);
        let c = Vec3::new(x, y, z);
        let handle = if p.handle == DoorHandle::PanicBar
            && side < 0.
            && p.mechanism != DoorMechanism::Revolving
        {
            DoorHandle::Pull
        } else {
            p.handle
        };
        match handle {
            DoorHandle::Lever | DoorHandle::Knob => {
                box_at(
                    a,
                    p.hardware_finish,
                    &label,
                    tf,
                    c,
                    Vec3::new(0.035, 0.14, 0.006),
                );
                pipe(
                    a,
                    p.hardware_finish,
                    &label,
                    tf,
                    &[c + Vec3::Z * side * 0.004, c + Vec3::Z * side * 0.037],
                    0.010,
                );
                let c = c + Vec3::Z * side * 0.043;
                if p.handle == DoorHandle::Knob {
                    a.part(p.hardware_finish, &label).ellipsoid(
                        Vec3::splat(0.025),
                        Transform::from_translation(tf.transform_point(c))
                            .with_rotation(tf.rotation),
                    );
                } else {
                    pipe(
                        a,
                        p.hardware_finish,
                        &label,
                        tf,
                        &[c, c + Vec3::X * handedness * 0.10],
                        0.009,
                    );
                }
            }
            DoorHandle::Pull => {
                pipe(
                    a,
                    p.hardware_finish,
                    &label,
                    tf,
                    &[
                        c - Vec3::Y * 0.16,
                        c - Vec3::Y * 0.16 + Vec3::Z * side * 0.04,
                        c + Vec3::Y * 0.16 + Vec3::Z * side * 0.04,
                        c + Vec3::Y * 0.16,
                    ],
                    0.010,
                );
            }
            DoorHandle::PanicBar => {
                let w = (s.x * 0.72).min(0.90);
                pipe(
                    a,
                    p.hardware_finish,
                    &label,
                    tf,
                    &[
                        Vec3::new(-w * 0.5, y, z),
                        Vec3::new(-w * 0.5, y, z + side * 0.045),
                        Vec3::new(w * 0.5, y, z + side * 0.045),
                        Vec3::new(w * 0.5, y, z),
                    ],
                    0.014,
                );
            }
            DoorHandle::Recessed => {
                // Disjoint escutcheon rails and dark shallow cup. No coplanar
                // decal or uncut hole falsely exposing the opposite panel face.
                box_at(
                    a,
                    Surface::Ink,
                    &label,
                    tf,
                    c,
                    Vec3::new(0.025, 0.115, 0.003),
                );
                for dx in [-0.018, 0.018] {
                    box_at(
                        a,
                        p.hardware_finish,
                        &label,
                        tf,
                        c + Vec3::X * dx + Vec3::Z * side * 0.004,
                        Vec3::new(0.007, 0.13, 0.007),
                    );
                }
            }
        }
    }
    if matches!(p.mechanism, DoorMechanism::Hinged | DoorMechanism::Paired) {
        for h in [0.22, s.y * 0.5, s.y - 0.22] {
            let pos = Vec3::new(
                handedness * s.x * 0.5,
                h - s.y * 0.5,
                -p.swing_side * s.z * 0.5,
            );
            a.part(p.hardware_finish, &label).cylinder(
                0.008,
                0.09,
                Transform::from_translation(tf.transform_point(pos)).with_rotation(tf.rotation),
            );
        }
    }
}
pub(super) fn build(d: &DoorInstallation, a: &mut Assembly, frame: bool) {
    let p = &d.program;
    let tf = d.transform();
    let frame_label = format!("door_frame#portal{}#finish{}", d.id, p.finish_variant);
    if frame {
        for side in [-1., 1.] {
            box_at(
                a,
                p.frame_finish,
                &frame_label,
                tf,
                Vec3::new(side * d.width * 0.5, (d.height - 0.018) * 0.5, 0.),
                Vec3::new(0.035, d.height - 0.018, d.wall_depth + 0.02),
            );
        }
        box_at(
            a,
            p.frame_finish,
            &frame_label,
            tf,
            Vec3::Y * d.height,
            Vec3::new(d.width + 0.035, 0.035, d.wall_depth + 0.02),
        );
    }
    if matches!(
        p.mechanism,
        DoorMechanism::Sliding | DoorMechanism::Telescoping
    ) {
        let hw_label = format!("door_hardware#portal{}", d.id);
        let z = p.swing_side * (d.wall_depth * 0.5 + 0.06);
        box_at(
            a,
            p.hardware_finish,
            &hw_label,
            tf,
            Vec3::new(p.handedness * d.width * 0.5, d.height - 0.065, z),
            Vec3::new(d.width * 2. + 0.084, 0.055, 0.06),
        );
    }
    if p.mechanism == DoorMechanism::Revolving {
        let r = d.width * 0.5 - 0.015;
        // Opposed curved glass side screens; front/back sectors stay open.
        for side in [0., PI] {
            for i in 0..16 {
                let t = side - 0.61 + 1.22 * i as f32 / 16.;
                let u = side - 0.61 + 1.22 * (i + 1) as f32 / 16.;
                let a0 = Vec3::new(r * t.cos(), d.height * 0.5, r * t.sin());
                let b0 = Vec3::new(r * u.cos(), d.height * 0.5, r * u.sin());
                let delta = b0 - a0;
                let rot = Quat::from_rotation_y(-delta.z.atan2(delta.x));
                let panel_tf = Transform::from_translation(tf.transform_point((a0 + b0) * 0.5))
                    .with_rotation(tf.rotation * rot);
                box_at(
                    a,
                    Surface::GlassInterior,
                    &frame_label,
                    panel_tf,
                    Vec3::ZERO,
                    Vec3::new(delta.length() - 0.001, d.height - 0.07, 0.008),
                );
            }
            for y in [0.018, d.height - 0.018] {
                let points: Vec<_> = (0..=32)
                    .map(|i| {
                        let t = side - 0.63 + 1.26 * i as f32 / 32.;
                        Vec3::new(r * t.cos(), y, r * t.sin())
                    })
                    .collect();
                pipe(a, p.frame_finish, &frame_label, tf, &points, 0.017);
            }
        }
        a.part(p.hardware_finish, &format!("door_hardware#portal{}", d.id))
            .cylinder(
                0.025,
                d.height - 0.025,
                Transform::from_translation(tf.transform_point(Vec3::Y * d.height * 0.5))
                    .with_rotation(tf.rotation),
            );
    }
    use super::super::super::motion::{construction as motion, MotionKind};
    let seed = (d.id as u64).wrapping_mul(1777)
        ^ d.center.x.to_bits() as u64
        ^ ((d.center.z.to_bits() as u64) << 32);
    if !a.motion_selected(MotionKind::Door, seed) {
        for leaf in d.leaves() {
            panel(a, d, leaf);
        }
        return;
    }
    let mut end = d.clone();
    end.program.opening = motion::endpoint(seed, p.opening, 1.);
    end.program.rotor_phase += if seed.is_multiple_of(2) { 1.3 } else { -1.3 };
    for (index, leaf) in d.leaves().into_iter().enumerate() {
        let target = end.leaves()[index];
        let q = target.transform.rotation * leaf.transform.rotation.inverse();
        let (axis, angle) = q.to_axis_angle();
        // Recover the analytic hinge from the leaf centre, preserving the exact
        // circular arc rather than interpolating its centre along a chord.
        let mut pivot = Vec3::ZERO;
        let mut translation = target.transform.translation - leaf.transform.translation;
        if angle.abs() > 1e-6 {
            let delta = target.transform.translation - q * leaf.transform.translation;
            let planar = delta - axis * axis.dot(delta);
            pivot = planar * 0.5 + axis.cross(planar) * (0.5 / (angle * 0.5).tan());
            translation = axis * axis.dot(delta);
        }
        let mut part = Assembly::default();
        panel(&mut part, d, leaf);
        let mut joint = motion::rigid(MotionKind::Door, seed, pivot, axis, angle, translation);
        joint.frame = leaf.transform;
        joint.coupled = p.mechanism == DoorMechanism::Revolving;
        joint.attachment = Some(format!("portal{}", d.id));
        a.articulated(part, joint);
    }
}
