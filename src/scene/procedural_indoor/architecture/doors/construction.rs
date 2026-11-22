use super::*;
mod drum;
mod hardware;
mod joinery;
mod panel;
use joinery::{Fit, Joinery};

struct Labels {
    leaf: String,
    seal: String,
    glass: String,
    hardware: String,
}

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
fn build_leaf(
    a: &mut Assembly,
    d: &DoorInstallation,
    leaf: Leaf,
    index: usize,
    j: &Joinery,
    labels: &Labels,
) {
    let fit = Fit::new(d, leaf, index, j);
    panel::build(a, &d.program, leaf, &fit, j, index, labels);
    hardware::build(a, &d.program, leaf, index, &fit, j, &labels.hardware);
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
        drum::build(a, d, &frame_label);
    }
    use super::super::super::motion::{construction as motion, MotionKind};
    let seed = (d.id as u64).wrapping_mul(1777)
        ^ d.center.x.to_bits() as u64
        ^ ((d.center.z.to_bits() as u64) << 32);
    let stock = Joinery::sample(p);
    let labels = Labels {
        leaf: format!("door#portal{}#finish{}", d.id, p.finish_variant),
        seal: format!("door#portal{}", d.id),
        glass: format!("door#portal{}#finish{}", d.id, stock.glass.slot()),
        hardware: format!("door_hardware#portal{}#finish{}", d.id, p.finish_variant),
    };
    if !a.motion_selected(MotionKind::Door, seed) {
        for (index, leaf) in d.leaves().into_iter().enumerate() {
            build_leaf(a, d, leaf, index, &stock, &labels);
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
        build_leaf(&mut part, d, leaf, index, &stock, &labels);
        let mut joint = motion::rigid(MotionKind::Door, seed, pivot, axis, angle, translation);
        joint.frame = leaf.transform;
        joint.coupled = p.mechanism == DoorMechanism::Revolving;
        joint.attachment = Some(format!("portal{}", d.id));
        a.articulated(part, joint);
    }
}

#[cfg(test)]
mod tests;
