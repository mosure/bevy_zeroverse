use super::*;
use crate::scene::procedural_indoor::geometry::stock::{GrainAxis, TimberCuts};
mod drum;
mod hardware;
mod hinges;
mod joinery;
mod panel;
use joinery::{Fit, Joinery};

impl DoorInstallation {
    /// Installed sheets in moving-leaf coordinates, including cutout and frame clearances.
    pub fn kick_plates(&self) -> Vec<FittedKickPlate> {
        if self.program.kick_plate.is_none() {
            return Vec::new();
        }
        let j = Joinery::sample(&self.program);
        let mut plates = Vec::new();
        for (index, leaf) in self.leaves().into_iter().enumerate() {
            let fit = Fit::new(self, leaf, index, &j);
            plates.extend(
                [-1., 1.]
                    .into_iter()
                    .filter_map(|side| fit.protective_face(&self.program, leaf, index, side, &j)),
            );
        }
        plates
    }

    /// Realized cutout bounds (leaf X, height above floor) and corner radius in metres.
    pub(crate) fn vision_apertures(&self) -> Vec<(Rect, f32)> {
        if self.program.panel != DoorPanel::VisionLite {
            return Vec::new();
        }
        let j = Joinery::sample(&self.program);
        self.leaves()
            .into_iter()
            .enumerate()
            .filter_map(|(i, leaf)| {
                let mut hole = Fit::new(self, leaf, i, &j).opening?;
                let r = self.program.vision_light.map_or(0., |v| v.radius(hole));
                let elevation = leaf.transform.translation.y - self.center.y;
                hole.min.y += elevation;
                hole.max.y += elevation;
                Some((hole, r))
            })
            .collect()
    }
}

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
    if let Some(hanging) = d.program.swing_hanging {
        let fit = hanging.fit(d, index);
        hinges::wing(
            a.part(d.program.hardware_finish, &labels.hardware),
            d,
            hanging,
            &fit,
            leaf.transform
                .mul_transform(Transform::from_translation(-fit.offset)),
            hinges::Wing::Leaf,
        );
    }
    if let Some(track) = &j.track {
        track.build_hangers(a, d, leaf, &labels.hardware);
    }
}
pub(super) fn build(d: &DoorInstallation, a: &mut Assembly, frame: bool) {
    let p = &d.program;
    let tf = d.transform();
    let frame_label = format!("door_frame#portal{}#finish{}", d.id, p.finish_variant);
    if frame {
        let mut cuts = TimberCuts::new(Joinery::stock_seed(p), 0);
        let jamb = Vec3::new(FRAME_WIDTH_M, d.height - 0.018, d.wall_depth + 0.02);
        let head = Vec3::new(d.width + FRAME_WIDTH_M, FRAME_WIDTH_M, d.wall_depth + 0.02);
        let g = a.part(p.frame_finish, &frame_label);
        for (center, size, axis) in [
            (
                Vec3::new(-d.width * 0.5, jamb.y * 0.5, 0.),
                jamb,
                GrainAxis::Y,
            ),
            (
                Vec3::new(d.width * 0.5, jamb.y * 0.5, 0.),
                jamb,
                GrainAxis::Y,
            ),
            (Vec3::Y * d.height, head, GrainAxis::X),
        ] {
            cuts.cuboid(
                g,
                p.frame_finish,
                size,
                0.0015_f32.min(size.min_element() * 0.15),
                Transform::from_translation(tf.transform_point(center)).with_rotation(tf.rotation),
                axis,
            );
        }
        if let Some(hanging) = p.swing_hanging {
            hinges::stops(g, d, hanging, &mut cuts);
        }
    }
    let stock = Joinery::sample(p);
    if p.mechanism == DoorMechanism::Revolving {
        drum::build(a, d, &frame_label);
    }
    use super::super::super::motion::{construction as motion, MotionKind};
    let seed = (d.id as u64).wrapping_mul(1777)
        ^ d.center.x.to_bits() as u64
        ^ ((d.center.z.to_bits() as u64) << 32);
    let labels = Labels {
        leaf: format!("door#portal{}#finish{}", d.id, p.finish_variant),
        seal: format!("door#portal{}", d.id),
        glass: format!("door#portal{}#finish{}", d.id, stock.glass.slot()),
        hardware: format!("door_hardware#portal{}#finish{}", d.id, p.finish_variant),
    };
    if let Some(track) = &stock.track {
        track.build_fixed(a, d, &labels.hardware);
    }
    if let Some(hanging) = p.swing_hanging {
        let g = a.part(p.hardware_finish, &labels.hardware);
        for i in 0..hanging.leaf_count(p) {
            let fit = hanging.fit(d, i);
            hinges::wing(
                g,
                d,
                hanging,
                &fit,
                tf.mul_transform(Transform::from_translation(fit.pin)),
                hinges::Wing::Jamb,
            );
        }
    }
    if !a.motion_selected(MotionKind::Door, seed) {
        for (index, leaf) in d.leaves().into_iter().enumerate() {
            build_leaf(a, d, leaf, index, &stock, &labels);
        }
        return;
    }
    let mut end = d.clone();
    end.program.opening = motion::endpoint(seed, p.opening, 1.);
    end.program.rotor_phase += if seed.is_multiple_of(2) { 1.3 } else { -1.3 };
    for (index, (leaf, target)) in d.leaves().into_iter().zip(end.leaves()).enumerate() {
        let (pivot, axis, angle, translation) = if let Some(hanging) = p.swing_hanging {
            let fit = hanging.fit(d, index);
            let angle = fit.side * p.swing_side * (end.program.opening - p.opening) * FRAC_PI_2;
            (tf.transform_point(fit.pin), Vec3::Y, angle, Vec3::ZERO)
        } else {
            relative_motion(leaf, target)
        };
        let mut part = Assembly::default();
        build_leaf(&mut part, d, leaf, index, &stock, &labels);
        let mut joint = motion::rigid(MotionKind::Door, seed, pivot, axis, angle, translation);
        joint.frame = leaf.transform;
        joint.coupled = matches!(
            p.mechanism,
            DoorMechanism::Revolving | DoorMechanism::Telescoping
        );
        joint.attachment = Some(format!("portal{}", d.id));
        a.articulated(part, joint);
    }
}

fn relative_motion(leaf: Leaf, target: Leaf) -> (Vec3, Vec3, f32, Vec3) {
    let q = target.transform.rotation * leaf.transform.rotation.inverse();
    let (axis, angle) = q.to_axis_angle();
    if angle.abs() <= 1e-6 {
        return (
            Vec3::ZERO,
            axis,
            angle,
            target.transform.translation - leaf.transform.translation,
        );
    }
    // Revolving wings and archived swing leaves describe their pivot implicitly.
    let delta = target.transform.translation - q * leaf.transform.translation;
    let translation = axis * axis.dot(delta);
    let planar = delta - translation;
    let pivot = planar * 0.5 + axis.cross(planar) * (0.5 / (angle * 0.5).tan());
    (pivot, axis, angle, translation)
}

#[cfg(test)]
mod tests;
