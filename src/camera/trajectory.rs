//! Camera paths and rigid transforms; independent of lens projection.
use super::{EditorCameraGizmoConfigGroup, ExtrinsicsSampler};
use bevy::{curve::cubic_splines::CubicBSpline, prelude::*};

#[derive(Clone, Debug, Reflect)]
#[type_path = "bevy_zeroverse::camera"]
pub enum TrajectorySampler {
    /// Piecewise linear, constant metric speed with smoothly interpolated aims.
    WaypointPath {
        positions: Vec<Vec3>,
        rotations: Vec<Quat>,
    },
    /// Deterministic metric path with independently aimed endpoint orientations.
    CubicBezier {
        positions: [Vec3; 4],
        rotations: [Quat; 2],
    },
    Static {
        start: ExtrinsicsSampler,
    },
    Linear {
        start: ExtrinsicsSampler,
        end: ExtrinsicsSampler,
    },
    Avoidant {
        start: ExtrinsicsSampler,
        end: ExtrinsicsSampler,
        bend_away_from: Vec3,
        radius: f32,
    },
    AvoidantXZ {
        start: ExtrinsicsSampler,
        end: ExtrinsicsSampler,
        bend_away_from: Vec3,
        radius: f32,
    },
    CubicBSpline {
        control_points: Vec<ExtrinsicsSampler>,
    },
    // TODO: formal curve support /w composition
    // TODO: support varying intrinsics across trajectory
}

impl Default for TrajectorySampler {
    fn default() -> Self {
        Self::Static {
            start: ExtrinsicsSampler::default(),
        }
    }
}

impl TrajectorySampler {
    pub fn sample(&mut self, progress: f32) -> Transform {
        match self {
            TrajectorySampler::WaypointPath {
                positions,
                rotations,
            } => {
                if positions.len() < 2 || rotations.len() != positions.len() {
                    return Transform::IDENTITY;
                }
                let length: f32 = positions.windows(2).map(|p| p[0].distance(p[1])).sum();
                let mut distance = progress.clamp(0.0, 1.0) * length;
                for (i, p) in positions.windows(2).enumerate() {
                    let segment = p[0].distance(p[1]);
                    if distance <= segment || i + 2 == positions.len() {
                        let t = if segment > 1e-6 {
                            (distance / segment).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        return Transform::from_translation(p[0].lerp(p[1], t))
                            .with_rotation(rotations[i].slerp(rotations[i + 1], t));
                    }
                    distance -= segment;
                }
                Transform::IDENTITY
            }
            TrajectorySampler::CubicBezier {
                positions: p,
                rotations: r,
            } => {
                let t = progress.clamp(0.0, 1.0);
                let a = p[0].lerp(p[1], t).lerp(p[1].lerp(p[2], t), t);
                let b = p[1].lerp(p[2], t).lerp(p[2].lerp(p[3], t), t);
                Transform::from_translation(a.lerp(b, t)).with_rotation(r[0].slerp(r[1], t))
            }
            TrajectorySampler::Static { start } => start.sample_cache(),
            TrajectorySampler::Linear { start, end } => {
                let progress = progress.clamp(0.0, 1.0);
                let start = start.sample_cache();
                let end = end.sample_cache();

                let pos = start.translation.lerp(end.translation, progress);

                let mut rot = start.rotation.slerp(end.rotation, progress);
                let (yaw, pitch, _roll) = rot.to_euler(EulerRot::YXZ); // extract yaw-pitch-roll
                rot = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) // roll = 0 → y-up
                    .normalize();

                Transform {
                    translation: pos,
                    rotation: rot,
                    ..Default::default()
                }
            }
            TrajectorySampler::Avoidant {
                start,
                end,
                bend_away_from,
                radius,
            } => {
                let progress = progress.clamp(0.0, 1.0);
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();

                let mid_point = start_transform
                    .translation
                    .lerp(end_transform.translation, 0.5);

                let path_dir = end_transform.translation - start_transform.translation;
                let path_length = path_dir.length();

                let avoidance_dir = (mid_point - *bend_away_from).normalize();
                let perpendicular_dir =
                    avoidance_dir - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                let perpendicular_dir = perpendicular_dir.normalize();

                let effective_radius = radius.min(path_length * 0.5);
                let control_point = mid_point + perpendicular_dir * effective_radius;

                let t = progress;
                let pos = (1.0 - t).powi(2) * start_transform.translation
                    + 2.0 * (1.0 - t) * t * control_point
                    + t.powi(2) * end_transform.translation;

                let rot = start_transform
                    .rotation
                    .slerp(end_transform.rotation, progress);

                Transform::from_translation(pos).mul_transform(Transform::from_rotation(rot))
            }
            TrajectorySampler::AvoidantXZ {
                start,
                end,
                bend_away_from,
                radius,
            } => {
                let progress = progress.clamp(0.0, 1.0);
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();

                let mid_point = start_transform
                    .translation
                    .lerp(end_transform.translation, 0.5);

                let path_dir =
                    (end_transform.translation - start_transform.translation).with_y(0.0);
                let path_length = path_dir.length();

                let avoidance_vec = (mid_point - *bend_away_from).with_y(0.0);

                let perpendicular_dir = if avoidance_vec.length_squared() < 1e-6 {
                    Vec3::new(-path_dir.z, 0.0, path_dir.x).normalize()
                } else {
                    let avoidance_dir = avoidance_vec.normalize();
                    let projected = avoidance_dir
                        - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                    projected.normalize()
                };

                let effective_radius = radius.min(path_length * 0.5);
                let control_point = mid_point + perpendicular_dir * effective_radius;

                let t = progress;
                let pos = (1.0 - t).powi(2) * start_transform.translation
                    + 2.0 * (1.0 - t) * t * control_point
                    + t.powi(2) * end_transform.translation;

                let rot = start_transform
                    .rotation
                    .slerp(end_transform.rotation, progress);

                Transform::from_translation(pos).mul_transform(Transform::from_rotation(rot))
            }
            TrajectorySampler::CubicBSpline { control_points } => {
                if control_points.is_empty() {
                    return Transform::default();
                }

                let mut transforms = Vec::with_capacity(control_points.len());
                for sampler in control_points.iter_mut() {
                    transforms.push(sampler.sample_cache());
                }

                let fallback = transforms.first().cloned().unwrap_or_default();
                if transforms.len() < 4 {
                    return fallback;
                }

                let mut translations = Vec::with_capacity(transforms.len());
                for transform in &transforms {
                    translations.push(transform.translation);
                }
                let translation_curve = match CubicBSpline::new(translations).to_curve() {
                    Ok(curve) => curve,
                    Err(_) => return fallback,
                };

                let mut quaternion_points = Vec::with_capacity(transforms.len());
                let mut previous = None;
                for transform in &transforms {
                    let quat = transform.rotation.normalize();
                    let mut vec = Vec4::new(quat.x, quat.y, quat.z, quat.w);
                    if let Some(prev_vec) = previous {
                        if vec.dot(prev_vec) < 0.0 {
                            vec = -vec;
                        }
                    }
                    previous = Some(vec);
                    quaternion_points.push(vec);
                }

                let t = progress.clamp(0.0, 1.0);
                let segment_count = translation_curve.segments().len().max(1) as f32;
                let curve_t = t * segment_count;
                let translation = translation_curve.position(curve_t);

                let rotation_curve = match CubicBSpline::new(quaternion_points).to_curve() {
                    Ok(curve) => curve,
                    Err(_) => {
                        let mut output = fallback;
                        output.translation = translation;
                        return output;
                    }
                };

                let mut quat_vec = rotation_curve.position(curve_t);
                if quat_vec.length_squared() == 0.0 {
                    quat_vec = Vec4::new(0.0, 0.0, 0.0, 1.0);
                } else {
                    quat_vec = quat_vec.normalize();
                }

                let rotation =
                    Quat::from_xyzw(quat_vec.x, quat_vec.y, quat_vec.z, quat_vec.w).normalize();

                let mut output = fallback;
                output.translation = translation;
                output.rotation = rotation;
                output
            }
        }
    }

    pub fn draw_gizmos(
        &mut self,
        gizmos: &mut Gizmos<EditorCameraGizmoConfigGroup>,
        transform: Transform,
        color: Color,
    ) {
        match self {
            TrajectorySampler::CubicBezier { .. } | TrajectorySampler::WaypointPath { .. } => {
                let mut last = transform.transform_point(self.sample(0.0).translation);
                for i in 1..=32 {
                    let next = transform.transform_point(self.sample(i as f32 / 32.0).translation);
                    gizmos.line(last, next, color);
                    last = next;
                }
            }
            TrajectorySampler::Linear { start, end } => {
                let start = transform.transform_point(start.sample_cache().translation);
                let end = transform.transform_point(end.sample_cache().translation);
                gizmos.line(start, end, color);
            }
            TrajectorySampler::Avoidant {
                start,
                end,
                bend_away_from,
                radius,
            } => {
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();
                let mid_point = start_transform
                    .translation
                    .lerp(end_transform.translation, 0.5);

                let path_dir = end_transform.translation - start_transform.translation;
                let path_length = path_dir.length();

                let avoidance_vec = mid_point - *bend_away_from;

                let perpendicular_dir = if avoidance_vec.length_squared() < 1e-6 {
                    path_dir.any_orthonormal_vector().normalize()
                } else {
                    let avoidance_dir = avoidance_vec.normalize();
                    let projected = avoidance_dir
                        - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                    projected.normalize()
                };

                let effective_radius = radius.min(path_length * 0.5);
                let control_point = mid_point + perpendicular_dir * effective_radius;

                let curve_length = start_transform.translation.distance(control_point)
                    + control_point.distance(end_transform.translation);
                let segments = (curve_length * 10.0).ceil().max(10.0) as usize;

                let step = 1.0 / segments as f32;
                let mut prev_point = transform.transform_point(start_transform.translation);
                for i in 1..=segments {
                    let t = step * i as f32;
                    let point = (1.0 - t).powi(2) * start_transform.translation
                        + 2.0 * (1.0 - t) * t * control_point
                        + t.powi(2) * end_transform.translation;
                    let transformed_point = transform.transform_point(point);
                    gizmos.line(prev_point, transformed_point, color);
                    prev_point = transformed_point;
                }
            }
            TrajectorySampler::AvoidantXZ {
                start,
                end,
                bend_away_from,
                radius,
            } => {
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();
                let mid_point = start_transform
                    .translation
                    .lerp(end_transform.translation, 0.5);

                let path_dir =
                    (end_transform.translation - start_transform.translation).with_y(0.0);
                let path_length = path_dir.length();

                let avoidance_vec = (mid_point - *bend_away_from).with_y(0.0);

                let perpendicular_dir = if avoidance_vec.length_squared() < 1e-6 {
                    Vec3::new(-path_dir.z, 0.0, path_dir.x).normalize()
                } else {
                    let avoidance_dir = avoidance_vec.normalize();
                    let projected = avoidance_dir
                        - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                    projected.normalize()
                };

                let effective_radius = radius.min(path_length * 0.5);
                let control_point =
                    (mid_point + perpendicular_dir * effective_radius).with_y(mid_point.y);

                let curve_length = start_transform.translation.distance(control_point)
                    + control_point.distance(end_transform.translation);
                let segments = (curve_length * 10.0).ceil().max(10.0) as usize;

                let step = 1.0 / segments as f32;
                let mut prev_point = transform.transform_point(start_transform.translation);
                for i in 1..=segments {
                    let t = step * i as f32;
                    let point = (1.0 - t).powi(2) * start_transform.translation
                        + 2.0 * (1.0 - t) * t * control_point
                        + t.powi(2) * end_transform.translation;
                    let transformed_point = transform.transform_point(point);
                    gizmos.line(prev_point, transformed_point, color);
                    prev_point = transformed_point;
                }
            }
            TrajectorySampler::CubicBSpline { control_points } => {
                if control_points.is_empty() {
                    return;
                }

                let mut transforms = Vec::with_capacity(control_points.len());
                for sampler in control_points.iter_mut() {
                    transforms.push(sampler.sample_cache());
                }

                let control_color = Color::srgb(1.0, 1.0, 0.0);
                for transform_entry in &transforms {
                    let point = transform.transform_point(transform_entry.translation);
                    gizmos.sphere(point, 0.05, control_color);
                }

                if transforms.len() < 4 {
                    let mut prev = None;
                    for transform_entry in &transforms {
                        let point = transform.transform_point(transform_entry.translation);
                        if let Some(prev_point) = prev {
                            gizmos.line(prev_point, point, color);
                        }
                        prev = Some(point);
                    }
                    return;
                }

                let mut translations = Vec::with_capacity(transforms.len());
                for transform_entry in &transforms {
                    translations.push(transform_entry.translation);
                }

                if let Ok(curve) = CubicBSpline::new(translations).to_curve() {
                    let segments = curve.segments().len();
                    let subdivisions = (segments * 16).max(16);

                    let mut prev = None;
                    for position in curve.iter_positions(subdivisions) {
                        let point = transform.transform_point(position);
                        if let Some(prev_point) = prev {
                            gizmos.line(prev_point, point, color);
                        }
                        prev = Some(point);
                    }
                }
            }
            _ => {}
        }
    }
}
