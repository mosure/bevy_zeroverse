use super::*;
use bevy::{camera::primitives::Aabb, mesh::VertexAttributeValues};

pub struct IndoorMotionPlugin;
impl Plugin for IndoorMotionPlugin {
    fn build(&self, app: &mut App) {
        super::appearance::install(app);
        app.init_resource::<crate::sample::CaptureFailure>()
            .add_systems(
                Update,
                (
                    animate.after(crate::camera::update_camera_trajectory),
                    bounds,
                )
                    .chain()
                    .run_if(any_with_component::<AnimatedPart>),
            )
            .add_systems(
                Update,
                animate_lights
                    .after(crate::camera::update_camera_trajectory)
                    .run_if(any_with_component::<AnimatedLight>),
            );
    }
}
#[derive(Component)]
pub(crate) struct AnimatedBounds;
/// Direct emitters follow the same scene-local rigid delta as their mesh.
/// This also handles lamps carried on a cart without reparenting annotation roots.
#[derive(Component)]
pub(crate) struct AnimatedLight {
    joint: Joint,
    rest: Transform,
    last: f32,
}
impl AnimatedLight {
    pub(crate) fn for_object(
        scene: &super::super::layout::IndoorManifest,
        id: usize,
        rest: Transform,
    ) -> Option<Self> {
        let m = scene.indoor_motion.as_ref()?.mechanism_for_object(id)?;
        let mut joint =
            construction::rigid(m.kind, 0, m.pivot, m.axis, m.angle_radians, m.translation_m);
        joint.interval = m.interval;
        joint.operation_cycle = m.operation_cycle;
        joint.lift_m = m.lift_m;
        Some(Self {
            joint,
            rest,
            last: 0.,
        })
    }
}
fn animate_lights(
    time: Res<crate::camera::Playback>,
    mut lights: Query<(&mut AnimatedLight, &mut Transform)>,
) {
    let t = time.trajectory_progress();
    for (mut light, mut tf) in &mut lights {
        let amount = light.joint.amount(t);
        if amount != light.last {
            *tf = light.joint.transform(amount).mul_transform(light.rest);
            light.last = amount;
        }
    }
}

/// Topology and UVs live on the render mesh. Only retain the endpoint
/// attributes that can actually change during playback.
struct MorphVertices {
    positions: [Vec<[f32; 3]>; 2],
    normals: Option<[Vec<[f32; 3]>; 2]>,
}

#[derive(Component)]
pub(crate) struct AnimatedPart {
    joint: Joint,
    morph: Option<MorphVertices>,
    last: Option<f32>,
}
impl AnimatedPart {
    pub(crate) fn new(joint: &Joint, key: &(Surface, String), geometry: &Geometry) -> Self {
        let sampler = Joint {
            program_seed: joint.program_seed,
            operation_cycle: joint.operation_cycle,
            kind: joint.kind,
            interval: joint.interval,
            pivot: joint.pivot,
            frame: joint.frame,
            cloth_motion: joint.cloth_motion.filter(|law| law.applies(&key.1)),
            axis: joint.axis,
            angle: joint.angle,
            translation: joint.translation,
            lift_m: joint.lift_m,
            parts: vec![],
            morph: default(),
            accepted: true,
            coupled: joint.coupled,
            contact: joint.contact,
            attachment: None,
            adjustment: joint.adjustment,
        };
        let morph = joint.morph.get(key).map(|end| MorphVertices {
            positions: [geometry.positions.clone(), end.positions.clone()],
            normals: (!matches!(
                sampler.cloth_motion,
                Some(ClothMotion::ProjectionScreen { .. })
            ))
            .then(|| [geometry.normals.clone(), end.normals.clone()]),
        });
        Self {
            joint: sampler,
            morph,
            last: Some(0.),
        }
    }
}
fn animate(
    time: Res<crate::camera::Playback>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut failure: ResMut<crate::sample::CaptureFailure>,
    mut parts: Query<(&mut AnimatedPart, &Mesh3d, &mut Transform, &mut Aabb)>,
) {
    let t = time.trajectory_progress();
    for (mut part, mesh, mut transform, mut aabb) in &mut parts {
        let amount = part.joint.amount(t);
        if part.last == Some(amount) {
            continue;
        }
        if let Some(morph) = &part.morph {
            let [start, end] = &morph.positions;
            let Some(mut mesh) = meshes.get_mut(&mesh.0) else {
                continue;
            };
            let t = amount * part.joint.adjustment.morph;
            let Some(VertexAttributeValues::Float32x3(positions)) =
                mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
            else {
                failure.0 = Some("articulated cloth is missing its position buffer".into());
                return;
            };
            let law = part.joint.cloth_motion.map(|law| law.at(t));
            let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            if positions.len() != start.len() || positions.len() != end.len() {
                failure.0 = Some("articulated cloth position buffer changed vertex count".into());
                return;
            }
            for ((out, &a), &b) in positions.iter_mut().zip(start).zip(end) {
                let (a, b) = (Vec3::from_array(a), Vec3::from_array(b));
                let p = if t == 0. {
                    a
                } else if t == 1. {
                    b
                } else if let Some(sample) = law {
                    sample.interpolate(a, b, t)
                } else {
                    a.lerp(b, t)
                };
                *out = p.to_array();
                lo = lo.min(p);
                hi = hi.max(p);
            }
            *aabb = Aabb::from_min_max(lo, hi);
            // A flat screen only feeds along a chart axis. Every face retains
            // its analytic normal and tangent, including the narrow stock edges.
            let Some(normals) = &morph.normals else {
                part.last = Some(amount);
                continue;
            };
            update_normals(&mut mesh, &morph.positions, normals, t);
            if mesh.attribute(Mesh::ATTRIBUTE_TANGENT).is_some() {
                if let Err(error) = mesh.generate_tangents() {
                    failure.0 = Some(format!(
                        "articulated cloth tangent generation failed: {error}"
                    ));
                    return;
                }
                super::super::geometry::repair_tangent_frames(&mut mesh);
            }
        } else {
            *transform = part.joint.transform(amount);
        }
        part.last = Some(amount);
    }
}
/// Authored endpoint normals include smooth bevels and cloth creases that a
/// triangle average cannot recover. Intermediate folds still need a new frame.
fn update_normals(
    mesh: &mut Mesh,
    [start, end]: &[Vec<[f32; 3]>; 2],
    [a, b]: &[Vec<[f32; 3]>; 2],
    t: f32,
) {
    if t == 0. || t == 1. {
        let frame = if t == 0. { a } else { b };
        if let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
        {
            normals.copy_from_slice(frame);
        } else {
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, frame.clone());
        }
        return;
    }
    mesh.compute_smooth_normals();
    if let Some(VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        for (i, n) in normals.iter_mut().enumerate() {
            let normal = Vec3::from_array(*n);
            if start[i] == end[i] && a[i] == b[i] {
                *n = a[i];
            } else if !normal.is_finite() || normal.length_squared() < 1e-8 {
                *n = Vec3::from_array(a[i])
                    .lerp(Vec3::from_array(b[i]), t)
                    .normalize_or(Vec3::from_array(a[i]))
                    .to_array();
            }
        }
    }
}
fn bounds(
    mut roots: Query<(&Children, &mut Aabb), With<AnimatedBounds>>,
    parts: Query<(&Transform, &Aabb), Without<AnimatedBounds>>,
) {
    for (children, mut bb) in &mut roots {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for child in children {
            let Ok((tf, b)) = parts.get(*child) else {
                continue;
            };
            let center = tf.transform_point(b.center.into());
            let matrix = Mat3::from_quat(tf.rotation) * Mat3::from_diagonal(tf.scale);
            let h = Vec3::from(b.half_extents);
            let extent =
                matrix.x_axis.abs() * h.x + matrix.y_axis.abs() * h.y + matrix.z_axis.abs() * h.z;
            lo = lo.min(center - extent);
            hi = hi.max(center + extent);
        }
        if lo.is_finite() && hi.is_finite() {
            *bb = Aabb::from_min_max(lo, hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn playback_changes_real_mesh_transforms_bounds_and_preserves_correspondence() {
        use crate::camera::{Playback, PlaybackMode, TrajectorySampler, ZeroverseCamera};
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Playback>()
            .init_resource::<Time>()
            .init_resource::<crate::app::BevyZeroverseConfig>()
            .add_message::<crate::scene::RegenerateSceneEvent>()
            .add_message::<crate::scene::SceneLoadedEvent>()
            .add_systems(Update, crate::camera::update_camera_trajectory)
            .add_plugins(IndoorMotionPlugin);
        let mut g = Geometry::default();
        g.cuboid(Vec3::ONE, 0., Transform::IDENTITY);
        let key = (Surface::Wood, "door".into());
        let joint = construction::rigid(MotionKind::Door, 1, Vec3::ZERO, Vec3::Y, 0.8, Vec3::X);
        let animated = AnimatedPart::new(&joint, &key, &g);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(g.into_mesh());
        let root = app
            .world_mut()
            .spawn((
                AnimatedBounds,
                Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5)),
            ))
            .id();
        let part = app
            .world_mut()
            .spawn((
                animated,
                Mesh3d(handle.clone()),
                Transform::IDENTITY,
                Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5)),
                ChildOf(root),
            ))
            .id();
        for t in [0., 0.3, 1., 0.3, 0.] {
            app.world_mut()
                .resource_mut::<crate::camera::Playback>()
                .progress = t;
            app.update();
            let expected = joint.transform(joint.amount(t));
            assert_eq!(*app.world().get::<Transform>(part).unwrap(), expected);
            assert!(
                Vec3::from(app.world().get::<Aabb>(root).unwrap().center)
                    .distance(expected.translation)
                    < 1e-5
            );
            assert_eq!(app.world().get::<Mesh3d>(part).unwrap().0.id(), handle.id());
        }
        // Drive the actual camera and the moving mesh from the same clock,
        // including the return leg and the start of the next cycle.
        let camera = app
            .world_mut()
            .spawn((
                Transform::IDENTITY,
                ZeroverseCamera {
                    trajectory: TrajectorySampler::WaypointPath {
                        positions: vec![Vec3::ZERO, Vec3::X],
                        rotations: vec![Quat::IDENTITY; 2],
                    },
                    ..default()
                },
            ))
            .id();
        app.world_mut()
            .resource_mut::<Playback>()
            .set_mode(PlaybackMode::EaseInOutCubic);
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_millis(250));
        for t in [0.0625, 0.5, 0.9375, 1., 0.9375, 0.5, 0.0625, 0., 0.0625] {
            app.update();
            assert_eq!(
                app.world().get::<Transform>(camera).unwrap().translation,
                t * Vec3::X
            );
            assert_eq!(
                *app.world().get::<Transform>(part).unwrap(),
                joint.transform(joint.amount(t))
            );
        }
    }
}
