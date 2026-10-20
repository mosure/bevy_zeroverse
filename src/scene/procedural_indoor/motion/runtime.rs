use super::*;
use bevy::camera::primitives::Aabb;

pub struct IndoorMotionPlugin;
impl Plugin for IndoorMotionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<crate::sample::CaptureFailure>()
            .add_systems(
                Update,
                (
                    animate.after(crate::camera::update_camera_trajectory),
                    bounds,
                )
                    .chain()
                    .run_if(any_with_component::<AnimatedPart>),
            );
    }
}
#[derive(Component)]
pub(crate) struct AnimatedBounds;
#[derive(Component)]
pub(crate) struct AnimatedPart {
    joint: Joint,
    morph: Option<(Geometry, Geometry)>,
    last: Option<f32>,
}
impl AnimatedPart {
    pub(crate) fn new(joint: &Joint, key: &(Surface, String), geometry: &Geometry) -> Self {
        let sampler = Joint {
            kind: joint.kind,
            interval: joint.interval,
            pivot: joint.pivot,
            frame: joint.frame,
            axis: joint.axis,
            angle: joint.angle,
            translation: joint.translation,
            parts: vec![],
            morph: default(),
            accepted: true,
            coupled: joint.coupled,
            attachment: None,
            adjustment: joint.adjustment,
        };
        Self {
            joint: sampler,
            morph: joint
                .morph
                .get(key)
                .map(|end| (geometry.clone(), end.clone())),
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
    let t = time.mode.map_progress(time.progress);
    for (mut part, mesh, mut transform, mut aabb) in &mut parts {
        let amount = part.joint.amount(t);
        if part.last == Some(amount) {
            continue;
        }
        if let Some((start, end)) = &part.morph {
            let Some(mut mesh) = meshes.get_mut(&mesh.0) else {
                continue;
            };
            let p: Vec<_> = start
                .positions
                .iter()
                .zip(&end.positions)
                .map(|(a, b)| {
                    Vec3::from_array(*a)
                        .lerp(Vec3::from_array(*b), amount * part.joint.adjustment.morph)
                        .to_array()
                })
                .collect();
            let (lo, hi) = p.iter().map(|p| Vec3::from_array(*p)).fold(
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                |(a, b), p| (a.min(p), b.max(p)),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, p);
            // Actual deformed surface frames: no stale normals or tangents when
            // folding cloth. Indices, entity/mesh IDs and metric UVs stay fixed.
            mesh.compute_smooth_normals();
            // Rounded mechanical parts contain collapsed bevel triangles with
            // analytic normals. Preserve stationary vertices and use the
            // validated endpoint frame when a triangulated normal vanishes.
            if let Some(bevy::mesh::VertexAttributeValues::Float32x3(normals)) =
                mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
            {
                for (i, n) in normals.iter_mut().enumerate() {
                    let normal = Vec3::from_array(*n);
                    if start.positions[i] == end.positions[i] {
                        *n = start.normals[i];
                    } else if !normal.is_finite() || normal.length_squared() < 1e-8 {
                        *n = Vec3::from_array(start.normals[i])
                            .lerp(
                                Vec3::from_array(end.normals[i]),
                                amount * part.joint.adjustment.morph,
                            )
                            .normalize_or(Vec3::from_array(start.normals[i]))
                            .to_array();
                    }
                }
            }
            if mesh.attribute(Mesh::ATTRIBUTE_TANGENT).is_some() {
                if let Err(error) = mesh.generate_tangents() {
                    failure.0 = Some(format!(
                        "articulated cloth tangent generation failed: {error}"
                    ));
                    return;
                }
                super::super::geometry::repair_tangent_frames(&mut mesh);
            }
            *aabb = Aabb::from_min_max(lo, hi);
        } else {
            *transform = part.joint.transform(amount);
        }
        part.last = Some(amount);
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
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<crate::camera::Playback>()
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
    }
}
