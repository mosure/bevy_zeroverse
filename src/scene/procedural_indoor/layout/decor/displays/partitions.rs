//! Faces of opaque interior walls adjoining the selected viewing area. The
//! original span remains the collision host even if only part belongs to the zone.
use super::*;

pub(super) struct PartitionFace {
    pub(super) index: usize,
    pub(super) tf: Transform,
    pub(super) span: f32,
    pub(super) bounds: Bounds,
}

impl PartitionFace {
    pub(super) fn supports(&self, scene: &IndoorManifest, o: &IndoorObject) -> bool {
        let local = self.tf.rotation.inverse() * (o.position - self.tf.translation);
        let offset = crate::scene::procedural_indoor::objects::wall::mount_offset(o);
        offset.is_some_and(|z| (local.z - z).abs() < 0.001)
            && (self.tf.rotation * Vec3::Z).dot(o.transform().rotation * Vec3::Z) > 0.99999
            && scene.attachment_clear(self.tf, self.span, o)
    }
}

impl IndoorManifest {
    pub(super) fn partition_faces(&self) -> impl Iterator<Item = PartitionFace> + '_ {
        let (min, max) = self.primary_room_bounds();
        self.program
            .iter()
            .flat_map(|p| p.partitions.iter().enumerate())
            .filter(|(_, p)| p.axis < 2 && p.glazing_fraction == 0.)
            .flat_map(move |(index, p)| {
                let axis = p.axis;
                let along = 1 - axis;
                let normal = if axis == 0 { Vec3::X } else { Vec3::Z };
                p.spans().into_iter().flat_map(move |(lo, hi)| {
                    let bounds = p.span_obstacle(lo, hi, self.room_size.y);
                    // Space for fitted T-junctions and jambs on both ends.
                    let start = (lo + 0.12).max(min[along]);
                    let end = (hi - 0.12).min(max[along]);
                    [-1., 1.].into_iter().filter_map(move |sign| {
                        let inside = p.coordinate + sign * (p.thickness * 0.5 + 0.06);
                        if start >= end || inside <= min[axis] || inside >= max[axis] {
                            return None;
                        }
                        let n = normal * sign;
                        let center = p.position((start + end) * 0.5, 0.) + n * (p.thickness * 0.5);
                        Some(PartitionFace {
                            index,
                            bounds,
                            span: end - start,
                            tf: Transform::from_translation(center)
                                .with_rotation(Quat::from_rotation_y(n.x.atan2(n.z))),
                        })
                    })
                })
            })
    }
}
