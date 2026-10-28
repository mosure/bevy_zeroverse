//! Room installation practices and relational worktop socket proposals.
use super::*;
use crate::scene::procedural_indoor::objects::wall;

/// Installation practice is coherent within a room, but not a dataset-wide
/// constant. Small per-instance tolerances remain in the admitted proposals.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ElectricalProgram {
    pub standard: u32,
    pub switches: usize,
    pub outlet_height: f32,
    pub switch_height: f32,
    pub pitch: f32,
    pub plate_height: f32,
    pub plate_depth: f32,
    pub worktop_probability: f32,
}
pub fn electrical_parameters(scene: &IndoorManifest) -> ElectricalProgram {
    let mut r = stream(scene.seed, 0x454c454354524943);
    ElectricalProgram {
        standard: r.random_range(0..3),
        switches: r.random_range(1..=4),
        outlet_height: r.random_range(0.19..0.62),
        switch_height: r.random_range(0.96..1.43),
        pitch: r.random_range(0.065..0.090),
        plate_height: r.random_range(0.095..0.137),
        plate_depth: r.random_range(0.016..0.026),
        worktop_probability: r.random_range(0.25..0.85),
    }
}
impl IndoorManifest {
    pub(in super::super) fn wall_hardware(&mut self, rng: &mut ChaCha8Rng) {
        let electrical = electrical_parameters(self);
        let worktops = worktop_anchors(self);
        let count =
            ((self.room_size.x + self.room_size.z) * rng.random_range(0.65..0.95)).round() as usize;
        for i in 0..count.clamp(10, 32) {
            let switch = i < electrical.switches;
            let kind = if switch {
                ObjectKind::LightSwitch
            } else {
                ObjectKind::WallOutlet
            };
            let gangs = rng.random_range(1..=3);
            let size = Vec3::new(
                electrical.pitch * gangs as f32 * rng.random_range(0.96..1.04),
                electrical.plate_height * rng.random_range(0.96..1.04),
                electrical.plate_depth,
            );
            let raised = !switch
                && !worktops.is_empty()
                && rng.random_bool(electrical.worktop_probability as f64);
            let (tf, span, along, worktop) = if raised {
                let a = &worktops[rng.random_range(0..worktops.len())];
                (
                    a.wall,
                    a.span,
                    rng.random_range(a.along.0..a.along.1),
                    Some(a),
                )
            } else {
                let (tf, span) = self.attachment_wall(rng);
                (tf, span, rng.random_range(-0.40..0.40) * span, None)
            };
            let anchor = tf.transform_point(Vec3::new(along, 0., 0.));
            let floor = self.floor_height(anchor.xz());
            let mut height = floor
                + if switch {
                    electrical.switch_height
                } else {
                    electrical.outlet_height
                };
            if let Some(worktop) = worktop {
                let o = &self.objects[worktop.support];
                let q = o
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(anchor)
                    .xz();
                let half = o.size.xz() * 0.5;
                if q.distance(q.clamp(-half, half)) > 0.45 {
                    continue;
                }
                height = worktop.top + rng.random_range(0.12..0.26);
            }
            height += rng.random_range(-0.025..0.025);
            let position = anchor + Vec3::Y * height;
            let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            let mut obj = self.candidate(kind, position, size, yaw, rng);
            obj.solid = false;
            obj.position += tf.rotation * Vec3::Z * wall::mount_offset(&obj).unwrap();
            // One electrical convention per scene; finishes/gang counts can vary.
            obj.variant = electrical.standard;
            let (lo, hi) = obj.bounds();
            if !self.attachment_clear(tf, span, &obj)
                || self
                    .objects
                    .iter()
                    .map(IndoorObject::bounds)
                    .chain(self.columns())
                    .any(|(a, b)| {
                        lo.cmplt(b + Vec3::splat(0.015)).all()
                            && hi.cmpgt(a - Vec3::splat(0.015)).all()
                    })
            {
                continue;
            }
            self.objects.push(obj);
        }
    }
}

struct WorktopAnchor {
    wall: Transform,
    span: f32,
    along: (f32, f32),
    top: f32,
    support: usize,
}

/// Project the true oriented furniture footprint into each wall's frame once.
/// This keeps relational placement bounded instead of adding rejection searches.
fn worktop_anchors(scene: &IndoorManifest) -> Vec<WorktopAnchor> {
    let walls: Vec<_> = if let Some(e) = &scene.envelope {
        e.walls
            .iter()
            .map(|w| {
                let a = e.footprint[w.edge];
                let b = e.footprint[(w.edge + 1) % e.footprint.len()];
                (e.wall_transform(w.edge), a.distance(b))
            })
            .collect()
    } else {
        FacadeSide::ALL
            .into_iter()
            .map(|s| (s.transform(scene.room_size), s.span(scene.room_size)))
            .collect()
    };
    let mut anchors = Vec::new();
    for (support, o) in scene.objects.iter().enumerate().filter(|(_, o)| {
        matches!(
            o.kind,
            ObjectKind::Desk
                | ObjectKind::KitchenCounter
                | ObjectKind::ServiceCounter
                | ObjectKind::Cabinet
        ) && o.support.is_none()
            && !o.neighbor
    }) {
        let top = o.position.y + o.size.y;
        let floor = scene.floor_height(o.position.xz());
        if !(floor + 0.55..floor + 1.4).contains(&top) {
            continue;
        }
        for &(wall, span) in &walls {
            let inverse = wall.compute_affine().inverse();
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for x in [-0.5, 0.5] {
                for z in [-0.5, 0.5] {
                    let q = inverse.transform_point3(o.transform().transform_point(Vec3::new(
                        x * o.size.x,
                        0.,
                        z * o.size.z,
                    )));
                    lo = lo.min(q);
                    hi = hi.max(q);
                }
            }
            // No socket is associated with distant furniture or a support on
            // the back side of a wall. Openings are checked during admission.
            let along = (
                (lo.x + 0.08).max(-span * 0.5 + 0.18),
                (hi.x - 0.08).min(span * 0.5 - 0.18),
            );
            if lo.z >= -0.02 && lo.z < 0.45 && along.1 - along.0 > 0.08 {
                anchors.push(WorktopAnchor {
                    wall,
                    span,
                    along,
                    top,
                    support,
                });
            }
        }
    }
    anchors
}
