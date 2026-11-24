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
// Preserve the first draw for existing wall installations and expose it to
// portable equipment without sampling the rest of the installation.
fn standard_and_stream(seed: u64) -> (u32, ChaCha8Rng) {
    let mut r = stream(seed, 0x454c454354524943);
    (r.random_range(0..3), r)
}
pub(in super::super) fn electrical_standard(seed: u64) -> u32 {
    standard_and_stream(seed).0
}
pub fn electrical_parameters(scene: &IndoorManifest) -> ElectricalProgram {
    let (standard, mut r) = standard_and_stream(scene.seed);
    ElectricalProgram {
        standard,
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
    pub(in super::super) fn wall_hardware(&mut self) {
        let mut count_rng = sampling::Domain::Electrical.rng(self.seed);
        let electrical = electrical_parameters(self);
        let worktops = worktop_anchors(self);
        let mut occupied: Vec<_> = self
            .objects
            .iter()
            .map(IndoorObject::bounds)
            .chain(self.columns())
            .collect();
        let count = ((self.room_size.x + self.room_size.z) * count_rng.random_range(0.65..0.95))
            .round() as usize;
        for i in 0..count.clamp(10, 32) {
            let mut local = sampling::Domain::Electrical.proposal(self.seed, i);
            let rng = &mut local;
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
            if switch && self.generator_version >= wall::switches::VERSION {
                wall::switches::install(&mut obj, self.seed, electrical.standard);
            } else {
                obj.variant = if switch {
                    electrical.standard
                } else {
                    wall::installed_variant(electrical.standard)
                };
            }
            // One circuit may be operated on entry. Other zone controls and
            // service sockets keep independent unconstrained wall proposals.
            if let Some(entry) = (i == 0)
                .then(|| self.fit_entry_switch(&obj, height - floor, &occupied))
                .flatten()
            {
                obj = entry;
            } else if !self.attachment_clear(tf, span, &obj)
                || !installation_clear(obj.bounds(), &occupied)
            {
                continue;
            }
            occupied.push(obj.bounds());
            self.objects.push(obj);
        }
    }
}

pub(super) fn installation_clear((lo, hi): (Vec3, Vec3), occupied: &[(Vec3, Vec3)]) -> bool {
    let gap = Vec3::splat(equipment::services::INSTALLATION_GAP);
    occupied
        .iter()
        .all(|&(a, b)| !(lo.cmplt(b + gap).all() && hi.cmpgt(a - gap).all()))
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
