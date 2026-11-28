//! Optional basketball stations in a volleyball-configured multipurpose hall.
use super::*;
use crate::scene::procedural_indoor::objects::{public::basketball, wall};

pub(in crate::scene::procedural_indoor::layout::functional) fn goal_access(
    o: &IndoorObject,
) -> Reservation {
    let forward = (o.transform().rotation * Vec3::Z).xz().round();
    let back = o.position.xz() - forward * (o.size.z * 0.5 - 0.06);
    let depth = o.size.z + 2.35;
    let mid = back + forward * (0.5 * depth);
    let half = forward.abs() * (0.5 * depth) + forward.perp().abs() * (0.5 * o.size.x + 0.32);
    Reservation {
        min: mid - half,
        max: mid + half,
        role: ReservationRole::EquipmentAccess,
    }
}

pub(in crate::scene::procedural_indoor::layout::functional) fn valid_goal(
    scene: &IndoorManifest,
    o: &IndoorObject,
    court: VolleyballCourt,
) -> bool {
    let (lo, hi) = o.bounds();
    let (min, max) = court.protected_bounds();
    let rim = o.position.y + basketball::BasketballGoal::for_object(o).rim_height();
    let eye = (o.transform().rotation * Vec3::Z).xz();
    let approach = goal_access(o);
    !o.neighbor
        && !o.solid
        && o.support.is_none()
        && (2.57..=3.051).contains(&rim)
        && eye.abs().max_element() > 0.99999
        && eye.dot(court.center - o.position.xz()) > 0.
        && (lo.xz().cmpge(max + 0.02).any() || hi.xz().cmple(min - 0.02).any())
        && !scene.entry_approach_overlaps(
            approach.min.extend(0.).xzy(),
            approach.max.extend(rim).xzy(),
        )
        && scene.envelope.as_ref().is_some_and(|e| {
            crate::scene::procedural_indoor::envelope::polygon::box_inside(
                &e.footprint,
                approach.min,
                approach.max,
                0.02,
            )
        })
        && scene.wall_attachment_clear(o)
}

/// Wall frame and usable projection behind the volleyball free zone. Both
/// ends of a paired installation order the same bracket length.
fn wall_for(
    scene: &IndoorManifest,
    court: VolleyballCourt,
    normal: Vec3,
) -> Option<(Transform, f32)> {
    let e = scene.envelope.as_ref()?;
    let (min, max) = court.protected_bounds();
    e.walls.iter().find_map(|w| {
        // The entry partition has separate glazing, door leaves and circulation.
        if e.shared_edge(w.edge, scene.room_size) {
            return None;
        }
        let tf = e.wall_transform(w.edge);
        if (tf.rotation * Vec3::Z).dot(normal) < 0.99999 {
            return None;
        }
        let wall = tf.translation.xz();
        let gap = (min - wall)
            .dot(normal.xz())
            .min((max - wall).dot(normal.xz()))
            - 0.04;
        (gap >= basketball::MIN_PROJECTION).then_some((tf, gap))
    })
}

fn station_at(
    scene: &IndoorManifest,
    court: VolleyballCourt,
    prototype: &IndoorObject,
    tf: Transform,
    aim: Vec3,
    depth: f32,
    obstacles: &[(Vec3, Vec3)],
) -> Option<IndoorObject> {
    let mut o = prototype.clone();
    o.size.z = depth;
    o.yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
    let x = (tf.rotation.inverse() * (aim - tf.translation)).x;
    o.position = tf.transform_point(Vec3::new(x, prototype.position.y, wall::mount_offset(&o)?));
    if !valid_goal(scene, &o, court) {
        return None;
    }
    let (lo, hi) = o.bounds();
    let access = goal_access(&o);
    let blocked = |(a, b): (Vec3, Vec3)| {
        (lo.cmplt(b + 0.035).all() && hi.cmpgt(a - 0.035).all())
            || (a.y < hi.y && b.y > 0.03 && access.overlaps(a, b))
    };
    if obstacles.iter().copied().any(blocked)
        || scene
            .objects
            .iter()
            .filter(|other| !other.neighbor)
            .flat_map(IndoorObject::obstacle_bounds)
            .any(blocked)
    {
        return None;
    }
    Some(o)
}

impl IndoorManifest {
    pub(super) fn install_practice_goals(&mut self, c: VolleyballCourt) {
        if self.generator_version < basketball::VERSION {
            return;
        }
        let mut rng = stream(self.seed, u64::from_le_bytes(*b"GymHoops"));
        if !rng.random_bool(0.70) {
            return;
        }
        let kit: u64 = rng.random();
        let size = geometry::dimensions(ObjectKind::BasketballGoal, kit);
        let height = if rng.random_bool(0.76) {
            3.05
        } else {
            rng.random_range(2.58..2.89)
        };
        let long_first = rng.random_bool(0.66);
        let sign = if rng.random_bool(0.5) { 1. } else { -1. };
        let wanted = 1 + usize::from(rng.random_bool(0.70));
        let ends = std::array::from_fn::<_, 2, _>(|_| {
            let mut o = self.candidate(ObjectKind::BasketballGoal, Vec3::ZERO, size, 0., &mut rng);
            o.solid = false;
            o.finish_seed = Some(kit);
            o.position.y = height - basketball::BasketballGoal::for_object(&o).rim_height();
            o
        });
        let obstacles = self.camera_obstacles();
        let mut best = Vec::with_capacity(wanted);
        'axes: for along in [long_first, !long_first] {
            let axis = if along == c.long_x { Vec3::X } else { Vec3::Z };
            let tangent = Vec3::Y.cross(axis);
            let span = if along { c.width } else { c.length };
            let walls = [sign, -sign].map(|k| wall_for(self, c, axis * k));
            let available = walls.iter().flatten().count();
            if available <= best.len() {
                continue;
            }
            let limit = walls
                .iter()
                .flatten()
                .map(|(_, gap)| *gap)
                .fold(basketball::MAX_PROJECTION, f32::min);
            // Sample within the feasible interval without piling samples against
            // its boundary or shrinking the ring, board or approach.
            let q = (size.z - basketball::MIN_PROJECTION)
                / (basketball::MAX_PROJECTION - basketball::MIN_PROJECTION);
            let depth = basketball::MIN_PROJECTION + q * (limit - basketball::MIN_PROJECTION);
            let hand = if rng.random_bool(0.5) { 1. } else { -1. };
            let offsets = [
                rng.random_range(-0.15..0.15),
                hand * rng.random_range(0.19..0.42),
                -hand * rng.random_range(0.19..0.42),
            ];
            for shift in offsets {
                let aim = Vec3::new(c.center.x, 0., c.center.y) + tangent * (shift * span);
                let mut found = Vec::with_capacity(wanted);
                for (end, wall) in ends.iter().zip(walls) {
                    let Some((tf, _)) = wall else {
                        continue;
                    };
                    if let Some(o) = station_at(self, c, end, tf, aim, depth, &obstacles) {
                        found.push(o);
                        if found.len() == wanted {
                            best = found;
                            break 'axes;
                        }
                    }
                }
                if found.len() > best.len() {
                    best = found;
                }
                if best.len() == available {
                    break;
                }
            }
        }
        for mut o in best {
            o.id = self.objects.len();
            self.assembly_diagnostics.programs[0]
                .reservations
                .push(goal_access(&o));
            self.objects.push(o);
        }
    }
}
