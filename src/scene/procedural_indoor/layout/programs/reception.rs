//! A staffed check-in station anchors a reception room to its entrance.
use super::*;
use crate::scene::procedural_indoor::objects::public::{self, reception::Counter};

pub(super) const INTRODUCED: u32 = 297;

impl IndoorManifest {
    fn reception_arrival(&self, min: Vec2, max: Vec2) -> Vec2 {
        let entry = Vec2::new(self.door_x, self.room_size.z * 0.5);
        let near_zone = |p: Vec2| p.distance_squared(p.clamp(min, max)) < 0.65 * 0.65;
        if near_zone(entry) {
            return entry;
        }
        self.program
            .iter()
            .flat_map(|p| &p.partitions)
            .map(|partition| partition.position(partition.door_center, 0.).xz())
            .filter(|p| near_zone(*p))
            .min_by(|a, b| {
                a.distance_squared(entry)
                    .total_cmp(&b.distance_squared(entry))
            })
            .unwrap_or_else(|| entry.clamp(min, max))
    }

    pub(super) fn furnish_reception(&mut self) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"Check-In"));
        let (min, max) = self.primary_room_bounds();
        if r.random_bool(0.16) || (max - min).element_product() < 14. {
            return;
        }
        let entry = self.reception_arrival(min, max);
        let structure = PlacementStructure::new();
        // After trying the full product, fit a compact commercial module to
        // constrained foyers. Standing space and working heights stay full sized.
        let initial = public::dimensions(ObjectKind::ReceptionDesk, r.random());
        let compact = Vec3::new(
            r.random_range(1.48..1.82),
            initial.y,
            r.random_range(0.78..0.87),
        );
        for size in [initial, compact] {
            let mut desk =
                self.floor_candidate(ObjectKind::ReceptionDesk, Vec3::ZERO, size, 0., &mut r);
            for trial in 0..24 {
                let fraction = Vec2::new(r.random_range(0.10..0.90), r.random_range(0.10..0.90));
                let aim = min + (max - min) * fraction;
                let direction = entry - aim;
                let heading = direction.x.atan2(direction.y);
                let yaw = if trial % 3 == 0 {
                    let quarter = std::f32::consts::FRAC_PI_2;
                    (heading / quarter).round() * quarter + r.random_range(-0.09..0.09)
                } else {
                    heading + r.random_range(-0.52..0.52)
                };
                desk.yaw = yaw;
                desk.position = Vec3::ZERO;
                let (a, b) = super::access::envelope(&desk).unwrap().bounds();
                // Sample feasible centres at this orientation. Never reserve a
                // neighbouring room or narrow a clearance strip to force a fit.
                let inset = Vec2::splat(0.07);
                let lo = min + inset - a.xz();
                let hi = max - inset - b.xz();
                if lo.cmpge(hi).any() {
                    continue;
                }
                let site = lo + (hi - lo) * fraction;
                let arrival = entry - site;
                if arrival.length_squared() < 2.25
                    || arrival.normalize().dot(Vec2::new(yaw.sin(), yaw.cos())) < 0.75
                {
                    continue;
                }
                desk.position = Vec3::new(site.x, self.floor_height(site), site.y);
                if !self.placement_clear_with_structure(&desk, 0.055, false, &structure) {
                    continue;
                }
                let id = desk.id;
                let tf = desk.transform();
                let millwork = Counter::sample(&desk);
                let x = (millwork.ledge(&desk).center().x * 0.55 + r.random_range(-0.12..0.12))
                    .clamp(-0.30, 0.30);
                let chair = tf.transform_point(Vec3::new(
                    x,
                    0.,
                    -size.z * 0.5 - r.random_range(0.46..0.63),
                ));
                let yaw = desk.yaw + std::f32::consts::PI;
                self.objects.push(desk);
                if r.random_bool(0.83) {
                    self.chair_in_family_at_surface(
                        chair,
                        yaw,
                        Some(r.random_range(0..2)),
                        Some(id),
                        &mut r,
                    );
                }
                return;
            }
        }
    }

    pub(in crate::scene::procedural_indoor::layout) fn dress_reception(
        &mut self,
        desk: &IndoorObject,
    ) {
        let p = Counter::sample(desk);
        let mut r = stream(desk.seed, u64::from_le_bytes(*b"Register"));
        let rear = p.worktop(desk);
        if r.random_bool(0.72) {
            let x = p.ledge(desk).center().x + r.random_range(-0.16..0.16);
            let width = r.random_range(0.30..0.43);
            let size = Vec3::new(
                width,
                r.random_range(0.19..0.28),
                width * r.random_range(0.64..0.83),
            );
            self.supported_prop(
                desk,
                ObjectKind::Laptop,
                Vec3::new(
                    x,
                    p.worktop_m - desk.size.y,
                    rear.center().y + r.random_range(-0.035..0.035),
                ),
                size,
                std::f32::consts::PI + r.random_range(-0.22..0.22),
                &mut r,
            );
        }
        if self.clutter.sample(desk.seed) > 0. && r.random_bool(0.69) {
            let lower = p.signing_wing(desk).filter(|_| r.random_bool(0.70));
            let (level, deck) =
                lower.map_or((desk.size.y, p.ledge(desk)), |wing| (p.worktop_m, wing));
            let size = Vec3::new(
                r.random_range(0.095..0.16),
                r.random_range(0.008..0.025),
                r.random_range(0.135..0.195),
            );
            let x = deck.center().x + deck.width() * r.random_range(-0.24..0.24);
            let z = if lower.is_some() {
                rear.max.y + 0.065
            } else {
                deck.center().y
            };
            self.supported_prop(
                desk,
                ObjectKind::Notepad,
                Vec3::new(x, level - desk.size.y, z),
                size,
                r.random_range(-0.34..0.34),
                &mut r,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{layout::SupportPlane, validation::validate_layout};
    #[test]
    fn reception_arrival_and_staff_access_survive_furnishing() {
        let mut counters = 0;
        let mut staffed = 0;
        let mut computers = 0;
        let mut visitors = 0;
        for seed in 0..96 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Reception, 0.65, 0, 0.35)
                    .unwrap();
            validate_layout(&scene).unwrap();
            if seed < 3 {
                assert_eq!(
                    scene,
                    IndoorManifest::generate_with_humans(
                        seed,
                        IndoorLayout::Reception,
                        0.65,
                        0,
                        0.35
                    )
                    .unwrap()
                );
            }
            let desks: Vec<_> = scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::ReceptionDesk)
                .collect();
            assert!(desks.len() <= 1);
            let Some(&desk) = desks.first() else { continue };
            counters += 1;
            let (lo, hi) = scene.primary_room_bounds();
            let front = (desk.transform().rotation * Vec3::Z).xz();
            let entry = scene.reception_arrival(lo, hi);
            assert!((entry - desk.position.xz()).normalize().dot(front) >= 0.7499);
            assert!(scene.placement_clear(desk, 0.));
            for obj in &scene.objects {
                if obj.id == desk.id {
                    continue;
                }
                if obj.kind == ObjectKind::Chair && obj.interaction_target == Some(desk.id) {
                    assert!((obj.position - desk.position).xz().dot(front) < -desk.size.z * 0.5);
                    staffed += 1;
                } else if obj.support == Some(desk.id) {
                    assert!(scene.prop_clear(obj, desk, 0.01));
                    if obj.kind == ObjectKind::Laptop {
                        computers += 1;
                        assert_eq!(scene.support_plane(obj), Some(SupportPlane::Worktop));
                    } else if obj.kind == ObjectKind::Notepad {
                        visitors += 1;
                        assert!(scene.support_plane(obj).is_some());
                    }
                } else if !obj.neighbor && obj.solid && obj.support.is_none() {
                    assert!(!super::super::access::blocks(desk, obj));
                }
            }
        }
        assert!((40..95).contains(&counters) && staffed>20 && computers>20 && visitors>8,
                "counters {counters}, chairs {staffed}, computers {computers}, visitor items {visitors}");
    }
}
