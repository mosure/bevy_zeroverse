//! A bag set down beside a seat keeps floor contact and the shared aisle reservations.
use super::*;
use crate::scene::procedural_indoor::{objects::bags::BESIDE_SEAT, program::Zone};
use std::f32::consts::PI;

impl IndoorManifest {
    pub(super) fn place_floor_bag(
        &mut self,
        zone: &Zone,
        size: Vec3,
        edge: Vec3,
        edge_yaw: f32,
        rng: &mut ChaCha8Rng,
    ) {
        // Allocate the product once. Rejecting a side must not resample its shape or
        // consume the next piece of furniture's construction/placement stream.
        let mut bag = self.floor_candidate(ObjectKind::Bag, edge, size, edge_yaw, rng);
        let original = bag.position;
        let mut r = stream(bag.seed, 0x4241_4750_4c41_4345);
        if r.random_bool(0.67) {
            let zone_contains = |p: Vec2| p.cmpge(zone.min).all() && p.cmple(zone.max).all();
            let seat = self
                .objects
                .iter()
                .filter(|o| {
                    !o.neighbor
                        && o.support.is_none()
                        && o.solid
                        && matches!(
                            o.kind,
                            ObjectKind::Chair | ObjectKind::Sofa | ObjectKind::Bench
                        )
                        && zone_contains(o.position.xz())
                        && (o.position.y - original.y).abs() < 0.04
                })
                .min_by(|a, b| {
                    a.position
                        .distance_squared(original)
                        .total_cmp(&b.position.distance_squared(original))
                });
            if let Some(seat) = seat {
                let side = if r.random_bool(0.5) { 1. } else { -1. };
                let yaw = seat.yaw
                    + r.random_range(-1.50..1.50)
                    + if r.random_bool(0.5) { PI } else { 0. };
                let half = super::super::super::placement::rotated_half(size.xz(), yaw - seat.yaw);
                let across = seat.size.x * 0.5 + half.x + r.random_range(0.070..0.22);
                let along = seat.size.z * r.random_range(-0.36..0.36);
                bag.yaw = yaw;
                for side in [side, -side] {
                    let site =
                        seat.transform()
                            .transform_point(Vec3::new(side * across, 0., along));
                    bag.position = site.with_y(self.floor_height(site.xz()));
                    let (lo, hi) = bag.bounds();
                    if (bag.position.y - seat.position.y).abs() < 0.04
                        && zone_contains(lo.xz())
                        && zone_contains(hi.xz())
                        && self.placement_clear(&bag, 0.06)
                    {
                        bag.variant |= BESIDE_SEAT;
                        self.objects.push(bag);
                        return;
                    }
                }
            }
        }
        // A set-down bag can follow a nearby wall or face freely into the room.
        bag.position = original;
        bag.yaw = if r.random_bool(0.38) {
            edge_yaw
        } else {
            r.random_range(-PI..PI)
        };
        if !self.placement_clear(&bag, 0.06) {
            if bag.yaw == edge_yaw {
                self.rejected_placements += 1;
                return;
            }
            bag.yaw = edge_yaw;
            if !self.placement_clear(&bag, 0.06) {
                self.rejected_placements += 1;
                return;
            }
        }
        self.objects.push(bag);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::validation::validate_layout;

    #[test]
    fn settled_rooms_keep_loose_bags_on_the_floor_and_clear_of_furniture_and_exits() {
        let mut beside = 0;
        let mut ordinary = 0;
        let mut free_yaw = 0;
        for seed in 400..560 {
            let room =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25)
                    .unwrap();
            validate_layout(&room).unwrap();
            for bag in room
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::Bag && !o.neighbor && o.support.is_none())
            {
                let (lo, hi) = bag.bounds();
                assert!((bag.position.y - room.floor_height(bag.position.xz())).abs() < 0.002);
                assert!(room.floor_support_clear(lo, hi));
                assert!(!room.entry_approach_overlaps(lo, hi));
                // Chairs can move toward their tables after the bag was set down.
                assert!(room.placement_clear(bag, 0.), "room {seed}, bag {}", bag.id);
                let angle = bag.yaw.rem_euclid(PI * 0.5);
                free_yaw += usize::from(angle.min(PI * 0.5 - angle) > 0.21);
                if bag.variant & BESIDE_SEAT == 0 {
                    ordinary += 1;
                } else {
                    beside += 1;
                }
            }
        }
        assert!(beside > 3 && ordinary > 3 && free_yaw > 3);
    }

    #[test]
    fn admitted_bags_use_both_sides_of_a_seat_and_reserve_the_rotated_footprint() {
        use crate::scene::procedural_indoor::placement::rotated_half;

        let mut sides = [0; 2];
        for seed in 400..432 {
            let mut room =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25)
                    .unwrap();
            let Some(program) = &room.program else {
                continue;
            };
            let mut rng = stream(seed, 824);
            for zone in program.zones.clone() {
                let seats: Vec<_> = room
                    .objects
                    .iter()
                    .filter(|o| {
                        !o.neighbor
                            && o.solid
                            && o.support.is_none()
                            && matches!(
                                o.kind,
                                ObjectKind::Chair | ObjectKind::Sofa | ObjectKind::Bench
                            )
                            && o.position.xz().cmpge(zone.min).all()
                            && o.position.xz().cmple(zone.max).all()
                    })
                    .cloned()
                    .collect();
                for seat in seats {
                    let count = room.objects.len();
                    // Start at a known seat, with its occupied centre blocking the
                    // ordinary floor fallback. Inspect each admission before any seat moves.
                    room.place_floor_bag(
                        &zone,
                        Vec3::new(0.43, 0.33, 0.19),
                        seat.position,
                        seat.yaw,
                        &mut rng,
                    );
                    if room.objects.len() == count {
                        continue;
                    }
                    let bag = room.objects.pop().unwrap();
                    assert_ne!(bag.variant & BESIDE_SEAT, 0);
                    let p = seat
                        .transform()
                        .compute_affine()
                        .inverse()
                        .transform_point3(bag.position);
                    let half = rotated_half(bag.size.xz(), bag.yaw - seat.yaw);
                    let gap = p.x.abs() - seat.size.x * 0.5 - half.x;
                    assert!((0.06999..0.22001).contains(&gap), "room {seed}, {gap}");
                    assert!(p.z.abs() <= seat.size.z * 0.36 + 1e-5);
                    assert!(room.placement_clear(&bag, 0.06));
                    sides[usize::from(p.x > 0.)] += 1;
                }
            }
        }
        assert!(sides.iter().all(|&n| n > 5), "{sides:?}");
    }
}
