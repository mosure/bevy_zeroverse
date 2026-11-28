//! Park linen carts within reach of an installed station, at the edge of a
//! standing or circulation aisle. Parked carts never occupy the reserved route.
use super::*;
use crate::scene::procedural_indoor::{objects::public::laundry::cart, placement::rotated_half};
use rand::seq::SliceRandom;

fn host(activity: IndoorLayout, kind: ObjectKind) -> bool {
    match activity {
        IndoorLayout::Laundromat => matches!(
            kind,
            ObjectKind::WashingMachine | ObjectKind::TumbleDryer | ObjectKind::LaundryFoldingTable
        ),
        IndoorLayout::LockerRoom => matches!(kind, ObjectKind::Locker | ObjectKind::Bench),
        _ => false,
    }
}

fn approach(route: &Reservation, cart: &IndoorObject) -> bool {
    if !matches!(
        route.role,
        ReservationRole::EquipmentAccess
            | ReservationRole::DoorApproach
            | ReservationRole::Circulation
    ) {
        return false;
    }
    let standing = cart
        .transform()
        .transform_point(Vec3::Z * (cart.size.z * 0.5 + 0.46))
        .xz();
    let half = Vec2::splat(0.16);
    (standing - half).cmpge(route.min).all() && (standing + half).cmple(route.max).all()
}

fn nearby(cart: &IndoorObject, station: &IndoorObject) -> bool {
    let local = station.transform().rotation.inverse() * (cart.position - station.position);
    let gap = (local.xz().abs() - station.size.xz() * 0.5).max(Vec2::ZERO);
    gap.length_squared() <= 1.85 * 1.85
}

pub(in super::super) fn valid_site(
    scene: &IndoorManifest,
    zone: &ProgramAssembly,
    cart: &IndoorObject,
) -> bool {
    cart.solid
        && cart.support.is_none()
        && !cart.neighbor
        && cart
            .interaction_target
            .filter(|id| zone.members.contains(id))
            .and_then(|id| scene.objects.get(id))
            .is_some_and(|station| host(zone.activity, station.kind) && nearby(cart, station))
        && zone.reservations.iter().any(|r| approach(r, cart))
}

impl IndoorManifest {
    pub(crate) fn valid_laundry_cart(&self, o: &IndoorObject) -> bool {
        self.functional_programs()
            .iter()
            .any(|p| p.members.contains(&o.id) && valid_site(self, p, o))
    }

    pub(super) fn furnish_laundry_carts(&mut self) {
        if self.generator_version < cart::VERSION {
            return;
        }
        for index in 0..self.functional_programs().len() {
            let zone = &self.functional_programs()[index];
            let rate = match zone.activity {
                IndoorLayout::Laundromat => 0.89,
                IndoorLayout::LockerRoom => 0.36,
                _ => continue,
            };
            let mut r = stream(self.seed, u64::from_le_bytes(*b"CartPark") ^ index as u64);
            if !r.random_bool(rate * (0.8 + 0.2 * f64::from(self.density))) {
                continue;
            }
            let mut hosts: Vec<_> = zone
                .members
                .iter()
                .copied()
                .filter(|&id| host(zone.activity, self.objects[id].kind))
                .collect();
            if hosts.is_empty() {
                continue;
            }
            let count = 1 + usize::from(
                hosts.len() >= 4 && r.random_bool(0.18 + 0.25 * f64::from(self.density)),
            );
            let mut stock = r.random();
            let structure = PlacementStructure::new();
            for ordinal in 0..count {
                if ordinal > 0 && r.random_bool(0.23) {
                    stock = r.random();
                }
                hosts.shuffle(&mut r);
                let mut cart = self.floor_candidate(
                    ObjectKind::LaundryCart,
                    Vec3::ZERO,
                    crate::scene::procedural_indoor::objects::public::laundry::cart::dimensions(
                        stock,
                    ),
                    0.,
                    &mut r,
                );
                cart.finish_seed = Some(stock);
                if let Some(station) =
                    self.park_laundry_cart(index, &hosts, &mut cart, &structure, &mut r)
                {
                    self.assembly_diagnostics.programs[index]
                        .members
                        .push(cart.id);
                    self.objects.push(cart);
                    hosts.retain(|&id| id != station);
                }
            }
        }
    }

    fn park_laundry_cart(
        &self,
        index: usize,
        hosts: &[usize],
        cart: &mut IndoorObject,
        structure: &PlacementStructure,
        r: &mut ChaCha8Rng,
    ) -> Option<usize> {
        let zone = &self.functional_programs()[index];
        let mut routes: Vec<_> = zone
            .reservations
            .iter()
            .filter(|route| {
                matches!(
                    route.role,
                    ReservationRole::EquipmentAccess
                        | ReservationRole::Circulation
                        | ReservationRole::DoorApproach
                )
            })
            .collect();
        // Bound the trials even when a laundry has dozens of machines. Filter
        // distant edges before invoking structure, floor and collision checks.
        for &id in hosts.iter().take(4) {
            let station = &self.objects[id];
            let at = station.position.xz();
            routes.shuffle(r);
            routes.sort_by(|a, b| {
                let distance = |r: &Reservation| at.distance_squared(at.clamp(r.min, r.max));
                distance(a).total_cmp(&distance(b))
            });
            cart.interaction_target = Some(id);
            for &route in routes.iter().take(3) {
                let mut edges = [(0, -1.), (0, 1.), (1, -1.), (1, 1.)];
                edges.shuffle(r);
                for (axis, sign) in edges {
                    let along = 1 - axis;
                    let edge = if sign < 0. {
                        route.min[axis]
                    } else {
                        route.max[axis]
                    };
                    if (at[axis] - edge).abs() > station.size.x.max(station.size.z) * 0.5 + 2.35 {
                        continue;
                    }
                    let mut normal = Vec2::ZERO;
                    normal[axis] = -sign;
                    cart.yaw = normal.x.atan2(normal.y) + r.random_range(-0.52..0.52);
                    let half = rotated_half(cart.size.xz(), cart.yaw);
                    let lo = route.min[along] + 0.24;
                    let hi = route.max[along] - 0.24;
                    if lo > hi {
                        continue;
                    }
                    let mut pos = Vec2::ZERO;
                    pos[axis] = edge + sign * (half[axis] + r.random_range(0.04..0.12));
                    let side = if r.random_bool(0.5) { 1. } else { -1. };
                    for offset in [side, -side] {
                        pos[along] =
                            (at[along] + offset * r.random_range(0.37..1.55)).clamp(lo, hi);
                        cart.position = Vec3::new(pos.x, self.floor_height(pos), pos.y);
                        let (a, b) = cart.bounds();
                        if a.xz().cmpge(zone.min).all()
                            && b.xz().cmple(zone.max).all()
                            && nearby(cart, station)
                            && approach(route, cart)
                            && self.placement_clear_with_structure(cart, 0.035, false, structure)
                        {
                            return Some(id);
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::validation::validate_layout;

    #[test]
    fn cart_parking_preserves_working_aisles_station_capacity_and_host_identity() {
        for activity in [IndoorLayout::Laundromat, IndoorLayout::LockerRoom] {
            let mut occupied = 0;
            let mut malformed_checked = false;
            for seed in 0..48 {
                let scene = IndoorManifest::generate_with_humans(
                    seed,
                    activity,
                    (seed % 3) as f32 / 2.,
                    3,
                    0.25,
                )
                .unwrap();
                validate_layout(&scene).unwrap();
                for zone in scene.functional_programs() {
                    let mut count = 0;
                    for &id in &zone.members {
                        let cart = &scene.objects[id];
                        if cart.kind != ObjectKind::LaundryCart {
                            continue;
                        }
                        count += 1;
                        assert!(
                            scene.valid_laundry_cart(cart) && !zone.counts_toward_capacity(cart)
                        );
                        let (lo, hi) = cart.bounds();
                        assert!(zone.reservations.iter().all(|r| !r.overlaps(lo, hi)));
                        assert!(
                            (cart.position.y - scene.floor_height(cart.position.xz())).abs()
                                < 0.001
                        );
                        if !malformed_checked {
                            let mut invalid = scene.clone();
                            invalid.objects[id].interaction_target = Some(id);
                            assert!(validate_layout(&invalid).is_err());
                            invalid.objects[id].interaction_target = None;
                            assert!(validate_layout(&invalid).is_err());
                            malformed_checked = true;
                        }
                    }
                    assert!(count <= 2);
                    occupied += usize::from(count > 0);
                }
            }
            assert!(
                (1..48).contains(&occupied) && malformed_checked,
                "{activity:?}: {occupied}"
            );
        }
    }
}
