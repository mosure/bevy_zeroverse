//! Reserve a continuous hall and each door approach before admitting furniture
//! on either side. Furniture in an adjoining room belongs to that room's receipt.
use super::*;
use crate::scene::procedural_indoor::program::hallway::{across, front_normal};

fn rect(axis: usize, cross: [f32; 2], along: [f32; 2], role: ReservationRole) -> Reservation {
    let mut min = Vec2::ZERO;
    let mut max = Vec2::ZERO;
    min[axis] = cross[0];
    max[axis] = cross[1];
    min[1 - axis] = along[0];
    max[1 - axis] = along[1];
    Reservation { min, max, role }
}

pub(in crate::scene::procedural_indoor::layout::functional) fn routes(
    scene: &IndoorManifest,
    hall: &Zone,
    parameters: &ProgramParameters,
) -> Vec<Reservation> {
    let a = across(hall);
    let t = 1 - a;
    let mid = (hall.min[a] + hall.max[a]) * 0.5;
    let half = parameters.aisle_metres * 0.5 + 0.12;
    let mut result = vec![rect(
        a,
        [mid - half, mid + half],
        [hall.min[t] + 0.28, hall.max[t] - 0.28],
        ReservationRole::Circulation,
    )];
    let mut branch = |at: f32, width: f32, face: f32| {
        let clear = (width * 0.44).min(0.64);
        result.push(rect(
            a,
            [mid.min(face), mid.max(face)],
            [at - clear, at + clear],
            ReservationRole::DoorApproach,
        ));
    };
    for wall in &scene.program.as_ref().unwrap().partitions {
        if let Some(inward) = front_normal(hall, wall) {
            branch(
                wall.door_center,
                wall.door_width,
                wall.coordinate + inward * (wall.thickness * 0.5 + 0.15),
            );
        }
    }
    if a == 1 {
        branch(scene.door_x, scene.entry_width(), hall.max.y - 0.24);
    } else {
        // Entering at the end of the hall may require a lateral turn.
        let door_half = scene.entry_width() * 0.42;
        result.push(rect(
            a,
            [
                mid.min(scene.door_x - door_half),
                mid.max(scene.door_x + door_half),
            ],
            [hall.max[t] - 1.15, hall.max[t] - 0.23],
            ReservationRole::DoorApproach,
        ));
    }
    result
}

impl IndoorManifest {
    pub(super) fn furnish_hallway(&mut self) {
        let zones = self.program.as_ref().unwrap().zones.clone();
        for (i, zone) in zones.iter().enumerate() {
            let mut r = stream(self.seed, u64::from_le_bytes(*b"HallFits") + i as u64);
            let parameters = ProgramParameters::sample(self.seed, i, self.density);
            let (role, unit, reservations) = match zone.activity {
                IndoorLayout::Hallway => (
                    SpaceRole::Transit,
                    CapacityUnit::SeatingStations,
                    routes(self, zone, &parameters),
                ),
                IndoorLayout::Lounge => (SpaceRole::Waiting, CapacityUnit::SeatingStations, vec![]),
                _ => (SpaceRole::Work, CapacityUnit::ChairStations, vec![]),
            };
            // The entrance is allowed up to the finished wall face. Suite
            // furniture still goes through the actual footprint/portal gate.
            let (min, max) = if i == 0 {
                (zone.min, zone.max)
            } else {
                (zone.min + Vec2::splat(0.16), zone.max - Vec2::splat(0.16))
            };
            let center = (min + max) * 0.5;
            let focus = if i == 0 {
                let branches = &reservations[1..];
                let branch = &branches[r.random_range(0..branches.len())];
                (branch.min + branch.max) * 0.5
            } else {
                center
            };
            self.assembly_diagnostics.programs.push(ProgramAssembly {
                zone: i,
                activity: zone.activity,
                role,
                capacity_unit: unit,
                parameters,
                min,
                max,
                reservations,
                members: vec![],
                required: vec![],
                services: vec![],
                requested_capacity: 0,
                accepted_capacity: 0,
                rejected_candidates: 0,
                focus: Vec3::new(focus.x, 1.24, focus.y),
            });
            let first = self.objects.len();
            if i == 0 {
                self.hall_seating(zone, &mut r);
            } else {
                self.hall_suite(i, zone, &mut r);
            }
            let p = &mut self.assembly_diagnostics.programs[i];
            p.members = (first..self.objects.len()).collect();
            p.accepted_capacity = p.realized_capacity(&self.objects);
        }
    }

    fn hall_seating(&mut self, hall: &Zone, r: &mut ChaCha8Rng) {
        let a = across(hall);
        let t = 1 - a;
        let stations = if r.random_bool(0.26 + 0.48 * self.density as f64) {
            r.random_range(1..=3)
        } else {
            0
        };
        self.assembly_diagnostics.programs[0].requested_capacity = stations;
        for _ in 0..stations {
            let size = geometry::dimensions(ObjectKind::Banquette, r.random());
            for _ in 0..6 {
                let low = r.random_bool(0.5);
                let yaw = match (a, low) {
                    (0, true) => FRAC_PI_2,
                    (0, false) => -FRAC_PI_2,
                    (_, true) => 0.,
                    (_, false) => PI,
                };
                let gap = size.z * 0.5 + r.random_range(0.12..0.32);
                let mut p = Vec2::ZERO;
                p[a] = if low {
                    hall.min[a] + gap
                } else {
                    hall.max[a] - gap
                };
                p[t] = r.random_range(hall.min[t] + size.x * 0.55..hall.max[t] - size.x * 0.55);
                if self
                    .public_fit(
                        0,
                        ObjectKind::Banquette,
                        Vec3::new(p.x, 0., p.y),
                        size,
                        yaw,
                        r,
                    )
                    .is_some()
                {
                    break;
                }
            }
        }
    }

    fn hall_suite(&mut self, index: usize, zone: &Zone, r: &mut ChaCha8Rng) {
        let min = self.assembly_diagnostics.programs[index].min;
        let max = self.assembly_diagnostics.programs[index].max;
        let work = zone.activity == IndoorLayout::OpenOffice;
        let requested = 1 + usize::from(
            (max - min).element_product() > 21. && r.random_bool(0.56 * self.density as f64),
        );
        self.assembly_diagnostics.programs[index].requested_capacity = requested;
        for _ in 0..requested {
            let size = if work {
                Vec3::new(
                    r.random_range(1.02..1.64),
                    r.random_range(0.72..0.79),
                    r.random_range(0.58..0.82),
                )
            } else {
                geometry::dimensions(ObjectKind::Banquette, r.random())
            };
            let phase = r.random_range(0..4);
            let points = variation::recovery_points(3, r);
            let mut installed = false;
            for offset in 0..2 {
                for (k, &q) in points.iter().enumerate() {
                    let pos = (min + Vec2::splat(0.48)) + (max - min - Vec2::splat(0.96)) * q;
                    let pos = Vec3::new(pos.x, 0., pos.y);
                    let yaw =
                        ((phase + k + offset) % 4) as f32 * FRAC_PI_2 + r.random_range(-0.14..0.14);
                    if work {
                        let first = self.objects.len();
                        let assemblies = self.assembly_diagnostics.accepted.len();
                        if self.workstation(ObjectKind::Desk, pos, size, yaw, r) {
                            if self.objects[first..].iter().all(|o| {
                                let (lo, hi) = o.bounds();
                                lo.xz().cmpge(min).all() && hi.xz().cmple(max).all()
                            }) {
                                installed = true;
                                break;
                            }
                            self.objects.truncate(first);
                            self.assembly_diagnostics.accepted.truncate(assemblies);
                        }
                    } else if self
                        .public_fit(index, ObjectKind::Banquette, pos, size, yaw, r)
                        .is_some()
                    {
                        installed = true;
                        break;
                    }
                }
                if installed {
                    break;
                }
            }
        }
        // An occasional low table or storage cabinet gives otherwise similar
        // neighboring suites different sightlines through the glass.
        let coffee = !work && r.random_bool(0.68);
        if coffee || r.random_bool(0.52) {
            let (kind, size) = if coffee {
                (
                    ObjectKind::CoffeeTable,
                    Vec3::new(
                        r.random_range(0.62..1.06),
                        r.random_range(0.36..0.48),
                        r.random_range(0.46..0.74),
                    ),
                )
            } else {
                (
                    ObjectKind::Bookcase,
                    Vec3::new(
                        r.random_range(0.75..1.46),
                        r.random_range(0.94..1.86),
                        r.random_range(0.31..0.44),
                    ),
                )
            };
            for q in variation::recovery_points(2, r) {
                let pos = min + (max - min) * q;
                let yaw = r.random_range(0..4) as f32 * FRAC_PI_2 + r.random_range(-0.08..0.08);
                if self
                    .public_fit(index, kind, Vec3::new(pos.x, 0., pos.y), size, yaw, r)
                    .is_some()
                {
                    break;
                }
            }
        }
    }
}
