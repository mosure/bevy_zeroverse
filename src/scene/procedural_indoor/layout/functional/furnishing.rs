use super::*;
mod dining;
mod exhibition;
mod teaching;
mod transit;
use super::super::super::objects::public as geometry;
use std::f32::consts::{FRAC_PI_2, PI};

impl IndoorManifest {
    /// Equipment and exhibits are functional contents, independent of the loose
    /// clutter sampler. They still pass the ordinary exact support/overlap gate.
    pub(super) fn program_prop(
        &mut self,
        support: &IndoorObject,
        kind: ObjectKind,
        offset: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> bool {
        let mut o = self.candidate(
            kind,
            support
                .transform()
                .transform_point(offset + Vec3::Y * support.size.y),
            size,
            support.yaw + yaw,
            rng,
        );
        o.solid = false;
        o.support = Some(support.id);
        o.finish_seed = support.finish_seed;
        if !self.prop_clear(&o, support, 0.012) {
            return false;
        }
        if let Some(p) = self
            .assembly_diagnostics
            .programs
            .iter_mut()
            .find(|p| p.members.contains(&support.id))
        {
            p.members.push(o.id);
        }
        self.objects.push(o);
        true
    }
    pub(in crate::scene::procedural_indoor::layout) fn furnish_public_programs(&mut self) {
        let zones = self.program.as_ref().unwrap().zones.clone();
        for (zone_index, zone) in zones.into_iter().enumerate() {
            let (min, max) = usable_region(self, &zone);
            let center = (min + max) * 0.5;
            let parameters = ProgramParameters::sample(self.seed, zone_index, self.density);
            let (role, required) = match zone.activity {
                IndoorLayout::ElevatorLanding => (SpaceRole::Transit, vec![ObjectKind::Elevator]),
                IndoorLayout::Cafe => (
                    SpaceRole::Service,
                    vec![ObjectKind::ServiceCounter, ObjectKind::DisplayCase],
                ),
                IndoorLayout::Cafeteria => (
                    SpaceRole::Dining,
                    vec![ObjectKind::ServiceCounter, ObjectKind::TrayReturn],
                ),
                IndoorLayout::Classroom | IndoorLayout::LectureHall => (
                    SpaceRole::Teaching,
                    vec![ObjectKind::TeachingBoard, ObjectKind::Lectern],
                ),
                IndoorLayout::Gallery => (
                    SpaceRole::Exhibition,
                    vec![ObjectKind::ExhibitionPanel, ObjectKind::ExhibitPlinth],
                ),
                IndoorLayout::Showroom => (
                    SpaceRole::Demonstration,
                    vec![ObjectKind::ExhibitPlinth, ObjectKind::InformationStand],
                ),
                _ => unreachable!("public zone is explicitly typed"),
            };
            let aisle = parameters.aisle_metres * 0.5;
            let mut reservations = vec![Reservation {
                role: ReservationRole::Circulation,
                min: Vec2::new(center.x - aisle, min.y + 2.7),
                max: Vec2::new(center.x + aisle, max.y),
            }];
            if self.layout == IndoorLayout::PublicSuite {
                reservations.push(Reservation {
                    role: ReservationRole::Circulation,
                    min: Vec2::new(min.x, max.y - parameters.aisle_metres),
                    max,
                });
            }
            let index = self.assembly_diagnostics.programs.len();
            self.assembly_diagnostics.programs.push(ProgramAssembly {
                zone: zone_index,
                activity: zone.activity,
                role,
                capacity_unit: match role {
                    SpaceRole::Transit => CapacityUnit::ElevatorCars,
                    SpaceRole::Exhibition | SpaceRole::Demonstration => {
                        CapacityUnit::DisplayStations
                    }
                    _ => CapacityUnit::ChairStations,
                },
                parameters: parameters.clone(),
                min,
                max,
                members: vec![],
                required: required.clone(),
                requested_capacity: 0,
                accepted_capacity: 0,
                rejected_candidates: 0,
                reservations,
                services: vec![],
                focus: Vec3::new(center.x, 1.25, min.y + 1.2),
            });
            let first = self.objects.len();
            let mut r = stream(self.seed, 8240 + zone_index as u64);
            // Required anchors are placed first. Failure is recorded and validation
            // rejects the program; unrelated furniture cannot satisfy this contract.
            let sizes: Vec<_> = required
                .iter()
                .map(|&kind| {
                    let mut size = geometry::dimensions(kind, r.random());
                    if kind == ObjectKind::ServiceCounter {
                        size.x = parameters.service_length_metres;
                    }
                    size
                })
                .collect();
            let total = sizes.iter().map(|s| s.x).sum::<f32>() + 0.30 * (sizes.len() - 1) as f32;
            let mut cursor = center.x - total * 0.5;
            for (i, &kind) in required.iter().enumerate() {
                let mut size = sizes[i];
                if kind == ObjectKind::Elevator {
                    size.y = 2.50;
                }
                let mut x = cursor + size.x * 0.5;
                cursor += size.x + 0.30;
                if kind == ObjectKind::TeachingBoard {
                    x = center.x;
                    size.x = (max.x - min.x) * 0.40;
                }
                if kind == ObjectKind::Lectern {
                    x = center.x - (max.x - min.x) * 0.29;
                }
                let z = min.y
                    + size.z * 0.5
                    + if kind == ObjectKind::ServiceCounter {
                        1.25
                    } else {
                        0.2
                    };
                let position = Vec3::new(x, 0., z);
                if self
                    .public_fit(index, kind, position, size, 0., &mut r)
                    .is_none()
                {
                    'search: for row in 0..10 {
                        for col in 0..10 {
                            let span = max - min - size.xz() - Vec2::splat(0.04);
                            if span.min_element() < 0. {
                                break 'search;
                            }
                            let q = min
                                + size.xz() * 0.5
                                + Vec2::splat(0.02)
                                + span * Vec2::new(col as f32 / 9., row as f32 / 9.);
                            if self
                                .public_fit(index, kind, Vec3::new(q.x, 0., q.y), size, 0., &mut r)
                                .is_some()
                            {
                                break 'search;
                            }
                        }
                    }
                }
            }
            match zone.activity {
                IndoorLayout::ElevatorLanding => self.transit_program(index, &mut r),
                IndoorLayout::Cafe | IndoorLayout::Cafeteria => self.dining_program(index, &mut r),
                IndoorLayout::Classroom | IndoorLayout::LectureHall => {
                    self.teaching_program(index, &mut r)
                }
                IndoorLayout::Gallery | IndoorLayout::Showroom => {
                    self.exhibition_program(index, &mut r)
                }
                _ => unreachable!(),
            }
            if matches!(zone.activity, IndoorLayout::Cafe | IndoorLayout::Cafeteria) {
                self.assembly_diagnostics.programs[index].required.push(
                    if zone.activity == IndoorLayout::Cafe {
                        ObjectKind::CoffeeMaker
                    } else {
                        ObjectKind::DrinkDispenser
                    },
                );
            }
            if matches!(
                zone.activity,
                IndoorLayout::Cafe
                    | IndoorLayout::Cafeteria
                    | IndoorLayout::Classroom
                    | IndoorLayout::LectureHall
            ) {
                'capacity: for row in 0..8 {
                    for col in 0..8 {
                        if self.objects[first..]
                            .iter()
                            .filter(|o| o.kind == ObjectKind::Chair)
                            .count()
                            >= 2
                        {
                            break 'capacity;
                        }
                        let q = min
                            + Vec2::splat(0.70)
                            + (max - min - Vec2::splat(1.4))
                                * Vec2::new(col as f32 / 7., row as f32 / 7.);
                        if matches!(
                            zone.activity,
                            IndoorLayout::Classroom | IndoorLayout::LectureHall
                        ) {
                            if let Some(id) = self.public_fit(
                                index,
                                ObjectKind::Chair,
                                Vec3::new(q.x, 0., q.y),
                                Vec3::new(0.72, 1.04, 0.72),
                                0.,
                                &mut r,
                            ) {
                                self.objects[id].variant = 10;
                            }
                        } else {
                            self.dining_table(index, Vec3::new(q.x, 0., q.y), 0.90, 0., &mut r);
                        }
                    }
                }
            }
            self.assembly_diagnostics.programs[index].members =
                (first..self.objects.len()).collect();
            self.assembly_diagnostics.programs[index].accepted_capacity =
                self.assembly_diagnostics.programs[index].realized_capacity(&self.objects);
        }
    }
    pub(super) fn public_fit(
        &mut self,
        index: usize,
        kind: ObjectKind,
        pos: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> Option<usize> {
        // Deterministic local alternatives, all subject to the same floor/portal,
        // physical-occupancy and circulation constraints as the rest of the engine.
        for delta in [
            Vec2::ZERO,
            Vec2::new(0., 0.35),
            Vec2::new(-0.45, 0.),
            Vec2::new(0.45, 0.),
            Vec2::new(0., 0.70),
            Vec2::new(-0.85, 0.35),
            Vec2::new(0.85, 0.35),
        ] {
            let p = pos + Vec3::new(delta.x, 0., delta.y);
            let mut o = self.floor_candidate(kind, p, size, yaw, rng);
            o.finish_seed = Some(self.seed.wrapping_add(index as u64));
            let (a, b) = o.bounds();
            let program = &self.assembly_diagnostics.programs[index];
            if a.x < program.min.x
                || b.x > program.max.x
                || a.z < program.min.y
                || b.z > program.max.y
                || b.y > self.ceiling_height(o.position.xz()) - 0.30
                || !self.placement_clear(&o, 0.035)
            {
                self.assembly_diagnostics.programs[index].rejected_candidates += 1;
                continue;
            }
            let id = o.id;
            let operational = match kind {
                ObjectKind::ServiceCounter => Some(Reservation {
                    role: ReservationRole::StaffAccess,
                    min: Vec2::new(a.x, a.z - 1.05),
                    max: Vec2::new(b.x, a.z - 0.05),
                }),
                ObjectKind::Elevator => Some(Reservation {
                    role: ReservationRole::DoorApproach,
                    min: Vec2::new(o.position.x - o.size.x * 0.32, b.z + 0.035),
                    max: Vec2::new(o.position.x + o.size.x * 0.32, b.z + 1.05),
                }),
                _ => None,
            };
            if let Some(route) = &operational {
                if !(route.min.cmpge(program.min).all()
                    && route.max.cmple(program.max).all()
                    && self.objects.iter().filter(|o| o.solid).all(|o| {
                        let (a, b) = o.bounds();
                        !route.overlaps(a, b)
                    }))
                {
                    self.assembly_diagnostics.programs[index].rejected_candidates += 1;
                    continue;
                }
            }
            self.objects.push(o);
            if let Some(route) = operational {
                self.assembly_diagnostics.programs[index]
                    .reservations
                    .push(route);
            }
            return Some(id);
        }
        None
    }
}
