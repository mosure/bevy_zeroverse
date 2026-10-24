use super::*;
mod dining;
mod exhibition;
mod queue;
mod teaching;
mod transit;
mod variation;
use super::super::super::objects::public as geometry;
use std::f32::consts::{FRAC_PI_2, PI};

impl IndoorManifest {
    /// Admit functional contents through the ordinary exact support/overlap gate.
    /// Loose table belongings also honor the shared per-table selection policy.
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
        if !self.permits_table_item(support, kind) {
            return false;
        }
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
                let mut x = cursor + size.x * 0.5;
                cursor += size.x + 0.30;
                let yaw = variation::anchor_yaw(kind, &mut r);
                if kind == ObjectKind::TeachingBoard {
                    x = center.x + (max.x - min.x) * r.random_range(-0.08..0.08);
                    size.x = size.x.min((max.x - min.x) * r.random_range(0.27..0.46));
                }
                if kind == ObjectKind::Lectern {
                    let side = if r.random_bool(0.5) { -1. } else { 1. };
                    x = center.x + side * (max.x - min.x) * r.random_range(0.23..0.34);
                }
                let z = min.y
                    + size.z * 0.5
                    + if kind == ObjectKind::ServiceCounter {
                        1.25
                    } else {
                        r.random_range(0.14..0.48)
                    };
                let position = Vec3::new(x, 0., z);
                if self
                    .public_fit(index, kind, position, size, yaw, &mut r)
                    .is_none()
                {
                    let (s, c) = yaw.sin_cos();
                    let extent = Vec2::new(
                        size.x * c.abs() + size.z * s.abs(),
                        size.x * s.abs() + size.z * c.abs(),
                    );
                    let span = max - min - extent - Vec2::splat(0.04);
                    if span.min_element() >= 0. {
                        for uv in variation::recovery_points(10, &mut r) {
                            let q = min + extent * 0.5 + Vec2::splat(0.02) + span * uv;
                            if self
                                .public_fit(index, kind, Vec3::new(q.x, 0., q.y), size, yaw, &mut r)
                                .is_some()
                            {
                                break;
                            }
                        }
                    }
                    if kind == ObjectKind::Elevator
                        && !self.objects[first..].iter().any(|o| o.kind == kind)
                    {
                        self.recover_elevator_anchor(index, size, &mut r);
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
            ) && self.objects[first..]
                .iter()
                .filter(|o| o.kind == ObjectKind::Chair)
                .count()
                < 2
            {
                let collection = variation::Collection::sample(self.seed, zone_index);
                for uv in variation::recovery_points(8, &mut r) {
                    if self.objects[first..]
                        .iter()
                        .filter(|o| o.kind == ObjectKind::Chair)
                        .count()
                        >= 2
                    {
                        break;
                    }
                    let q = min + Vec2::splat(0.70) + (max - min - Vec2::splat(1.4)) * uv;
                    if matches!(
                        zone.activity,
                        IndoorLayout::Classroom | IndoorLayout::LectureHall
                    ) {
                        let size = collection.chair(&mut r);
                        let yaw = r.random_range(-0.12..0.12);
                        if let Some(id) = self.public_fit(
                            index,
                            ObjectKind::Chair,
                            Vec3::new(q.x, 0., q.y),
                            size,
                            yaw,
                            &mut r,
                        ) {
                            self.objects[id].variant = if zone.activity == IndoorLayout::LectureHall
                            {
                                11
                            } else {
                                10
                            };
                        }
                    } else {
                        self.dining_table(index, Vec3::new(q.x, 0., q.y), 0.90, 0., &mut r);
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
        let core_passes = if kind == ObjectKind::Elevator { 2 } else { 1 };
        let alternatives = variation::local_alternatives(rng);
        for pass in 0..core_passes {
            for delta in alternatives {
                let p = pos + Vec3::new(delta.x, 0., delta.y);
                let mut o = self.floor_candidate(kind, p, size, yaw, rng);
                o.finish_seed = Some(self.seed.wrapping_add(index as u64));
                if kind == ObjectKind::Elevator {
                    o.variant = 20;
                    // A low soffit may prevent the full-height shaft, but the
                    // fallback still needs a masonry enclosure and car headroom.
                    o.size.y = o.size.y.max(geometry::elevator::MIN_CORE_HEIGHT);
                }
                if kind == ObjectKind::Elevator && pass == 0 {
                    let (lo, hi) = o.bounds();
                    let roof = [
                        lo.xz(),
                        hi.xz(),
                        Vec2::new(lo.x, hi.z),
                        Vec2::new(hi.x, lo.z),
                    ]
                    .into_iter()
                    .map(|p| self.ceiling_height(p))
                    .fold(f32::INFINITY, f32::min);
                    o.size.y = roof - o.position.y - 0.012;
                    o.variant = 20;
                    if o.size.y < 2.70 {
                        continue;
                    }
                }
                let (a, b) = o.bounds();
                let program = &self.assembly_diagnostics.programs[index];
                if a.x < program.min.x
                    || b.x > program.max.x
                    || a.z < program.min.y
                    || b.z > program.max.y
                    || b.y > self.ceiling_height(o.position.xz()) - o.overhead_clearance()
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
                    ObjectKind::Elevator => Some(transit::door_approach(&o)),
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
        }
        None
    }
}
