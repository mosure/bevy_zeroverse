use super::*;

/// Resolve the landing after admission. Program member IDs are finalized after furnishing.
pub(super) fn identify_landing(scene: &IndoorManifest, index: usize, o: &mut IndoorObject) {
    use geometry::signage::floor_guide::FloorGuide;
    let p = &scene.assembly_diagnostics.programs[index];
    let guide = match p.activity {
        IndoorLayout::ElevatorLanding => scene
            .objects
            .iter()
            .filter(|car| {
                car.kind == ObjectKind::Elevator
                    && !car.neighbor
                    && car.position.xz().cmpge(p.min).all()
                    && car.position.xz().cmple(p.max).all()
            })
            .find_map(|car| geometry::elevator::Levels::decode(car.variant))
            .map(|l| FloorGuide::lift(l.floor, l.highest)),
        IndoorLayout::StairHall => scene
            .envelope
            .as_ref()
            .and_then(|e| e.mezzanine.as_ref())
            .and_then(|m| {
                FloorGuide::stair(
                    o.position.y,
                    scene.floor_height(o.position.xz()),
                    m.deck.height,
                )
            }),
        _ => None,
    };
    if let Some(guide) = guide {
        guide.identify(o);
    }
}

pub(super) fn door_approach(o: &IndoorObject) -> Reservation {
    // Keep the established unrotated bank's exact arithmetic. A recovered bank
    // must reserve the space in front of its actual doors, not world +Z.
    let (_, hi) = o.bounds();
    if o.yaw == 0. {
        return Reservation {
            role: ReservationRole::DoorApproach,
            min: Vec2::new(o.position.x - o.size.x * 0.32, hi.z + 0.035),
            max: Vec2::new(o.position.x + o.size.x * 0.32, hi.z + 1.05),
        };
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for x in [-0.32, 0.32] {
        for offset in [0.035, 1.05] {
            let point = o
                .transform()
                .transform_point(Vec3::new(x * o.size.x, 0., o.size.z * 0.5 + offset))
                .xz();
            min = min.min(point);
            max = max.max(point);
        }
    }
    Reservation {
        role: ReservationRole::DoorApproach,
        min,
        max,
    }
}

impl IndoorManifest {
    pub(super) fn recover_elevator_anchor(
        &mut self,
        index: usize,
        size: Vec3,
        rng: &mut ChaCha8Rng,
    ) {
        // A low duct can block the front bank while a side-wall bank fits beside
        // the central aisle. Try other cardinal orientations only after ordinary
        // placement fails, with the same size and all existing clearance gates.
        let p = self.assembly_diagnostics.programs[index].clone();
        for yaw in [FRAC_PI_2, -FRAC_PI_2, PI] {
            let extent = if yaw.abs() == FRAC_PI_2 {
                Vec2::new(size.z, size.x)
            } else {
                size.xz()
            };
            let span = p.max - p.min - extent - Vec2::splat(0.04);
            if span.min_element() < 0. {
                continue;
            }
            for row in 0..10 {
                for col in 0..10 {
                    let point = p.min
                        + extent * 0.5
                        + Vec2::splat(0.02)
                        + span * Vec2::new(col as f32 / 9., row as f32 / 9.);
                    if self
                        .public_fit(
                            index,
                            ObjectKind::Elevator,
                            Vec3::new(point.x, 0., point.y),
                            size,
                            yaw,
                            rng,
                        )
                        .is_some()
                    {
                        return;
                    }
                }
            }
        }
        // Position/orientation changes cannot fit an oversized product into a
        // narrow bay between a truss and the aisle (reported seed 65437985).
        // Only after exhausting the original size, try progressively smaller
        // catalog sizes. Keep a full-height car and its masonry enclosure; all
        // candidates still pass public_fit, including door-approach clearance.
        // This bounded search costs nothing for normally admitted anchors.
        let minimum = Vec3::new(
            geometry::elevator::MIN_FOOTPRINT.x,
            geometry::elevator::MIN_CORE_HEIGHT,
            geometry::elevator::MIN_FOOTPRINT.y,
        );
        for fraction in [0.35, 0.70, 1.] {
            let compact = size.max(minimum).lerp(minimum, fraction).max(minimum);
            for yaw in [0., FRAC_PI_2, -FRAC_PI_2, PI] {
                let extent = if yaw.abs() == FRAC_PI_2 {
                    Vec2::new(compact.z, compact.x)
                } else {
                    compact.xz()
                };
                let span = p.max - p.min - extent - Vec2::splat(0.04);
                if span.min_element() < 0. {
                    continue;
                }
                // Stratified, shuffled positions avoid introducing a fixed
                // grid/corner preference in recovered room samples.
                for uv in variation::recovery_points(6, rng) {
                    let point = p.min + extent * 0.5 + Vec2::splat(0.02) + span * uv;
                    if self
                        .public_fit(
                            index,
                            ObjectKind::Elevator,
                            Vec3::new(point.x, 0., point.y),
                            compact,
                            yaw,
                            rng,
                        )
                        .is_some()
                    {
                        return;
                    }
                }
            }
        }
        self.recover_elevator_circulation(index, size, rng);
    }

    fn recover_elevator_circulation(&mut self, index: usize, size: Vec3, rng: &mut ChaCha8Rng) {
        // Roof beams and services can leave an otherwise useful landing with
        // no car-sized bay beside its *centered* aisle. Recover the layout,
        // not the clearance contract: retain the full aisle width, all ceiling
        // members, minimum car dimensions and the door's approach reservation.
        // Only required-anchor failure reaches this bounded search.
        let p = self.assembly_diagnostics.programs[index].clone();
        let Some(route) = p
            .reservations
            .first()
            .filter(|r| r.role == ReservationRole::Circulation)
        else {
            return;
        };
        let minimum = Vec3::new(
            geometry::elevator::MIN_FOOTPRINT.x,
            geometry::elevator::MIN_CORE_HEIGHT,
            geometry::elevator::MIN_FOOTPRINT.y,
        );
        let structure = self.structural_obstacles_with_ceiling(true);
        for fraction in [0., 0.08, -0.08, 0.16, -0.16] {
            let offset = Vec2::X * (p.max.x - p.min.x) * fraction;
            let mut moved = route.clone();
            moved.min += offset;
            moved.max += offset;
            let floor = self.floor_height((moved.min + moved.max) * 0.5);
            if !moved.min.cmpge(p.min).all()
                || !moved.max.cmple(p.max).all()
                || structure.iter().any(|&(lo, hi)| {
                    lo.y < floor + 2.0 && hi.y > floor + 0.02 && moved.overlaps(lo, hi)
                })
                || self.objects.iter().filter(|o| o.solid).any(|o| {
                    let (lo, hi) = o.bounds();
                    moved.overlaps(lo, hi)
                })
            {
                continue;
            }
            self.assembly_diagnostics.programs[index].reservations[0] = moved;
            for compact in [
                size.max(minimum),
                size.max(minimum).lerp(minimum, 0.7),
                minimum,
            ] {
                for yaw in [0., FRAC_PI_2, -FRAC_PI_2, PI] {
                    for point in recovery::anchor_points(self, index, compact, yaw, rng) {
                        if self
                            .public_fit(
                                index,
                                ObjectKind::Elevator,
                                Vec3::new(point.x, 0., point.y),
                                compact,
                                yaw,
                                rng,
                            )
                            .is_some()
                        {
                            return;
                        }
                    }
                }
            }
        }
        self.assembly_diagnostics.programs[index].reservations = p.reservations;
    }

    pub(super) fn transit_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        self.assembly_diagnostics.programs[index].requested_capacity = 1;
        // Banks are sampled as complete closed cavities, rather than doors pasted
        // over an opaque wall. Additional banks compete for genuine clear space.
        for side in [-1., 1.] {
            if r.random_bool(0.70) {
                self.assembly_diagnostics.programs[index].requested_capacity += 1;
                let size = geometry::dimensions(ObjectKind::Elevator, r.random());
                self.public_fit(
                    index,
                    ObjectKind::Elevator,
                    Vec3::new(
                        center.x + side * r.random_range(2.25..3.15),
                        0.,
                        p.min.y + size.z * 0.5 + 0.2,
                    ),
                    size,
                    0.,
                    r,
                );
            }
        }
        let mut seats = stream(self.seed, u64::from_le_bytes(*b"LobbyUse") + index as u64);
        let occupied = seats.random_bool(0.84 + 0.10 * f64::from(self.density));
        if occupied {
            let count = 1 + usize::from(seats.random_bool(0.30 + 0.40 * f64::from(self.density)));
            self.waiting_seats(index, 0, count);
        }
        self.lobby_directories(index);
    }

    fn lobby_directories(&mut self, index: usize) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"LiftSign") + index as u64);
        if !r.random_bool(0.56 + 0.22 * f64::from(self.density)) {
            return;
        }
        let p = &self.functional_programs()[index];
        let spine = p.reservations[0].clone();
        let mut banks: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                o.kind == ObjectKind::Elevator
                    && !o.neighbor
                    && o.position.xz().cmpge(p.min).all()
                    && o.position.xz().cmple(p.max).all()
            })
            .map(|o| (o.transform(), o.size))
            .collect();
        let two =
            banks.len() >= 3 && (p.max - p.min).element_product() > 60. && r.random_bool(0.30);
        use rand::seq::SliceRandom;
        banks.shuffle(&mut r);
        let structure = PlacementStructure::new();
        for (bank, cabinet) in banks.into_iter().take(if two { 2 } else { 1 }) {
            let kind = ObjectKind::InformationStand;
            let size = geometry::dimensions(kind, r.random());
            let side = if r.random_bool(0.5) { 1. } else { -1. };
            for flank in [side, -side] {
                let pos = bank.transform_point(Vec3::new(
                    flank * (cabinet.x * 0.5 + size.x * 0.5 + r.random_range(0.12..0.48)),
                    0.,
                    cabinet.z * 0.5 + r.random_range(0.10..0.96),
                ));
                // A visitor reads the face from the public route to the car.
                let approach = pos.xz().clamp(spine.min, spine.max) - pos.xz();
                if approach.length_squared() < 0.25 {
                    continue;
                }
                let yaw = approach.x.atan2(approach.y) + r.random_range(-0.18..0.18);
                let site = FloorProposal {
                    kind,
                    size,
                    position: pos,
                    yaw,
                };
                if self
                    .public_fit_at_offsets(index, site, &mut r, &structure, &[Vec2::ZERO], |_| true)
                    .is_some()
                {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impossible_elevator_recovery_restores_the_aisle_and_retains_obstacles() {
        let mut scene = IndoorManifest::initial_mixed_fixture(69_608_412, 0.65, 0, 0.).unwrap();
        scene.objects.clear();
        let p = scene.assembly_diagnostics.programs[0].clone();
        // A completely occupied overhead volume is deliberately infeasible.
        // Recovery must leave the unsatisfied contract and architecture intact.
        let ceiling = scene.ceiling_installation.as_mut().unwrap();
        let mut blocked = ceiling.modules[0].clone();
        blocked.center = Vec3::new((p.min.x + p.max.x) * 0.5, 1.8, (p.min.y + p.max.y) * 0.5);
        blocked.size = Vec3::new(p.max.x - p.min.x, 1., p.max.y - p.min.y);
        blocked.suspension = None;
        ceiling.modules = vec![blocked];
        let before = scene.clone();
        scene.recover_elevator_circulation(0, Vec3::new(2.4, 2.8, 1.8), &mut stream(42, 7));
        assert_eq!(scene, before);
        assert!(scene.validate_public_programs().is_err());
    }
}
