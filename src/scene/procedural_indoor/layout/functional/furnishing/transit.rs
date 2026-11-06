use super::*;

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
            let size = geometry::dimensions(ObjectKind::Banquette, r.random());
            self.public_fit(
                index,
                ObjectKind::Banquette,
                Vec3::new(
                    if side < 0. {
                        p.min.x + 0.5
                    } else {
                        p.max.x - 0.5
                    },
                    0.,
                    center.y + 0.8,
                ),
                size,
                side * FRAC_PI_2,
                r,
            );
            let size = geometry::dimensions(ObjectKind::InformationStand, r.random());
            self.public_fit(
                index,
                ObjectKind::InformationStand,
                Vec3::new(center.x + side * 1.25, 0., center.y),
                size,
                side * 0.25,
                r,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impossible_elevator_recovery_restores_the_aisle_and_retains_obstacles() {
        let mut scene =
            IndoorManifest::generate_with_humans(69_608_412, IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
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
