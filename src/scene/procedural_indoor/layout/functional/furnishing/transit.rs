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
                let mut size = geometry::dimensions(ObjectKind::Elevator, r.random());
                size.y = 2.50;
                self.public_fit(
                    index,
                    ObjectKind::Elevator,
                    Vec3::new(center.x + side * 2.6, 0., p.min.y + size.z * 0.5 + 0.2),
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
