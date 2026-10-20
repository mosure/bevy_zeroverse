use super::*;
impl IndoorManifest {
    pub(super) fn dining_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        let communal = p.activity == IndoorLayout::Cafeteria;
        let pitch = r.random_range(2.55..3.15);
        let stagger = r.random_range(-0.45..0.45);
        let setback = r.random_range(3.15..3.65);
        let disorder = r.random_range(0.04..0.22);
        let rows = ((p.max.y - p.min.y - 3.2) / pitch).floor().max(1.) as usize;
        for row in 0..rows {
            let row_shift = r.random_range(-0.22..0.22);
            for side in [-1., 1.] {
                let x = if side < 0. {
                    (p.min.x + center.x) * 0.5 - 0.18
                } else {
                    (p.max.x + center.x) * 0.5 + 0.18
                } + row_shift
                    + r.random_range(-disorder..disorder);
                let z = p.min.y
                    + setback
                    + row as f32 * pitch
                    + side * stagger
                    + r.random_range(-disorder..disorder);
                self.dining_table(
                    index,
                    Vec3::new(x, 0., z),
                    if communal {
                        r.random_range(1.35..1.95)
                    } else {
                        r.random_range(0.78..1.2)
                    },
                    p.parameters.group_rotation_radians + r.random_range(-0.24..0.24),
                    r,
                );
            }
        }
        if !communal {
            let size = Vec3::new((p.max.y - p.min.y - 3.5).min(2.5), 1.1, 0.68);
            self.public_fit(
                index,
                ObjectKind::Banquette,
                Vec3::new(p.min.x + 0.36, 0., center.y + 1.1),
                size,
                FRAC_PI_2,
                r,
            );
        }
        let counters: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                o.kind == ObjectKind::ServiceCounter
                    && o.position.x >= p.min.x
                    && o.position.x <= p.max.x
                    && o.position.z >= p.min.y
                    && o.position.z <= p.max.y
            })
            .cloned()
            .collect();
        for counter in counters {
            for (offset, kind) in [
                (-0.23, ObjectKind::CoffeeMaker),
                (0.24, ObjectKind::DrinkDispenser),
            ] {
                let size = if kind == ObjectKind::CoffeeMaker {
                    Vec3::new(
                        r.random_range(0.29..0.44),
                        r.random_range(0.34..0.53),
                        r.random_range(0.32..0.47),
                    )
                } else {
                    geometry::dimensions(kind, r.random())
                };
                self.program_prop(
                    &counter,
                    kind,
                    Vec3::new(
                        counter.size.x * (offset + r.random_range(-0.045..0.045)),
                        0.,
                        counter.size.z * r.random_range(-0.12..0.12),
                    ),
                    size,
                    r.random_range(-0.14..0.14),
                    r,
                );
            }
        }
        // A bounded serving queue alongside the service zone; circulation and
        // the public/staff sides are exported separately from obstacle bounds.
        let queue = Reservation {
            role: ReservationRole::Queue,
            min: Vec2::new(p.min.x, p.min.y + 1.45),
            max: Vec2::new(center.x - 0.80, p.min.y + 2.55),
        };
        if queue.min.cmplt(queue.max).all()
            && self.objects.iter().filter(|o| o.solid).all(|o| {
                let (a, b) = o.bounds();
                !queue.overlaps(a, b)
            })
        {
            self.assembly_diagnostics.programs[index]
                .reservations
                .push(queue.clone());
            if communal || r.random_bool(0.50) {
                self.public_fit(
                    index,
                    ObjectKind::QueueBarrier,
                    Vec3::new((queue.min.x + queue.max.x) * 0.5, 0., queue.max.y + 0.22),
                    Vec3::new((queue.max.x - queue.min.x).min(1.8), 0.92, 0.30),
                    0.,
                    r,
                );
            }
        }
    }
    pub(super) fn dining_table(
        &mut self,
        index: usize,
        pos: Vec3,
        width: f32,
        yaw: f32,
        r: &mut ChaCha8Rng,
    ) {
        self.assembly_diagnostics.programs[index].requested_capacity += 2;
        let collection = variation::Collection::sample(self.seed, index);
        let table_size = collection.table(width, r);
        let start = self.objects.len();
        let Some(id) = self.public_fit(index, ObjectKind::Table, pos, table_size, yaw, r) else {
            return;
        };
        let table = self.objects[id].clone();
        for side in [-1., 1.] {
            if side < 0.
                && self.assembly_diagnostics.programs[index].activity == IndoorLayout::Cafe
                && r.random_bool(0.48)
            {
                let depth = r.random_range(0.63..0.76);
                let p = table.position
                    + Quat::from_rotation_y(yaw)
                        * Vec3::new(
                            r.random_range(-0.10..0.10),
                            0.,
                            -(table.size.z * 0.5 + depth * 0.5 + r.random_range(0.14..0.26)),
                        );
                if self
                    .public_fit(
                        index,
                        ObjectKind::Banquette,
                        p,
                        Vec3::new(width.max(1.15), r.random_range(0.96..1.17), depth),
                        yaw,
                        r,
                    )
                    .is_some()
                {
                    continue;
                }
            }
            let size = collection.chair(r);
            let p = table.position
                + Quat::from_rotation_y(yaw)
                    * Vec3::new(
                        r.random_range(-0.16..0.16),
                        0.,
                        side * (table.size.z * 0.5 + size.z * 0.5 + r.random_range(0.12..0.27)),
                    );
            if let Some(chair) = self.public_fit(
                index,
                ObjectKind::Chair,
                p,
                size,
                yaw + if side < 0. { PI } else { 0. } + r.random_range(-0.16..0.16),
                r,
            ) {
                self.objects[chair].variant = if r.random_bool(0.5) { 3 } else { 4 };
                self.objects[chair].interaction_target = Some(id);
            }
        }
        if self.objects.len() == start + 1 {
            self.objects.truncate(start);
        }
    }
}
