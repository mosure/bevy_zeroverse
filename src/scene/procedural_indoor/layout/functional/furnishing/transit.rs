use super::*;
impl IndoorManifest {
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
