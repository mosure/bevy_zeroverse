use super::*;
impl IndoorManifest {
    pub(super) fn exhibition_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        let rows = ((p.max.y - p.min.y - 3.0) / p.parameters.display_spacing_metres)
            .floor()
            .max(1.) as usize;
        self.assembly_diagnostics.programs[index].requested_capacity = rows * 2
            + if p.activity == IndoorLayout::Gallery {
                2
            } else {
                1
            };
        for row in 0..rows {
            for side in [-1., 1.] {
                let kind = if p.activity == IndoorLayout::Gallery && (row % 2 == 0) {
                    ObjectKind::ExhibitionPanel
                } else if r.random_bool(0.30) {
                    ObjectKind::DisplayCase
                } else {
                    ObjectKind::ExhibitPlinth
                };
                let size = geometry::dimensions(kind, r.random());
                let x = if side < 0. {
                    (p.min.x + center.x) * 0.5 - 0.3
                } else {
                    (p.max.x + center.x) * 0.5 + 0.3
                };
                let pos = Vec3::new(
                    x,
                    0.,
                    p.min.y + 3.2 + row as f32 * p.parameters.display_spacing_metres,
                );
                if let Some(id) = self.public_fit(
                    index,
                    kind,
                    pos,
                    size,
                    side * p.parameters.group_rotation_radians + r.random_range(-0.12..0.12),
                    r,
                ) {
                    if kind == ObjectKind::DisplayCase {
                        self.objects[id].variant = 16;
                    }
                }
            }
        }
    }
    pub(in crate::scene::procedural_indoor::layout) fn dress_public_programs(&mut self) {
        let mut r = stream(self.seed, 8250);
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind.is_surface() && !o.neighbor)
            .cloned()
            .collect();
        for support in surfaces {
            let kind = match support.kind {
                ObjectKind::ExhibitPlinth => {
                    if self.assembly_diagnostics.programs.iter().any(|p| {
                        p.activity == IndoorLayout::Showroom && p.members.contains(&support.id)
                    }) {
                        [
                            ObjectKind::Sculpture,
                            ObjectKind::Vase,
                            ObjectKind::Laptop,
                            ObjectKind::DeskLamp,
                        ][stream(support.finish_seed.unwrap_or(self.seed), 8252).random_range(0..4)]
                    } else {
                        ObjectKind::Sculpture
                    }
                }
                ObjectKind::Desk => ObjectKind::Notebook,
                ObjectKind::Table => {
                    if r.random_bool(0.5) {
                        ObjectKind::Dishware
                    } else {
                        ObjectKind::CoffeeCup
                    }
                }
                _ => continue,
            };
            let size = match kind {
                ObjectKind::Sculpture => geometry::dimensions(kind, r.random()),
                ObjectKind::Laptop => Vec3::new(0.37, 0.30, 0.34),
                ObjectKind::DeskLamp => Vec3::new(0.22, 0.43, 0.22),
                ObjectKind::Vase => Vec3::new(0.28, 0.45, 0.28),
                ObjectKind::Notebook => Vec3::new(0.21, 0.025, 0.28),
                ObjectKind::Dishware => Vec3::new(0.26, 0.11, 0.26),
                _ => Vec3::new(0.095, 0.13, 0.095),
            };
            let start = self.objects.len();
            let yaw = r.random_range(-0.2..0.2);
            if support.kind == ObjectKind::ExhibitPlinth {
                self.program_prop(&support, kind, Vec3::ZERO, size, yaw, &mut r);
            } else {
                self.prop(&support, kind, Vec3::ZERO, size, yaw, &mut r);
            }
            if support.kind == ObjectKind::ExhibitPlinth {
                for item in &mut self.objects[start..] {
                    item.finish_seed = support.finish_seed;
                }
            }
            if support.kind == ObjectKind::Table {
                let (drink, size) = crate::scene::procedural_indoor::layout::decor::drink(&mut r);
                self.prop(
                    &support,
                    drink,
                    Vec3::new(support.size.x * 0.28, 0., 0.16),
                    size,
                    0.,
                    &mut r,
                );
            }
        }
        let anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                matches!(
                    o.kind,
                    ObjectKind::ExhibitPlinth
                        | ObjectKind::ExhibitionPanel
                        | ObjectKind::InformationStand
                )
            })
            .cloned()
            .collect();
        for o in anchors {
            let Some(index) = self
                .functional_programs()
                .iter()
                .position(|p| p.members.contains(&o.id))
            else {
                continue;
            };
            let center = o.position
                + Quat::from_rotation_y(o.yaw) * Vec3::new(0., 0., o.size.z * 0.5 + 0.58);
            let route = Reservation {
                role: ReservationRole::Viewing,
                min: center.xz() - Vec2::new(0.43, 0.35),
                max: center.xz() + Vec2::new(0.43, 0.35),
            };
            let p = &self.functional_programs()[index];
            if route.min.cmpge(p.min).all()
                && route.max.cmple(p.max).all()
                && self.objects.iter().filter(|o| o.solid).all(|o| {
                    let (a, b) = o.bounds();
                    !route.overlaps(a, b)
                })
            {
                self.assembly_diagnostics.programs[index]
                    .reservations
                    .push(route);
            }
        }
    }
}
