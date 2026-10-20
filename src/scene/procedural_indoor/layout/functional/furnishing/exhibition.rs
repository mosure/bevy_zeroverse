use super::*;
impl IndoorManifest {
    pub(super) fn exhibition_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        let panel_fraction = r.random_range(0.20..0.65);
        let case_fraction = r.random_range(0.12..0.48);
        let stagger = r.random_range(-0.50..0.50);
        let disorder = r.random_range(0.08..0.32);
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
                let kind = if p.activity == IndoorLayout::Gallery && r.random_bool(panel_fraction) {
                    ObjectKind::ExhibitionPanel
                } else if r.random_bool(case_fraction) {
                    ObjectKind::DisplayCase
                } else {
                    ObjectKind::ExhibitPlinth
                };
                let size = geometry::dimensions(kind, r.random());
                let x = if side < 0. {
                    (p.min.x + center.x) * 0.5 - 0.3
                } else {
                    (p.max.x + center.x) * 0.5 + 0.3
                } + r.random_range(-disorder..disorder);
                let pos = Vec3::new(
                    x,
                    0.,
                    p.min.y
                        + 3.2
                        + row as f32 * p.parameters.display_spacing_metres
                        + side * stagger
                        + r.random_range(-disorder..disorder),
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
                        showroom_exhibit(&support)
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
                ObjectKind::Laptop => Vec3::new(
                    r.random_range(0.27..0.41),
                    r.random_range(0.23..0.33),
                    r.random_range(0.30..0.40),
                ),
                ObjectKind::DeskLamp => {
                    let base = r.random_range(0.17..0.27);
                    Vec3::new(base, r.random_range(0.32..0.51), base)
                }
                ObjectKind::Vase => {
                    let diameter = r.random_range(0.17..0.33);
                    Vec3::new(diameter, r.random_range(0.27..0.53), diameter)
                }
                ObjectKind::Notebook => {
                    super::super::super::super::items::sizing::loose(kind, &mut r).unwrap()
                }
                ObjectKind::Dishware => Vec3::new(0.26, 0.11, 0.26) * r.random_range(0.80..1.15),
                _ => super::super::super::super::items::sizing::loose(kind, &mut r).unwrap(),
            };
            let start = self.objects.len();
            let yaw: f32 = r.random_range(-0.45..0.45);
            // Offset against the rotated item's envelope, not against its center
            // alone. Small stands may correctly admit almost-centered exhibits.
            let (s, c) = yaw.sin_cos();
            let projected = Vec2::new(
                size.x * c.abs() + size.z * s.abs(),
                size.x * s.abs() + size.z * c.abs(),
            );
            let available =
                ((support.size.xz() - projected) * 0.5 - Vec2::splat(0.025)).max(Vec2::ZERO);
            let offset =
                available * Vec2::new(r.random_range(-0.8..0.8), r.random_range(-0.8..0.8));
            if support.kind == ObjectKind::ExhibitPlinth {
                self.program_prop(
                    &support,
                    kind,
                    Vec3::new(offset.x, 0., offset.y),
                    size,
                    yaw,
                    &mut r,
                );
            } else {
                self.prop(
                    &support,
                    kind,
                    Vec3::new(offset.x, 0., offset.y),
                    size,
                    yaw,
                    &mut r,
                );
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

fn showroom_exhibit(support: &IndoorObject) -> ObjectKind {
    // Finish identity expresses a coordinated installation, not a product class.
    [
        ObjectKind::Sculpture,
        ObjectKind::Vase,
        ObjectKind::Laptop,
        ObjectKind::DeskLamp,
    ][stream(support.seed, 8252).random_range(0..4)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhibit_identity_uses_instance_seed_and_is_invariant_to_finish() {
        let mut kinds = std::collections::BTreeSet::new();
        let mut support =
            crate::scene::procedural_indoor::objects::audit::fixture(ObjectKind::ExhibitPlinth, 0);
        for seed in 0..64 {
            support.seed = seed;
            support.finish_seed = Some(17);
            let kind = showroom_exhibit(&support);
            kinds.insert(format!("{kind:?}"));
            support.finish_seed = Some(901);
            assert_eq!(kind, showroom_exhibit(&support));
        }
        assert_eq!(kinds.len(), 4);
    }
}
