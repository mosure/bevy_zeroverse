use super::*;
impl IndoorManifest {
    pub(super) fn exhibition_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        let panel_fraction = r.random_range(0.20..0.65);
        let case_fraction = r.random_range(0.12..0.48);
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
                let yaw = side * p.parameters.group_rotation_radians + r.random_range(-0.24..0.24);
                let (s, c) = yaw.sin_cos();
                let half = Vec2::new(
                    size.x * c.abs() + size.z * s.abs(),
                    size.x * s.abs() + size.z * c.abs(),
                ) * 0.5;
                let aisle = p.parameters.aisle_metres * 0.5;
                let (left, right) = if side < 0. {
                    (p.min.x + half.x + 0.05, center.x - aisle - half.x - 0.05)
                } else {
                    (center.x + aisle + half.x + 0.05, p.max.x - half.x - 0.05)
                };
                let near = p.min.y + 2.9 + half.y;
                let far = p.max.y - half.y - 0.05;
                if right <= left || far <= near {
                    continue;
                }
                // Aisle-connected viewing fields, not two repeated centerlines.
                // Stratification limits clumping without fixing station pitch.
                let z =
                    near + (far - near) * (row as f32 + r.random_range(0.02..0.98)) / rows as f32;
                let pos = Vec3::new(r.random_range(left..right), 0., z);
                if let Some(id) = self.public_fit(index, kind, pos, size, yaw, r) {
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
            if support.kind == ObjectKind::ExhibitPlinth {
                // Choose the product and its dimensions jointly with its support,
                // before constructing either mesh. Independent wide size priors
                // otherwise leave small, valid plinths without their exhibit.
                for _ in 0..24 {
                    let yaw: f32 = r.random_range(-0.45..0.45);
                    let Some(size) = exhibit_dimensions(kind, support.size.xz(), yaw, &mut r)
                    else {
                        continue;
                    };
                    let (s, c) = yaw.sin_cos();
                    let projected = Vec2::new(
                        size.x * c.abs() + size.z * s.abs(),
                        size.x * s.abs() + size.z * c.abs(),
                    );
                    let slack = ((support.size.xz() - projected) * 0.5 - Vec2::splat(0.018))
                        .max(Vec2::ZERO);
                    let offset =
                        slack * Vec2::new(r.random_range(-0.90..0.90), r.random_range(-0.90..0.90));
                    if self.program_prop(
                        &support,
                        kind,
                        Vec3::new(offset.x, 0., offset.y),
                        size,
                        yaw,
                        &mut r,
                    ) {
                        break;
                    }
                }
                continue;
            }
            let size = match kind {
                ObjectKind::Notebook => {
                    super::super::super::super::items::sizing::loose(kind, &mut r).unwrap()
                }
                ObjectKind::Dishware => Vec3::new(0.26, 0.11, 0.26) * r.random_range(0.80..1.15),
                _ => super::super::super::super::items::sizing::loose(kind, &mut r).unwrap(),
            };
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
            self.prop(
                &support,
                kind,
                Vec3::new(offset.x, 0., offset.y),
                size,
                yaw,
                &mut r,
            );
            if support.kind == ObjectKind::Table {
                let (drink, size) = crate::scene::procedural_indoor::layout::decor::drink(&mut r);
                let (offset, yaw) =
                    crate::scene::procedural_indoor::layout::surfaces::scatter_proposal(
                        &self.objects,
                        &support,
                        drink,
                        size,
                        &mut r,
                    );
                self.prop(&support, drink, offset, size, yaw, &mut r);
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

/// Sample physical dimensions conditioned on a rotated support rectangle. This
/// is a truncated product-size distribution, not a scale applied after admission.
fn exhibit_dimensions(
    kind: ObjectKind,
    support: Vec2,
    yaw: f32,
    r: &mut ChaCha8Rng,
) -> Option<Vec3> {
    let (min, max) = match kind {
        ObjectKind::Sculpture => (Vec3::new(0.24, 0.26, 0.22), Vec3::new(0.78, 1.05, 0.70)),
        ObjectKind::Laptop => (Vec3::new(0.27, 0.23, 0.30), Vec3::new(0.41, 0.33, 0.40)),
        ObjectKind::DeskLamp => (Vec3::new(0.17, 0.32, 0.17), Vec3::new(0.27, 0.51, 0.27)),
        ObjectKind::Vase => (Vec3::new(0.10, 0.18, 0.10), Vec3::new(0.33, 0.65, 0.33)),
        _ => unreachable!("exhibit category"),
    };
    let available = support - Vec2::splat(0.04);
    let (s, c) = yaw.sin_cos();
    let (s, c) = (s.abs().max(1e-6), c.abs().max(1e-6));
    let width_max = max
        .x
        .min((available.x - s * min.z) / c)
        .min((available.y - c * min.z) / s);
    if width_max <= min.x {
        return None;
    }
    let width = r.random_range(min.x..width_max);
    let depth_max = max
        .z
        .min((available.x - c * width) / s)
        .min((available.y - s * width) / c);
    if depth_max <= min.z {
        return None;
    }
    let depth = r.random_range(min.z..depth_max);
    // Turned vessels and lamp bases retain round metric footprints.
    let (width, depth) = if matches!(kind, ObjectKind::Vase | ObjectKind::DeskLamp) {
        (width.min(depth), width.min(depth))
    } else {
        (width, depth)
    };
    Some(Vec3::new(width, r.random_range(min.y..max.y), depth))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhibits_sample_inside_small_rotated_supports_without_collapsing_yaw() {
        let mut r = stream(91, 8253);
        for kind in [
            ObjectKind::Sculpture,
            ObjectKind::Laptop,
            ObjectKind::Vase,
            ObjectKind::DeskLamp,
        ] {
            let mut accepted = 0;
            let mut yaw_bins = std::collections::BTreeSet::new();
            for _ in 0..512 {
                let support = Vec2::new(r.random_range(0.42..1.12), r.random_range(0.40..1.08));
                let yaw: f32 = r.random_range(-0.45..0.45);
                if let Some(size) = exhibit_dimensions(kind, support, yaw, &mut r) {
                    let (s, c) = yaw.sin_cos();
                    let extent = Vec2::new(
                        size.x * c.abs() + size.z * s.abs(),
                        size.x * s.abs() + size.z * c.abs(),
                    );
                    assert!(extent.cmple(support - Vec2::splat(0.0399)).all());
                    assert!(size.is_finite() && size.min_element() > 0.09);
                    accepted += 1;
                    yaw_bins.insert((yaw * 10.).floor() as i32);
                }
            }
            assert!(accepted > 450, "{kind:?}: {accepted}");
            assert_eq!(yaw_bins.len(), 10, "{kind:?}");
        }
    }

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
