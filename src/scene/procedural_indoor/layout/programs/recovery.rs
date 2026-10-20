//! Complete accepted activity areas when ordinary workstation repair exhausts.
use super::*;
use crate::scene::procedural_indoor::objects::standing_lamp;
use rand::seq::SliceRandom;

fn side_along(r: &mut ChaCha8Rng) -> [f32; 7] {
    let mut positions = std::array::from_fn(|i| i as f32 * 0.5 - 1.5 + r.random_range(-0.18..0.18));
    positions.shuffle(r);
    positions
}

impl IndoorManifest {
    pub(super) fn complete_lounge_groups(&mut self) {
        // Complete each seating area. A desk in a different activity zone is
        // not a usable surface for this sofa.
        let before = (self.objects.len(), self.rejected_placements);
        self.add_lounge_surfaces();
        // An unsuccessful, repeatable repair is a no-op on the manifest; do not
        // accumulate diagnostic counters when clients recheck a complete room.
        if self.objects.len() == before.0 {
            self.rejected_placements = before.1;
        }
    }

    /// Check population after perimeter/service furnishing has finished. An
    /// accepted activity surface can still have an incomplete seating/storage
    /// group and fail the unchanged furnishing minimum.
    pub(in crate::scene::procedural_indoor::layout) fn complete_underfilled_activity_groups(
        &mut self,
    ) {
        self.complete_library_storage();
        // Service pieces can meet the population minimum while every work or
        // coffee surface was rejected. Complete that activity independently of
        // the count, using the same supported-floor and collision checks.
        if self.main_furniture_count() < self.minimum_main_objects()
            || self.primary_activity_surface_missing()
        {
            self.add_lounge_surfaces();
            self.add_lounge_side_pieces();
            self.add_work_area_side_pieces();
            self.add_activity_side_storage();
        }
    }

    fn primary_activity_surface_missing(&self) -> bool {
        !self.objects.iter().any(|o| {
            !o.neighbor
                && (matches!(
                    o.kind,
                    ObjectKind::Desk | ObjectKind::Table | ObjectKind::CoffeeTable
                ) || self.layout.is_kitchen() && o.kind == ObjectKind::KitchenCounter)
        })
    }

    // A moved entrance can reject the perimeter bookcase. Reuse admitted
    // storage first; adding another footprint can obstruct constrained rooms.
    fn complete_library_storage(&mut self) {
        if self.layout != IndoorLayout::Library
            || self
                .objects
                .iter()
                .any(|o| !o.neighbor && o.kind == ObjectKind::Bookcase)
        {
            return;
        }
        if let Some(cabinet) = self
            .objects
            .iter_mut()
            .find(|o| !o.neighbor && o.kind == ObjectKind::Cabinet)
        {
            cabinet.kind = ObjectKind::Bookcase;
            return;
        }
        // Storage completes this activity even if chairs and other service
        // pieces already satisfy the generic population minimum.
        let Some(program) = &self.program else { return };
        let zones = program.zones.clone();
        let mut rng = stream(self.seed, 6134);
        let rejected = self.rejected_placements;
        for width in [0.8, 0.55] {
            let size = Vec3::new(width, 1.4, 0.32);
            for zone in &zones {
                for side in 0..4 {
                    for i in 0..12 {
                        let t = (i as f32 + 0.5) / 12.;
                        let point = zone.min.lerp(zone.max, t);
                        let (x, z, yaw) = match side {
                            0 => (zone.min.x + 0.34, point.y, -std::f32::consts::FRAC_PI_2),
                            1 => (zone.max.x - 0.34, point.y, std::f32::consts::FRAC_PI_2),
                            2 => (point.x, zone.min.y + 0.34, 0.),
                            _ => (point.x, zone.max.y - 0.34, std::f32::consts::PI),
                        };
                        if self
                            .add(
                                ObjectKind::Bookcase,
                                Vec3::new(x, 0., z),
                                size,
                                yaw,
                                &mut rng,
                            )
                            .is_some()
                        {
                            return;
                        }
                    }
                }
            }
        }
        self.rejected_placements = rejected;
    }

    fn add_activity_side_storage(&mut self) {
        if self.main_furniture_count() >= self.minimum_main_objects() {
            return;
        }
        let anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Sofa | ObjectKind::Desk | ObjectKind::Table
                    )
            })
            .cloned()
            .collect();
        let mut rng = stream(self.seed, 6133);
        for anchor in anchors {
            let diameter = rng.random_range(0.28..0.40);
            for (kind, size) in [
                (
                    ObjectKind::TrashCan,
                    Vec3::new(diameter, rng.random_range(0.43..0.72), diameter),
                ),
                (
                    ObjectKind::SideTable,
                    Vec3::new(
                        rng.random_range(0.38..0.60),
                        rng.random_range(0.43..0.60),
                        rng.random_range(0.36..0.54),
                    ),
                ),
            ] {
                if self.main_furniture_count() >= self.minimum_main_objects() {
                    return;
                }
                if self.objects.iter().any(|o| {
                    o.kind == kind
                        && !o.neighbor
                        && o.position.distance(anchor.position)
                            < anchor.size.xz().length() * 0.5 + 1.
                }) {
                    continue;
                }
                let phase = rng.random_range(0.0..std::f32::consts::TAU);
                'placement: for band in 0..3 {
                    let gap = 0.14 + band as f32 * 0.14 + rng.random_range(0.0..0.10);
                    for sector in 0..16 {
                        let angle = phase
                            + std::f32::consts::TAU * sector as f32 / 16.
                            + rng.random_range(-0.12..0.12);
                        let direction = Vec2::from_angle(angle);
                        let offset = direction
                            * (anchor.size.xz() * 0.5 + size.xz() * 0.5 + Vec2::splat(gap));
                        let p = anchor.position
                            + Quat::from_rotation_y(anchor.yaw) * Vec3::new(offset.x, 0., offset.y);
                        if self
                            .add(kind, p, size, anchor.yaw + angle, &mut rng)
                            .is_some()
                        {
                            break 'placement;
                        }
                    }
                }
            }
        }
    }

    fn main_furniture_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| o.solid && !o.neighbor)
            .count()
    }

    fn add_work_area_side_pieces(&mut self) {
        if self.main_furniture_count() >= self.minimum_main_objects() {
            return;
        }
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && o.solid
                    && matches!(
                        o.kind,
                        ObjectKind::Desk | ObjectKind::Table | ObjectKind::CoffeeTable
                    )
            })
            .cloned()
            .collect();
        // A desk/chair pair can exhaust the usable floor around steps, portals
        // and columns. Compact storage or a reading lamp still completes an
        // existing activity area, rather than leaving the room underfilled.
        // A separate stream preserves the rest of the scene's sampled details.
        let mut rng = stream(self.seed, 6132);
        for surface in surfaces {
            let rotation = Quat::from_rotation_y(surface.yaw);
            for (kind, size) in [
                (
                    if self.layout == IndoorLayout::Library {
                        ObjectKind::Bookcase
                    } else {
                        ObjectKind::Cabinet
                    },
                    Vec3::new(
                        rng.random_range(0.55..0.76),
                        rng.random_range(0.82..1.32),
                        rng.random_range(0.28..0.40),
                    ),
                ),
                (
                    ObjectKind::FloorLamp,
                    standing_lamp::dimensions(rng.random()),
                ),
            ] {
                let yaw = if kind == ObjectKind::FloorLamp {
                    rng.random_range(-std::f32::consts::PI..std::f32::consts::PI)
                } else {
                    surface.yaw
                };
                for side in [-1., 1.] {
                    if self.main_furniture_count() >= self.minimum_main_objects() {
                        return;
                    }
                    // Small modular storage units can form a pair alongside
                    // an activity area; one reading lamp per side is enough.
                    let limit = if kind == ObjectKind::FloorLamp { 1 } else { 2 };
                    let mut placed = 0;
                    'candidates: for band in 0..3 {
                        let gap = 0.14 + band as f32 * 0.15 + rng.random_range(0.0..0.08);
                        for along in side_along(&mut rng) {
                            let p = surface.position
                                + rotation
                                    * Vec3::new(
                                        side * ((surface.size.x + size.x) * 0.5 + gap),
                                        0.,
                                        along * surface.size.z * 0.5,
                                    );
                            if self.add(kind, p, size, yaw, &mut rng).is_some() {
                                if self.main_furniture_count() >= self.minimum_main_objects() {
                                    return;
                                }
                                placed += 1;
                                if placed == limit {
                                    break 'candidates;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_lounge_side_pieces(&mut self) {
        let sofas: Vec<_> = self
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.solid && o.kind == ObjectKind::Sofa)
            .cloned()
            .collect();
        let mut rng = stream(self.seed, 6131);
        for sofa in sofas {
            let rotation = Quat::from_rotation_y(sofa.yaw);
            // A side table or reading lamp can complete an alcove where the
            // front is reserved for circulation, a pillar or floor transitions.
            // Each proposal uses the same supported-floor and clearance gates
            // as ordinary furniture; no existing objects are moved or removed.
            for (kind, size) in [
                (
                    ObjectKind::CoffeeTable,
                    Vec3::new(
                        rng.random_range(0.48..0.72),
                        rng.random_range(0.40..0.56),
                        rng.random_range(0.40..0.60),
                    ),
                ),
                (
                    ObjectKind::FloorLamp,
                    standing_lamp::dimensions(rng.random()),
                ),
            ] {
                let yaw = if kind == ObjectKind::FloorLamp {
                    rng.random_range(-std::f32::consts::PI..std::f32::consts::PI)
                } else {
                    sofa.yaw
                };
                for side in [-1., 1.] {
                    if self.main_furniture_count() >= self.minimum_main_objects() {
                        if !self.primary_activity_surface_missing() {
                            return;
                        }
                        if kind != ObjectKind::CoffeeTable {
                            // Keep trying surfaces at the remaining sofas;
                            // another lamp cannot complete this activity.
                            continue;
                        }
                    }
                    'candidates: for band in 0..3 {
                        let gap = 0.14 + band as f32 * 0.15 + rng.random_range(0.0..0.08);
                        for along in side_along(&mut rng) {
                            let p = sofa.position
                                + rotation
                                    * Vec3::new(
                                        side * ((sofa.size.x + size.x) * 0.5 + gap),
                                        0.,
                                        along * sofa.size.z * 0.5,
                                    );
                            if self.add(kind, p, size, yaw, &mut rng).is_some() {
                                break 'candidates;
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_lounge_surfaces(&mut self) {
        let sofas: Vec<_> = self
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.solid && o.kind == ObjectKind::Sofa)
            .cloned()
            .collect();
        let mut rng = stream(self.seed, 6130);
        for sofa in sofas {
            let rotation = Quat::from_rotation_y(sofa.yaw);
            if self.objects.iter().any(|o| {
                if o.neighbor
                    || o.kind != ObjectKind::CoffeeTable
                    || (o.position.y - sofa.position.y).abs() > 0.01
                {
                    return false;
                }
                let local = rotation.inverse() * (o.position - sofa.position);
                local.x.abs() < (sofa.size.x + o.size.x) * 0.5
                    && local.z < -sofa.size.z * 0.5
                    && local.z > -sofa.size.z * 0.5 - o.size.z - 1.0
            }) {
                continue;
            }
            // Search in sofa coordinates: negative Z is the front of the seat.
            // Smaller tables and off-centre placement fit compact alcoves while
            // retaining human-scale furniture, access gaps and floor support.
            'candidates: for width in [
                rng.random_range(0.95_f32..1.15),
                rng.random_range(0.70_f32..0.90),
            ] {
                let size = Vec3::new(
                    width.min(sofa.size.x * 0.8),
                    rng.random_range(0.34..0.48),
                    rng.random_range(0.40..0.60),
                );
                let lateral = (sofa.size.x - size.x) * 0.5;
                let mut offsets = std::array::from_fn::<_, 5, _>(|i| {
                    (i as f32 - 2.) * 0.42 + rng.random_range(-0.15..0.15)
                });
                offsets.shuffle(&mut rng);
                let yaw = sofa.yaw + rng.random_range(-0.12..0.12);
                for offset in offsets {
                    for band in 0..3 {
                        let gap = 0.30 + band as f32 * 0.15 + rng.random_range(0.0..0.12);
                        let p = sofa.position
                            + rotation
                                * Vec3::new(
                                    offset * lateral,
                                    0.0,
                                    -(sofa.size.z + size.z) * 0.5 - gap,
                                );
                        if self
                            .add(ObjectKind::CoffeeTable, p, size, yaw, &mut rng)
                            .is_some()
                        {
                            break 'candidates;
                        }
                    }
                }
            }
        }
    }
}
