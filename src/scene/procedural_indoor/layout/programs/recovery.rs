//! Complete accepted activity areas when ordinary workstation repair exhausts.
use super::*;

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
        if self.main_furniture_count() < self.minimum_main_objects() {
            self.add_lounge_surfaces();
            self.add_lounge_side_pieces();
            self.add_work_area_side_pieces();
            self.add_activity_side_storage();
        }
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
            for (kind, size) in [
                (ObjectKind::TrashCan, Vec3::new(0.36, 0.6, 0.36)),
                (ObjectKind::SideTable, Vec3::new(0.50, 0.52, 0.44)),
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
                'placement: for gap in [0.18, 0.32, 0.48] {
                    for sector in 0..16 {
                        let angle = std::f32::consts::TAU * sector as f32 / 16.
                            + rng.random_range(-0.06..0.06);
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
                    Vec3::new(0.64, 1.12, 0.32),
                ),
                // The lamp base has a 0.22 m minimum radius; reserve its full
                // mesh footprint, including the same clearance as lounge lamps.
                (ObjectKind::FloorLamp, Vec3::new(0.46, 1.50, 0.46)),
            ] {
                for side in [-1., 1.] {
                    if self.main_furniture_count() >= self.minimum_main_objects() {
                        return;
                    }
                    // Small modular storage units can form a pair alongside
                    // an activity area; one reading lamp per side is enough.
                    let limit = if kind == ObjectKind::FloorLamp { 1 } else { 2 };
                    let mut placed = 0;
                    'candidates: for gap in [0.15, 0.30, 0.45] {
                        for along in [0., -0.5, 0.5, -1., 1., -1.5, 1.5] {
                            let p = surface.position
                                + rotation
                                    * Vec3::new(
                                        side * ((surface.size.x + size.x) * 0.5 + gap),
                                        0.,
                                        along * surface.size.z * 0.5,
                                    );
                            if self.add(kind, p, size, surface.yaw, &mut rng).is_some() {
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
                (ObjectKind::CoffeeTable, Vec3::new(0.60, 0.52, 0.50)),
                (ObjectKind::FloorLamp, Vec3::new(0.46, 1.55, 0.46)),
            ] {
                for side in [-1., 1.] {
                    if self.main_furniture_count() >= self.minimum_main_objects() {
                        return;
                    }
                    'candidates: for gap in [0.15, 0.30, 0.45] {
                        for along in [0., -0.5, 0.5, -1., 1., 1.5] {
                            let p = sofa.position
                                + rotation
                                    * Vec3::new(
                                        side * ((sofa.size.x + size.x) * 0.5 + gap),
                                        0.,
                                        along * sofa.size.z * 0.5,
                                    );
                            if self.add(kind, p, size, sofa.yaw, &mut rng).is_some() {
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
            'candidates: for width in [1.05_f32, 0.85] {
                let size = Vec3::new(width.min(sofa.size.x * 0.8), 0.42, 0.50);
                let lateral = (sofa.size.x - size.x) * 0.5;
                for offset in [0.0, 0.5, -0.5, 1.0, -1.0] {
                    for gap in [0.35, 0.50, 0.65] {
                        let p = sofa.position
                            + rotation
                                * Vec3::new(
                                    offset * lateral,
                                    0.0,
                                    -(sofa.size.z + size.z) * 0.5 - gap,
                                );
                        if self
                            .add(ObjectKind::CoffeeTable, p, size, sofa.yaw, &mut rng)
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
