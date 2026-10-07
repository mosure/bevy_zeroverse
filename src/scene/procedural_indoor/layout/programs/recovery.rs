//! Complete already accepted seating when ordinary workstation repair exhausts.
use super::*;

impl IndoorManifest {
    pub(super) fn complete_lounge_groups(&mut self) {
        // Activity mixtures can leave only sofas in a globally office-labelled
        // room. Another desk plus chair may not fit, but its existing seating
        // can still support a useful lounge group. Keep ordinary successful
        // scenes and their random streams unchanged.
        if self.objects.iter().any(|o| {
            !o.neighbor
                && matches!(
                    o.kind,
                    ObjectKind::Desk | ObjectKind::Table | ObjectKind::CoffeeTable
                )
        }) {
            return;
        }
        self.add_lounge_surfaces();
    }

    /// Check population after perimeter/service furnishing has finished. An
    /// existing desk does not imply that every accepted sofa has its own usable
    /// surface, or that the room meets the unchanged furnishing minimum.
    pub(in crate::scene::procedural_indoor::layout) fn complete_underfilled_lounge_groups(
        &mut self,
    ) {
        if self.main_furniture_count() < self.minimum_main_objects() {
            self.add_lounge_surfaces();
            self.add_lounge_side_pieces();
        }
    }

    fn main_furniture_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| o.solid && !o.neighbor)
            .count()
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
            let mut placed = false;
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
                            placed = true;
                            break 'candidates;
                        }
                    }
                }
            }
            if placed && self.main_furniture_count() >= self.minimum_main_objects() {
                break;
            }
        }
    }
}
