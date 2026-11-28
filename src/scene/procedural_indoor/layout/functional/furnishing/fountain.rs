//! A plumbed station needs both a solid mounting face and an unobstructed approach.
use super::*;
use crate::scene::procedural_indoor::{layout::programs::access, objects::wall};
use geometry::fountain as stock;
use rand::seq::SliceRandom;

impl IndoorManifest {
    pub(super) fn furnish_drinking_fountain(&mut self) {
        if self.generator_version < stock::VERSION {
            return;
        }
        let rate = match self.layout {
            IndoorLayout::Gymnasium => 0.74,
            IndoorLayout::Hallway => 0.57,
            IndoorLayout::LockerRoom => 0.48,
            IndoorLayout::FitnessRoom => 0.40,
            _ => return,
        };
        let mut r = stream(self.seed, u64::from_le_bytes(*b"PlumbBay"));
        if !r.random_bool(rate) {
            return;
        }
        let Some(index) = self
            .functional_programs()
            .iter()
            .position(|p| p.activity == self.layout)
        else {
            return;
        };
        let kind = ObjectKind::DrinkingFountain;
        let mut o = self.candidate(kind, Vec3::ZERO, Vec3::ONE, 0., &mut r);
        o.size = geometry::dimensions(kind, o.seed);
        let region = self.functional_programs()[index].member_region(self, &o);
        let Some(e) = &self.envelope else { return };
        // Envelope walls have physical backing; the separate glazed entrance is excluded.
        let mut walls: Vec<_> = e.walls.iter().map(|w| w.edge).collect();
        walls.shuffle(&mut r);
        let structure = PlacementStructure::new();
        let offset = wall::mount_offset(&o).expect("wall-mounted fountain");
        for edge in walls {
            let tf = e.wall_transform(edge);
            let extent = 0.5
                * (e.footprint[edge].distance(e.footprint[(edge + 1) % e.footprint.len()])
                    - o.size.x)
                - 0.14;
            if extent <= 0. {
                continue;
            }
            let tangent = (tf.rotation * Vec3::X).xz();
            let zone_mid = ((region.0 + region.1) * 0.5 - tf.translation.xz()).dot(tangent);
            let zone_half = ((region.1 - region.0) * 0.5).dot(tangent.abs());
            let start = (-extent).max(zone_mid - zone_half + o.size.x * 0.5);
            let end = extent.min(zone_mid + zone_half - o.size.x * 0.5);
            if end <= start {
                continue;
            }
            o.yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            let phase = r.random_range(0_f32..1.);
            for k in 0..3 {
                let along = start + (end - start) * (phase + k as f32 / 3.).fract();
                let center = tf.transform_point(Vec3::new(along, 0., offset));
                o.position =
                    center.with_y(e.floor_height(center.xz()) + stock::mount_height_m(o.seed));
                let (lo, hi) = access::envelope(&o).expect("fountain approach").bounds();
                let (body_min, body_max) = o.bounds();
                if lo.xz().cmpge(region.0).all()
                    && hi.xz().cmple(region.1).all()
                    && self.placement_clear_with_structure(&o, 0.04, false, &structure)
                    && self
                        .objects
                        .iter()
                        .filter(|other| !other.solid && !other.neighbor)
                        .all(|other| {
                            let (a, b) = other.bounds();
                            !(body_min.cmplt(b + 0.01).all() && body_max.cmpgt(a - 0.01).all())
                        })
                {
                    self.assembly_diagnostics.programs[index].members.push(o.id);
                    self.objects.push(o);
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::validation::validate_layout;

    #[test]
    fn fountains_require_backing_and_approach_and_leave_program_capacity_intact() {
        for layout in [
            IndoorLayout::Hallway,
            IndoorLayout::Gymnasium,
            IndoorLayout::LockerRoom,
            IndoorLayout::FitnessRoom,
        ] {
            let mut installations = 0;
            for seed in 76000..76040 {
                let scene =
                    IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.35).unwrap();
                validate_layout(&scene)
                    .unwrap_or_else(|error| panic!("{seed}/{layout:?}: {error}"));
                let Some(o) = scene
                    .objects
                    .iter()
                    .find(|o| o.kind == ObjectKind::DrinkingFountain)
                else {
                    continue;
                };
                installations += 1;
                let zone = scene
                    .functional_programs()
                    .iter()
                    .find(|p| p.members.contains(&o.id))
                    .unwrap();
                assert_eq!(zone.activity, layout);
                assert!(o.solid && o.support.is_none() && !zone.counts_toward_capacity(o));
                assert_eq!(
                    zone.members
                        .iter()
                        .filter(|&&i| matches!(
                            scene.objects[i].kind,
                            ObjectKind::DrinkingFountain | ObjectKind::WaterCooler
                        ))
                        .count(),
                    1
                );
                let (lo, hi) = access::envelope(o).unwrap().bounds();
                assert!(zone.reservations.iter().all(|r| !r.overlaps(lo, hi)));
                assert!(!scene.entry_approach_overlaps(lo, hi));
                assert!(scene.wall_attachment_clear(o) && scene.placement_clear(o, 0.));
                let mut detached = o.clone();
                detached.position += o.transform().rotation * Vec3::Z * 0.03;
                assert!(!scene.placement_clear(&detached, 0.));
                detached = o.clone();
                detached.position.y += 0.025;
                assert!(!scene.placement_clear(&detached, 0.));
                let mut blocker = o.clone();
                blocker.id = scene.objects.len();
                blocker.kind = ObjectKind::Chair;
                blocker.size = Vec3::new(0.20, 0.34, 0.18);
                blocker.position = o
                    .transform()
                    .transform_point(Vec3::Z * (o.size.z * 0.5 + 0.34))
                    .with_y(lo.y);
                assert!(access::blocks(o, &blocker));
                assert!(!scene.placement_clear(&blocker, 0.));
                let mut archived = scene.clone();
                archived.generator_version = stock::VERSION - 1;
                archived.furnish_drinking_fountain();
                assert_eq!(archived.objects.len(), scene.objects.len());
            }
            assert!(
                (1..39).contains(&installations),
                "{layout:?} installed {installations}/40"
            );
        }
    }
}
