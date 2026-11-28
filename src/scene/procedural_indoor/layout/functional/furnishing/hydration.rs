//! Site one optional self-contained water station after the primary equipment.
//! All poses share a product. A rejected pose costs neither a new product nor
//! a retry of the room. Its +Z working space participates in later admissions.
use super::*;
use crate::scene::procedural_indoor::{layout::programs::access, objects::public::cooler};
use rand::seq::SliceRandom;

impl IndoorManifest {
    pub(super) fn furnish_water_coolers(&mut self) {
        if self.generator_version < cooler::VERSION {
            return;
        }
        for index in 0..self.assembly_diagnostics.programs.len() {
            let program = &self.assembly_diagnostics.programs[index];
            let rate = match program.activity {
                IndoorLayout::LockerRoom => 0.24,
                IndoorLayout::FitnessRoom => 0.67,
                IndoorLayout::CyclingStudio => 0.59,
                IndoorLayout::YogaStudio => 0.48,
                IndoorLayout::Gymnasium => 0.63,
                IndoorLayout::Hallway => 0.37,
                IndoorLayout::ElevatorLanding => 0.19,
                _ => continue,
            };
            if self.generator_version >= super::geometry::fountain::VERSION
                && program
                    .members
                    .iter()
                    .any(|&id| self.objects[id].kind == ObjectKind::DrinkingFountain)
            {
                continue;
            }
            let (lo, hi) = (program.min, program.max);
            let domain =
                u64::from_le_bytes(*b"WaterBay") ^ ((program.activity as u64) << 32) ^ index as u64;
            let mut r = stream(self.seed, domain);
            if !r.random_bool(rate * (0.84 + 0.25 * f64::from(self.density))) {
                continue;
            }
            let seed = r.random();
            let size = geometry::dimensions(ObjectKind::WaterCooler, seed);
            let mut o = self.floor_candidate(ObjectKind::WaterCooler, Vec3::ZERO, size, 0., &mut r);
            o.seed = seed;
            let mut sides = [0, 1, 2, 3];
            sides.shuffle(&mut r);
            let structure = PlacementStructure::new();
            'site: for side in sides {
                let phase = r.random_range(0_f32..1.);
                for i in 0..3 {
                    let u = 0.06 + 0.88 * (phase + i as f32 / 3.).fract();
                    let inset = size.z * 0.5 + r.random_range(0.16..0.41);
                    let x = lo.x + (hi.x - lo.x) * u;
                    let z = lo.y + (hi.y - lo.y) * u;
                    let (pos, yaw) = match side {
                        0 => (Vec2::new(x, lo.y + inset), 0.),
                        1 => (Vec2::new(x, hi.y - inset), PI),
                        2 => (Vec2::new(lo.x + inset, z), FRAC_PI_2),
                        _ => (Vec2::new(hi.x - inset, z), -FRAC_PI_2),
                    };
                    o.yaw = yaw + r.random_range(-0.165..0.165);
                    o.position = Vec3::new(pos.x, self.floor_height(pos), pos.y);
                    let (a, b) = access::envelope(&o)
                        .expect("water cooler has a cup approach")
                        .bounds();
                    if a.xz().cmpge(lo).all()
                        && b.xz().cmple(hi).all()
                        && self.placement_clear_with_structure(&o, 0.045, false, &structure)
                    {
                        self.assembly_diagnostics.programs[index].members.push(o.id);
                        self.objects.push(o);
                        break 'site;
                    }
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
    fn hydration_access_survives_furnishing_and_does_not_count_as_a_seat_or_station() {
        for profile in [
            IndoorLayout::Hallway,
            IndoorLayout::FitnessRoom,
            IndoorLayout::YogaStudio,
            IndoorLayout::CyclingStudio,
            IndoorLayout::Gymnasium,
            IndoorLayout::ElevatorLanding,
        ] {
            let mut installed = 0;
            for seed in 0..48 {
                let scene =
                    IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.25).unwrap();
                validate_layout(&scene).unwrap();
                for zone in scene.functional_programs() {
                    let mut stations = zone
                        .members
                        .iter()
                        .map(|&i| &scene.objects[i])
                        .filter(|o| o.kind == ObjectKind::WaterCooler);
                    let Some(o) = stations.next() else {
                        continue;
                    };
                    assert!(
                        stations.next().is_none() && o.solid && !o.neighbor && o.support.is_none()
                    );
                    assert!(!zone.counts_toward_capacity(o));
                    assert!((o.position.y - scene.floor_height(o.position.xz())).abs() < 0.001);
                    let (lo, hi) = access::envelope(o).unwrap().bounds();
                    assert!(lo.xz().cmpge(zone.min).all() && hi.xz().cmple(zone.max).all());
                    assert!(!zone.reservations.iter().any(|r| r.overlaps(lo, hi)));
                    assert!(
                        scene
                            .objects
                            .iter()
                            .filter(|b| b.id != o.id && !b.neighbor)
                            .all(|b| !access::blocks(o, b)),
                        "{profile:?} seed {seed} lost access"
                    );
                    let inward = (zone.min + zone.max) * 0.5 - o.position.xz();
                    assert!(inward.dot((o.transform().rotation * Vec3::Z).xz()) > 0.);
                    installed += 1;
                }
            }
            assert!(
                (1..48).contains(&installed),
                "{profile:?}: {installed} stations"
            );
        }
    }
}
