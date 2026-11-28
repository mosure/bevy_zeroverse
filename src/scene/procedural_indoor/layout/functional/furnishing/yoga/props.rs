//! Assemble loose class equipment after all instructor and participant stations
//! and their access zones have been admitted.
use super::*;
use crate::scene::procedural_indoor::{objects::public::mats, placement::rotated_half};
use rand::seq::SliceRandom;
#[cfg(test)]
mod tests;

struct PracticeItem {
    kind: ObjectKind,
    size: Vec3,
    stock: u64,
    end: f32,
}

impl IndoorManifest {
    pub(in super::super) fn furnish_practice_props(&mut self) {
        if self.generator_version < mats::props::VERSION || self.density <= 0. {
            return;
        }
        for index in 0..self.functional_programs().len() {
            let zone = &self.functional_programs()[index];
            let rate = match zone.activity {
                IndoorLayout::YogaStudio => 0.91,
                IndoorLayout::FitnessRoom => 0.34,
                _ => continue,
            };
            let mut r = stream(self.seed, u64::from_le_bytes(*b"ClassKit") ^ index as u64);
            if !r.random_bool(rate * (0.52 + 0.48 * f64::from(self.density))) {
                continue;
            }
            let mut stations: Vec<_> = zone
                .members
                .iter()
                .copied()
                .filter(|&id| self.objects[id].kind == ObjectKind::ExerciseMat)
                .collect();
            stations.shuffle(&mut r);
            let fleet = r.random::<u64>();
            let coverage = r.random_range(0.24..0.95);
            let bolster_rate = match r.random_range(0..5) {
                0 | 1 => 0.,
                2 => r.random_range(0.77..1.),
                _ => r.random_range(0.12..0.64),
            };
            let pair_rate = r.random_range(0.24..0.86);
            let cap = r.random_range(4..=14);
            let mut fitted = 0;
            for id in stations {
                if fitted == cap {
                    break;
                }
                let mat = self.objects[id].clone();
                let mut place = stream(mat.seed, u64::from_le_bytes(*b"MatPlace"));
                if !place.random_bool(coverage) {
                    continue;
                }
                let stock = if place.random_bool(0.23) {
                    place.random()
                } else {
                    fleet
                };
                let kind = if place.random_bool(bolster_rate) {
                    ObjectKind::YogaBolster
                } else {
                    ObjectKind::YogaBlock
                };
                let size = mats::props::dimensions(kind, stock);
                let pair = kind == ObjectKind::YogaBlock && place.random_bool(pair_rate);
                let end = if place.random_bool(0.5) { 1. } else { -1. };
                let area = mats::PracticeMat::for_object(&mat).prop_area(mat.size);
                for i in 0..=usize::from(pair) {
                    if fitted == cap {
                        break;
                    }
                    // Some blocks stand on their narrow side; most bear on the broad face.
                    let size = if kind == ObjectKind::YogaBlock && place.random_bool(0.27) {
                        Vec3::new(size.x, size.z, size.y)
                    } else {
                        size
                    };
                    let end = if i == 1 && place.random_bool(0.31) {
                        -end
                    } else {
                        end
                    };
                    if self.fit_practice_item(
                        &mat,
                        area,
                        PracticeItem {
                            kind,
                            size,
                            stock,
                            end,
                        },
                        &mut place,
                    ) {
                        fitted += 1;
                    }
                }
            }
        }
    }

    fn fit_practice_item(
        &mut self,
        mat: &IndoorObject,
        area: (Vec2, Vec2),
        item: PracticeItem,
        r: &mut ChaCha8Rng,
    ) -> bool {
        let crosswise = r.random_bool(0.80);
        for attempt in 0..6 {
            let yaw = if item.kind == ObjectKind::YogaBolster {
                let cross = crosswise ^ (attempt % 2 == 1);
                (if cross { FRAC_PI_2 } else { 0. })
                    + (if r.random_bool(0.5) { PI } else { 0. })
                    + r.random_range(-0.34..0.34)
            } else {
                r.random_range(-PI..PI)
            };
            let half = rotated_half(item.size.xz(), yaw) + 0.016;
            let low = area.0 + half;
            let high = area.1 - half;
            if low.cmpgt(high).any() {
                continue;
            }
            let x = r.random_range(low.x..=high.x);
            let z = item.end * r.random_range(low.y..=high.y);
            if self.program_prop(mat, item.kind, Vec3::new(x, 0., z), item.size, yaw, r) {
                self.objects.last_mut().unwrap().finish_seed = Some(item.stock);
                return true;
            }
        }
        false
    }
}
