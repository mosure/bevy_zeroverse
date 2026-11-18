//! Furnish the adjoining bay in its own metric frame. The entry and a strip
//! along the shared wall remain passable from either end of the room.
use super::*;
use std::f32::consts::PI;

struct Bay {
    outline: Vec<Vec2>,
    exterior_length: f32,
    passage_z: f32,
}

impl Bay {
    fn new(scene: &IndoorManifest, passage: f32) -> Self {
        let outline = scene.adjoining_space().footprint(scene.room_size);
        let exterior_length = super::super::super::envelope::polygon::edges(&outline)
            .skip(1)
            .map(|(a, b)| a.distance(b))
            .sum();
        Self {
            outline,
            exterior_length,
            passage_z: scene.room_size.z * 0.5 + passage,
        }
    }

    /// Sample physical wall length. Both sides of a return can receive furniture.
    /// The product faces into the room and its realized rotated footprint fits the wall.
    fn propose(&self, item: &mut IndoorObject, rng: &mut ChaCha8Rng) -> bool {
        let mut along = rng.random_range(0.0..self.exterior_length);
        for (a, b) in super::super::super::envelope::polygon::edges(&self.outline).skip(1) {
            let length = a.distance(b);
            if along >= length {
                along -= length;
                continue;
            }
            let tangent = (b - a) / length;
            let inward = Vec2::new(-tangent.y, tangent.x);
            let heading = inward.x.atan2(inward.y);
            item.yaw = match item.kind {
                ObjectKind::Plant => rng.random_range(-PI..PI),
                ObjectKind::Chair => heading + PI + rng.random_range(-0.33..0.33),
                _ => heading + rng.random_range(-0.16..0.16),
            };
            let (lo, hi) = item.bounds();
            let half = (hi - lo).xz() * 0.5;
            let end = 0.31 + tangent.abs().dot(half);
            if length <= 2. * end {
                return false;
            }
            let gap = 0.31 + inward.abs().dot(half) + rng.random_range(0.0..0.22);
            let p = a + tangent * rng.random_range(end..length - end) + inward * gap;
            item.position = Vec3::new(p.x, 0., p.y);
            return true;
        }
        false
    }

    fn fits<'a>(
        &self,
        scene: &IndoorManifest,
        item: &IndoorObject,
        mut peers: impl Iterator<Item = &'a IndoorObject>,
    ) -> bool {
        let (lo, hi) = item.bounds();
        lo.z >= self.passage_z
            && scene
                .adjoining_space()
                .contains_bounds(scene.room_size, lo.xz(), hi.xz(), 0.30)
            && !scene.entry_approach_overlaps(lo, hi)
            && hi.y <= scene.minimum_ceiling_height(lo.xz(), hi.xz()) - item.overhead_clearance()
            && peers.all(|peer| {
                let (a, b) = peer.bounds();
                !lo.xz().cmplt(b.xz() + 0.025).all() || !hi.xz().cmpgt(a.xz() - 0.025).all()
            })
    }
}

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn furnish_neighbor(&mut self) {
        let mut use_rng = stream(self.seed, 0x4241_5955_5345);
        let bay = Bay::new(self, use_rng.random_range(0.82..1.18));
        let start = self.objects.len();
        let use_draw = use_rng.random::<f32>();
        let work = if self.layout.is_public() { 0.24 } else { 0.48 };
        let anchor = if use_draw < work {
            ObjectKind::Desk
        } else if use_draw < work + 0.29 {
            if use_rng.random_bool(0.48) {
                ObjectKind::Chair
            } else {
                ObjectKind::Bench
            }
        } else if use_draw < 0.91 {
            ObjectKind::Bookcase
        } else {
            return; // Clear circulation along the adjoining facade.
        };
        let storage = use_rng.random_bool((0.22 + 0.48 * self.density) as f64);
        let planted = use_rng.random_bool((0.30 + 0.42 * self.density) as f64);
        let visitor_seat = anchor != ObjectKind::Desk && use_rng.random_bool(0.72);
        let products = [
            Some(anchor),
            storage.then_some(ObjectKind::Cabinet),
            planted.then_some(ObjectKind::Plant),
            visitor_seat.then_some(ObjectKind::Chair),
        ];
        for (slot, kind) in products.into_iter().enumerate() {
            let Some(kind) = kind else { continue };
            let mut product = stream(self.seed, 0x4241_5950_524f_4400 + slot as u64);
            let mut placement = stream(self.seed, 0x4241_5950_4f53_4500 + slot as u64);
            let mut item = self.neighbor_product(kind, &mut product);
            // A workstation is admitted as a complete group, with its seat
            // facing the actual desk. Failed poses leave no partial group.
            let mut chair = (kind == ObjectKind::Desk).then(|| {
                let family = [0, 1, 2, 3, 4, 5, 9][product.random_range(0..7)];
                let height = product.random_range(0.83..1.44);
                let mut chair = self.chair_product(
                    Vec3::ZERO,
                    0.,
                    height,
                    Some(family),
                    Some(item.seed),
                    &mut product,
                );
                chair.id = item.id + 1;
                chair.neighbor = true;
                chair.interaction_target = Some(item.id);
                chair
            });
            for _ in 0..12 {
                if !bay.propose(&mut item, &mut placement)
                    || !bay.fits(self, &item, self.objects[start..].iter())
                {
                    self.rejected_placements += 1;
                    continue;
                }
                if let Some(seat) = &mut chair {
                    seat.yaw = item.yaw + placement.random_range(-0.34..0.34);
                    seat.position = item.transform().transform_point(Vec3::new(
                        item.size.x * placement.random_range(-0.20..0.20),
                        0.,
                        (item.size.z + seat.size.z) * 0.5 + placement.random_range(0.12..0.32),
                    ));
                    if !bay.fits(self, seat, self.objects[start..].iter().chain([&item])) {
                        self.rejected_placements += 1;
                        continue;
                    }
                }
                self.objects.push(item);
                self.objects.extend(chair);
                break;
            }
        }
    }

    fn neighbor_product(&self, kind: ObjectKind, rng: &mut ChaCha8Rng) -> IndoorObject {
        let size = match kind {
            ObjectKind::Chair => {
                let family = [0, 2, 3, 4, 9][rng.random_range(0..5)];
                let height = rng.random_range(0.79..1.15);
                let mut chair = self.chair_product(Vec3::ZERO, 0., height, Some(family), None, rng);
                chair.neighbor = true;
                return chair;
            }
            ObjectKind::Desk => Vec3::new(
                rng.random_range(0.96..1.85),
                rng.random_range(0.70..0.80),
                rng.random_range(0.56..0.86),
            ),
            ObjectKind::Bench => Vec3::new(
                rng.random_range(1.08..2.32),
                rng.random_range(0.67..0.98),
                rng.random_range(0.46..0.72),
            ),
            ObjectKind::Bookcase => Vec3::new(
                rng.random_range(0.62..1.52),
                rng.random_range(1.08..1.96),
                rng.random_range(0.29..0.48),
            ),
            ObjectKind::Cabinet => Vec3::new(
                rng.random_range(0.65..1.72),
                rng.random_range(0.64..1.36),
                rng.random_range(0.32..0.56),
            ),
            ObjectKind::Plant => {
                let species = rng.random_range(0..crate::scene::procedural_indoor::plants::SPECIES);
                let size =
                    crate::scene::procedural_indoor::plants::sample_size(species, false, rng);
                let mut plant = self.candidate(kind, Vec3::ZERO, size, 0., rng);
                plant.neighbor = true;
                plant.variant = species;
                return plant;
            }
            _ => unreachable!("adjoining bay catalog"),
        };
        let mut item = self.candidate(kind, Vec3::ZERO, size, 0., rng);
        item.neighbor = true;
        if kind == ObjectKind::Bench {
            let bench =
                crate::scene::procedural_indoor::objects::services::bench::parameters(&item);
            if !bench.backed && !bench.arms {
                item.size.y = bench.seat_height;
            }
        }
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::validation::validate_layout;
    use std::collections::BTreeSet;

    #[test]
    fn adjacent_spaces_vary_across_office_and_public_programs_and_keep_a_passage() {
        for layout in [IndoorLayout::Conference, IndoorLayout::Gallery] {
            let mut compositions = BTreeSet::new();
            let mut desks = 0;
            let mut benches = 0;
            for seed in 0..64 {
                let scene =
                    IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.).unwrap();
                validate_layout(&scene).unwrap();
                let mut kinds = Vec::new();
                for o in scene.objects.iter().filter(|o| o.solid && o.neighbor) {
                    kinds.push(o.kind as usize);
                    let (lo, _) = o.bounds();
                    assert!(lo.z >= scene.room_size.z * 0.5 + 0.8199);
                    if o.kind == ObjectKind::Desk {
                        let seat = scene
                            .objects
                            .iter()
                            .find(|s| {
                                s.kind == ObjectKind::Chair
                                    && s.neighbor
                                    && s.interaction_target == Some(o.id)
                            })
                            .unwrap();
                        let toward = (o.position - seat.position).normalize();
                        assert!((seat.transform().rotation * Vec3::NEG_Z).dot(toward) > 0.65);
                        desks += 1;
                    }
                    benches += usize::from(o.kind == ObjectKind::Bench);
                }
                compositions.insert(kinds);
            }
            assert!(
                compositions.len() >= 8 && desks >= 4 && benches >= 4,
                "{layout:?}: {compositions:?}, {desks}, {benches}"
            );
        }
    }
}
