//! Loose cutlery on completed counter runs. Fit around dishes, supplies and cables
//! with an independent stream so a missed placement never resamples the kitchen.
use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, SurfacePattern};
use rand::seq::IteratorRandom;

impl IndoorManifest {
    pub(in super::super) fn finish_counter_flatware(&mut self) {
        for k in 0..self.assembly_diagnostics.kitchens.len() {
            for i in 0..self.assembly_diagnostics.kitchens[k].units.len() {
                let id = self.assembly_diagnostics.kitchens[k].units[i];
                let support = &self.objects[id];
                if support.kind != ObjectKind::KitchenCounter {
                    continue;
                }
                let mut r = stream(support.seed, 0x5554_454e_5255_4e53);
                let amount = self.clutter.sample(support.seed);
                if amount > 0. && r.random_bool(f64::from(0.16 + 0.54 * amount)) {
                    if let Some(utensil) = self.counter_utensil(support, &mut r) {
                        self.assembly_diagnostics.kitchens[k].props.push(utensil.id);
                        self.objects.push(utensil);
                    }
                }
            }
        }
    }

    fn counter_utensil(&self, support: &IndoorObject, r: &mut ChaCha8Rng) -> Option<IndoorObject> {
        let kind = ObjectKind::Flatware;
        let seed = r.random();
        let size = crate::scene::procedural_indoor::items::sizing::loose(kind, r)?;
        let mut prop = self.candidate(kind, support.position, size, support.yaw, r);
        prop.seed = seed;
        prop.neighbor = support.neighbor;
        prop.solid = false;
        prop.support = Some(support.id);
        let spoon = geometry::flatware::parameters(&prop).tines == 0;
        let tf = support.transform();
        let bounds = geometry::counter_top_bounds(support);
        let anchor = self
            .objects
            .iter()
            .filter(|o| {
                o.support == Some(support.id)
                    && (matches!(o.kind, ObjectKind::Dishware | ObjectKind::CuttingBoard)
                        || spoon && matches!(o.kind, ObjectKind::Mug | ObjectKind::CoffeeCup))
            })
            .choose(r)
            .map(|o| {
                let pos = (tf.rotation.inverse() * (o.position - support.position)).xz();
                (pos, rotated_half(o.size.xz(), o.yaw - support.yaw))
            });
        let pattern = SurfacePattern::sample(r);
        for attempt in 0..8 {
            let yaw = pattern.yaw(r);
            let center = if let Some((pos, half)) = anchor.filter(|_| attempt < 4) {
                let direction = Vec2::from_angle(r.random_range(-PI..PI));
                let envelope =
                    half + rotated_half(size.xz(), yaw) + Vec2::splat(r.random_range(0.016..0.072));
                pos + direction * (envelope / direction.abs().max(Vec2::splat(1e-6))).min_element()
            } else {
                let Some(free) = pattern.propose(bounds, size.xz(), yaw, 0.015, r) else {
                    continue;
                };
                free.center
            };
            prop.position = tf.transform_point(Vec3::new(center.x, support.size.y, center.y));
            prop.yaw = support.yaw + yaw;
            if self.prop_clear(&prop, support, 0.012)
                && crate::scene::procedural_indoor::humans::supported_prop_clear(self, &prop)
            {
                return Some(prop);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counter_flatware_clears_an_occupied_preparation_station() {
        use crate::scene::procedural_indoor::{humans, validation};
        let scene = IndoorManifest::generate_with_humans(56633, IndoorLayout::Mixed, 0.65, 3, 0.25)
            .unwrap();
        let worker = scene
            .humans
            .iter()
            .find(|h| h.workstation.is_some())
            .unwrap();
        let support = worker.workstation.unwrap();
        let unoccupied =
            IndoorManifest::generate_with_humans(56633, IndoorLayout::Mixed, 0.65, 3, 0.).unwrap();
        let original = unoccupied
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::Flatware && o.support == Some(support))
            .unwrap();
        assert!(humans::supported_prop_clear(&unoccupied, original));
        assert!(!humans::supported_prop_clear(&scene, original));
        assert!(scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Flatware)
            .all(|o| humans::supported_prop_clear(&scene, o)));
        validation::validate_layout(&scene).unwrap();
        validation::validate_geometry(&scene).unwrap();
    }

    #[test]
    fn loose_spoons_and_forks_fit_counters_at_varied_headings() {
        let mut counters = 0;
        let mut spoons = 0;
        let mut forks = 0;
        let mut headings = std::collections::BTreeSet::new();
        for seed in 0..64 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Kitchen, 0.65, 0, 0.)
                    .unwrap();
            for group in &scene.assembly_diagnostics.kitchens {
                counters += group
                    .units
                    .iter()
                    .filter(|&&i| scene.objects[i].kind == ObjectKind::KitchenCounter)
                    .count();
                for &id in &group.props {
                    let o = &scene.objects[id];
                    if o.kind != ObjectKind::Flatware {
                        continue;
                    }
                    let support = &scene.objects[o.support.unwrap()];
                    assert_eq!(support.kind, ObjectKind::KitchenCounter);
                    assert!(scene.clutter.sample(support.seed) > 0.);
                    assert!((support.position.y + support.size.y - o.position.y).abs() < 1e-5);
                    assert!(scene.prop_clear(o, support, 0.012));
                    headings.insert(((o.yaw - support.yaw).rem_euclid(TAU) * 8. / TAU) as u32);
                    if geometry::flatware::parameters(o).tines == 0 {
                        spoons += 1;
                    } else {
                        forks += 1;
                    }
                }
            }
        }
        assert!(
            spoons > 3 && forks > 3 && forks + spoons < counters && headings.len() >= 6,
            "{spoons} spoons, {forks} forks, {counters} counters, {headings:?}"
        );
    }
}
