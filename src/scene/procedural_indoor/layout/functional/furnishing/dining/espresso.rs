use super::*;
use crate::scene::procedural_indoor::{objects::public::espresso, placement::rotated_half};

impl IndoorManifest {
    /// Fit the complete kit on the staff edge. Keep one free landing space for
    /// cups and pickup. A cramped counter can still accept the compact brewer.
    pub(super) fn fit_espresso_bar(&mut self, counter: &IndoorObject) -> bool {
        if self.generator_version < espresso::VERSION {
            return false;
        }
        let mut r = stream(counter.seed, u64::from_le_bytes(*b"Barista!"));
        if !r.random_bool(0.73) {
            return false;
        }
        let kinds = [ObjectKind::EspressoMachine, ObjectKind::CoffeeGrinder];
        let seeds: [u64; 2] = r.random();
        let sizes = [
            espresso::dimensions(kinds[0], seeds[0]),
            espresso::dimensions(kinds[1], seeds[1]),
        ];
        let kit = r.random();
        let side = if r.random_bool(0.5) { 1. } else { -1. };
        for attempt in 0..6 {
            let side = if attempt % 2 == 0 { side } else { -side };
            let yaws = [
                PI + r.random_range(-0.10..0.10),
                PI + r.random_range(-0.24..0.24),
            ];
            let halves = [
                rotated_half(sizes[0].xz(), yaws[0]),
                rotated_half(sizes[1].xz(), yaws[1]),
            ];
            let gap = r.random_range(0.054..0.205);
            let landing = r.random_range(0.205..0.34);
            let run = 2. * (halves[0].x + halves[1].x) + gap;
            let spare = counter.size.x - run - landing - 0.055;
            if spare < 0.
                || halves
                    .iter()
                    .any(|h| counter.size.z - 2. * h.y - 0.023 < 0.022)
            {
                continue;
            }
            // Grinder at either end of the pair, landing beyond the machine.
            let edge = counter.size.x * 0.5 - 0.027 - r.random_range(0.0..=spare);
            let xs = [
                side * (edge - 2. * halves[1].x - gap - halves[0].x),
                side * (edge - halves[1].x),
            ];
            let mut pair: [_; 2] = std::array::from_fn(|i| {
                let available = counter.size.z - 2. * halves[i].y - 0.023;
                let back = r.random_range(0.022..=available.min(0.115));
                let local = Vec3::new(
                    xs[i],
                    counter.size.y,
                    -counter.size.z * 0.5 + back + halves[i].y,
                );
                let mut o = self.candidate(
                    kinds[i],
                    counter.transform().transform_point(local),
                    sizes[i],
                    counter.yaw + yaws[i],
                    &mut r,
                );
                o.id += i;
                o.seed = seeds[i];
                o.finish_seed = Some(kit);
                o.solid = false;
                o.support = Some(counter.id);
                o
            });
            pair[1].interaction_target = Some(pair[0].id);
            if !pair.iter().all(|o| self.prop_clear(o, counter, 0.012)) {
                continue;
            }
            self.objects.extend(pair);
            let limit = counter.size.x * 0.5 + side * xs[0] - halves[0].x;
            // Place loose cups only on the pickup side of the machine.
            for _ in 0..r.random_range(0..=3) {
                let diameter = r.random_range(0.074..0.103);
                let end = counter.size.x * 0.5 - 0.088;
                let inner = end - (limit - 0.18).max(0.);
                let x = -side * r.random_range(inner..=end);
                let z = counter.size.z * r.random_range(-0.21..0.32);
                let h = r.random_range(0.068..0.126);
                let yaw = r.random_range(-PI..PI);
                self.program_prop(
                    counter,
                    ObjectKind::CoffeeCup,
                    Vec3::new(x, 0., z),
                    Vec3::new(diameter, h, diameter),
                    yaw,
                    &mut r,
                );
            }
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::validation;
    #[test]
    fn espresso_bar_faces_staff_and_admits_full_pairs_or_compact_stock() {
        let mut sides = [0; 2];
        let mut compact = 0;
        for seed in 0..64 {
            let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::Cafe, 0.65, 0, 0.25)
                .unwrap();
            validation::validate_layout(&s).unwrap();
            let c = s
                .objects
                .iter()
                .find(|o| o.kind == ObjectKind::ServiceCounter && !o.neighbor)
                .unwrap();
            let on = |k| {
                s.objects
                    .iter()
                    .find(|o| o.kind == k && o.support == Some(c.id))
            };
            let (Some(m), Some(g)) = (
                on(ObjectKind::EspressoMachine),
                on(ObjectKind::CoffeeGrinder),
            ) else {
                assert!(
                    on(ObjectKind::CoffeeMaker).is_some()
                        && on(ObjectKind::CoffeeGrinder).is_none()
                        && on(ObjectKind::EspressoMachine).is_none()
                );
                compact += 1;
                continue;
            };
            assert_eq!(g.interaction_target, Some(m.id));
            assert!(on(ObjectKind::CoffeeMaker).is_none());
            let inv = c.transform().compute_affine().inverse();
            let x = |o: &IndoorObject| inv.transform_point3(o.position).x;
            let half = |o: &IndoorObject| rotated_half(o.size.xz(), o.yaw - c.yaw).x;
            let sign = (x(g) - x(m)).signum();
            let gap = (x(g) - x(m)).abs() - half(m) - half(g);
            assert!((0.053..0.206).contains(&gap));
            assert!(c.size.x * 0.5 + sign * x(m) - half(m) > 0.20);
            for o in [m, g] {
                assert!(s.prop_clear(o, c, 0.0119) && (o.yaw - c.yaw).cos() < -0.96);
                assert!((o.position.y - c.position.y - c.size.y).abs() < 0.0001);
            }
            sides[usize::from(sign > 0.)] += 1;
            if sides.iter().sum::<i32>() == 1 {
                let replay =
                    IndoorManifest::generate_with_humans(seed, IndoorLayout::Cafe, 0.65, 0, 0.25)
                        .unwrap();
                assert_eq!(s.objects, replay.objects);
                let mut broken = s.clone();
                broken.objects[g.id].yaw += PI;
                assert!(validation::validate_layout(&broken).is_err());
                broken.objects[g.id].interaction_target = None;
                assert!(validation::validate_layout(&broken).is_err());
            }
        }
        assert!(
            compact >= 8 && sides.iter().all(|n| *n >= 8),
            "{compact} {sides:?}"
        );
    }
}
