//! Greenery on real furniture tops, after shelves and functional equipment exist.
//! The entire canopy is admitted, not only the pot's smaller contact patch.
use super::*;
use crate::scene::procedural_indoor::{objects, plants};
use std::f32::consts::TAU;

/// Eligible, genuinely flat tops. Cooking/washing surfaces, appliance lids and
/// display-case glazing are deliberately not used as decorative plant supports.
fn support_weight(kind: ObjectKind) -> Option<f32> {
    use ObjectKind::*;
    Some(match kind {
        Desk | Table | KitchenCounter => 1.,
        CoffeeTable | SideTable | DrawerUnit => 1.25,
        Cabinet | Bookcase | WallShelf => 1.65,
        _ => return None,
    })
}

fn top_bounds(support: &IndoorObject) -> (Vec2, Vec2) {
    let mut half = support.size.xz() * 0.5;
    if support.kind == ObjectKind::KitchenCounter {
        return objects::kitchen::counter_top_bounds(support);
    }
    if support.kind == ObjectKind::Cabinet {
        // Open leaves occupy part of the declared envelope, not of the shelf.
        let p = objects::storage::parameters(support);
        half.y -= support.size.x / p.columns as f32 * p.opening_radians.sin();
    }
    (-half, half)
}

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn furnish_supported_botany(&mut self) {
        let program = BotanyProgram::sample(self.seed);
        // A bounded pass over existing supports; no navigation or mesh build is
        // needed. Per-support streams prevent a failed fit shifting every plant.
        let supports: Vec<_> = self
            .objects
            .iter()
            .filter(|o| !o.neighbor && support_weight(o.kind).is_some())
            .cloned()
            .collect();
        for support in supports {
            let mut rng = stream(self.seed ^ support.seed, 6262);
            let probability = program.tabletop_fraction
                * (0.40 + program.abundance * 0.60)
                * support_weight(support.kind).unwrap();
            if !self.permits_table_item(&support, ObjectKind::Plant)
                || !rng.random_bool(probability.min(0.95) as f64)
            {
                continue;
            }
            let cluster = matches!(
                support.kind,
                ObjectKind::Cabinet | ObjectKind::Bookcase | ObjectKind::WallShelf
            ) && support.size.x > 1.0
                && rng.random_bool((program.clustering * 0.55) as f64);
            for _ in 0..if cluster { 2 } else { 1 } {
                self.plant_on_surface(&support, &program, &mut rng);
            }
        }
    }

    fn plant_on_surface(
        &mut self,
        support: &IndoorObject,
        program: &BotanyProgram,
        rng: &mut ChaCha8Rng,
    ) -> bool {
        let (min, max) = top_bounds(support);
        for _ in 0..12 {
            let variant = program.species(rng);
            let mut size = plants::sample_size(variant, true, rng);
            // A shallow shelf gets a naturally small cultivar; retain the whole
            // plant's proportions and sample spare space instead of a tight fit.
            let available = (max - min).min_element() - 0.05;
            let scale = (available * rng.random_range(0.46..0.86) / size.x.max(size.z)).min(1.);
            size *= scale;
            if size.x.min(size.z) < 0.10 || size.y < 0.08 {
                continue;
            }
            let yaw = rng.random_range(0.0..TAU);
            let (s, c) = yaw.sin_cos();
            let half = Vec2::new(
                c.abs() * size.x + s.abs() * size.z,
                s.abs() * size.x + c.abs() * size.z,
            ) * 0.5;
            let lo = min + half + Vec2::splat(0.025);
            let hi = max - half - Vec2::splat(0.025);
            if lo.cmpge(hi).any() {
                continue;
            }
            let mut uv = Vec2::new(rng.random(), rng.random());
            // Most work-desk plants leave a usable central work area, but no
            // compulsory back edge, corner, axis or equally spaced grid exists.
            if support.kind == ObjectKind::Desk && rng.random_bool(0.70) {
                uv.x = if uv.x < 0.5 {
                    uv.x * 0.40
                } else {
                    0.80 + (uv.x - 0.5) * 0.40
                };
            }
            let xy = lo + (hi - lo) * uv;
            let position =
                support
                    .transform()
                    .transform_point(Vec3::new(xy.x, support.size.y, xy.y));
            let mut plant =
                self.candidate(ObjectKind::Plant, position, size, support.yaw + yaw, rng);
            plant.variant = variant;
            plant.solid = false;
            plant.support = Some(support.id);
            if !self.plant_surface_clear(&plant, support) {
                self.rejected_placements += 1;
                continue;
            }
            self.objects.push(plant);
            return true;
        }
        false
    }

    fn plant_surface_clear(&self, plant: &IndoorObject, support: &IndoorObject) -> bool {
        if !self.prop_clear(plant, support, 0.012) {
            return false;
        }
        let (lo, hi) = plant.bounds();
        // Tall foliage must also miss unrelated upper cabinets and plants on
        // adjacent supports, which the same-top clutter test does not cover.
        !self
            .objects
            .iter()
            .filter(|o| o.id != support.id && o.id != plant.id && o.neighbor == plant.neighbor)
            .any(|o| {
                o.obstacle_bounds().into_iter().any(|(a, b)| {
                    lo.cmplt(b + Vec3::splat(0.012)).all() && hi.cmpgt(a - Vec3::splat(0.012)).all()
                })
            })
    }
}

#[cfg(test)]
mod tests;
