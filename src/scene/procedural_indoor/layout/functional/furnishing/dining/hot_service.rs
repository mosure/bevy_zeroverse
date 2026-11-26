//! Install a staffed hot line only on worktop left after the beverage service.
//! The clear guest-side strip lets customers set a plate down. The controls
//! face the same rear working aisle reserved for the host counter.
use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint, SurfacePattern};

use crate::scene::procedural_indoor::layout::refreshments::{serving_envelope, PLATE_LANDING};
const MAX_YAW: f32 = 0.14;

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn fit_hot_food_well(
        &mut self,
        counter: &IndoorObject,
    ) -> f32 {
        let mut r = stream(counter.seed, u64::from_le_bytes(*b"HotLine "));
        if !r.random_bool(0.82) {
            return 0.;
        }
        let seed = r.random();
        let kind = ObjectKind::HotFoodWell;
        let mut size = geometry::dimensions(kind, seed);
        let bounds = counter.size.xz() * 0.5;
        let pattern = SurfacePattern::sample(&mut r);
        let mut o = self.candidate(kind, Vec3::ZERO, size, counter.yaw, &mut r);
        o.seed = seed;
        o.support = Some(counter.id);
        o.solid = false;
        o.neighbor = counter.neighbor;
        o.finish_seed = counter.finish_seed;
        loop {
            for _ in 0..16 {
                let yaw = r.random_range(-MAX_YAW..MAX_YAW);
                let Some(site) = pattern.propose(
                    (-bounds, bounds),
                    size.xz() + Vec2::Y * PLATE_LANDING,
                    yaw,
                    0.019,
                    &mut r,
                ) else {
                    continue;
                };
                let center = site.center - Vec2::new(yaw.sin(), yaw.cos()) * (PLATE_LANDING * 0.5);
                o.size = size;
                o.yaw = counter.yaw + yaw;
                o.position = counter.transform().transform_point(Vec3::new(
                    center.x,
                    counter.size.y,
                    center.y,
                ));
                if self.hot_food_access(&o, counter) {
                    self.objects.push(o);
                    return size.x * (size.z + PLATE_LANDING);
                }
            }
            if size.x < 0.92 {
                return 0.;
            }
            // Swap a two-well chassis for a real single unit on a tight counter.
            size.x *= 0.5;
        }
    }

    pub(in crate::scene::procedural_indoor::layout::functional) fn hot_food_access(
        &self,
        o: &IndoorObject,
        counter: &IndoorObject,
    ) -> bool {
        if counter.kind != ObjectKind::ServiceCounter
            || o.support != Some(counter.id)
            || (o.yaw - counter.yaw).cos() < MAX_YAW.cos() - 1e-5
            || (o.position.y - counter.position.y - counter.size.y).abs() > 0.003
        {
            return false;
        }
        let envelope = serving_envelope(o);
        if !self.prop_clear(&envelope, counter, 0.012) {
            return false;
        }
        let inv = counter.transform().compute_affine().inverse();
        let local = |o: &IndoorObject| Footprint {
            center: inv.transform_point3(o.position).xz(),
            half: rotated_half(o.size.xz(), o.yaw - counter.yaw),
            yaw: o.yaw - counter.yaw,
        };
        let well = local(&envelope);
        // Keep a cup-sized landing in front of any existing self-service taps.
        self.objects
            .iter()
            .filter(|p| p.support == o.support && p.kind == ObjectKind::DrinkDispenser)
            .all(|tap| !well.overlaps(local(&serving_envelope(tap)), 0.012))
    }
}
