//! Fit the canopy to the work surface and the narrower flue to its own roof contact.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::polygon,
    layout::equipment::intersects_installed_ceiling_light,
    objects::{
        kitchen,
        public::service::{hood_parameters, HoodProgram},
    },
};

fn flue_corners(hood: &IndoorObject, product: &HoodProgram) -> [Vec2; 4] {
    let (center, width) = product.chimney(hood.size);
    let half = width * 0.5;
    let tf = hood.transform();
    [
        (-half.x, -half.y),
        (half.x, -half.y),
        (half.x, half.y),
        (-half.x, half.y),
    ]
    .map(|(x, z)| tf.transform_point(center + Vec3::new(x, 0., z)).xz())
}

fn footprint_bounds(corners: [Vec2; 4]) -> (Vec2, Vec2) {
    corners.into_iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(p), hi.max(p)),
    )
}

impl IndoorManifest {
    pub(super) fn mount_height(&self, unit: &IndoorObject) -> f32 {
        if unit.kind == ObjectKind::ExtractionHood && self.generator_version >= 123 {
            let (a, b) = footprint_bounds(flue_corners(unit, &hood_parameters(unit)));
            self.minimum_ceiling_height(a, b)
        } else {
            self.ceiling_height(unit.position.xz())
        }
    }

    pub(super) fn hood_over(&self, host: &IndoorObject) -> Option<IndoorObject> {
        let stove = kitchen::parameters(host);
        let (center, stove_size) = kitchen::appliances::cooking_surface(host, &stove);
        let mut rng = stream(host.seed, 8270);
        let clearance = if stove.gas {
            rng.random_range(0.73..0.89)
        } else {
            rng.random_range(0.62..0.83)
        };
        let bottom = center.y + clearance;
        let size = Vec3::new(
            stove_size.x * rng.random_range(1.0..1.22),
            1.,
            stove_size.y * rng.random_range(0.86..1.12),
        );
        let mut hood = self.candidate(
            ObjectKind::ExtractionHood,
            center.with_y(bottom),
            size,
            host.yaw,
            &mut rng,
        );
        hood.solid = false;
        let product = hood_parameters(&hood);
        let chimney = flue_corners(&hood, &product);
        let (a, b) = footprint_bounds(chimney);
        let contact = self.minimum_ceiling_height(a, b);
        hood.size.y = contact - bottom - 0.004;
        // The conservative axis-aligned roof query may include a soffit outside
        // the rotated flue. A real corner must come within the collar allowance.
        if hood.size.y < 0.30
            || chimney
                .into_iter()
                .all(|p| self.ceiling_height(p) > contact + 0.015)
        {
            return None;
        }
        let (lo, hi) = hood.bounds();
        let canopy_top = bottom + product.canopy_top(hood.size.y);
        if self.minimum_ceiling_height(lo.xz(), hi.xz()) < canopy_top + 0.012
            || self
                .envelope
                .as_ref()
                .is_some_and(|e| !polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), 0.001))
        {
            return None;
        }
        let occupied = [
            (lo, hi.with_y(canopy_top)),
            (Vec3::new(a.x, bottom, a.y), Vec3::new(b.x, hi.y, b.y)),
        ];
        let clear = occupied.into_iter().all(|(lo, hi)| {
            !intersects_installed_ceiling_light(self, lo, hi)
                && self
                    .objects
                    .iter()
                    .flat_map(IndoorObject::obstacle_bounds)
                    .chain(self.columns())
                    .all(|(a, b)| {
                        !lo.cmplt(b + Vec3::splat(0.012)).all()
                            || !hi.cmpgt(a - Vec3::splat(0.012)).all()
                    })
        });
        clear.then_some(hood)
    }
}
