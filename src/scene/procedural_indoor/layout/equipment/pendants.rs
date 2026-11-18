//! The shade, suspension and canopy belong to one product. Fitting it over a
//! table or exhibit may change its anchor, without re-rolling the product.
use super::*;

impl IndoorManifest {
    pub(super) fn pendant_equipment(&mut self) {
        for index in 0..self.objects.len() {
            let host = &self.objects[index];
            if host.neighbor
                || !matches!(
                    host.kind,
                    ObjectKind::Table | ObjectKind::CoffeeTable | ObjectKind::SideTable
                )
                || !sampling::Domain::Pendants
                    .rng(host.seed)
                    .random_bool(0.23 + self.density as f64 * 0.25)
            {
                continue;
            }
            if let Some(lamp) = self.pendant_over(host) {
                self.objects.push(lamp);
            }
        }
    }

    pub(in crate::scene::procedural_indoor::layout) fn pendant_over(
        &self,
        host: &IndoorObject,
    ) -> Option<IndoorObject> {
        let mut product = stream(host.seed, 0x50454e4450524f44);
        let diameter = product
            .random_range(0.13_f32..0.55)
            .min(host.size.x.min(host.size.z) * 0.83);
        let drop = product.random_range(0.22..0.88);
        let yaw = product.random_range(-PI..PI);
        let mut lamp = self.candidate(
            ObjectKind::CeilingPendant,
            Vec3::ZERO,
            Vec3::new(diameter, drop, diameter),
            yaw,
            &mut product,
        );
        lamp.solid = false;
        let mut mounting = stream(host.seed, 0x50454e444d4f554e);
        let centered = mounting.random_bool(0.19);
        for attempt in 0..4 {
            let local = if centered && attempt == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(
                    host.size.x * mounting.random_range(-0.29..0.29),
                    0.,
                    host.size.z * mounting.random_range(-0.29..0.29),
                )
            };
            let anchor = host.transform().transform_point(local);
            let bottom = self.ceiling_height(anchor.xz()) - 0.004 - drop;
            if bottom
                < (host.position.y + host.size.y + 0.95).max(self.floor_height(anchor.xz()) + 2.05)
            {
                continue;
            }
            lamp.position = anchor.with_y(bottom);
            if ceiling_fixture_clear(self, &lamp) {
                return Some(lamp);
            }
        }
        None
    }
}
