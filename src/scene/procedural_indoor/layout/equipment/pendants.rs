//! The shade, suspension and canopy belong to one product. Fitting it over a
//! table or exhibit may change its anchor, without re-rolling the product.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::ceiling::{highest_floor, CeilingGrid},
    objects::equipment::pendant::Canopy,
};

pub(super) fn mounted_clear(scene: &IndoorManifest, o: &IndoorObject, lo: Vec3, hi: Vec3) -> bool {
    let c = Canopy::sample(o);
    let center = o.position.xz();
    if !canopy_support_height(scene, center, c).is_some_and(|y| (hi.y - y).abs() < 0.001)
        || scene.envelope.as_ref().is_some_and(|e| {
            !super::super::super::envelope::polygon::box_inside(
                &e.footprint,
                lo.xz(),
                hi.xz(),
                0.02,
            )
        })
        || lo.y < highest_floor(scene, lo.xz(), hi.xz()) + 2.05
    {
        return false;
    }
    // The canopy skirt and lower shade/suspension have separate ceiling
    // reserves. A broad shade does not occupy the roof plane.
    if CeilingGrid::for_scene(scene).is_none()
        && (hi.y - c.pad_height_m
            > scene.minimum_ceiling_height(center - c.radius_m, center + c.radius_m)
            || hi.y - c.height_m > scene.minimum_ceiling_height(lo.xz(), hi.xz()))
    {
        return false;
    }
    let clear =
        |(a, b): (Vec3, Vec3)| !((lo - 0.003).cmplt(b).all() && (hi + 0.003).cmpgt(a).all());
    scene
        .ceiling_installation
        .as_ref()
        .is_none_or(|installation| {
            installation.modules.iter().all(|m| clear(m.bounds()))
                && installation
                    .framing
                    .as_ref()
                    .is_none_or(|f| f.bounds().all(clear))
                && installation
                    .room_grid
                    .as_ref()
                    .is_none_or(|g| g.air_reservations(scene.room_size.y).all(clear))
        })
}

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
        let canopy = Canopy::sample(&lamp);
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
            let Some(ceiling) = canopy_support_height(self, anchor.xz(), canopy) else {
                continue;
            };
            let bottom = ceiling - drop;
            if bottom
                < (host.position.y + host.size.y + 0.95).max(self.floor_height(anchor.xz()) + 2.05)
            {
                continue;
            }
            lamp.position = anchor.with_y(bottom);
            let (lo, hi) = lamp.bounds();
            if bottom
                >= highest_floor(self, lo.xz(), hi.xz())
                    + 2.05
                    + super::super::floor::area::MAX_THICKNESS_M
                && ceiling_fixture_clear(self, &lamp)
            {
                return Some(lamp);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        architecture::ceiling::{CeilingInstallation, CeilingModule, CeilingSystem},
        envelope::FloorPatch,
        objects::audit,
    };

    #[test]
    fn narrow_mount_fits_a_rake_while_low_soffits_and_services_block_the_head() {
        let mut scene =
            IndoorManifest::generate_with_humans(7, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
        scene.ceiling_installation = Some(CeilingInstallation::default());
        let e = scene.envelope.as_mut().unwrap();
        e.soffits.clear();
        e.floor_patches.clear();
        e.mezzanine = None;
        e.ceiling_drop = Vec2::new(scene.room_size.x * 0.065, 0.);
        scene.rug_supports.clear();
        let mut o = audit::fixture(ObjectKind::CeilingPendant, 171);
        o.position = Vec3::ZERO;
        o.size = Vec3::new(0.50, 0.36, 0.50);
        let c = Canopy::sample(&o);
        let y = canopy_support_height(&scene, Vec2::ZERO, c).unwrap();
        assert!((scene.ceiling_height(Vec2::X * c.pad_radius_m) - y).abs() < 1e-6);
        o.position.y = y - o.size.y;
        let (lo, hi) = o.bounds();
        assert!(mounted_clear(&scene, &o, lo, hi));
        assert!(scene.ceiling_height(o.position.xz() + Vec2::X * o.size.x * 0.5) < hi.y - 0.004);
        scene
            .ceiling_installation
            .as_mut()
            .unwrap()
            .modules
            .push(CeilingModule {
                center: o.position + Vec3::Y * 0.15,
                size: Vec3::new(0.5, 0.1, 0.1),
                system: CeilingSystem::RoundDuct,
                support_top_m: y,
                pitch_m: 0.4,
                thickness_m: 0.02,
                seed: 1,
                detail: None,
                suspension: None,
            });
        assert!(!mounted_clear(&scene, &o, lo, hi));
        scene.ceiling_installation.as_mut().unwrap().modules.clear();
        // This strip crosses the shade without covering the centre or a corner.
        let e = scene.envelope.as_mut().unwrap();
        e.soffits.push(FloorPatch {
            min: Vec2::new(0.15, -0.4),
            max: Vec2::new(0.18, 0.4),
            height: y - 0.1,
        });
        assert!(!mounted_clear(&scene, &o, lo, hi));
        scene.envelope.as_mut().unwrap().soffits[0].min.x = 0.003;
        assert!(canopy_support_height(&scene, Vec2::ZERO, c).is_none());
    }
}
