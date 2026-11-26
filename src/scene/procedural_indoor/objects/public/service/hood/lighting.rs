//! The optical rail is one switched circuit. Its apertures share rated output;
//! a single shadowed spot represents their combined beam on the cooking surface.
use super::*;
use crate::scene::procedural_indoor::{
    layout::IndoorManifest,
    objects::kitchen,
    photometry::{lighting_seed, FixtureBeam, SourcePower},
};

impl FilterDeck {
    pub(crate) fn power(&self, scene: &IndoorManifest, o: &IndoorObject) -> SourcePower {
        let mut product = stream(o.seed, u64::from_le_bytes(*b"HoodFlux"));
        let rated_lumens = match self.lamp {
            LampStyle::Puck => self.emitting_area_m2() * product.random_range(65_000.0..153_000.0),
            LampStyle::Linear => {
                self.lens_size.x * self.lamps as f32 * product.random_range(450.0..980.0)
            }
        };
        let kelvin = product.random_range(2700.0..5300.0);
        let tint = product.random_range(-0.024..0.024);
        let mut switch = stream(
            lighting_seed(scene) ^ o.seed.rotate_left(27),
            u64::from_le_bytes(*b"HoodDial"),
        );
        let dimmer = if switch.random_bool(0.25) {
            0.
        } else {
            switch.random_range(0.12_f32..1.).sqrt()
        };
        let scale = scene
            .appearance
            .as_ref()
            .map_or(1., |a| a.illumination_scale);
        SourcePower {
            kelvin,
            tint,
            dimmer,
            lumens: rated_lumens * dimmer * scale,
            color: scene.illuminant_model().lamp_srgb(kelvin, tint),
        }
    }

    pub(crate) fn beam(&self, scene: &IndoorManifest, o: &IndoorObject) -> FixtureBeam {
        // Just below the bezel. The midpoint may sit between lenses, so the
        // probe must still trace each individual optical face.
        let centroid = (self.lens_center(0) + self.lens_center(self.lamps - 1)) * 0.5;
        let position = o.transform().transform_point(centroid - Vec3::Y * 0.002);
        let target = scene
            .assembly_diagnostics
            .mounts
            .iter()
            .find(|m| m.object == o.id)
            .and_then(|m| scene.objects.get(m.host))
            .filter(|host| matches!(host.kind, ObjectKind::Range | ObjectKind::Cooktop))
            .map_or(o.position - Vec3::Y * 0.72, |host| {
                kitchen::appliances::cooking_surface(host, &kitchen::parameters(host)).0
            });
        let mut optic = stream(o.seed, u64::from_le_bytes(*b"HoodBeam"));
        let outer = optic.random_range(49.0_f32..68.).to_radians();
        FixtureBeam {
            position,
            direction: (target - position).try_normalize().unwrap_or(Vec3::NEG_Y),
            radius: (self.emitting_area_m2() / (std::f32::consts::PI * self.lamps as f32)).sqrt(),
            inner: outer * optic.random_range(0.46..0.74),
            outer,
        }
    }
}
