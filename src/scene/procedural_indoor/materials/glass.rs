//! Optical priors for architectural glazing. Values use metres and linear
//! attenuation; opacity is used only by the explicitly reduced portable renderer.
use super::Surface;
use crate::scene::procedural_indoor::{layout::stream, IndoorQuality};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
pub(crate) mod carboy;
pub mod display;
pub(crate) mod door;
pub(crate) mod mirror;
mod sheet;
pub(crate) mod tableware;
pub use sheet::SheetGauge;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlassRecipe {
    pub roughness: f32,
    pub ior: f32,
    pub thickness_m: f32,
    pub attenuation: [f32; 3],
    pub attenuation_distance_m: f32,
}
impl GlassRecipe {
    /// Photopic transmittance of a flat pane in air. Cosine is relative to the
    /// pane normal. Sum internal reflections incoherently and apply the same
    /// linear Beer attenuation and metric thickness as the PBR material.
    pub(crate) fn luminous_transmittance(&self, cosine: f32) -> f32 {
        let c = cosine.clamp(0., 1.);
        if c == 0. {
            return 0.;
        }
        let n = self.ior;
        let ct = (1. - (1. - c * c) / (n * n)).sqrt();
        let rs = ((c - n * ct) / (c + n * ct)).powi(2);
        let rp = ((n * c - ct) / (n * c + ct)).powi(2);
        let path = self.thickness_m / (ct * self.attenuation_distance_m);
        let transmitted = Vec3::from_array(self.attenuation.map(|channel| {
            let absorb = channel.powf(path);
            let plate = |r: f32| {
                (1. - r).powi(2) * absorb / (1. - (r * absorb).powi(2)).max(f32::MIN_POSITIVE)
            };
            0.5 * (plate(rs) + plate(rp))
        }));
        transmitted.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    /// Cosine-weighted transmission under an isotropic sky. Three-point
    /// Gauss-Legendre quadrature over the incident hemisphere.
    pub(crate) fn sky_transmittance(&self) -> f32 {
        [
            (0.11270166, 5. / 18.),
            (0.5, 4. / 9.),
            (0.88729835, 5. / 18.),
        ]
        .into_iter()
        .map(|(c, w)| 2. * w * c * self.luminous_transmittance(c))
        .sum()
    }
    pub fn sample(seed: u64, surface: Surface) -> Self {
        let interior = surface == Surface::GlassInterior;
        let mut rng = stream(seed, if interior { 152 } else { 151 });
        let tint = rng.random_range(0.0..1.0_f32).powf(2.0);
        let hue = rng.random_range(0.0..1.0);
        Self {
            roughness: if interior && rng.random_bool(0.28) {
                rng.random_range(0.18..0.48)
            } else {
                // Only the GGX reflection helper clamps to 0.089. Transmission
                // uses the original roughness squared as its blur radius. The
                // old reflection-floor minimum made every clear pane hazy.
                rng.random_range(0.005..0.045)
            },
            ior: rng.random_range(1.46..1.55),
            thickness_m: if interior { 0.010 } else { 0.008 },
            // Continuous mixtures of neutral smoke, green and bronze absorption.
            attenuation: [
                1.0 - tint * (0.18 + hue * 0.52),
                1.0 - tint * 0.24,
                1.0 - tint * (0.65 - hue * 0.35),
            ],
            attenuation_distance_m: rng.random_range(0.018..0.16),
        }
    }
    pub fn apply(&self, material: &mut StandardMaterial, quality: IndoorQuality) {
        material.base_color = Color::WHITE;
        material.perceptual_roughness = self.roughness;
        material.ior = self.ior;
        material.thickness = self.thickness_m;
        material.attenuation_color = Color::linear_rgb(
            self.attenuation[0],
            self.attenuation[1],
            self.attenuation[2],
        );
        material.attenuation_distance = self.attenuation_distance_m;
        if quality.specular_transmission() {
            material.specular_transmission = 1.0;
        } else {
            let transmitted = self
                .attenuation
                .map(|c| c.powf(self.thickness_m / self.attenuation_distance_m));
            material.base_color = Color::linear_rgba(
                transmitted[0],
                transmitted[1],
                transmitted[2],
                (0.06 + self.roughness * 0.65).clamp(0.06, 0.40),
            );
            material.alpha_mode = AlphaMode::Blend;
        }
        // Architectural panes are closed slabs. Rendering their exit faces as
        // additional entry surfaces doubles refraction and alpha attenuation.
        material.cull_mode = Some(bevy::render::render_resource::Face::Back);
        material.double_sided = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pane_transmission_matches_clear_plate_and_diffuse_integral() {
        let mut clear = GlassRecipe::sample(7, Surface::Glass);
        clear.attenuation = [1.; 3];
        clear.ior = 1.5;
        assert!((clear.luminous_transmittance(1.) - 0.96 / 1.04).abs() < 1e-6);
        assert_eq!(clear.luminous_transmittance(0.), 0.);
        assert!(clear.luminous_transmittance(1e-9).is_finite());
        for seed in 0..64 {
            let recipe = GlassRecipe::sample(seed, Surface::Glass);
            let integral: f64 = (0..256)
                .map(|i| {
                    let c = (i as f32 + 0.5) / 256.;
                    f64::from(2. * c * recipe.luminous_transmittance(c)) / 256.
                })
                .sum();
            assert!((integral - f64::from(recipe.sky_transmittance())).abs() < 0.003);
            let mut thicker = recipe.clone();
            thicker.thickness_m *= 2.;
            assert!(thicker.luminous_transmittance(0.7) <= recipe.luminous_transmittance(0.7));
            assert!(recipe.luminous_transmittance(0.05) < recipe.luminous_transmittance(0.70));
        }
    }
    #[test]
    fn glazing_roles_span_clear_tinted_and_frosted_optics() {
        let mut frosted = 0;
        for seed in 0..256 {
            let exterior = GlassRecipe::sample(seed, Surface::Glass);
            let interior = GlassRecipe::sample(seed, Surface::GlassInterior);
            assert!((0.005..0.045).contains(&exterior.roughness));
            if interior.roughness > 0.18 {
                frosted += 1;
            }
            assert_ne!(exterior.attenuation, interior.attenuation);
            for q in [IndoorQuality::Auto, IndoorQuality::Portable] {
                let mut m = StandardMaterial::default();
                interior.apply(&mut m, q);
                assert_eq!(m.thickness, 0.01);
                assert!(m.attenuation_distance.is_finite());
            }
        }
        assert!((40..100).contains(&frosted));
    }
}
