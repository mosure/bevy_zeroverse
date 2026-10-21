//! Blade-aligned venation, chlorophyll patches and wax; leaf UVs are an atlas.
use super::{
    field::*,
    hash, periodic_noise,
    program::{MaterialRecipe, Texel},
    Surface,
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeafRecipe {
    pub vein_pairs: u32,
    pub vein_rake: f32,
    pub vein_curve: f32,
    pub vein_width: f32,
    pub midrib_width: f32,
    pub chlorophyll_variation: f32,
    pub variegation: f32,
    pub patch_cells: [u32; 2],
    pub pale_color: [f32; 3],
    pub wax: f32,
    pub transmission: f32,
    /// Reference blade width/length for converting atlas derivatives to metres.
    pub reference_size_m: [f32; 2],
}
impl LeafRecipe {
    pub fn sample(seed: u64, surface: Surface) -> Self {
        let mut rng = stream(seed, 0x4c454146);
        let n = rng.random_range(0.62..0.90);
        Self {
            vein_pairs: rng.random_range(5..=18),
            vein_rake: rng.random_range(0.15..0.70),
            vein_curve: rng.random_range(-0.18..0.18),
            vein_width: rng.random_range(0.0015..0.006),
            midrib_width: rng.random_range(0.003..0.013),
            chlorophyll_variation: rng.random_range(0.03..0.30),
            variegation: if surface == Surface::LeafVariegated {
                rng.random_range(0.20..0.85)
            } else {
                rng.random_range(0.0..0.10)
            },
            patch_cells: [rng.random_range(2..=7), rng.random_range(3..=11)],
            pale_color: [
                n,
                n * rng.random_range(0.97..1.03),
                n * rng.random_range(0.65..0.91),
            ],
            wax: rng.random_range(0.0..0.80),
            transmission: rng.random_range(0.015..0.09),
            reference_size_m: [rng.random_range(0.04..0.10), rng.random_range(0.12..0.30)],
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(3..=24).contains(&self.vein_pairs)
            || self.patch_cells.iter().any(|n| !(1..=16).contains(n))
            || [
                self.vein_rake,
                self.vein_width,
                self.midrib_width,
                self.chlorophyll_variation,
                self.variegation,
                self.wax,
                self.transmission,
            ]
            .iter()
            .chain(self.pale_color.iter())
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || !self.vein_curve.is_finite()
            || self.vein_curve.abs() > 0.3
            || self.vein_width <= 0.
            || self.midrib_width <= 0.
            || self
                .reference_size_m
                .iter()
                .any(|v| !v.is_finite() || !(0.02..=0.5).contains(v))
        {
            return Err("invalid botanical material program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, mat: &mut bevy::prelude::StandardMaterial) {
        mat.diffuse_transmission = self.transmission;
        mat.clearcoat = self.wax * 0.25;
        mat.clearcoat_perceptual_roughness = 0.18 + (1. - self.wax) * 0.20;
        mat.reflectance = 0.45;
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, [u, v]: [f32; 2]) -> Texel {
        let x = u - 0.5;
        let side = u32::from(x > 0.);
        let branch = v - x.abs() * self.vein_rake - x * x * self.vein_curve;
        let row = (branch * self.vein_pairs as f32).round();
        let jitter = (hash(row.max(0.) as u32, side, r.seed) - 0.5) * 0.15;
        let d = ((branch * self.vein_pairs as f32 - row - jitter) / self.vein_pairs as f32).abs();
        let secondary = (1. - smooth(d / self.vein_width)) * smooth(x.abs() / self.midrib_width);
        let midrib = 1. - smooth(x.abs() / self.midrib_width);
        let n = |nx, ny, s| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(s));
        let cells = deposit([u, v], self.patch_cells, 0.08, r.seed.wrapping_add(29));
        let patch = smooth((cells - (0.88 - self.variegation * 0.52)) * 9.);
        let green = tint(
            r.color,
            1. + (n(7, 11, 41) - 0.5) * self.chlorophyll_variation,
        );
        let veins = (midrib * 0.20 + secondary * 0.06).min(0.25);
        let color = mix(mix(green, self.pale_color, patch), self.pale_color, veins);
        let micro = n(83, 79, 47) - 0.5;
        Texel {
            color,
            height: r.relief_m * (midrib * 0.65 + secondary * 0.25 + micro * 0.12),
            roughness: (r.roughness + micro * 0.06 + veins * 0.12 + patch * 0.03).clamp(0.20, 0.85),
            occlusion: 1.,
        }
    }
}

/// Thick succulent/cactus cuticle uses tileable wax bloom and stomatal pores.
/// Closed fleshy volumes do not carry a blade midrib, leaf veins or thin-leaf
/// transmission. Units and UV repeat follow the metric surface vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CuticleRecipe {
    pub chlorophyll_variation: f32,
    pub wax: f32,
    pub bloom: f32,
    pub pore_cells: [u32; 2],
    pub pore_contrast: f32,
    pub transmission: f32,
}
impl CuticleRecipe {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x43555449434c45);
        Self {
            chlorophyll_variation: rng.random_range(0.025..0.14),
            wax: rng.random_range(0.10..0.85),
            bloom: rng.random_range(0.0_f32..1.0).powf(2.3) * 0.65,
            pore_cells: [rng.random_range(21..=47), rng.random_range(23..=51)],
            pore_contrast: rng.random_range(0.02..0.09),
            transmission: rng.random_range(0.002..0.025),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.pore_cells.iter().any(|n| !(8..=64).contains(n))
            || [
                self.chlorophyll_variation,
                self.wax,
                self.bloom,
                self.pore_contrast,
                self.transmission,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
        {
            return Err("invalid succulent cuticle program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, mat: &mut bevy::prelude::StandardMaterial) {
        mat.diffuse_transmission = self.transmission;
        mat.clearcoat = self.wax * (1. - self.bloom) * 0.25;
        mat.clearcoat_perceptual_roughness = 0.20 + self.bloom * 0.28;
        mat.reflectance = 0.45;
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, [u, v]: [f32; 2]) -> Texel {
        let macro_n = periodic_noise(u, v, 5, 7, r.seed.wrapping_add(73));
        let meso = periodic_noise(u, v, 13, 17, r.seed.wrapping_add(79));
        let pore = periodic_noise(
            u,
            v,
            self.pore_cells[0],
            self.pore_cells[1],
            r.seed.wrapping_add(83),
        );
        let green = tint(
            r.color,
            1. + (macro_n - 0.5) * self.chlorophyll_variation - (1. - pore) * self.pore_contrast,
        );
        let bloom = self.bloom * (0.82 + meso * 0.18);
        Texel {
            color: mix(green, [0.54, 0.64, 0.57], bloom),
            height: r.relief_m * (pore - 0.5),
            roughness: (r.roughness + (0.5 - pore) * 0.035 + bloom * 0.12).clamp(0.25, 0.82),
            occlusion: 1.,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn succulent_cuticle_is_metric_tileable_seeded_and_has_no_leaf_atlas() {
        let mut wax_bins = std::collections::BTreeSet::new();
        let mut bloom_bins = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let recipes = super::super::program::sample(seed);
            let r = &recipes[Surface::Succulent as usize];
            let c = r.cuticle.as_ref().unwrap();
            c.validate().unwrap();
            assert!(r.leaf.is_none() && (0.000005..0.00005).contains(&r.relief_m));
            let base = super::super::program::sample_base(seed);
            assert_eq!(
                *base[Surface::Succulent as usize].cuticle.as_ref().unwrap(),
                CuticleRecipe::sample(r.seed)
            );
            let replay = super::super::program::sample(seed);
            assert_eq!(
                c,
                replay[Surface::Succulent as usize]
                    .cuticle
                    .as_ref()
                    .unwrap()
            );
            for t in [0., 0.23, 0.57, 1.] {
                let left = r.texel(0., t, 0);
                let right = r.texel(1., t, 0);
                let bottom = r.texel(t, 0., 0);
                let top = r.texel(t, 1., 0);
                assert!(left
                    .color
                    .iter()
                    .zip(right.color)
                    .all(|(a, b)| (a - b).abs() < 1e-6));
                assert!((left.height - right.height).abs() < 1e-10);
                assert!(bottom
                    .color
                    .iter()
                    .zip(top.color)
                    .all(|(a, b)| (a - b).abs() < 1e-6));
                assert!((bottom.roughness - top.roughness).abs() < 1e-6);
            }
            wax_bins.insert((c.wax * 10.) as u32);
            bloom_bins.insert((c.bloom * 10.) as u32);
            let mut mat = bevy::prelude::StandardMaterial::default();
            c.apply(&mut mat);
            assert!(mat.diffuse_transmission <= 0.025 && mat.clearcoat <= 0.25);
        }
        assert!(wax_bins.len() >= 7 && bloom_bins.len() >= 5);
    }
}
