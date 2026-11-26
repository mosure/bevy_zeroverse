//! The outer surface of an electronics case can be dielectric even on a metal chassis.
use super::super::{filter, periodic_noise, sheet, Surface};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use bevy::prelude::*;
use rand::Rng;

const SIDE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Process {
    Molded,
    PowderCoated,
    Brushed,
    BeadBlasted,
}

impl Process {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Molded => "molded_polymer",
            Self::PowderCoated => "powder_coat",
            Self::Brushed => "brushed_aluminium",
            Self::BeadBlasted => "bead_blasted_aluminium",
        }
    }

    pub(super) fn exposed_alloy(self) -> bool {
        matches!(self, Self::Brushed | Self::BeadBlasted)
    }
}

pub(super) struct Finish {
    pub process: Process,
    pub roughness: f32,
    period_m: Vec2,
    relief_m: f32,
    roughness_spread: f32,
    field_seed: u64,
    alloy_srgb: f32,
    anisotropy: f32,
}

impl Finish {
    pub fn sample(seed: u64, surface: Surface, lot: usize) -> Self {
        let mut r = stream(
            seed,
            u64::from_le_bytes(*b"CaseSkin") ^ ((surface as u64) << 8 | lot as u64),
        );
        let process = if surface == Surface::Plastic {
            Process::Molded
        } else if matches!(lot, 1 | 2) && r.random_bool(0.74) {
            if r.random_bool(0.58) {
                Process::Brushed
            } else {
                Process::BeadBlasted
            }
        } else {
            Process::PowderCoated
        };
        let (roughness, period) = match process {
            Process::Molded => (0.41..0.69, 0.010..0.026),
            Process::PowderCoated => (0.43..0.72, 0.013..0.030),
            Process::Brushed => (0.31..0.49, 0.00052..0.00105),
            Process::BeadBlasted => (0.44..0.66, 0.007..0.016),
        };
        let roughness = r.random_range(roughness);
        let transverse_m = r.random_range(period);
        let relief_um = match process {
            Process::Molded => 30.0..76.0,
            Process::PowderCoated => 32.0..88.0,
            Process::BeadBlasted => 14.0..42.0,
            Process::Brushed => {
                // A shallow, broad scratch can round to a flat 8-bit normal. Couple
                // the tool depth to its transverse pitch so the lay survives encoding.
                let pitch_um = transverse_m * 1e6 / 15.;
                0.075 * pitch_um..0.11 * pitch_um
            }
        };
        let relief_m = r.random_range(relief_um) * 1e-6;
        let spread = if process == Process::Brushed {
            0.018..0.052
        } else {
            0.08..0.19
        };
        let roughness_spread = r.random_range(spread);
        let field_seed = r.random();
        let alloy_srgb = r.random_range(0.87..0.95);
        // Tool travel spans centimetres, while individual scratches are tens of
        // micrometres apart. A square period makes a large back cover look embossed.
        let (period_m, anisotropy) = if process == Process::Brushed {
            (
                Vec2::new(r.random_range(0.045..0.095), transverse_m),
                r.random_range(0.23..0.49),
            )
        } else {
            (Vec2::splat(transverse_m), 0.)
        };
        Self {
            process,
            roughness,
            period_m,
            relief_m,
            roughness_spread,
            field_seed,
            alloy_srgb,
            anisotropy,
        }
    }

    fn maps(&self, detail: f32) -> [Image; 2] {
        let mut heights = Vec::with_capacity(SIDE * SIDE);
        let mut packed = Vec::with_capacity(SIDE * SIDE * 4);
        let metal = if self.process.exposed_alloy() { 255 } else { 0 };
        for y in 0..SIDE {
            for x in 0..SIDE {
                let uv = (Vec2::new(x as f32, y as f32) + 0.5) / SIDE as f32;
                let noise =
                    |nx, ny, salt| periodic_noise(uv.x, uv.y, nx, ny, self.field_seed ^ salt);
                let (grain, finish) = match self.process {
                    Process::Molded => (noise(14, 13, 0), noise(6, 5, 1)),
                    Process::PowderCoated => (noise(6, 7, 0), noise(14, 13, 1)),
                    Process::Brushed => (noise(2, 15, 0), noise(3, 9, 1)),
                    Process::BeadBlasted => (noise(15, 14, 0), noise(11, 12, 1)),
                };
                heights.push((0.8 * grain + 0.2 * finish) * self.relief_m * detail);
                let r = self.roughness
                    + (0.65 * grain + 0.35 * finish - 0.5) * self.roughness_spread * detail;
                packed.extend([255, (r * 255.).round() as u8, metal, 255]);
            }
        }
        let normals = sheet::metric_normals::<SIDE>(&heights, self.period_m);
        filter::relief_images(normals, packed, SIDE as u32)
    }

    pub fn apply(
        &self,
        mat: &mut StandardMaterial,
        images: &mut impl AssetStore<Image>,
        detail: f32,
    ) {
        if self.process.exposed_alloy() {
            let v = self.alloy_srgb;
            mat.base_color = Color::srgb(v, v * 0.994, v * 0.986);
            mat.metallic = 1.;
        } else {
            mat.metallic = 0.;
        }
        mat.perceptual_roughness = self.roughness;
        mat.anisotropy_strength = self.anisotropy * detail;
        if self.process == Process::Brushed {
            // The gradient varies across the tool marks, along the bitangent.
            mat.anisotropy_rotation = std::f32::consts::FRAC_PI_2;
        }
        if detail > 0. {
            let [normal, packed] = self.maps(detail);
            mat.normal_map_texture = Some(images.add(normal));
            mat.metallic_roughness_texture = Some(images.add(packed));
            mat.perceptual_roughness = 1.;
            mat.uv_transform = bevy::math::Affine2::from_scale(self.period_m.recip());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_relief_is_metric_linear_filtered_and_disappears_at_zero_detail() {
        use bevy::render::render_resource::TextureFormat;
        use std::collections::BTreeSet;

        let mut processes = BTreeSet::new();
        for seed in 0..512 {
            for surface in [Surface::Plastic, Surface::Metal] {
                let f = Finish::sample(seed, surface, (seed % 4) as usize);
                if !processes.insert(f.process.name()) && f.process != Process::Brushed {
                    continue;
                }
                assert_eq!(surface == Surface::Plastic, f.process == Process::Molded);
                let mut images = Assets::<Image>::default();
                let mut m = StandardMaterial::default();
                f.apply(&mut m, &mut images, 0.);
                assert!(images.is_empty());
                assert_eq!(m.perceptual_roughness, f.roughness);
                assert_eq!(m.metallic, if f.process.exposed_alloy() { 1. } else { 0. });
                f.apply(&mut m, &mut images, 1.);
                assert!(m.base_color_texture.is_none());
                let period = Vec2::new(
                    m.uv_transform.matrix2.x_axis.x,
                    m.uv_transform.matrix2.y_axis.y,
                )
                .recip();
                assert!((period - f.period_m).abs().max_element() < 1e-8);
                if f.process == Process::Brushed {
                    assert!(period.x > period.y * 40. && period.y / 15. < 0.000075);
                    assert!(
                        m.anisotropy_strength > 0.2
                            && m.anisotropy_rotation == std::f32::consts::FRAC_PI_2
                    );
                } else {
                    assert_eq!(m.anisotropy_strength, 0.);
                }
                let n = images.get(m.normal_map_texture.as_ref().unwrap()).unwrap();
                let r = images
                    .get(m.metallic_roughness_texture.as_ref().unwrap())
                    .unwrap();
                for image in [n, r] {
                    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba8Unorm);
                    assert_eq!(image.texture_descriptor.mip_level_count, 6);
                    assert_eq!(image.data.as_ref().unwrap().len(), 5460);
                }
                let normals = n.data.as_ref().unwrap();
                let packed = r.data.as_ref().unwrap();
                let half = f.maps(0.5);
                let slopes = |bytes: &[u8]| {
                    bytes[..SIDE * SIDE * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .fold(Vec2::ZERO, |v, p| {
                            let delta = Vec2::new(p[0] as f32, p[1] as f32) - 127.5;
                            v + delta * delta
                        })
                };
                let xy = slopes(normals);
                let half_xy = slopes(half[0].data.as_ref().unwrap());
                assert!(xy.min_element() > 0. && xy.x + xy.y > half_xy.x + half_xy.y);
                if f.process == Process::Brushed {
                    assert!(xy.y > xy.x * 4., "brushed finish {seed}: {xy:?}");
                }
                assert!(normals[normals.len() - 4..][..2]
                    .iter()
                    .all(|v| (127..=128).contains(v)));
                assert!(packed
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| p[0] == 255 && p[2] == (255. * m.metallic) as u8 && p[3] == 255));
                let mean = packed[..SIDE * SIDE * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| p[1] as f32 / 255.)
                    .sum::<f32>()
                    / (SIDE * SIDE) as f32;
                assert!((mean - f.roughness).abs() < 0.04);
                assert!(packed[packed.len() - 3] as f32 / 255. + 1. / 255. >= mean);
                let replay = Finish::sample(seed, surface, (seed % 4) as usize).maps(1.);
                assert_eq!(replay[0].data, n.data);
                assert_eq!(replay[1].data, r.data);
            }
        }
        assert_eq!(processes.len(), 4);
    }
}
