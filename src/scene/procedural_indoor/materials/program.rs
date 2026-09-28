//! Bounded, metric material programs. A recipe describes correlated colour,
//! roughness and relief layers; seed changes do not merely recolour one bitmap.
use super::{periodic_noise, Surface};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialRecipe {
    #[serde(default)]
    pub layers: Option<super::layers::FinishLayers>,
    pub surface: Surface,
    pub seed: u64,
    pub color: [f32; 3],
    pub roughness: f32,
    pub period_m: f32,
    pub relief_m: f32,
    pub grain_frequency: f32,
    pub cross_frequency: f32,
    pub warp: f32,
    pub contrast: f32,
    pub weathering: f32,
    pub mineral_mix: f32,
    pub weave_mix: f32,
    pub phase: [f32; 2],
    pub panel_count: [u32; 2],
    pub joint_width: f32,
}
pub const SURFACES: [Surface; 35] = [
    Surface::Paint,
    Surface::Accent,
    Surface::Wood,
    Surface::WoodEdge,
    Surface::Floor,
    Surface::Ceiling,
    Surface::Metal,
    Surface::Chrome,
    Surface::Plastic,
    Surface::Fabric,
    Surface::FabricAlt,
    Surface::Glass,
    Surface::Ceramic,
    Surface::Soil,
    Surface::Leaf,
    Surface::LeafLight,
    Surface::Paper,
    Surface::Screen,
    Surface::Ink,
    Surface::Light,
    Surface::Concrete,
    Surface::Art,
    Surface::Rubber,
    Surface::LeafVariegated,
    Surface::Terracotta,
    Surface::Bark,
    Surface::GlassInterior,
    Surface::ContainerGlass,
    Surface::Liquid,
    Surface::Drink,
    Surface::PhoneScreen,
    Surface::PrintedPaper,
    Surface::Leather,
    Surface::Whiteboard,
    Surface::Television,
];
pub fn sample(seed: u64) -> Vec<MaterialRecipe> {
    let mut palette = stream(seed, 71);
    let hue = palette.random_range(0.0..360.0);
    let wood_light = palette.random_range(0.25..0.78);
    let wood_saturation = palette.random_range(0.10..0.70);
    let temperature = palette.random_range(-0.025..0.04);
    let accent_hue = hue + palette.random_range(90.0..210.0);
    SURFACES
        .into_iter()
        .map(|surface| {
            let mut rng = stream(seed, 100 + surface as u64);
            let cloth = matches!(surface, Surface::Fabric | Surface::FabricAlt);
            let wood = matches!(surface, Surface::Wood | Surface::WoodEdge | Surface::Bark);
            let chromatic = bevy::prelude::Color::hsl(
                if matches!(surface, Surface::FabricAlt | Surface::Art) {
                    accent_hue
                } else {
                    hue
                } + rng.random_range(-14.0..14.0),
                rng.random_range(0.12..0.43),
                rng.random_range(0.16..0.51),
            )
            .to_srgba();
            let neutral = rng.random_range(0.68..0.91);
            let color = match surface {
                Surface::Wood | Surface::WoodEdge => [
                    wood_light,
                    wood_light * (1.0 - 0.42 * wood_saturation),
                    wood_light * (1.0 - 0.72 * wood_saturation),
                ],
                Surface::Fabric | Surface::FabricAlt | Surface::Accent | Surface::Art => {
                    [chromatic.red, chromatic.green, chromatic.blue]
                }
                Surface::Paint if rng.random_bool(0.28) => {
                    let tint = bevy::prelude::Color::hsl(
                        hue,
                        rng.random_range(0.04..0.25),
                        rng.random_range(0.38..0.85),
                    )
                    .to_srgba();
                    [tint.red, tint.green, tint.blue]
                }
                Surface::Paint | Surface::Ceiling => {
                    [neutral, neutral, neutral * (1.0 - temperature)]
                }
                Surface::Floor => {
                    let n = rng.random_range(0.3..0.7);
                    [
                        n,
                        n * (1.0 - wood_saturation * 0.18),
                        n * (1.0 - wood_saturation * 0.32),
                    ]
                }
                Surface::Metal | Surface::Plastic => {
                    let n = rng.random_range(0.035..0.25);
                    [n, n * 1.03, n * 1.06]
                }
                Surface::Chrome => {
                    let n = rng.random_range(0.56..0.76);
                    [n, n * 1.01, n * 1.02]
                }
                Surface::Glass
                | Surface::GlassInterior
                | Surface::ContainerGlass
                | Surface::Liquid => [1.0; 3],
                Surface::Drink => [0.09, 0.038, 0.016],
                Surface::Leather => {
                    let n = rng.random_range(0.055..0.40);
                    [
                        n,
                        n * rng.random_range(0.40..0.82),
                        n * rng.random_range(0.25..0.65),
                    ]
                }
                Surface::Ceramic | Surface::Paper | Surface::PrintedPaper => [
                    neutral,
                    neutral * rng.random_range(0.94..1.02),
                    neutral * rng.random_range(0.88..1.01),
                ],
                Surface::Soil => {
                    let n = rng.random_range(0.075..0.16);
                    [n, n * 0.58, n * 0.29]
                }
                Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated => {
                    let green = rng.random_range(0.18..0.42);
                    [
                        green * rng.random_range(0.30..0.70),
                        green,
                        green * rng.random_range(0.14..0.38),
                    ]
                }
                Surface::Concrete => {
                    let n = rng.random_range(0.39..0.68);
                    [n, n * 0.99, n * 0.95]
                }
                Surface::Terracotta => {
                    let n = rng.random_range(0.42..0.64);
                    [n, n * rng.random_range(0.42..0.68), n * 0.3]
                }
                Surface::Bark => {
                    let n = rng.random_range(0.12..0.31);
                    [n, n * 0.7, n * 0.4]
                }
                Surface::Rubber => [0.035, 0.036, 0.037],
                Surface::Ink => [0.04, 0.065, 0.09],
                Surface::Light
                | Surface::Screen
                | Surface::PhoneScreen
                | Surface::Television
                | Surface::Whiteboard => [1.0; 3],
            };
            let roughness = if cloth {
                rng.random_range(0.72..0.98)
            } else if wood {
                rng.random_range(0.18..0.82)
            } else {
                match surface {
                    Surface::Glass | Surface::GlassInterior => {
                        super::glass::GlassRecipe::sample(seed, surface).roughness
                    }
                    Surface::Chrome => rng.random_range(0.14..0.32),
                    Surface::ContainerGlass | Surface::Liquid => rng.random_range(0.035..0.08),
                    Surface::Drink => rng.random_range(0.12..0.22),
                    Surface::Metal => rng.random_range(0.32..0.50),
                    Surface::Plastic => rng.random_range(0.32..0.60),
                    Surface::Ceramic => rng.random_range(0.18..0.38),
                    Surface::Art => rng.random_range(0.28..0.72),
                    Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated => {
                        rng.random_range(0.36..0.63)
                    }
                    Surface::Soil | Surface::Rubber | Surface::Paper | Surface::PrintedPaper => {
                        rng.random_range(0.83..0.99)
                    }
                    Surface::Screen | Surface::PhoneScreen | Surface::Television => {
                        rng.random_range(0.15..0.34)
                    }
                    Surface::Leather => rng.random_range(0.32..0.58),
                    Surface::Whiteboard => rng.random_range(0.18..0.30),
                    _ => rng.random_range(0.62..0.93),
                }
            };
            let period_m = if cloth {
                rng.random_range(0.10..0.28)
            } else if wood {
                rng.random_range(0.28..2.2)
            } else if surface == Surface::Floor {
                rng.random_range(1.2..5.0)
            } else if matches!(
                surface,
                Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
            ) {
                1.0
            } else if surface == Surface::Soil {
                0.2
            } else if matches!(
                surface,
                Surface::Metal
                    | Surface::Chrome
                    | Surface::Plastic
                    | Surface::Rubber
                    | Surface::Paper
                    | Surface::Leather
            ) {
                rng.random_range(0.045..0.16)
            } else {
                rng.random_range(0.4..1.4)
            };
            let relief_m = if cloth {
                rng.random_range(0.00008..0.00035)
            } else if wood {
                rng.random_range(0.000025..0.00022)
            } else if matches!(surface, Surface::Metal | Surface::Chrome) {
                rng.random_range(0.000008..0.000025)
            } else if matches!(
                surface,
                Surface::Plastic | Surface::Paper | Surface::Leather
            ) {
                rng.random_range(0.000010..0.000030)
            } else {
                rng.random_range(0.00002..0.0003)
            };
            MaterialRecipe {
                layers: Some(super::layers::FinishLayers::sample(surface, &mut rng)),
                surface,
                seed: rng.random(),
                color,
                roughness,
                period_m,
                relief_m,
                grain_frequency: rng.random_range(16.0..80.0),
                cross_frequency: rng.random_range(1.0..8.0),
                warp: rng.random_range(0.005..0.09),
                contrast: match surface {
                    Surface::Metal | Surface::Chrome | Surface::Paper | Surface::PrintedPaper => {
                        rng.random_range(0.005..0.025)
                    }
                    Surface::Plastic | Surface::Paint | Surface::Ceiling | Surface::Accent => {
                        rng.random_range(0.015..0.045)
                    }
                    _ => rng.random_range(0.035..0.23),
                },
                weathering: match surface {
                    Surface::Metal | Surface::Chrome | Surface::Paper | Surface::PrintedPaper => {
                        rng.random_range(0.0..0.004)
                    }
                    Surface::Plastic | Surface::Paint | Surface::Ceiling | Surface::Accent => {
                        rng.random_range(0.0..0.018)
                    }
                    _ => rng.random_range(0.0..0.14),
                },
                mineral_mix: rng.random_range(0.0..1.0),
                weave_mix: rng.random_range(0.0..1.0),
                phase: [rng.random(), rng.random()],
                panel_count: [rng.random_range(3..15), rng.random_range(1..6)],
                joint_width: rng.random_range(0.001..0.0035),
            }
        })
        .collect()
}
impl MaterialRecipe {
    /// Integer repetitions preserve tileability while constraining the actual
    /// plank/tile dimensions rather than drawing arbitrary counts per texture.
    pub fn floor_repetitions(&self, floor_style: u32) -> [f32; 2] {
        let width = if floor_style == 0 {
            0.10 + self.panel_count[0] as f32 * 0.010
        } else {
            0.30 + self.panel_count[0] as f32 * 0.055
        };
        let length = if floor_style == 0 {
            0.75 + self.panel_count[1] as f32 * 0.20
        } else {
            0.30 + self.panel_count[1] as f32 * 0.10
        };
        [
            (self.period_m / width).round().max(1.0),
            (self.period_m / length).round().max(1.0),
        ]
    }
    /// Tileable multi-scale layers, evaluated in a physical repeat domain.
    pub fn evaluate(&self, u: f32, v: f32, floor_style: u32) -> (f32, f32, f32) {
        let (u, v) = self.layers.as_ref().map_or((u, v), |l| l.rotate(u, v));
        let u = (u + self.phase[0]).rem_euclid(1.0);
        let v = (v + self.phase[1]).rem_euclid(1.0);
        let noise = |nx, ny, s| periodic_noise(u, v, nx, ny, self.seed.wrapping_add(s));
        let macro_n = noise(3, 3, 1);
        let meso = noise(17, 19, 2);
        let micro = noise(97, 91, 3);
        let mineral = (0.55 * macro_n + 0.3 * meso + 0.15 * micro) * (1.0 - self.mineral_mix)
            + self.mineral_mix * (0.65 * meso + 0.35 * (TAU * u * 5.0 + macro_n * 8.0).sin().abs());
        let count = (self.period_m / 0.0025).round().clamp(12.0, 100.0);
        let threads = (TAU * u * count).sin() * (TAU * v * count).sin();
        // An odd thread count must not leave a half cycle at the tile boundary.
        let twill = (TAU * (u + v) * (count * 0.5).round()).sin();
        let weave = 0.5 + 0.25 * (threads * (1.0 - self.weave_mix) + twill * self.weave_mix);
        let wood = matches!(
            self.surface,
            Surface::Wood | Surface::WoodEdge | Surface::Bark
        ) || (self.surface == Surface::Floor && floor_style == 0);
        let cloth = matches!(self.surface, Surface::Fabric | Surface::FabricAlt)
            || (self.surface == Surface::Floor && floor_style == 1);
        let value = if matches!(self.surface, Surface::Metal | Surface::Chrome) {
            // Long machining grooves; color stays restrained while normal and
            // roughness maps carry the brushed finish under grazing light.
            noise(103, 5, 44) * 0.78 + micro * 0.22
        } else if matches!(
            self.surface,
            Surface::Plastic | Surface::Rubber | Surface::Paper | Surface::Leather
        ) {
            micro * 0.72 + meso * 0.28
        } else if wood {
            super::timber::grain(self, u, v)
        } else if cloth {
            weave * 0.72 + micro * 0.28
        } else {
            mineral
        };
        let stain = ((macro_n - 0.5) * 3.0).max(0.0) * self.weathering;
        let mut shade = 0.96 + (value - 0.5) * self.contrast - stain;
        let mut height = (value - 0.5) * self.relief_m;
        let finish_variation = if matches!(
            self.surface,
            Surface::Metal | Surface::Chrome | Surface::Plastic | Surface::Paper
        ) {
            0.14
        } else if wood || cloth {
            0.26
        } else {
            0.34
        };
        let mut roughness = self.roughness
            + (value - 0.5) * finish_variation
            + (macro_n - 0.5) * self.weathering
            + stain * 0.3;
        if self.surface == Surface::Floor && floor_style != 1 {
            let [nx, ny] = self.floor_repetitions(floor_style);
            let strip = (u * nx).floor() as u32;
            let stagger = if floor_style == 0 {
                (strip % 2) as f32 * 0.5
            } else {
                0.0
            };
            let x = (u * nx).fract();
            let y = (v * ny + stagger).fract();
            let distance =
                (x.min(1.0 - x) * self.period_m / nx).min(y.min(1.0 - y) * self.period_m / ny);
            let filter_width = self.joint_width + self.period_m / 256.0;
            let seam =
                (1.0 - distance / filter_width).clamp(0.0, 1.0) * (self.joint_width / filter_width);
            // Staggered boards cross the texture boundary halfway along their
            // length. Wrap their identity as well as UVs to avoid a false joint.
            let board = super::hash(
                strip,
                (v * ny + stagger).floor() as u32 % ny as u32,
                self.seed,
            );
            shade = shade * (1.0 - 0.22 * seam) + (board - 0.5) * self.contrast;
            height -= seam * self.joint_width * 0.25;
            roughness += seam * 0.12;
        }
        let mut finish = [shade, height, roughness];
        if let Some(layers) = &self.layers {
            layers.apply(
                self,
                [u, v],
                [macro_n, meso, micro],
                &mut finish,
                floor_style,
            );
        }
        (
            finish[0].clamp(0.25, 1.0),
            finish[1],
            finish[2].clamp(0.12, 1.0),
        )
    }
}
