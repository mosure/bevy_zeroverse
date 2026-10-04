//! Bounded, metric material programs. A recipe describes correlated colour,
//! roughness and relief layers; seed changes do not merely recolour one bitmap.
use super::{periodic_noise, Surface};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialRecipe {
    #[serde(default)]
    pub layers: Option<super::layers::FinishLayers>,
    #[serde(default)]
    pub textile: Option<super::textile::TextileRecipe>,
    #[serde(default)]
    pub leather: Option<super::leather::LeatherRecipe>,
    #[serde(default)]
    pub mineral: Option<super::mineral::MineralRecipe>,
    #[serde(default)]
    pub coating: Option<super::coating::CoatingRecipe>,
    #[serde(default)]
    pub wood: Option<super::timber::WoodFinish>,
    #[serde(default)]
    pub leaf: Option<super::botanical::LeafRecipe>,
    pub surface: Surface,
    pub seed: u64,
    pub color: [f32; 3],
    pub roughness: f32,
    /// Longitudinal repeat. Timber's transverse cut width is given by `period_uv`.
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
    let mut recipes: Vec<MaterialRecipe> = SURFACES
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
                    let n = rng.random_range(0.78..0.92);
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
                Surface::Ceramic if rng.random_bool(0.42) => {
                    let c = bevy::prelude::Color::hsl(
                        hue + rng.random_range(-35.0..35.0),
                        rng.random_range(0.10..0.60),
                        rng.random_range(0.14..0.68),
                    )
                    .to_srgba();
                    [c.red, c.green, c.blue]
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
                    Surface::Chrome => rng.random_range(0.06..0.24),
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
                    | Surface::Paint
                    | Surface::Accent
                    | Surface::Ceiling
                    | Surface::Ceramic
            ) {
                rng.random_range(0.045..0.16)
            } else {
                rng.random_range(0.4..1.4)
            };
            let relief_m = if cloth {
                rng.random_range(0.00016..0.00055)
            } else if wood {
                rng.random_range(0.000025..0.00022)
            } else if surface == Surface::Chrome {
                // Polished plating has microscopic scratches, not corrugated relief.
                rng.random_range(0.0000002..0.0000015)
            } else if surface == Surface::Metal {
                rng.random_range(0.000008..0.000025)
            } else if matches!(
                surface,
                Surface::Plastic | Surface::Paper | Surface::Ceramic
            ) {
                rng.random_range(0.000010..0.000030)
            } else if surface == Surface::Leather {
                rng.random_range(0.00016..0.00048)
            } else if matches!(surface, Surface::Paint | Surface::Accent | Surface::Ceiling) {
                rng.random_range(0.000045..0.00014)
            } else {
                rng.random_range(0.00002..0.0003)
            };
            let mut recipe = MaterialRecipe {
                layers: Some(super::layers::FinishLayers::sample(surface, &mut rng)),
                textile: None,
                leather: None,
                mineral: None,
                coating: None,
                wood: None,
                leaf: None,
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
                    Surface::Wood | Surface::WoodEdge | Surface::Bark => {
                        rng.random_range(0.25..0.65)
                    }
                    Surface::Fabric | Surface::FabricAlt | Surface::Leather => {
                        rng.random_range(0.16..0.40)
                    }
                    Surface::Metal | Surface::Chrome | Surface::Paper | Surface::PrintedPaper => {
                        rng.random_range(0.005..0.025)
                    }
                    Surface::Plastic
                    | Surface::Paint
                    | Surface::Ceiling
                    | Surface::Accent
                    | Surface::Ceramic => rng.random_range(0.015..0.045),
                    _ => rng.random_range(0.035..0.23),
                },
                weathering: match surface {
                    Surface::Metal | Surface::Chrome | Surface::Paper | Surface::PrintedPaper => {
                        rng.random_range(0.0..0.004)
                    }
                    Surface::Plastic
                    | Surface::Paint
                    | Surface::Ceiling
                    | Surface::Accent
                    | Surface::Ceramic => rng.random_range(0.0..0.018),
                    Surface::Leather | Surface::Wood | Surface::WoodEdge => {
                        rng.random_range(0.0..0.035)
                    }
                    _ => rng.random_range(0.0..0.14),
                },
                mineral_mix: rng.random_range(0.0..1.0),
                weave_mix: rng.random_range(0.0..1.0),
                phase: [rng.random(), rng.random()],
                panel_count: [rng.random_range(3..15), rng.random_range(1..6)],
                joint_width: rng.random_range(0.001..0.0035),
            };
            recipe.specialize();
            recipe
        })
        .collect();
    // End grain and face grain share stain/varnish, while keeping independent cuts.
    recipes[Surface::WoodEdge as usize].wood = recipes[Surface::Wood as usize].wood.clone();
    recipes
}

/// sRGB reflectance (absolute or relative per recipe), metric relief, roughness, AO.
pub(super) struct Texel {
    pub color: [f32; 3],
    pub height: f32,
    pub roughness: f32,
    pub occlusion: f32,
}
/// Specialize the floor recipe once, before recording it in a scene manifest.
/// Carpet needs a textile-scale repeat, not the metre-wide plank/tile domain.
pub(crate) fn floor_finish(recipes: &mut [MaterialRecipe], style: u32) {
    let textile_color = recipes[Surface::Fabric as usize].color;
    let r = &mut recipes[Surface::Floor as usize];
    let choice = super::hash(17, 89, r.seed);
    match style {
        0 => {
            r.roughness = 0.26 + choice * 0.42;
            // The floor atlas spans many boards. Millimetre fibres are below
            // its texel footprint; exaggerating the surviving broad bands
            // makes timber look like a striped sheet across the whole room.
            r.contrast = (r.contrast * 0.70).clamp(0.04, 0.14);
            r.relief_m = 0.00005 + choice * 0.00014;
        }
        1 => {
            if let Some(t) = &mut r.textile {
                t.pile = 0.50 + choice * 0.50;
            }
            if choice > 0.55 {
                r.color = textile_color.map(|c| (c * 0.85).clamp(0.08, 0.80));
            }
            r.period_m = (r.period_m / 16.).clamp(0.12, 0.30);
            r.roughness = 0.88 + choice * 0.10;
            r.relief_m = 0.00030 + choice * 0.00045;
        }
        _ => {
            if let Some(m) = &r.mineral {
                r.roughness = (0.80 - m.polish * 0.58 - m.marble_mix * 0.15).clamp(0.14, 0.90);
                r.relief_m = 0.00002 + (1. - m.polish) * 0.00016;
                let n = 0.18 + super::hash(7, 37, r.seed) * 0.65;
                let warm = super::hash(11, 41, r.seed) * 0.12 - 0.05;
                r.color = [n, n * (1. - warm), (n * (1. - warm * 1.7)).min(0.95)];
            } else {
                r.roughness = 0.22 + choice * 0.53;
                r.relief_m = 0.000025 + choice * 0.00010;
            }
        }
    }
}

/// Sample the palette with the floor's metric scale and finish resolved.
/// Styles are timber (0), carpet (1) and stone/tile (2).
pub fn sample_with_floor(seed: u64, style: u32) -> Result<Vec<MaterialRecipe>, String> {
    if style > 2 {
        return Err("floor style must be 0..2".into());
    }
    let mut recipes = sample(seed);
    floor_finish(&mut recipes, style);
    Ok(recipes)
}

impl MaterialRecipe {
    pub fn validate(&self) -> Result<(), String> {
        if !self
            .color
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.).contains(v))
            || !self.roughness.is_finite()
            || !(0.0..=1.).contains(&self.roughness)
            || !self.period_m.is_finite()
            || self.period_m <= 0.
            || !self.relief_m.is_finite()
            || !(0.0..0.02).contains(&self.relief_m)
            || [
                self.warp,
                self.contrast,
                self.weathering,
                self.mineral_mix,
                self.weave_mix,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || self.phase.iter().any(|v| !v.is_finite())
            || !self.grain_frequency.is_finite()
            || !(1.0..=256.).contains(&self.grain_frequency)
            || !self.cross_frequency.is_finite()
            || !(1.0..=64.).contains(&self.cross_frequency)
            || self.panel_count.iter().any(|v| !(1..=64).contains(v))
            || !self.joint_width.is_finite()
            || !(0.0..=0.02).contains(&self.joint_width)
        {
            return Err("invalid metric material recipe".into());
        }
        if self.leaf.is_some()
            && (!matches!(
                self.surface,
                Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
            ) || self.period_m != 1.)
            || self.coating.is_some()
                && !matches!(
                    self.surface,
                    Surface::Paint | Surface::Accent | Surface::Ceiling | Surface::Ceramic
                )
            || self.wood.is_some()
                && !matches!(
                    self.surface,
                    Surface::Wood | Surface::WoodEdge | Surface::Floor
                )
            || self.mineral.is_some()
                && !matches!(
                    self.surface,
                    Surface::Concrete | Surface::Terracotta | Surface::Soil | Surface::Floor
                )
            || self.textile.is_some()
                && !matches!(
                    self.surface,
                    Surface::Fabric | Surface::FabricAlt | Surface::Floor
                )
            || self.leather.is_some() && self.surface != Surface::Leather
        {
            return Err("substrate program does not match material role".into());
        }
        if let Some(m) = &self.mineral {
            m.validate()?;
            if m.casting.is_some() && self.surface != Surface::Concrete {
                return Err("concrete casting requires a concrete role".into());
            }
        }
        if let Some(c) = &self.coating {
            c.validate()?;
            if c.glaze.is_some() && self.surface != Surface::Ceramic
                || c.application.is_some() && self.surface == Surface::Ceramic
            {
                return Err("glaze/wall application does not match material role".into());
            }
        }
        if let Some(w) = &self.wood {
            w.validate()?;
        }
        if let Some(l) = &self.leaf {
            l.validate()?;
        }
        if let Some(t) = &self.textile {
            t.validate()?;
        }
        if let Some(l) = &self.leather {
            l.validate()?;
        }
        if let Some(l) = &self.layers {
            l.validate()?;
        }
        Ok(())
    }
    /// Bounded resolution: floor atlases cover many independently cut panels.
    pub fn map_size(&self, floor_style: u32) -> u32 {
        if self.surface == Surface::Floor && floor_style != 1
            || self.surface == Surface::Concrete
                && self.mineral.as_ref().is_some_and(|m| m.casting.is_some())
        {
            512
        } else {
            256
        }
    }
    /// Production substrate maps with sRGB color, tangent-space normal, and
    /// linear R=AO/G=perceptual roughness/B=metalness channels, including mips.
    pub fn maps(&self, floor_style: u32) -> [bevy::prelude::Image; 3] {
        super::mapped_images(
            super::texture_maps(
                self.surface,
                floor_style,
                self.seed,
                self.roughness,
                Some(self),
            ),
            self.map_size(floor_style),
            Some(self),
        )
    }
    fn specialize(&mut self) {
        let mut rng = stream(self.seed, 0x5355425354524154);
        if matches!(
            self.surface,
            Surface::Fabric | Surface::FabricAlt | Surface::Floor
        ) {
            let t = super::textile::TextileRecipe::sample(self.seed);
            if self.surface != Surface::Floor {
                self.period_m = rng.random_range(0.0012..0.0035) * t.yarns[1] as f32;
                self.relief_m = rng.random_range(0.00006..0.00022);
                self.roughness = (0.74 + t.fuzz * 0.20 - t.lustre * 0.30).clamp(0.40, 0.98);
                if rng.random_bool(0.55) {
                    let n = rng.random_range(0.08..0.84);
                    let warmth = rng.random_range(-0.03..0.06);
                    self.color = [
                        n,
                        (n * (1. - warmth)).min(0.95),
                        (n * (1. - warmth * 1.7)).min(0.95),
                    ];
                }
            }
            self.textile = Some(t);
        }
        if self.surface == Surface::Leather {
            let l = super::leather::LeatherRecipe::sample(self.seed);
            self.period_m = rng.random_range(0.0008..0.0024) * l.cells[1] as f32;
            self.relief_m = rng.random_range(0.000015..0.00010) * (1. - l.polish * 0.55);
            self.roughness = (0.52 - l.polish * 0.26 + l.nap * 0.36).clamp(0.20, 0.91);
            // Upholstery may be dyed neutral, cream or chromatic as well as brown.
            if rng.random_bool(0.35) {
                let c = bevy::prelude::Color::hsl(
                    rng.random_range(0.0..360.),
                    rng.random_range(0.02..0.38),
                    rng.random_range(0.09..0.75),
                )
                .to_srgba();
                self.color = [c.red, c.green, c.blue];
            }
            self.leather = Some(l);
        }
        if matches!(
            self.surface,
            Surface::Concrete | Surface::Floor | Surface::Terracotta | Surface::Soil
        ) {
            let m = super::mineral::MineralRecipe::sample(self.seed, self.surface);
            if self.surface == Surface::Concrete {
                self.period_m = rng.random_range(0.35..0.90);
                self.relief_m = rng.random_range(0.00006..0.00045) * (1. - m.polish * 0.85);
                self.roughness = 0.88 - m.polish * 0.60;
            } else if matches!(self.surface, Surface::Terracotta | Surface::Soil) {
                self.period_m = rng.random_range(0.06..0.22);
                self.relief_m = rng.random_range(0.00008..0.0005);
            }
            self.mineral = Some(m);
        }
        if matches!(
            self.surface,
            Surface::Paint | Surface::Accent | Surface::Ceiling | Surface::Ceramic
        ) {
            let c = super::coating::CoatingRecipe::sample(self.seed, self.surface);
            self.period_m = if self.surface == Surface::Ceramic {
                rng.random_range(0.085..0.22)
            } else {
                rng.random_range(0.18..0.42)
            };
            self.roughness = if self.surface == Surface::Ceramic {
                0.80 - c.gloss * 0.72
            } else {
                0.95 - c.gloss * 0.72
            };
            self.relief_m = if self.surface == Surface::Ceramic {
                rng.random_range(0.000012..0.000065)
            } else {
                0.000018 + c.texture_mix.powi(2) * 0.00065
            };
            self.coating = Some(c);
        }
        if matches!(
            self.surface,
            Surface::Wood | Surface::WoodEdge | Surface::Floor
        ) {
            self.wood = Some(super::timber::WoodFinish::sample(self.seed));
        }
        if self.surface == Surface::Bark {
            self.period_m = rng.random_range(0.08..0.25);
            self.relief_m = rng.random_range(0.00025..0.0012);
        }
        if matches!(
            self.surface,
            Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
        ) {
            let l = super::botanical::LeafRecipe::sample(self.seed, self.surface);
            self.period_m = 1.;
            self.relief_m = rng.random_range(0.000015..0.000065);
            self.roughness = 0.66 - l.wax * 0.38;
            self.leaf = Some(l);
        }
    }
    /// Bounded structure programs shared by the twelve finish slots.
    pub fn variant(&self, slot: usize) -> Self {
        let mut result = self.clone();
        let group = slot % super::variants::structure_count(self.surface);
        if group != 0
            && matches!(
                self.surface,
                Surface::Ceramic
                    | Surface::Fabric
                    | Surface::FabricAlt
                    | Surface::Leather
                    | Surface::Wood
                    | Surface::WoodEdge
                    | Surface::Leaf
                    | Surface::LeafLight
                    | Surface::LeafVariegated
            )
        {
            result.seed = stream(self.seed, 0x535452554354 + group as u64).random();
            result.specialize();
            result.color = self.color;
            if let Some(w) = &self.wood {
                result.wood = Some(super::timber::WoodFinish::sample(
                    stream(w.seed, 0x46494e495348 + group as u64).random(),
                ));
            }
        }
        result
    }
    pub(super) fn apply_pbr(&self, mat: &mut bevy::prelude::StandardMaterial) {
        if let Some(t) = &self.textile {
            if matches!(self.surface, Surface::Fabric | Surface::FabricAlt) {
                t.apply(self, mat);
            }
        }
        if let Some(l) = &self.leather {
            l.apply(mat);
        }
        if let Some(c) = &self.coating {
            c.apply(mat);
        }
        if let Some(w) = &self.wood {
            if self.surface != Surface::Floor {
                w.apply(mat);
            }
        }
        if let Some(l) = &self.leaf {
            l.apply(mat);
        }
    }
    /// Colored substrate maps contain absolute sRGB reflectance. Upholstery and
    /// modern ceramic glazes retain relative maps for independent pigment tinting.
    pub(super) fn absolute_color(&self, floor_style: u32) -> bool {
        self.leaf.is_some()
            || self.coating.as_ref().is_some_and(|c| c.glaze.is_none())
            || self.surface == Surface::Bark
            || matches!(self.surface, Surface::Wood | Surface::WoodEdge) && self.wood.is_some()
            || matches!(
                self.surface,
                Surface::Concrete | Surface::Terracotta | Surface::Soil
            ) && self.mineral.is_some()
            || self.surface == Surface::Floor
                && (floor_style == 0 && self.wood.is_some()
                    || floor_style == 2 && self.mineral.is_some())
    }
    /// Base-color multiplier paired with `maps`. Mineral/coating/foliage and
    /// stained timber maps carry absolute color; upholstery maps retain tinting.
    pub fn texture_base_color(&self, floor_style: u32) -> [f32; 3] {
        if self.absolute_color(floor_style) {
            [1.; 3]
        } else {
            self.color
        }
    }
    pub fn period_uv(&self) -> bevy::prelude::Vec2 {
        let across = if matches!(self.surface, Surface::Wood | Surface::WoodEdge) {
            (self.period_m * 0.32).clamp(0.22, 0.65)
        } else if let Some(l) = &self.leather {
            self.period_m * l.stretch
        } else {
            self.period_m
        };
        bevy::prelude::Vec2::new(across, self.period_m)
    }

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
        let t = self.texel(u, v, floor_style);
        (
            (t.color[0] + t.color[1] + t.color[2]) / 3.,
            t.height,
            t.roughness,
        )
    }
    pub(super) fn texel(&self, u: f32, v: f32, floor_style: u32) -> Texel {
        // Leaf UVs run from petiole to tip. Rotating/phasing a repeating wall
        // texture here would move the midrib away from the geometric fold.
        if let Some(l) = &self.leaf {
            return l.evaluate(self, [u, v]);
        }
        let (u, v) = self.layers.as_ref().map_or((u, v), |l| l.rotate(u, v));
        let u = (u + self.phase[0]).rem_euclid(1.0);
        let v = (v + self.phase[1]).rem_euclid(1.0);
        if let Some(c) = &self.coating {
            return c.evaluate(self, [u, v]);
        }
        if let Some(m) = &self.mineral {
            if self.surface != Surface::Floor || floor_style == 2 {
                return m.evaluate(self, [u, v], floor_style);
            }
        }
        if self.surface == Surface::Bark {
            return super::timber::bark(self, [u, v]);
        }
        if let Some(w) = &self.wood {
            if self.surface != Surface::Floor || floor_style == 0 {
                let mut t = w.evaluate(self, [u, v]);
                if self.surface == Surface::Floor {
                    self.floor_joints([u, v], &mut t);
                }
                return t;
            }
        }
        if matches!(self.surface, Surface::Fabric | Surface::FabricAlt)
            || self.surface == Surface::Floor && floor_style == 1
        {
            if let Some(t) = &self.textile {
                if self.surface != Surface::Floor {
                    return t.evaluate(self, [u, v]);
                }
                let mut t = t.clone();
                if t.pile == 0. {
                    t.pile = 0.85;
                }
                return t.evaluate(self, [u, v]);
            }
            let mut t = super::textile::TextileRecipe::sample(self.seed);
            if self.surface == Surface::Floor {
                t.pile = 0.85;
            }
            return t.evaluate(self, [u, v]);
        }
        if self.surface == Surface::Leather {
            return self.leather.as_ref().map_or_else(
                || super::leather::LeatherRecipe::sample(self.seed).evaluate(self, [u, v]),
                |l| l.evaluate(self, [u, v]),
            );
        }
        let noise = |nx, ny, s| periodic_noise(u, v, nx, ny, self.seed.wrapping_add(s));
        let macro_n = noise(3, 3, 1);
        let meso = noise(17, 19, 2);
        let micro = noise(97, 91, 3);
        let wood = matches!(
            self.surface,
            Surface::Wood | Surface::WoodEdge | Surface::Bark
        ) || (self.surface == Surface::Floor && floor_style == 0);
        let cloth = matches!(self.surface, Surface::Fabric | Surface::FabricAlt)
            || (self.surface == Surface::Floor && floor_style == 1);
        let value = super::microstructure::value(self, [u, v], [macro_n, meso, micro], floor_style);
        let stain = ((macro_n - 0.5) * 3.0).max(0.0) * self.weathering;
        let mut shade = if wood {
            // Leave headroom for light earlywood instead of clipping almost the
            // entire grain to white. Base tint remains the species/palette.
            0.88 + (value - 0.5) * self.contrast * 1.6 - stain
        } else {
            0.96 + (value - 0.5) * self.contrast - stain
        };
        let mut height = (value - 0.5) * self.relief_m;
        let finish_variation = if self.surface == Surface::Chrome {
            0.035
        } else if matches!(
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
        if matches!(
            self.surface,
            Surface::Leather | Surface::Wood | Surface::WoodEdge
        ) {
            // Pores and creases scatter more broadly than their polished tops.
            roughness += (0.5 - value) * 0.28;
        }
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
        Texel {
            color: [finish[0].clamp(0.25, 1.); 3],
            height: finish[1],
            roughness: finish[2].clamp(0.045, 1.),
            occlusion: 1.,
        }
    }
    fn floor_joints(&self, [u, v]: [f32; 2], t: &mut Texel) {
        let [nx, ny] = self.floor_repetitions(0);
        let strip = (u * nx).floor() as u32;
        let stagger = (strip % 2) as f32 * 0.5;
        let x = (u * nx).fract();
        let y = (v * ny + stagger).fract();
        let d = (x.min(1. - x) * self.period_m / nx).min(y.min(1. - y) * self.period_m / ny);
        let width = self.joint_width + self.period_m / self.map_size(0) as f32;
        let seam = (1. - d / width).clamp(0., 1.) * self.joint_width / width;
        let board = super::hash(
            strip,
            (v * ny + stagger).floor() as u32 % ny as u32,
            self.seed,
        );
        t.color = super::field::tint(
            t.color,
            1. - seam * 0.22 + (board - 0.5) * self.contrast * 0.35,
        );
        t.height -= seam * self.joint_width * 0.25;
        t.roughness = (t.roughness + seam * 0.12).min(1.);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_programs_replay_are_periodic_and_validate_their_physical_scales() {
        for seed in 0..256 {
            let rs = sample(seed);
            for surface in [
                Surface::Paint,
                Surface::Accent,
                Surface::Ceiling,
                Surface::Concrete,
                Surface::Ceramic,
            ] {
                let r = &rs[surface as usize];
                r.validate().unwrap();
                assert_eq!(r, &sample(seed)[surface as usize]);
                for p in [0., 0.19, 0.73, 1.] {
                    for (a, b) in [
                        (r.texel(0., p, 0), r.texel(1., p, 0)),
                        (r.texel(p, 0., 0), r.texel(p, 1., 0)),
                    ] {
                        assert!(
                            a.color
                                .iter()
                                .zip(b.color)
                                .all(|(a, b)| (a - b).abs() < 0.0001),
                            "{surface:?}"
                        );
                        assert!(
                            (a.height - b.height).abs() < 1e-7
                                && (a.roughness - b.roughness).abs() < 0.0001
                        );
                    }
                }
            }
            let concrete = rs[Surface::Concrete as usize].mineral.as_ref().unwrap();
            assert_eq!(concrete.marble_mix, 0., "concrete is not a marble slab");
            assert!(concrete.casting.is_some());
            let ceramic = &rs[Surface::Ceramic as usize];
            assert_eq!(
                ceramic.layers.as_ref().unwrap().quarter_turn % 2,
                0,
                "wheel marks must follow circumference"
            );
            assert!(ceramic.coating.as_ref().unwrap().glaze.is_some());
            assert_ne!(ceramic.variant(0).coating, ceramic.variant(1).coating);
            assert_ne!(ceramic.variant(1).coating, ceramic.variant(2).coating);
            assert_eq!(ceramic.variant(0), ceramic.variant(3));
        }
        let mut r = sample(7).remove(Surface::Ceramic as usize);
        r.coating
            .as_mut()
            .unwrap()
            .glaze
            .as_mut()
            .unwrap()
            .body_grain_m = 0.;
        assert!(r.validate().is_err());
        let mut r = sample(7).remove(Surface::Paint as usize);
        r.coating
            .as_mut()
            .unwrap()
            .application
            .as_mut()
            .unwrap()
            .roller_stretch = f32::NAN;
        assert!(r.validate().is_err());
        let mut r = sample(7).remove(Surface::Concrete as usize);
        r.mineral
            .as_mut()
            .unwrap()
            .casting
            .as_mut()
            .unwrap()
            .bughole_depth_m = 0.1;
        assert!(r.validate().is_err());
    }

    #[test]
    fn ceramic_pigment_is_applied_once_and_old_recorded_recipes_remain_readable() {
        let mut r = sample(81).remove(Surface::Ceramic as usize);
        let maps = r.maps(0);
        r.color = [0.08, 0.23, 0.60];
        assert_eq!(r.texture_base_color(0), r.color);
        for (a, b) in maps.iter().zip(r.maps(0)) {
            assert_eq!(
                a.data, b.data,
                "ceramic pigment was baked into a relative map"
            );
        }
        for surface in [Surface::Ceramic, Surface::Paint, Surface::Concrete] {
            let r = sample(81).remove(surface as usize);
            let mut json = serde_json::to_value(r).unwrap();
            if let Some(c) = json
                .get_mut("coating")
                .and_then(serde_json::Value::as_object_mut)
            {
                c.remove("glaze");
                c.remove("application");
            }
            if let Some(m) = json
                .get_mut("mineral")
                .and_then(serde_json::Value::as_object_mut)
            {
                m.remove("casting");
            }
            let old: MaterialRecipe = serde_json::from_value(json).unwrap();
            old.validate().unwrap();
            assert!(old.absolute_color(0));
            assert!(old.texel(0.12, 0.71, 0).height.is_finite());
        }
    }

    #[test]
    fn mineral_coating_timber_and_foliage_programs_are_bounded_and_seeded() {
        for seed in 0..1024 {
            let recipes = sample(seed);
            assert_eq!(recipes, sample(seed));
            assert_eq!(
                recipes[Surface::Wood as usize].wood,
                recipes[Surface::WoodEdge as usize].wood
            );
            for r in &recipes {
                if let Some(m) = &r.mineral {
                    m.validate().unwrap();
                }
                if let Some(c) = &r.coating {
                    c.validate().unwrap();
                }
                if let Some(w) = &r.wood {
                    w.validate().unwrap();
                }
                if let Some(l) = &r.leaf {
                    l.validate().unwrap();
                }
                for (u, v) in [(0.07, 0.13), (0.49, 0.31), (0.53, 0.87)] {
                    for floor in 0..3 {
                        let t = r.texel(u, v, floor);
                        assert!(
                            t.color
                                .iter()
                                .all(|c| c.is_finite() && (0.0..=1.).contains(c)),
                            "{:?}",
                            r.surface
                        );
                        assert!(t.height.is_finite() && t.height.abs() < 0.003);
                        assert!((0.045..=1.).contains(&t.roughness));
                        assert!((0.8..=1.).contains(&t.occlusion));
                    }
                }
            }
            for slot in 0..12 {
                assert_eq!(
                    recipes[Surface::Wood as usize].variant(slot).wood,
                    recipes[Surface::WoodEdge as usize].variant(slot).wood
                );
            }
        }
    }

    #[test]
    fn leaf_atlas_keeps_midrib_on_the_geometric_fold_and_preserves_pale_pigment() {
        let mut r = sample(31).remove(Surface::LeafVariegated as usize);
        let peak: f32 = (1..32)
            .map(|i| r.texel(0.5, i as f32 / 32., 0).height)
            .sum();
        let blade: f32 = (1..32)
            .map(|i| r.texel(0.25, i as f32 / 32., 0).height)
            .sum();
        assert!(peak > blade * 1.5);
        let before = r.texel(0.5, 0.31, 0);
        r.phase = [0.17, 0.81];
        r.layers.as_mut().unwrap().quarter_turn = 3;
        let after = r.texel(0.5, 0.31, 0);
        assert_eq!(before.color, after.color);
        assert_eq!(before.height, after.height);
        let l = r.leaf.as_mut().unwrap();
        l.variegation = 1.;
        l.pale_color = [0.85, 0.87, 0.68];
        let brightest = (0..32)
            .flat_map(|y| (0..32).map(move |x| (x, y)))
            .map(|(x, y)| r.texel(x as f32 / 32., y as f32 / 32., 0).color[0])
            .fold(0., f32::max);
        assert!(
            brightest > 0.75,
            "pale foliage was multiplied by a dark green base tint"
        );
    }

    #[test]
    fn malformed_substrate_parameters_are_rejected() {
        let mut rs = sample(7);
        let m = rs[Surface::Concrete as usize].mineral.as_mut().unwrap();
        m.aggregate_cells[0] = 0;
        assert!(m.validate().is_err());
        let c = rs[Surface::Paint as usize].coating.as_mut().unwrap();
        c.gloss = f32::NAN;
        assert!(c.validate().is_err());
        let l = rs[Surface::Leaf as usize].leaf.as_mut().unwrap();
        l.vein_width = 0.;
        assert!(l.validate().is_err());
        let w = rs[Surface::Wood as usize].wood.as_mut().unwrap();
        w.stain_strength = 1.1;
        assert!(w.validate().is_err());
    }

    #[test]
    fn weaving_and_leather_programs_replay_close_at_seams_and_span_parameters() {
        let mut topologies = std::collections::BTreeSet::new();
        for seed in 0..512 {
            let recipes = sample(seed);
            assert_eq!(recipes, sample(seed));
            for s in [Surface::Fabric, Surface::FabricAlt, Surface::Leather] {
                let r = &recipes[s as usize];
                if let Some(t) = &r.textile {
                    t.validate().unwrap();
                    topologies.insert((
                        t.repeat,
                        t.float_length,
                        t.advance,
                        t.bundle,
                        t.herringbone,
                    ));
                    let spacing = r.period_m / t.yarns[1] as f32;
                    assert!((0.0012..0.0035).contains(&spacing));
                }
                if let Some(l) = &r.leather {
                    l.validate().unwrap();
                }
                for t in [0.0, 0.123, 0.371, 0.825, 1.0] {
                    for (a, b) in [
                        (r.texel(0., t, 0), r.texel(1., t, 0)),
                        (r.texel(t, 0., 0), r.texel(t, 1., 0)),
                    ] {
                        assert!(a
                            .color
                            .iter()
                            .zip(b.color)
                            .all(|(x, y)| (x - y).abs() < 0.0001));
                        assert!(
                            (a.height - b.height).abs() < 1e-7
                                && (a.roughness - b.roughness).abs() < 0.0001
                        );
                        assert!(a.height.is_finite() && a.height.abs() < 0.001);
                        assert!(
                            (0.18..=1.).contains(&a.roughness) && (0.9..=1.).contains(&a.occlusion)
                        );
                    }
                }
                assert_eq!(r.variant(0), r.variant(3));
                assert_ne!(r.variant(0), r.variant(1));
                assert_ne!(r.variant(1), r.variant(2));
            }
        }
        assert!(
            topologies.len() > 150,
            "weaving topology collapsed: {}",
            topologies.len()
        );
    }

    #[test]
    fn malformed_weaving_programs_are_rejected_without_overflow() {
        let mut t = super::super::textile::TextileRecipe::sample(8);
        t.repeat = u32::MAX;
        assert!(t.validate().is_err());
        t.repeat = 0;
        assert!(t.validate().is_err());
    }

    #[test]
    fn upholstery_variants_change_structure_not_only_palette() {
        let recipes = sample(81);
        for s in [Surface::Fabric, Surface::FabricAlt, Surface::Leather] {
            let a = recipes[s as usize].maps(0);
            let b = recipes[s as usize].variant(1).maps(0);
            assert_ne!(a[1].data, b[1].data, "{s:?} shares all relief");
            assert_ne!(a[2].data, b[2].data, "{s:?} shares all scattering");
            assert!(a[2]
                .data
                .as_ref()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[2] == 0));
            let pixels = &a[0].data.as_ref().unwrap()[..256 * 256 * 4];
            let codes: std::collections::BTreeSet<_> =
                pixels.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
            assert!(codes.len() > 15, "{s:?} has flat albedo");
        }
    }

    #[test]
    fn timber_cut_spans_growth_rings_and_resolves_millimetre_pores() {
        for seed in 0..128 {
            for r in sample(seed)
                .into_iter()
                .filter(|r| matches!(r.surface, Surface::Wood | Surface::WoodEdge))
            {
                let period = r.period_uv();
                assert_eq!(period.y, r.period_m);
                assert!((0.002..0.009).contains(&(period.x / 109.)));
                assert!(period.x <= period.y);
            }
        }
    }

    #[test]
    fn recorded_floor_recipes_have_physical_scale_and_finish() {
        for seed in 0..128 {
            for style in 0..3 {
                let mut recipes = sample(seed);
                let original = recipes.clone();
                floor_finish(&mut recipes, style);
                for (r, old) in recipes.iter().zip(&original) {
                    if r.surface != Surface::Floor {
                        assert_eq!(r, old, "floor specialization modified another substrate");
                    }
                }
                let floor = &recipes[Surface::Floor as usize];
                assert!(floor.relief_m > 0. && floor.relief_m < 0.001);
                if style == 1 {
                    assert!((0.12..=0.30).contains(&floor.period_m));
                    assert!(floor.roughness >= 0.88);
                } else {
                    assert_eq!(floor.period_m, original[Surface::Floor as usize].period_m);
                    if style == 0 {
                        assert!((0.20..=0.76).contains(&floor.roughness));
                    } else {
                        assert!((0.12..=0.95).contains(&floor.roughness));
                    }
                }
            }
        }
    }
}
