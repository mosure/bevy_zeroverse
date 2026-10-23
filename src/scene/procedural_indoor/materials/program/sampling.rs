//! Seeded recipe construction and floor finish overlays. Keep RNG order stable.
use super::super as materials;
use super::*;

pub fn sample(seed: u64) -> Vec<MaterialRecipe> {
    let mut recipes = sample_base(seed);
    for r in &mut recipes {
        if r.surface == Surface::Floor {
            // The physical floor program is resolved after its style is known.
            r.construction = Some(materials::construction::ConstructionSample::sample(r.seed));
        } else {
            materials::construction::sample(r, 0);
        }
    }
    // End grain and face grain belong to the same finished timber stock.
    recipes[Surface::WoodEdge as usize].wood = recipes[Surface::Wood as usize].wood.clone();
    recipes
}

// Independent, retained sampler for bounded construction qualification. This
// also keeps unrelated RNG streams and legacy recipe generation explicit.
pub(in crate::scene::procedural_indoor::materials) fn sample_base(
    seed: u64,
) -> Vec<MaterialRecipe> {
    let palette = PaletteProgram::sample(seed);
    let hue = palette.hue;
    let wood_saturation = palette.wood_saturation;
    let accent_hue = hue + palette.harmony_angle;
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
                (palette.furnishing_saturation + rng.random_range(-0.06..0.06)).clamp(0.035, 0.68),
                (palette.furnishing_lightness + rng.random_range(-0.10..0.10)).clamp(0.08, 0.78),
            )
            .to_srgba();
            let neutral = rng.random_range(0.68..0.91);
            let color = match surface {
                Surface::Wood | Surface::WoodEdge => {
                    PaletteProgram::rgb(palette.wood_hue, wood_saturation, palette.wood_lightness)
                }
                Surface::Fabric | Surface::FabricAlt | Surface::Accent | Surface::Art => {
                    [chromatic.red, chromatic.green, chromatic.blue]
                }
                Surface::Paint => {
                    PaletteProgram::rgb(hue, palette.pigment_strength, palette.wall_lightness)
                }
                Surface::Ceiling => PaletteProgram::rgb(
                    hue,
                    palette.pigment_strength * 0.16,
                    palette.ceiling_lightness,
                ),
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
                Surface::Leaf
                | Surface::LeafLight
                | Surface::LeafVariegated
                | Surface::Succulent => {
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
                        materials::glass::GlassRecipe::sample(seed, surface).roughness
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
                construction: None,
                manufactured: None,
                detail_scale: None,
                flooring: None,
                carpet: None,
                layers: Some(materials::layers::FinishLayers::sample(surface, &mut rng)),
                textile: None,
                leather: None,
                mineral: None,
                coating: None,
                wood: None,
                leaf: None,
                cuticle: None,
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

/// Specialize the floor recipe once, before recording it in a scene manifest.
/// Carpet needs a textile-scale repeat, not the metre-wide plank/tile domain.
pub(crate) fn floor_finish(recipes: &mut [MaterialRecipe], style: u32) {
    floor_finish_base(recipes, style);
    let r = &mut recipes[Surface::Floor as usize];
    if r.construction.is_some() {
        materials::construction::sample(r, style);
    }
}

pub(in crate::scene::procedural_indoor::materials) fn floor_finish_base(
    recipes: &mut [MaterialRecipe],
    style: u32,
) {
    let textile_color = recipes[Surface::Fabric as usize].color;
    let wood_color = recipes[Surface::Wood as usize].color;
    let r = &mut recipes[Surface::Floor as usize];
    let choice = materials::hash(17, 89, r.seed);
    match style {
        0 => {
            r.color = wood_color;
            r.period_m = r.period_m.max(2.8);
            r.roughness = 0.26 + choice * 0.42;
            // The floor atlas spans many boards. Millimetre fibres are below
            // its texel footprint; exaggerating the surviving broad bands
            // makes timber look like a striped sheet across the whole room.
            r.contrast = (r.contrast * 1.1).clamp(0.09, 0.24);
            r.relief_m = 0.00005 + choice * 0.00014;
        }
        1 => {
            if let Some(t) = &mut r.textile {
                t.pile = 0.50 + choice * 0.50;
            }
            let tone = 0.14 + materials::hash(31, 67, r.seed) * 0.36;
            let warmth = materials::hash(13, 29, r.seed) * 0.07 - 0.025;
            let neutral = [tone, tone * (1. - warmth), tone * (1. - warmth * 1.6)];
            // Pigmented commercial carpet is usually heathered/diluted, not
            // the saturated accent upholstery spread across an entire room.
            r.color = if choice > 0.55 {
                [0, 1, 2].map(|i| neutral[i] * 0.65 + textile_color[i] * 0.35)
            } else {
                neutral
            };
            // Separate room-scale dye/installation from filtered pile. A
            // 15 cm upholstery swatch repeated across a room is conspicuous.
            r.period_m = 2.8 + choice * 1.4;
            r.roughness = 0.88 + choice * 0.10;
            r.relief_m = 0.00030 + choice * 0.00045;
        }
        _ => {
            if let Some(m) = &mut r.mineral {
                // Mineral floors range from quiet porcelain/limestone to
                // pronounced marble. Thick, bright level-set loops should
                // not dominate most rooms, especially on dark substrates.
                m.marble_mix = m.marble_mix.powi(4);
                m.vein_width *= 0.45;
                m.vein_strength *= 0.65;
                r.roughness = (0.80 - m.polish * 0.58 - m.marble_mix * 0.15).clamp(0.14, 0.90);
                r.relief_m = 0.00002 + (1. - m.polish) * 0.00016;
                let n = 0.18 + materials::hash(7, 37, r.seed) * 0.65;
                let warm = materials::hash(11, 41, r.seed) * 0.12 - 0.05;
                r.color = [n, n * (1. - warm), (n * (1. - warm * 1.7)).min(0.95)];
            } else {
                r.roughness = 0.22 + choice * 0.53;
                r.relief_m = 0.000025 + choice * 0.00010;
            }
        }
    }
    r.flooring = Some(materials::flooring::FloorFinish::sample(
        r.seed, style, r.period_m,
    ));
    if style == 2 {
        let mut finish = stream(r.seed, 0x4d494e464c4f4f52);
        let m = r.mineral.as_mut().unwrap();
        // Distinct binder/chip/vein balances span quiet porcelain, honed stone,
        // ground aggregate and poured terrazzo. Pigment and polish are coupled
        // to a substrate, not an independent coat of arbitrary white speckles.
        let aggregate: f32 = finish.random_range(0.0..1.);
        m.aggregate_exposure = aggregate.powi(2) * (1. - m.marble_mix);
        m.aggregate_radius = finish.random_range(0.12..0.44);
        if aggregate > 0.55 {
            let hue = finish.random_range(0.0..360.);
            m.aggregate_color = [0.14, 0.72].map(|light| {
                let c = bevy::prelude::Color::hsl(
                    hue,
                    finish.random_range(0.02..0.28),
                    light + finish.random_range(-0.08..0.08),
                )
                .to_srgba();
                [c.red, c.green, c.blue]
            });
        }
        if r.flooring.as_ref().unwrap().layout == materials::flooring::FloorLayout::Seamless {
            r.period_m = finish.random_range(0.80..1.70);
            m.aggregate_cells = [finish.random_range(48..=96), finish.random_range(48..=96)];
            m.marble_mix *= 0.12;
            m.porosity *= 0.15;
            r.flooring = Some(materials::flooring::FloorFinish::sample(
                r.seed, style, r.period_m,
            ));
        }
    }
    if style == 1 {
        r.carpet = Some(materials::carpet::CarpetDesign::sample(
            r.seed, r.color, false,
        ));
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
