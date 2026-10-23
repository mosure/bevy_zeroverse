//! Legacy palette defaults and sampled recipe overlays, in stable surface order.
use super::*;

pub(super) type Definition = (Surface, [f32; 3], f32, f32, f32);
pub(super) fn definitions(scene: &IndoorManifest) -> [Definition; Surface::COUNT] {
    let (wood, fabric, accent) = match scene.palette {
        0 => ([0.58, 0.37, 0.19], [0.16, 0.25, 0.28], [0.22, 0.34, 0.32]),
        1 => ([0.76, 0.60, 0.40], [0.31, 0.34, 0.35], [0.42, 0.48, 0.52]),
        2 => ([0.31, 0.19, 0.11], [0.26, 0.32, 0.40], [0.22, 0.29, 0.40]),
        3 => ([0.68, 0.50, 0.31], [0.43, 0.24, 0.17], [0.54, 0.32, 0.22]),
        4 => ([0.72, 0.64, 0.48], [0.27, 0.36, 0.23], [0.39, 0.45, 0.31]),
        _ => ([0.45, 0.29, 0.18], [0.37, 0.27, 0.33], [0.40, 0.33, 0.39]),
    };
    let floor = match scene.floor_style {
        0 => [wood[0] * 0.82, wood[1] * 0.82, wood[2] * 0.82],
        1 => [0.36, 0.37, 0.36],
        _ => [0.65, 0.64, 0.59],
    };
    let mut definitions = [
        (Surface::Paint, [0.84, 0.83, 0.79], 0.83, 0.0, 0.5),
        (Surface::Accent, accent, 0.83, 0.0, 0.5),
        (Surface::Wood, wood, 0.38, 0.0, 0.70),
        (Surface::WoodEdge, wood, 0.43, 0.0, 0.70),
        (
            Surface::Floor,
            floor,
            if scene.floor_style == 1 { 0.94 } else { 0.49 },
            0.0,
            2.0,
        ),
        (Surface::Ceiling, [0.88, 0.88, 0.85], 0.9, 0.0, 0.6),
        // Painted/powder-coated furniture metal reflects as a dielectric. Bare
        // polished/brushed metal uses Chrome, with a conductive base layer.
        (Surface::Metal, [0.12, 0.135, 0.15], 0.34, 0.0, 1.0),
        (Surface::Chrome, [0.64, 0.66, 0.68], 0.23, 1.0, 1.0),
        (Surface::Plastic, [0.065, 0.074, 0.08], 0.48, 0.0, 1.0),
        (Surface::Fabric, fabric, 0.92, 0.0, 0.12),
        (Surface::FabricAlt, accent, 0.93, 0.0, 0.12),
        (Surface::Glass, [0.88, 0.96, 0.96], 0.08, 0.0, 1.0),
        (Surface::Ceramic, [0.77, 0.75, 0.69], 0.24, 0.0, 1.0),
        (Surface::Soil, [0.095, 0.055, 0.025], 0.98, 0.0, 0.2),
        (Surface::Leaf, [0.10, 0.25, 0.055], 0.50, 0.0, 1.0),
        (Surface::LeafLight, [0.20, 0.34, 0.09], 0.53, 0.0, 1.0),
        (Surface::Paper, [0.86, 0.86, 0.81], 0.88, 0.0, 1.0),
        (Surface::Screen, [0.055, 0.09, 0.12], 0.27, 0.0, 1.0),
        (Surface::Ink, [0.055, 0.12, 0.19], 0.7, 0.0, 1.0),
        (Surface::Light, [0.90, 0.90, 0.84], 0.4, 0.0, 1.0),
        (Surface::Concrete, [0.52, 0.51, 0.47], 0.86, 0.0, 0.75),
        (Surface::Art, accent, 0.83, 0.0, 1.0),
        (Surface::Rubber, [0.033, 0.035, 0.038], 0.91, 0.0, 1.0),
        (Surface::LeafVariegated, [0.30, 0.42, 0.13], 0.43, 0.0, 1.0),
        (Surface::Terracotta, [0.53, 0.27, 0.16], 0.84, 0.0, 0.3),
        (Surface::Bark, [0.22, 0.16, 0.085], 0.91, 0.0, 0.2),
        (Surface::GlassInterior, [1.0; 3], 0.06, 0.0, 1.0),
        (Surface::ContainerGlass, [1.0; 3], 0.06, 0.0, 1.0),
        (Surface::Liquid, [1.0; 3], 0.04, 0.0, 1.0),
        (Surface::Drink, [0.09, 0.038, 0.016], 0.17, 0.0, 1.0),
        (Surface::PhoneScreen, [1.0; 3], 0.24, 0.0, 1.0),
        (Surface::PrintedPaper, [1.0; 3], 0.9, 0.0, 1.0),
        (Surface::Leather, [0.22, 0.12, 0.07], 0.42, 0.0, 0.14),
        (Surface::Whiteboard, [1.0; 3], 0.22, 0.0, 1.0),
        (Surface::Television, [1.0; 3], 0.24, 0.0, 1.0),
        (Surface::Succulent, [0.16, 0.32, 0.22], 0.47, 0.0, 0.10),
    ];
    if let Some(program) = &scene.program {
        for (surface, rgb, roughness, _, period) in &mut definitions {
            let Some(recipe) = program.materials.get(*surface as usize) else {
                continue;
            };
            if recipe.color != [1.0; 3] {
                *rgb = recipe.color;
            }
            *roughness = recipe.roughness;
            *period = recipe.period_m;
        }
    }
    definitions
}
