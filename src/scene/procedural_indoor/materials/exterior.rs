//! Bounded exterior finishes reuse baked PBR maps; no per-building atlases.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

pub(crate) const FIRST: usize = 200;
pub(crate) const COUNT: usize = 6;
const SURFACES: [Surface; 7] = [
    Surface::Concrete,
    Surface::Paint,
    Surface::Terracotta,
    Surface::WoodEdge,
    Surface::Metal,
    Surface::Glass,
    Surface::Soil,
];

pub(crate) fn supports(surface: Surface, slot: usize) -> bool {
    SURFACES.contains(&surface) && (FIRST..FIRST + COUNT).contains(&slot)
}

pub(crate) fn label(surface: Surface, name: &str, seed: u64) -> String {
    if SURFACES.contains(&surface) {
        let slot = stream(seed, 6392).random_range(FIRST..FIRST + COUNT);
        format!("{name}#finish{slot}")
    } else {
        name.into()
    }
}

/// Exterior pigments are independent of the coordinated interior palette. The
/// base map supplies mineral/coating/timber microstructure, normals and ORM.
pub(crate) fn apply(material: &mut StandardMaterial, seed: u64, surface: Surface, slot: usize) {
    let mut r = stream(seed, 6393 + slot as u64 * 41 + surface as u64);
    let (hue, sat, light) = match surface {
        Surface::Terracotta => (
            r.random_range(9.0..42.0),
            r.random_range(0.20..0.52),
            r.random_range(0.32..0.65),
        ),
        Surface::Concrete => (
            r.random_range(22.0..52.0),
            r.random_range(0.015..0.12),
            r.random_range(0.42..0.93),
        ),
        Surface::Soil => (
            r.random_range(25.0..60.0),
            r.random_range(0.12..0.32),
            r.random_range(0.28..0.65),
        ),
        Surface::WoodEdge => (
            r.random_range(18.0..43.0),
            r.random_range(0.09..0.38),
            r.random_range(0.30..0.92),
        ),
        Surface::Metal => (
            r.random_range(20.0..240.0),
            r.random_range(0.01..0.15),
            r.random_range(0.20..0.78),
        ),
        Surface::Glass => (
            r.random_range(170.0..235.0),
            r.random_range(0.02..0.22),
            r.random_range(0.08..0.30),
        ),
        _ => (
            r.random_range(0.0..360.0),
            r.random_range(0.015..0.26),
            r.random_range(0.35..0.94),
        ),
    };
    material.base_color = Color::hsl(hue, sat, light);
    material.uv_transform *=
        bevy::math::Affine2::from_translation(Vec2::new(r.random(), r.random()));
    if surface == Surface::Glass {
        // These distant panes are opaque facade proxies with real mass behind
        // them, not portals into an unmodelled building. Fresnel/reflections
        // remain dielectric; never turn windows into emissive or metallic tiles.
        material.base_color_texture = None;
        material.normal_map_texture = None;
        material.metallic_roughness_texture = None;
        material.occlusion_texture = None;
        material.alpha_mode = AlphaMode::Opaque;
        material.specular_transmission = 0.;
        material.diffuse_transmission = 0.;
        material.metallic = 0.;
        material.reflectance = 0.5;
        material.perceptual_roughness = r.random_range(0.06..0.30);
        material.clearcoat = 0.;
    } else {
        material.perceptual_roughness *= r.random_range(0.88..1.);
    }
}

pub(super) fn build(
    scene: &IndoorManifest,
    base: &[Handle<StandardMaterial>],
    materials: &mut impl AssetStore<StandardMaterial>,
    used: Option<&std::collections::BTreeSet<(Surface, usize)>>,
) -> std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>> {
    let mut result = std::collections::BTreeMap::new();
    for surface in SURFACES {
        for slot in FIRST..FIRST + COUNT {
            if used.is_some_and(|keys| !keys.contains(&(surface, slot))) {
                continue;
            }
            let mut material = materials.get(&base[surface as usize]).unwrap().clone();
            apply(&mut material, scene.material_seed(), surface, slot);
            result.insert((surface, slot), materials.add(material));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exterior_tags_resolve_after_annotation_ownership_and_keep_maps() {
        for surface in SURFACES {
            for seed in 0..64 {
                let label = format!("{}#exterior", label(surface, "window", seed));
                let slot = finish_slot(surface, &label).unwrap();
                assert!(registered_finish(surface, slot));
                let mut images = Assets::<Image>::default();
                let maps = images.add(Image::default());
                let mut a = StandardMaterial {
                    normal_map_texture: Some(maps.clone()),
                    metallic_roughness_texture: Some(maps),
                    ..default()
                };
                let base = a.clone();
                apply(&mut a, seed, surface, slot);
                assert!(a.perceptual_roughness > 0. && a.perceptual_roughness <= 1.);
                if surface == Surface::Glass {
                    assert_eq!(a.specular_transmission, 0.);
                    assert_eq!(a.metallic, 0.);
                    assert_eq!(a.alpha_mode, AlphaMode::Opaque);
                } else {
                    assert_eq!(a.normal_map_texture, base.normal_map_texture);
                    assert_eq!(
                        a.metallic_roughness_texture,
                        base.metallic_roughness_texture
                    );
                }
            }
        }
    }
}
