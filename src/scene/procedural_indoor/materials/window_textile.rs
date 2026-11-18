//! Window fabrics use the scene's relative yarn maps but choose their own dye
//! and lining. All openings using one stock share the same material handle.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lining {
    Unlined,
    Dimout,
    Blackout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowTextile {
    pub lining: Lining,
    /// Ivory, flax, mineral gray or independently dyed yarn.
    pub colorway: u8,
}

const FIRST: usize = 336;
const LININGS: [Lining; 3] = [Lining::Unlined, Lining::Dimout, Lining::Blackout];

pub(crate) fn receives_transmitted_shadows(
    surface: Surface,
    label: &crate::render::semantic::SemanticLabel,
) -> bool {
    use crate::render::semantic::SemanticLabel;
    matches!(surface, Surface::Fabric | Surface::FabricAlt)
        && matches!(label, SemanticLabel::Blinds | SemanticLabel::Curtain)
}

impl WindowTextile {
    pub(crate) fn sample(r: &mut impl Rng, curtain: bool) -> Self {
        let lining = match r.random_range(0..10) {
            0..=3 => Lining::Unlined,
            4..=7 => Lining::Dimout,
            _ => Lining::Blackout,
        };
        let colorway = match r.random_range(0..10) {
            0..=3 => 0,
            4..=6 => 1,
            7..=8 if !curtain => 2,
            _ => 3,
        };
        Self { lining, colorway }
    }

    pub(crate) fn slot(self) -> usize {
        FIRST + self.lining as usize * 4 + self.colorway as usize
    }

    fn apply(self, seed: u64, mat: &mut StandardMaterial) {
        let mut rng = stream(seed, 0x5348414445445945 + self.slot() as u64);
        let (hue, saturation, lightness) = match self.colorway {
            0 => (
                rng.random_range(37.0..57.),
                rng.random_range(0.03..0.18),
                rng.random_range(0.74..0.92),
            ),
            1 => (
                rng.random_range(27.0..49.),
                rng.random_range(0.13..0.36),
                rng.random_range(0.45..0.75),
            ),
            2 => (
                rng.random_range(190.0..250.),
                rng.random_range(0.02..0.13),
                rng.random_range(0.31..0.69),
            ),
            _ => (
                rng.random_range(0.0..360.),
                rng.random_range(0.14..0.43),
                rng.random_range(0.24..0.63),
            ),
        };
        mat.base_color = Color::hsl(hue, saturation, lightness);
        mat.metallic = 0.;
        mat.clearcoat = 0.;
        mat.specular_transmission = 0.;
        mat.diffuse_transmission = match self.lining {
            Lining::Unlined => rng.random_range(0.23..0.46),
            Lining::Dimout => rng.random_range(0.035..0.12),
            Lining::Blackout => 0.,
        };
        // The installed cloth is a closed shell with outward-facing triangles
        // on both sides. Sampling its inner backfaces doubles the yarn layers
        // at grazing angles and gives unstable lighting on tight folds.
        mat.cull_mode = Some(bevy::render::render_resource::Face::Back);
        mat.double_sided = false;
        mat.thickness = 0.0012;
        mat.reflectance = 0.35;
        // Perceptual roughness multiplies the mapped yarn roughness.
        mat.perceptual_roughness = rng.random_range(0.92..1.0);
        mat.uv_transform =
            bevy::math::Affine2::from_scale(Vec2::splat(rng.random_range(1.25..2.5)))
                * mat.uv_transform;
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    matches!(surface, Surface::Fabric | Surface::FabricAlt) && (FIRST..FIRST + 12).contains(&slot)
}

pub(super) fn install(
    scene: &IndoorManifest,
    base: &[Handle<StandardMaterial>],
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    for surface in [Surface::Fabric, Surface::FabricAlt] {
        for lining in LININGS {
            for colorway in 0..4 {
                let cloth = WindowTextile { lining, colorway };
                let key = (surface, cloth.slot());
                if selection.is_some_and(|s| !s.finishes.contains(&key)) {
                    continue;
                }
                let mut mat = materials.get(&base[surface as usize]).unwrap().clone();
                cloth.apply(scene.material_seed(), &mut mat);
                variants.insert(key, materials.add(mat));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_dyes_and_linings_share_all_yarn_maps_and_stay_opaque_to_annotations() {
        let maps = [
            Handle::<Image>::default(),
            Handle::default(),
            Handle::default(),
        ];
        let mut colors = std::collections::BTreeSet::new();
        let mut slots = std::collections::BTreeSet::new();
        for lining in LININGS {
            for colorway in 0..4 {
                let stock = WindowTextile { lining, colorway };
                assert!(slots.insert(stock.slot()));
                for seed in 0..32 {
                    let parent = StandardMaterial {
                        base_color: Color::srgb(0., 1., 0.),
                        base_color_texture: Some(maps[0].clone()),
                        normal_map_texture: Some(maps[1].clone()),
                        metallic_roughness_texture: Some(maps[2].clone()),
                        ..default()
                    };
                    let mut m = parent.clone();
                    stock.apply(seed, &mut m);
                    assert_eq!(m.base_color_texture, parent.base_color_texture);
                    assert_eq!(m.normal_map_texture, parent.normal_map_texture);
                    assert_eq!(
                        m.metallic_roughness_texture,
                        parent.metallic_roughness_texture
                    );
                    assert_eq!(m.alpha_mode, AlphaMode::Opaque);
                    assert_eq!(m.cull_mode, Some(bevy::render::render_resource::Face::Back));
                    assert!(!m.double_sided);
                    assert_eq!((m.specular_transmission, m.metallic), (0., 0.));
                    assert!(matches!(
                        (lining, m.diffuse_transmission),
                        (Lining::Blackout, 0.0)
                            | (Lining::Unlined, 0.23..0.46)
                            | (Lining::Dimout, 0.035..0.12)
                    ));
                    assert!(supports(Surface::Fabric, stock.slot()));
                    assert!(registered_finish(Surface::FabricAlt, stock.slot()));
                    assert!(!registered_finish(Surface::Metal, stock.slot()));
                    colors.insert(m.base_color.to_srgba().to_u8_array());
                }
            }
        }
        assert!(colors.len() > 345);
    }
}
