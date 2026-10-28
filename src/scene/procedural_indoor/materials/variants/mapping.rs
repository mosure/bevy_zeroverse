//! Independent cuts from a shared sheet of stock. All PBR maps use the same
//! positive, isotropic UV transform: no mirrored or rotated tangent normals.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct FinishMapping {
    /// Multiplier on the recipe's physical texture period, not mesh dimensions.
    pub period_scale: f32,
    /// Offset measured in repeats of the shared texture.
    pub phase: Vec2,
}

pub fn parameters(scene_seed: u64, surface: Surface, slot: usize) -> FinishMapping {
    let (family, low, high) = match surface {
        Surface::Wood | Surface::WoodEdge => (Surface::Wood, 0.65_f32, 1.65_f32),
        Surface::Fabric | Surface::FabricAlt => (Surface::Fabric, 0.70, 1.50),
        Surface::Leather => (surface, 0.72, 1.40),
        Surface::Ceramic | Surface::Metal | Surface::Chrome | Surface::Plastic => {
            (surface, 0.70, 1.60)
        }
        // Leaf veins and displayed documents have semantic UV coordinates.
        // Cropping them to disguise repetition would damage their geometry.
        _ => {
            return FinishMapping {
                period_scale: 1.,
                phase: Vec2::ZERO,
            }
        }
    };
    let mut r = stream(
        scene_seed,
        0x53544f434b435554 ^ (family as u64 * COUNT as u64 + slot as u64),
    );
    FinishMapping {
        period_scale: r.random_range(low.ln()..high.ln()).exp(),
        phase: Vec2::new(r.random(), r.random()),
    }
}

impl FinishMapping {
    pub(super) fn apply(self, material: &mut StandardMaterial) {
        material.uv_transform = bevy::math::Affine2::from_scale_angle_translation(
            Vec2::splat(self.period_scale.recip()),
            0.,
            self.phase,
        ) * material.uv_transform;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_cuts_change_period_and_phase_without_new_maps_or_tangent_rotation() {
        for surface in [
            Surface::Wood,
            Surface::Fabric,
            Surface::Leather,
            Surface::Metal,
            Surface::Chrome,
            Surface::Plastic,
            Surface::Ceramic,
        ] {
            let mut periods = std::collections::BTreeSet::new();
            let mut phases = std::collections::BTreeSet::new();
            for seed in 0..16 {
                for slot in 0..COUNT {
                    let p = parameters(seed, surface, slot);
                    assert_eq!(p, parameters(seed, surface, slot));
                    assert!((0.64..1.66).contains(&p.period_scale));
                    assert!(p.phase.cmpge(Vec2::ZERO).all() && p.phase.cmplt(Vec2::ONE).all());
                    periods.insert((p.period_scale * 100.).round() as u32);
                    phases.insert(((p.phase.x * 10.) as u32, (p.phase.y * 10.) as u32));
                    let mut m = StandardMaterial {
                        base_color_texture: Some(Handle::default()),
                        normal_map_texture: Some(Handle::default()),
                        metallic_roughness_texture: Some(Handle::default()),
                        uv_transform: bevy::math::Affine2::from_scale(Vec2::new(2., 3.)),
                        ..default()
                    };
                    let before = m.clone();
                    p.apply(&mut m);
                    assert_eq!(m.base_color_texture, before.base_color_texture);
                    assert_eq!(m.normal_map_texture, before.normal_map_texture);
                    assert_eq!(
                        m.metallic_roughness_texture,
                        before.metallic_roughness_texture
                    );
                    let u = m.uv_transform.transform_vector2(Vec2::X);
                    let v = m.uv_transform.transform_vector2(Vec2::Y);
                    assert!(u.x > 0. && u.y == 0. && v.x == 0. && v.y > 0.);
                    assert!((u.x / v.y - 2. / 3.).abs() < 1e-5);
                }
            }
            assert!(periods.len() > 50 && phases.len() > 65, "{surface:?}");
        }
    }

    #[test]
    fn related_stock_stays_coherent_and_semantic_uvs_stay_intact() {
        for slot in 0..COUNT {
            assert_eq!(
                parameters(42, Surface::Wood, slot),
                parameters(42, Surface::WoodEdge, slot)
            );
            assert_eq!(
                parameters(42, Surface::Fabric, slot),
                parameters(42, Surface::FabricAlt, slot)
            );
            for surface in [
                Surface::Leaf,
                Surface::LeafLight,
                Surface::LeafVariegated,
                Surface::Succulent,
                Surface::Screen,
                Surface::Television,
                Surface::PhoneScreen,
                Surface::Art,
                Surface::Whiteboard,
                Surface::PrintedPaper,
            ] {
                let mut m = StandardMaterial::default();
                parameters(42, surface, slot).apply(&mut m);
                assert_eq!(m.uv_transform, bevy::math::Affine2::IDENTITY);
            }
        }
    }
}
