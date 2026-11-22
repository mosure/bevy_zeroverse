//! Rounded locks and curved crests in metric rest-scalp coordinates. All views
//! and bone poses see the same opaque surface, including the cut at the nape.
use super::{groom::Groom, HairStyle};
use bevy::prelude::*;

/// Each actor modulates the room's shared coiled fibre maps. Unresolved turns
/// spread the highlight across the warped tangent chart.
pub(super) fn finish(material: &mut StandardMaterial, style: HairStyle, radius_m: f32) {
    let radius = ((radius_m - 0.006) / 0.021).clamp(0., 1.);
    let tight = 1. - radius;
    let (roughness, sheen) = if style == HairStyle::Afro {
        (0.86, 0.06)
    } else {
        (0.77, 0.12)
    };
    material.perceptual_roughness = roughness + 0.065 * tight;
    material.anisotropy_strength = sheen + 0.09 * radius;
    material.uv_transform *=
        bevy::math::Affine2::from_scale(Vec2::new(1.14 + 0.23 * tight, 0.87 + 0.22 * radius));
}

pub(super) fn sample(
    groom: &Groom,
    style: HairStyle,
    p: Vec3,
    local: Vec3,
    field: f32,
    uv: Vec2,
) -> (f32, f32, Vec2) {
    let afro = style == HairStyle::Afro;
    // Resolve groups of fibres at the mesh scale. Individual hair fibres are
    // represented by the mipmapped fibre arcs inside each lock.
    let pitch = 0.012 + groom.program.curl_radius_m * 0.64 + groom.program.clump_width_m * 0.37;
    let mut q = (p - groom.centre) / pitch;
    let flow = groom.noise(local + Vec3::new(0.37, 0.21, -0.16), 2.7);
    q.x += 0.42 * flow + q.y * groom.program.sweep * 0.20;
    q.z += 0.34 * flow;
    let lock = (0.5 + 0.87 * groom.noise(q, 0.84)).clamp(0., 1.);
    let bend = groom.noise(q + Vec3::new(1.6 * lock, -0.8 * lock, 1.73), 1.27);
    let crest = (1. - (bend.abs() * 3.0).min(1.)).powi(2) * (0.35 + 0.65 * lock);

    let crown = local.y.clamp(0., 1.);
    let fullness = if afro {
        (0.045 + 0.037 * groom.volume) * (0.42 + 0.58 * crown)
    } else {
        (0.004 + groom.length * groom.volume * (0.19 + 0.18 * groom.curl)) * (0.56 + 0.44 * crown)
    };
    let relief = (0.009 + 0.28 * pitch) * (0.57 * lock + 0.43 * crest);
    let hairline = field + pitch * (0.46 * (lock - 0.5) + 0.20 * flow);
    let taper = (hairline / if afro { 0.043 } else { 0.018 }).clamp(0., 1.);
    let height = 0.0012 + taper * taper * (3. - 2. * taper) * (fullness + relief);
    // Bend the tangent chart with its own group. Using the same displacement
    // on both sides of the periodic seam preserves both the atlas and its mips.
    let uv = uv + pitch * Vec2::new(0.55 * lock + 0.31 * flow, 0.28 * bend);
    (height, hairline, uv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::humans::{sample_person, HumanPoseKind};
    use std::f32::consts::TAU;

    #[test]
    fn coiled_finishes_keep_the_shared_pbr_maps_and_broaden_the_smallest_coils() {
        let mut images = Assets::<Image>::default();
        let map = images.add(Image::default());
        for style in [HairStyle::TightCurls, HairStyle::Afro] {
            let mut small = StandardMaterial {
                base_color_texture: Some(map.clone()),
                metallic_roughness_texture: Some(map.clone()),
                normal_map_texture: Some(map.clone()),
                ..default()
            };
            let mut large = small.clone();
            finish(&mut small, style, 0.006);
            finish(&mut large, style, 0.027);
            assert!(
                small.perceptual_roughness > large.perceptual_roughness
                    && small.anisotropy_strength < large.anisotropy_strength
            );
            for mat in [small, large] {
                assert_eq!(mat.base_color_texture.as_ref(), Some(&map));
                assert_eq!(mat.normal_map_texture.as_ref(), Some(&map));
                assert_eq!(mat.metallic_roughness_texture.as_ref(), Some(&map));
                assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
                assert_eq!(mat.metallic, 0.);
            }
        }
    }

    #[test]
    fn curl_group_relief_and_fibre_direction_follow_seeded_scalp_coordinates() {
        let size = Vec3::new(0.147, 0.258, 0.187);
        let center = Vec3::new(0., 1.62, 0.);
        for style in [HairStyle::TightCurls, HairStyle::Afro] {
            let mut signatures = std::collections::HashSet::new();
            for seed in 30..46 {
                let mut h = sample_person(
                    seed,
                    0,
                    Vec3::ZERO,
                    0.,
                    HumanPoseKind::StandingRelaxed,
                    None,
                    false,
                );
                h.hairstyle = style as u8;
                let g = Groom::new(&h, center - size * 0.5, center + size * 0.5);
                let y = 0.69_f32;
                let ring = (1. - y * y).sqrt();
                let mut offsets = Vec::new();
                let mut shifts = Vec::new();
                for i in 0..96 {
                    let angle = TAU * i as f32 / 96.;
                    let local = Vec3::new(angle.sin() * ring, y, angle.cos() * ring);
                    let p = center + local * size * 0.5;
                    let (height, _, uv) = sample(&g, style, p, local, 0.09, Vec2::ZERO);
                    assert!((0.001..0.18).contains(&height) && uv.is_finite());
                    offsets.push(height);
                    shifts.push(uv);
                    let (root, field, _) = sample(&g, style, p, local, -0.04, Vec2::ZERO);
                    assert!(field < 0. && (root - 0.0012).abs() < 1e-6);
                }
                let differences = offsets
                    .windows(4)
                    .map(|window| (window[3] - window[0]).abs());
                assert!(
                    differences.clone().fold(0., f32::max) > 0.0035,
                    "smooth cap for {seed}"
                );
                assert!(shifts.windows(4).any(|w| w[3].distance(w[0]) > 0.003));
                signatures.insert(offsets[..4].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
            assert_eq!(signatures.len(), 16);
        }
    }
}
