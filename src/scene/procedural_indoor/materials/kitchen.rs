//! Small shared food finishes. Only requested maps are synthesized; existing GPU
//! material bakes and the symbolic transport resolver use these same handles.
use super::super::preparation::AssetStore;
use super::*;
pub(super) fn build(
    scene: &IndoorManifest,
    base: &[Handle<StandardMaterial>],
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    used: Option<&std::collections::BTreeSet<(Surface, usize)>>,
) -> std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>> {
    let mut result = std::collections::BTreeMap::new();
    if !scene.objects.iter().any(|o| o.kind.is_kitchen()) {
        return result;
    }
    for (slot, surface, color) in [
        (100, Surface::Ceramic, [0.68, 0.12, 0.055]),
        (101, Surface::Ceramic, [0.42, 0.55, 0.13]),
        (102, Surface::Ceramic, [0.88, 0.64, 0.13]),
        (103, Surface::Terracotta, [0.53, 0.27, 0.085]),
        (104, Surface::Paper, [0.85, 0.69, 0.43]),
    ] {
        if used.is_some_and(|u| !u.contains(&(surface, slot))) {
            continue;
        }
        let mut mat = materials.get(&base[surface as usize]).unwrap().clone();
        let n = 64;
        let mut pixels = Vec::with_capacity(n * n * 4);
        for y in 0..n {
            for x in 0..n {
                let u = x as f32 / n as f32;
                let v = y as f32 / n as f32;
                let mottling =
                    periodic_noise(u, v, 7, 9, scene.material_seed().wrapping_add(slot as u64));
                let pores = periodic_noise(u, v, 23, 27, slot as u64);
                let variation = 0.85 + 0.19 * mottling + 0.07 * pores;
                for c in color {
                    pixels.push((c * variation * 255.).clamp(0., 255.) as u8);
                }
                pixels.push(255);
            }
        }
        mat.base_color = Color::WHITE;
        mat.base_color_texture = Some(images.add(mip_image(pixels, n as u32, MapType::Color)));
        mat.metallic = 0.;
        mat.perceptual_roughness = if slot < 103 { 0.35 } else { 0.86 };
        mat.clearcoat = if slot < 103 { 0.12 } else { 0. };
        mat.uv_transform = bevy::math::Affine2::from_scale(Vec2::splat(7.));
        result.insert((surface, slot), materials.add(mat));
    }
    result
}
