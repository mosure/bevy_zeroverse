//! Metric normal, roughness and color maps for manufactured sheet stock.
use super::*;
use crate::scene::procedural_indoor::preparation::AssetStore;

/// Color is authored in sRGB. Relief is in metres, so the gradient must
/// account for the actual length of both axes before filtering the normal map.
pub(super) fn sheet_material<const SIDE: usize>(
    images: &mut impl AssetStore<Image>,
    period: Vec2,
    clearcoat: f32,
    coat_roughness: f32,
    sample: impl Fn(Vec2) -> (Vec3, f32, f32),
) -> StandardMaterial {
    let mut color = Vec::with_capacity(SIDE * SIDE * 4);
    let mut orm = Vec::with_capacity(SIDE * SIDE * 4);
    let mut heights = Vec::with_capacity(SIDE * SIDE);
    let encode = |v: f32| (v.clamp(0., 1.) * 255.).round() as u8;
    for y in 0..SIDE {
        for x in 0..SIDE {
            let (pigment, h, r) = sample((Vec2::new(x as f32, y as f32) + 0.5) / SIDE as f32);
            color.extend(pigment.to_array().map(encode));
            color.push(255);
            heights.push(h);
            orm.extend([255, encode(r), 0, 255]);
        }
    }
    let scale = Vec2::splat(SIDE as f32 * 0.5) / period;
    let mut normals = Vec::with_capacity(SIDE * SIDE * 4);
    for y in 0..SIDE {
        for x in 0..SIDE {
            let dx = heights[y * SIDE + (x + 1) % SIDE] - heights[y * SIDE + (x + SIDE - 1) % SIDE];
            let dy = heights[(y + 1) % SIDE * SIDE + x] - heights[(y + SIDE - 1) % SIDE * SIDE + x];
            let n = Vec3::new(-dx * scale.x, -dy * scale.y, 1.).normalize();
            normals.extend((n * 0.5 + 0.5).to_array().map(encode));
            normals.push(255);
        }
    }
    let mut mat = StandardMaterial {
        base_color_texture: Some(images.add(mip_image(color, SIDE as u32, MapType::Color))),
        normal_map_texture: Some(images.add(mip_image(normals, SIDE as u32, MapType::Normal))),
        metallic_roughness_texture: Some(images.add(mip_image(orm, SIDE as u32, MapType::Data))),
        perceptual_roughness: 1.,
        metallic: 0.,
        clearcoat,
        clearcoat_perceptual_roughness: coat_roughness,
        uv_transform: bevy::math::Affine2::from_scale(period.recip()),
        ..default()
    };
    response::conformal_coat(&mut mat);
    mat
}
