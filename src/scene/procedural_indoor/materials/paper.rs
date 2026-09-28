//! Ruling and handwriting live in a filtered color map, never raised geometry.
use super::{raster::Canvas, *};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

pub(super) fn apply(seed: u64, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
    let mut rng = stream(seed, 864);
    let mut c = Canvas(vec![255; 256 * 256 * 4]);
    c.rect(0, 0, 256, 256, [245, 243, 236]);
    let spacing = rng.random_range(8..14);
    let ink = [64, 77, 92];
    // Both plain ruled pads and squared engineering paper, with a pale margin.
    for y in (24..243).step_by(spacing) {
        c.rect(13, y, 232, 1, [166, 190, 209]);
    }
    if rng.random_bool(0.3) {
        for x in (13..246).step_by(spacing) {
            c.rect(x, 24, 1, 219, [191, 208, 217]);
        }
    }
    c.rect(30, 10, 1, 241, [213, 166, 163]);
    for row in 0..rng.random_range(0..10) {
        let y = 24 + row * spacing as i32 - 3;
        let mut x = 37;
        for _ in 0..rng.random_range(1..5) {
            let width = rng.random_range(10..33);
            for dx in 0..width {
                let dy = rng.random_range(-1..2);
                c.rect(x + dx, y + dy, 1, 1, ink);
            }
            x += width + rng.random_range(4..8);
        }
    }
    let mut texture = mip_image(c.0, 256, MapType::Color);
    texture.sampler = ImageSampler::linear();
    mat.base_color = Color::WHITE;
    mat.base_color_texture = Some(images.add(texture));
    mat.uv_transform = bevy::math::Affine2::IDENTITY;
    mat.perceptual_roughness = 0.9;
}
