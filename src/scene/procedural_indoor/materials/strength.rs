//! Equipment stock. The coating, machined steel and pigmented discs have
//! separate dielectric/metal responses even when their base hues are similar.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    preparation::AssetStore,
};
use rand::Rng;
use std::{collections::BTreeMap, sync::OnceLock};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finish {
    Coat,
    Steel,
    Grip,
    Iron,
    Rubber,
    Green,
    Yellow,
    Blue,
    Red,
}

const STOCK: [Finish; 9] = [
    Finish::Coat,
    Finish::Steel,
    Finish::Grip,
    Finish::Iron,
    Finish::Rubber,
    Finish::Green,
    Finish::Yellow,
    Finish::Blue,
    Finish::Red,
];

impl Finish {
    pub(crate) fn key(self) -> (Surface, &'static str) {
        match self {
            Self::Coat => (Surface::Paint, "fitness_equipment#finish990"),
            Self::Steel => (Surface::Chrome, "fitness_equipment#finish991"),
            Self::Grip => (Surface::Metal, "fitness_equipment#finish992"),
            Self::Iron => (Surface::Metal, "fitness_equipment#finish993"),
            Self::Rubber => (Surface::Rubber, "fitness_equipment#finish994"),
            Self::Green => (Surface::Rubber, "fitness_equipment#finish995"),
            Self::Yellow => (Surface::Rubber, "fitness_equipment#finish996"),
            Self::Blue => (Surface::Rubber, "fitness_equipment#finish997"),
            Self::Red => (Surface::Rubber, "fitness_equipment#finish998"),
        }
    }
    fn stock_key(self) -> (Surface, usize) {
        (self.key().0, 990 + self as usize)
    }
    fn material(self, seed: u64) -> StandardMaterial {
        let mut r = stream(seed, u64::from_le_bytes(*b"RackCoat") ^ self as u64);
        let (rgb, metal, rough) = match self {
            Self::Coat => {
                let hue = match r.random_range(0..10) {
                    0..=3 => [0.14, 0.15, 0.16],
                    4 => [0.68, 0.68, 0.65],
                    5 => [0.40, 0.42, 0.45],
                    6 => [0.52, 0.11, 0.09],
                    7 => [0.13, 0.27, 0.39],
                    8 => [0.09, 0.26, 0.23],
                    _ => [0.65, 0.37, 0.09],
                };
                (hue, 0., r.random_range(0.35..0.59))
            }
            Self::Steel => ([0.76, 0.78, 0.80], 0.96, r.random_range(0.17..0.29)),
            Self::Grip => ([0.71, 0.73, 0.75], 0.93, r.random_range(0.32..0.47)),
            Self::Iron => ([0.25, 0.26, 0.27], 0.49, r.random_range(0.48..0.66)),
            Self::Rubber => ([0.13, 0.135, 0.14], 0., r.random_range(0.65..0.84)),
            Self::Green => ([0.10, 0.40, 0.24], 0., 0.70),
            Self::Yellow => ([0.80, 0.60, 0.15], 0., 0.69),
            Self::Blue => ([0.12, 0.28, 0.58], 0., 0.68),
            Self::Red => ([0.65, 0.16, 0.15], 0., 0.69),
        };
        let v = r.random_range(0.87..1.10);
        StandardMaterial {
            base_color: Color::srgb(rgb[0] * v, rgb[1] * v, rgb[2] * v),
            metallic: metal,
            perceptual_roughness: rough,
            ..default()
        }
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    slot.checked_sub(990)
        .and_then(|i| STOCK.get(i))
        .is_some_and(|f| f.key().0 == surface)
}

fn knurl() -> Image {
    // Cross-cut diagonal ridges, tiled in shaft coordinates. Normal-aware
    // mipmaps flatten them at room-view distances without repeating geometry.
    const N: usize = 32;
    let mut bytes = Vec::with_capacity(N * N * 4);
    for y in 0..N {
        for x in 0..N {
            let phase = std::f32::consts::TAU * 4. / N as f32;
            let a = ((x + y) as f32 * phase).sin();
            let b = ((x as f32 - y as f32) * phase).sin();
            let n = Vec3::new(0.26 * (a + b), 0.26 * (a - b), 1.).normalize();
            bytes.extend(n.to_array().map(|v| ((v * 0.5 + 0.5) * 255.).round() as u8));
            bytes.push(255);
        }
    }
    let mut image = mip_image(bytes, N as u32, MapType::Normal);
    // At 4x4 and below each texel covers an integer number of complete
    // grooves. Their exact mean normal is vertical. Repeated 8-bit filtering
    // alone introduces a small directional bias at these distances.
    let far = 4 * (N * N + (N / 2).pow(2) + (N / 4).pow(2));
    if let Some(data) = &mut image.data {
        for pixel in data[far..].as_chunks_mut::<4>().0 {
            *pixel = [128, 128, 255, 255];
        }
    }
    image
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if selection.is_none()
        && !scene
            .objects
            .iter()
            .any(|o| o.kind == ObjectKind::PowerRack)
    {
        return;
    }
    for f in STOCK {
        let key = f.stock_key();
        if selection.is_some_and(|s| !s.finishes.contains(&key)) {
            continue;
        }
        let mut mat = f.material(scene.material_seed());
        if f == Finish::Grip {
            static KNURL: OnceLock<Image> = OnceLock::new();
            mat.normal_map_texture = Some(images.add(KNURL.get_or_init(knurl).clone()));
            mat.uv_transform = bevy::math::Affine2::from_scale(Vec2::splat(1. / 0.0064));
        }
        variants.insert(key, materials.add(mat));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strength_finishes_have_stable_semantics_and_filtered_grip_relief() {
        for (index, f) in STOCK.into_iter().enumerate() {
            let (surface, name) = f.key();
            assert_eq!(name, format!("fitness_equipment#finish{}", 990 + index));
            assert!(supports(surface, 990 + index));
        }
        let seed = 101;
        assert!(
            Finish::Steel.material(seed).metallic > 0.9
                && Finish::Coat.material(seed).metallic == 0.
        );
        assert!(
            Finish::Grip.material(seed).perceptual_roughness
                > Finish::Steel.material(seed).perceptual_roughness
        );
        assert!(Finish::Rubber.material(seed).perceptual_roughness > 0.60);
        let image = knurl();
        let bytes = image.data.as_deref().unwrap();
        assert_eq!(image.texture_descriptor.mip_level_count, 6);
        let normals = bytes.as_chunks::<4>().0;
        assert!(normals[..1024].iter().any(|v| v[0] < 110));
        assert!(normals[32 * 32 + 16 * 16 + 8 * 8..]
            .iter()
            .all(|p| *p == [128, 128, 255, 255]));
    }
}
