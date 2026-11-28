//! One mipmapped print sheet shared by the recessed mass inserts on a rack.
use super::*;
use crate::scene::procedural_indoor::materials::raster::{number_rows, Canvas};
use crate::scene::procedural_indoor::objects::public::dumbbells::{HeadFinish, VERSION};

const SLOT: usize = 999;
pub(crate) const KEY: (Surface, &str) = (Surface::Ink, "fitness_equipment#finish999#uv_local");
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == KEY.0 && slot == SLOT
}

// Each 32-pixel cell has its own margin. At long distances the filtered
// atlas converges to the mean reflectance of the dark printed inserts.
pub(crate) fn uv(kg: u8, unit: Vec2) -> [f32; 2] {
    let cell = u32::from(kg / 2 - 1);
    ((Vec2::new((cell % 8) as f32 + 0.5, (cell / 8) as f32 + 0.5) + 0.475 * unit) / 8.).to_array()
}

fn image() -> Image {
    let mut c = Canvas::new([43, 45, 46]);
    for cell in 0..64 {
        let value = 2 * (1 + cell % 32);
        let x = (cell % 8) * 32;
        let y = (cell / 8) * 32;
        let digits = value.to_string();
        let width = (digits.len() as i32 * 4 - 1) * 2;
        for (place, digit) in digits.bytes().enumerate() {
            let left = x + (32 - width) / 2 + 8 * place as i32;
            for (row, bits) in number_rows(usize::from(digit - b'0'))
                .into_iter()
                .enumerate()
            {
                for col in 0..3 {
                    if bits & (4 >> col) != 0 {
                        c.rect(
                            left + 2 * col,
                            y + 8 + 2 * row as i32,
                            2,
                            2,
                            [223, 222, 212],
                        );
                    }
                }
            }
        }
        c.text("KG", x + 12, y + 20, 1, [185, 188, 184]);
    }
    printed_mip_image(c.0, 256)
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let key = (KEY.0, SLOT);
    let required = || {
        scene.objects.iter().any(|o| {
            o.kind == ObjectKind::DumbbellRack
                && HeadFinish::for_object(o) == HeadFinish::ColourCodedEnds
        })
    };
    if scene.generator_version < VERSION
        || !selection.map_or_else(required, |s| s.finishes.contains(&key))
    {
        return;
    }
    static ATLAS: OnceLock<Image> = OnceLock::new();
    variants.insert(
        key,
        materials.add(StandardMaterial {
            base_color_texture: Some(images.add(ATLAS.get_or_init(image).clone())),
            perceptual_roughness: 0.61,
            ..default()
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn printed_nominal_masses_have_distinct_padded_cells_and_a_full_mip_chain() {
        let atlas = image();
        let bytes = atlas.data.unwrap();
        assert_eq!(atlas.texture_descriptor.mip_level_count, 9);
        let mut patterns = std::collections::BTreeSet::new();
        for kg in (2..=64).step_by(2) {
            let [u, v] = uv(kg, Vec2::ZERO);
            let x = (u * 256.) as usize - 16;
            let y = (v * 256.) as usize - 16;
            let mut ink = Vec::with_capacity(1024);
            for row in 0..32 {
                for col in 0..32 {
                    let lit = bytes[4 * ((y + row) * 256 + x + col)] > 100;
                    assert!(!lit || (2..30).contains(&row) && (2..30).contains(&col));
                    ink.push(lit);
                }
            }
            assert!(patterns.insert(ink), "duplicate marking {kg}");
        }
    }
}
