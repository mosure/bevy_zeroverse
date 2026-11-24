//! Reflective status panels on office printers. A room shares each used atlas
//! tile across cameras and builds no tiles when no panel is visible in its mesh.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorObject, ObjectKind},
    objects::utilities::printer,
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};

const FIRST: usize = 640;
const STATES: [&str; 4] = ["READY", "COPY", "SCAN", "SLEEP"];
const LOTS: usize = 3;
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Ink && (FIRST..FIRST + LOTS * STATES.len()).contains(&slot)
}
fn state(o: &IndoorObject) -> usize {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"PrintLCD"));
    match r.random_range(0..12) {
        0..=5 => 0,
        6..=7 => 1,
        8 => 2,
        _ => 3,
    }
}
pub(crate) fn status(o: &IndoorObject) -> &'static str {
    STATES[state(o)]
}
fn slot(o: &IndoorObject) -> usize {
    let lot = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"LCDStock"),
    )
    .random_range(0..LOTS);
    FIRST + lot * STATES.len() + state(o)
}
pub(crate) fn label(o: &IndoorObject) -> String {
    format!("printer#finish{}#uv_local", slot(o))
}
fn image(slot: usize) -> Image {
    const N: usize = 64;
    let i = slot - FIRST;
    let bg = [
        [161, 172, 148, 255],
        [174, 180, 174, 255],
        [157, 180, 184, 255],
    ][i / STATES.len()];
    let fg = [46, 56, 50, 255];
    let mut pixels = bg.repeat(N * N);
    let word = STATES[i % STATES.len()];
    let start = (N - (word.len() * 8 - 2)) / 2;
    for (letter, ch) in word.bytes().enumerate() {
        let glyph = raster::letter_rows(ch).expect("LCD alphabet");
        for (row, bits) in glyph.into_iter().enumerate() {
            for col in 0..3 {
                if bits & (4 >> col) == 0 {
                    continue;
                }
                for dy in 0..5 {
                    for dx in 0..2 {
                        let offset =
                            4 * ((18 + row * 6 + dy) * N + start + letter * 8 + col * 2 + dx);
                        pixels[offset..offset + 4].copy_from_slice(&fg);
                    }
                }
            }
        }
    }
    mip_image(pixels, N as u32, MapType::Color)
}
pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < printer::OUTPUT_VERSION {
        return;
    }
    let slots: BTreeSet<_> = if let Some(selection) = selection {
        selection
            .finishes
            .iter()
            .filter(|&&(s, i)| supports(s, i))
            .map(|&(_, i)| i)
            .collect()
    } else {
        scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Printer && printer::parameters(o).screen)
            .map(slot)
            .collect()
    };
    for i in slots {
        variants.insert(
            (Surface::Ink, i),
            materials.add(StandardMaterial {
                base_color_texture: Some(images.add(image(i))),
                perceptual_roughness: 0.42,
                reflectance: 0.37,
                ..default()
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn panel_ink_keeps_its_semantics_and_complete_filtered_mip_chain() {
        for i in FIRST..FIRST + LOTS * STATES.len() {
            assert!(supports(Surface::Ink, i));
            assert!(!supports(Surface::Screen, i));
            let image = image(i);
            let d = &image.texture_descriptor;
            assert!(d.format.is_srgb());
            assert_eq!(d.mip_level_count, 7);
            let rgba = image.data.unwrap();
            assert_eq!(
                rgba.len(),
                (0..7)
                    .map(|l| (64_usize >> l).pow(2_u32) * 4)
                    .sum::<usize>()
            );
            let pixels = &rgba[..64 * 64 * 4];
            assert!(pixels.as_chunks::<4>().0.contains(&[46, 56, 50, 255]));
            assert!(pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
            assert!(rgba[rgba.len() - 4] > 100);
        }
    }
}
