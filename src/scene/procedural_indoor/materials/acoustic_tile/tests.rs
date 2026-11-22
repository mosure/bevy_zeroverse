use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;
use bevy::render::render_resource::TextureFormat;

#[test]
fn mineral_boards_have_seeded_pores_dielectric_response_and_filtered_normals() {
    for facing in FACINGS {
        let bake = |seed| {
            let mut images = Assets::default();
            let m = board_material(seed, facing, &mut images);
            let period = m.uv_transform.matrix2.x_axis.x.recip();
            assert!((0.145..=0.235).contains(&period));
            assert_eq!(m.metallic, 0.);
            let mut maps = Vec::new();
            for (handle, format) in [
                (m.base_color_texture, TextureFormat::Rgba8UnormSrgb),
                (m.normal_map_texture, TextureFormat::Rgba8Unorm),
                (m.metallic_roughness_texture, TextureFormat::Rgba8Unorm),
            ] {
                let image = images.get(handle.unwrap().id()).unwrap();
                assert_eq!(image.texture_descriptor.mip_level_count, 8);
                assert_eq!(image.texture_descriptor.format, format);
                maps.push(image.data.as_ref().unwrap().clone());
            }
            assert!(maps[2]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 255 && p[1] >= 200 && p[2] == 0 && p[3] == 255));
            // The base uses nearest byte and the mip encoder truncates each
            // channel. Decoding a byte to [-1,1] has a 2/255 quantization step.
            let step = 2.0_f32 / 255.;
            for (index, p) in maps[1].as_chunks::<4>().0.iter().enumerate() {
                let n = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 127.5 - 1.;
                let channel_error = if index < SIZE * SIZE {
                    step * 0.5
                } else {
                    step
                };
                assert!(
                    n.z > 0. && (n.length() - 1.).abs() <= 3.0_f32.sqrt() * channel_error + 1e-6
                );
            }
            let mut offset = 0;
            let mut width = SIZE;
            let decode = |i: usize| {
                let p = &maps[1][i * 4..];
                Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 127.5 - 1.
            };
            while width > 1 {
                let next = offset + width * width;
                for y in 0..width / 2 {
                    for x in 0..width / 2 {
                        let i = offset + 2 * y * width + 2 * x;
                        let direction =
                            (decode(i) + decode(i + 1) + decode(i + width) + decode(i + width + 1))
                                .normalize();
                        let actual = decode(next + y * (width / 2) + x);
                        assert!((direction - actual).abs().max_element() <= step + 1e-6);
                    }
                }
                width /= 2;
                offset = next;
            }
            assert!(maps[1][..SIZE * SIZE * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[0].abs_diff(128) > 10 || p[1].abs_diff(128) > 10));
            maps
        };
        assert_eq!(bake(29), bake(29));
        assert_ne!(bake(29), bake(41));
        for seed in 0..8 {
            let field = pores(facing, 0.19, &mut stream(seed, 7));
            let coverage = field.iter().filter(|&&p| p > 0.1).count() as f32 / field.len() as f32;
            assert!((0.01..0.45).contains(&coverage), "{facing:?}: {coverage}");
        }
    }
}

#[test]
fn partial_material_builds_only_require_the_requested_board_and_coated_runner() {
    let scene = IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut selection = MaterialSelection {
        surfaces: Default::default(),
        finishes: Default::default(),
        direct_surfaces: Default::default(),
    };
    let mut images = Assets::default();
    let mut mats = Assets::default();
    let mut table = BTreeMap::new();
    install(&scene, &mut images, &mut mats, Some(&selection), &mut table);
    assert!(images.is_empty() && mats.is_empty() && table.is_empty());
    selection.finishes.extend([
        (Surface::Ceiling, TileFacing::Pinhole.slot()),
        (Surface::Metal, GRID_SLOT),
    ]);
    install(&scene, &mut images, &mut mats, Some(&selection), &mut table);
    assert_eq!(images.len(), 3);
    assert_eq!(table.len(), 2);
    assert_eq!(mats.len(), 2);
}
