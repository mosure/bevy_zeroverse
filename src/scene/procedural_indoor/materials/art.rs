//! Coherent framed prints and painted subjects without image precursors.
//!
//! Composition is independent of substrate: a landscape, botanical study or
//! abstract print has meaningful large-scale structure, while canvas/ink
//! relief and roughness remain small, non-emissive PBR effects.
mod composition;
mod fields;
mod finish;
mod poster;
use super::{
    raster::{Canvas, N},
    *,
};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
pub use composition::ArtComposition;
pub use fields::FieldProgram;
pub use finish::ArtFinish;
use rand::Rng;

#[derive(Clone, Debug, serde::Serialize)]
pub struct FrameProgram {
    /// 0: stretched canvas; 1/4: timber; 2: black; 3/5: metal frames.
    pub style: u32,
    pub border_width: f32,
    pub mat_width: f32,
    pub shadow_gap: f32,
    pub picture_inset: f32,
    pub stepped: bool,
    pub outer_fraction: f32,
    pub recess_fraction: f32,
    pub mat_scale: Vec2,
    pub picture_bias: Vec2,
}
pub fn frame_parameters(o: &super::super::layout::IndoorObject) -> FrameProgram {
    let mut rng = stream(o.seed, 3094);
    let style = o.variant % 6;
    let mut section = stream(o.seed, 3095);
    FrameProgram {
        style,
        mat_scale: Vec2::new(
            section.random_range(0.65..1.35),
            section.random_range(0.65..1.50),
        ),
        picture_bias: Vec2::new(
            section.random_range(-0.55..0.55),
            section.random_range(-0.35..0.65),
        ),
        stepped: style != 0 && section.random_bool(0.56),
        outer_fraction: section.random_range(0.30..0.70),
        recess_fraction: section.random_range(0.18..0.62),
        border_width: if style == 0 {
            0.
        } else if style == 5 {
            rng.random_range(0.006..0.013_f32)
                .min(o.size.x.min(o.size.y) * 0.06)
        } else {
            rng.random_range(0.008..0.04_f32)
                .min(o.size.x.min(o.size.y) * 0.095)
        },
        mat_width: if style == 0 || style == 4 || rng.random_bool(0.36) {
            0.
        } else {
            rng.random_range(0.010..0.13_f32)
                .min(o.size.x.min(o.size.y) * 0.18)
        },
        shadow_gap: if style == 4 {
            rng.random_range(0.008..0.023_f32)
                .min(o.size.x.min(o.size.y) * 0.06)
        } else {
            0.
        },
        picture_inset: if style == 0 {
            0.001
        } else {
            rng.random_range(0.003..0.008_f32).min(o.size.z * 0.20)
        },
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ArtProgram {
    pub composition: ArtComposition,
    pub finish: ArtFinish,
    pub composed: bool,
    pub subject: u32,
    pub horizon: f32,
    pub relief: f32,
    pub viewpoint: f32,
    pub hue: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub density: f32,
    pub sun: [f32; 2],
    pub brush: f32,
    pub canvas: bool,
}
pub fn parameters(seed: u64) -> ArtProgram {
    let mut rng = stream(seed, 3090);
    let finish = ArtFinish::sample(seed);
    ArtProgram {
        composition: ArtComposition::sample(seed),
        canvas: finish.canvas > 0.,
        finish,
        composed: stream(seed, 0x4152544d4958).random_bool(0.96),
        subject: rng.random_range(0..14),
        horizon: rng.random_range(0.14..0.86),
        relief: rng.random_range(0.045..0.22),
        viewpoint: rng.random_range(0.05..0.95),
        hue: rng.random_range(-0.12..0.13),
        saturation: rng.random_range(0.30..0.85),
        brightness: rng.random_range(0.66..1.03),
        density: rng.random_range(0.18..0.95),
        sun: [rng.random_range(0.08..0.92), rng.random_range(0.10..0.36)],
        // Pigment variation stays below the depicted marks; visible canvas
        // texture belongs in the separately filtered physical finish.
        brush: rng.random_range(0.003..0.028),
    }
}
fn color(p: &ArtProgram, rgb: [u8; 3], distance: f32) -> [u8; 3] {
    let gray = rgb.iter().map(|&v| v as f32).sum::<f32>() / 3.;
    std::array::from_fn(|i| {
        let warm = match i {
            0 => p.hue * 70.,
            1 => p.hue * 10.,
            _ => -p.hue * 65.,
        };
        let value = gray + (rgb[i] as f32 - gray) * p.saturation + warm;
        (value * p.brightness * (1. - distance) + 210. * distance).clamp(0., 255.) as u8
    })
}
pub(super) fn landscape(seed: u64, p: &ArtProgram, coast: bool) -> Canvas {
    let mut rng = stream(seed, 3091);
    let mut c = Canvas::new([230, 234, 232]);
    let phase = rng.random_range(0.0..6.);
    let horizon = p.horizon * N as f32;
    for y in 0..N {
        let t = y as f32 / N as f32;
        let rgb = if t < p.horizon {
            [102. + t * 168., 152. + t * 103., 190. + t * 67.]
        } else if coast {
            [56. + t * 23., 111. + t * 37., 125. + t * 35.]
        } else {
            [96., 130., 107.]
        };
        c.rect(0, y, N, 1, color(p, rgb.map(|v| v as u8), 0.));
    }
    c.ellipse(
        p.sun.map(|v| v * N as f32),
        [8., 8.],
        color(p, [252, 236, 191], 0.),
    );
    for _ in 0..(p.density * 9.).ceil() as usize {
        let x = rng.random_range(-30.0..256.0);
        let y = rng.random_range(12.0..horizon.max(20.) - 5.);
        let w = rng.random_range(15.0..61.0);
        c.ellipse(
            [x, y],
            [w, rng.random_range(1.5..5.)],
            color(p, [224, 229, 224], 0.),
        );
    }
    for layer in 0..4 {
        let depth = layer as f32 / 4.;
        let mut points = vec![[0., N as f32]];
        for x in 0..=64 {
            let fx = x as f32 / 64.;
            let wave = (fx * (5.7 + depth * 3.) + phase + depth * 2.3).sin()
                + 0.37 * (fx * 18. + phase * 2. + depth).sin()
                + 0.19 * (fx * 43. + depth * 6.).sin();
            let y = horizon + depth * 51. - p.relief * N as f32 * wave * (0.65 + depth);
            points.push([fx * N as f32, y]);
        }
        points.push([N as f32, N as f32]);
        let base = if coast { [58, 82, 82] } else { [47, 82, 59] };
        c.polygon(&points, color(p, base, (1. - depth) * 0.50));
    }
    if coast {
        // Water reflects the sky in short, perspective-compressed highlights.
        c.polygon(
            &[
                [0., horizon + 43.],
                [p.viewpoint * 256., horizon + 32.],
                [256., 235.],
                [256., 256.],
                [0., 256.],
            ],
            color(p, [71, 122, 139], 0.),
        );
        for _ in 0..(p.density * 120.) as usize {
            let y = rng.random_range((horizon + 52.).min(245.) as i32..256);
            let x = rng.random_range(0..256);
            c.rect(
                x,
                y,
                rng.random_range(3..24),
                1,
                color(p, [165, 184, 179], 0.),
            );
        }
    } else {
        for _ in 0..(p.density * 18.).ceil() as usize {
            let x = rng.random_range(-5.0..261.0);
            let y = rng.random_range(176.0..260.0);
            let h = rng.random_range(18.0..52.0) * ((y - 150.) / 100.);
            c.line([x, y], [x, y - h], 2, color(p, [58, 59, 46], 0.));
            for b in 0..5 {
                let t = b as f32 / 5.;
                c.ellipse(
                    [x, y - h * (1. - t * 0.66)],
                    [h * (0.11 + t * 0.18), h * 0.18],
                    color(p, [36, 66, 48], 0.),
                );
            }
        }
    }
    c
}
pub fn pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    let mut rng = stream(seed, 3092);
    let mut c = if p.composed || p.subject >= 7 {
        composition::paint(&p)
    } else if p.subject < 2 {
        landscape(seed, &p, p.subject == 1)
    } else {
        Canvas::new(color(&p, [233, 225, 208], 0.))
    };
    if !p.composed {
        match p.subject {
            2 => {
                // Botanical studies: connected curved stems with alternating leaves.
                for stem in 0..(2. + p.density * 5.) as usize {
                    let base = 35. + stem as f32 * 192. / (2. + p.density * 5.);
                    let bend = rng.random_range(-32.0..32.0);
                    let top = rng.random_range(21.0..100.0);
                    let mut previous = [base, 242.];
                    for i in 1..18 {
                        let t = i as f32 / 17.;
                        let point = [base + bend * t * t, 242. + (top - 242.) * t];
                        c.line(previous, point, 2, color(&p, [58, 85, 54], 0.));
                        if i % 2 == 0 {
                            let side = if i % 4 == 0 { -1. } else { 1. };
                            let tip = [
                                point[0] + side * rng.random_range(15.0..30.0),
                                point[1] - 17.,
                            ];
                            c.polygon(
                                &[
                                    point,
                                    [point[0] + side * 10., point[1] - 17.],
                                    tip,
                                    [point[0] + side * 13., point[1] - 1.],
                                ],
                                color(&p, [67, 105, 63], 0.),
                            );
                            c.line(point, tip, 1, color(&p, [128, 148, 99], 0.));
                        }
                        previous = point;
                    }
                }
            }
            3 => {
                // Architectural elevation: common horizon, varied building rhythms.
                c.rect(0, 0, 256, 155, color(&p, [170, 192, 199], 0.));
                let mut x = -9;
                while x < N {
                    let w = rng.random_range(17..49);
                    let h = rng.random_range(43..163);
                    let facade = color(
                        &p,
                        [
                            rng.random_range(119..202),
                            rng.random_range(120..198),
                            rng.random_range(118..178),
                        ],
                        0.,
                    );
                    c.rect(x, 228 - h, w, h, facade);
                    for y in ((233 - h)..220).step_by(rng.random_range(10..17)) {
                        for wx in ((x + 4)..(x + w - 3)).step_by(9) {
                            c.rect(wx, y, 4, 6, color(&p, [63, 79, 87], 0.));
                        }
                    }
                    x += w + rng.random_range(1..7);
                }
                c.rect(0, 229, 256, 27, color(&p, [134, 130, 119], 0.));
            }
            4 | 5 => {
                // Intentional abstract compositions with bounded earth-tone palette.
                for _ in 0..(3. + p.density * 11.) as usize {
                    let rgb = [[137, 91, 69], [67, 105, 119], [203, 165, 104], [54, 63, 64]]
                        [rng.random_range(0..4)];
                    let x = rng.random_range(-20.0..250.0);
                    let y = rng.random_range(-20.0..250.0);
                    let w = rng.random_range(18.0..107.0);
                    let h = rng.random_range(18.0..106.0);
                    if p.subject == 4 {
                        c.ellipse([x, y], [w * 0.5, h * 0.5], color(&p, rgb, 0.));
                    } else {
                        c.polygon(
                            &[
                                [x, y],
                                [x + w, y + h * 0.18],
                                [x + w * 0.8, y + h],
                                [x - w * 0.2, y + h * 0.65],
                            ],
                            color(&p, rgb, 0.),
                        );
                    }
                }
            }
            6 => {
                // Restrained contour drawing, smooth nested lines and open margins.
                let cx = 80. + p.viewpoint * 100.;
                for layer in 0..(4. + p.density * 12.) as usize {
                    let r = 12. + layer as f32 * 5.;
                    for i in 0..80 {
                        let t = i as f32 / 80. * std::f32::consts::TAU;
                        let point = |t: f32| {
                            [
                                cx + r * (1. + 0.15 * (3. * t).sin()) * t.cos(),
                                126. + r * 1.18 * t.sin(),
                            ]
                        };
                        c.line(
                            point(t),
                            point(t + std::f32::consts::TAU / 80.),
                            1,
                            color(&p, [65, 78, 80], 0.),
                        );
                    }
                }
            }
            _ => (),
        }
    }
    // Brush pigment is subtle correlated variation, never a noisy subject mask.
    let noise = field::PreparedPeriodic::new([13, 17], seed ^ 3093, &mut 2048);
    for y in 0..N {
        for x in 0..N {
            let variation = noise.sample(x as f32 / 256., y as f32 / 256.) - 0.5;
            let i = ((y * N + x) * 4) as usize;
            for channel in 0..3 {
                c.0[i + channel] =
                    (c.0[i + channel] as f32 * (1. + variation * p.brush)).clamp(0., 255.) as u8;
            }
        }
    }
    c.0
}
/// Bounded image content without generating unused PBR maps or grain passes.
pub(super) fn illustration(seed: u64) -> Vec<u8> {
    composition::paint(&parameters(seed)).0
}

/// Production color/normal/ORM images, including coupled GGX mip filtering.
pub fn maps(seed: u64) -> [Image; 3] {
    parameters(seed).finish.maps(seed, pixels(seed))
}
pub(super) fn apply(seed: u64, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
    apply_prepared(seed, maps(seed), mat, images);
}
pub(super) fn apply_prepared(
    seed: u64,
    maps: [Image; 3],
    mat: &mut StandardMaterial,
    images: &mut impl AssetStore<Image>,
) {
    let p = parameters(seed);
    let [color, normal, data] = maps.map(|image| images.add(image));
    mat.base_color = Color::WHITE;
    mat.base_color_texture = Some(color);
    mat.normal_map_texture = Some(normal);
    mat.metallic_roughness_texture = Some(data.clone());
    mat.occlusion_texture = Some(data);
    mat.perceptual_roughness = 1.;
    mat.metallic = 0.;
    mat.clearcoat = p.finish.varnish;
    mat.clearcoat_perceptual_roughness = p.finish.varnish_roughness;
    response::conformal_coat(mat);
    mat.emissive = LinearRgba::BLACK;
    mat.emissive_texture = None;
    mat.uv_transform = bevy::math::Affine2::IDENTITY;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn art_finish_is_independent_of_the_depicted_image() {
        let finish = ArtFinish::sample(55);
        let white = finish.maps(55, vec![255; 256 * 256 * 4]);
        let subject = finish.maps(55, pixels(91));
        assert_ne!(white[0].data, subject[0].data);
        for i in 1..3 {
            assert_eq!(white[i].data, subject[i].data);
            assert_eq!(subject[i].texture_descriptor.mip_level_count, 9);
            assert_eq!(
                subject[i].data.as_ref().unwrap().len(),
                (0..9).map(|m| (256usize >> m).pow(2) * 4).sum::<usize>()
            );
        }
    }
    #[test]
    fn art_composition_and_substrate_have_independent_coverage() {
        let mut counts = std::collections::BTreeSet::new();
        let mut substrates = std::collections::BTreeSet::new();
        for seed in 0..512 {
            let p = parameters(seed);
            assert_eq!(
                serde_json::to_value(&p).unwrap(),
                serde_json::to_value(parameters(seed)).unwrap()
            );
            counts.insert(p.composition.regions.regions.len());
            substrates.insert((p.canvas, p.finish.paint_body_m > 0., p.finish.varnish > 0.));
            assert_eq!(p.canvas, p.finish.canvas > 0.);
            assert!((0.58..0.95).contains(&p.finish.roughness));
        }
        assert!(counts.len() >= 4);
        assert_eq!(substrates.len(), 8);
    }
    #[test]
    fn framed_art_uses_the_complete_front_image_and_fits_the_mount() {
        use super::super::super::{
            layout::{IndoorLayout, IndoorManifest, ObjectKind},
            objects,
        };
        let scene =
            IndoorManifest::generate_with_humans(7, IndoorLayout::Lounge, 0.5, 0, 0.).unwrap();
        let mut styles = std::collections::BTreeSet::new();
        for seed in 0..64 {
            for size in [Vec3::new(0.35, 1.25, 0.012), Vec3::new(1.2, 0.35, 0.045)] {
                let mut object = scene.objects[0].clone();
                object.kind = ObjectKind::WallArt;
                object.seed = seed;
                object.variant = seed as u32 % 6;
                object.size = size;
                let p = frame_parameters(&object);
                styles.insert(p.style);
                let assembly = objects::build_object(&object);
                let (lo, hi) = assembly.bounds();
                assert!(lo
                    .cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(0.00001))
                    .all());
                assert!(hi
                    .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.00001))
                    .all());
                let art = assembly
                    .parts
                    .iter()
                    .find(|((s, _), _)| *s == Surface::Art)
                    .unwrap()
                    .1;
                assert_eq!(art.positions.len(), 4);
                assert_eq!(art.indices.len(), 6);
                assert_eq!(art.uvs, [[0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
                assert!(art.normals.iter().all(|n| *n == [0., 0., 1.]));
                let picture_z = art.positions[0][2];
                let backing_surface = if p.style == 4 {
                    Surface::Plastic
                } else {
                    Surface::Paper
                };
                let backing = assembly
                    .parts
                    .iter()
                    .find(|((s, _), _)| *s == backing_surface)
                    .unwrap()
                    .1;
                assert!(backing
                    .positions
                    .iter()
                    .all(|point| point[2] < picture_z - 0.0001));
                let image_width =
                    size.x - 2. * (p.border_width + p.mat_width * p.mat_scale.x + p.shadow_gap);
                assert!((art.positions[1][0] - art.positions[0][0] - image_width).abs() < 0.00001);
                let center = Vec2::new(
                    (art.positions[0][0] + art.positions[2][0]) * 0.5,
                    (art.positions[0][1] + art.positions[2][1]) * 0.5,
                );
                let expected =
                    Vec2::new(0., size.y * 0.5) + p.mat_scale * p.mat_width * p.picture_bias;
                assert!(center.distance(expected) < 0.00001);
                // Asymmetry must never put the image beneath the frame rails.
                for point in &art.positions {
                    assert!(point[0].abs() <= size.x * 0.5 - p.border_width + 0.00001);
                    assert!(
                        point[1] >= p.border_width - 0.00001
                            && point[1] <= size.y - p.border_width + 0.00001
                    );
                }
            }
        }
        assert_eq!(styles.len(), 6);
    }
    #[test]
    fn portrait_whiteboards_keep_their_full_normalized_content() {
        use super::super::super::{
            layout::{IndoorLayout, IndoorManifest, ObjectKind},
            objects,
        };
        let scene =
            IndoorManifest::generate_with_humans(7, IndoorLayout::Lounge, 0.5, 0, 0.).unwrap();
        for size in [Vec3::new(0.7, 1.35, 0.12), Vec3::new(2.3, 0.65, 0.18)] {
            let mut object = scene.objects[0].clone();
            object.kind = ObjectKind::Whiteboard;
            object.size = size;
            let assembly = objects::build_object(&object);
            let board = assembly
                .parts
                .iter()
                .find(|((s, _), _)| *s == Surface::Whiteboard)
                .unwrap()
                .1;
            assert_eq!(board.uvs, [[0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
            assert_eq!(board.positions.len(), 4);
            assert!(board.normals.iter().all(|n| *n == [0., 0., 1.]));
        }
    }
    #[test]
    fn compositions_replay_and_cover_subjects_without_pixel_noise() {
        let mut subjects = std::collections::BTreeSet::new();
        let mut images = std::collections::BTreeSet::new();
        for seed in 0..128 {
            subjects.insert(parameters(seed).subject);
            let image = pixels(seed);
            assert_eq!(image.len(), 256 * 256 * 4);
            assert!(image.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 255));
            images.insert(image);
        }
        assert_eq!(subjects.len(), 14);
        assert_eq!(images.len(), 128);
        assert_eq!(pixels(299), pixels(299));
    }
    #[test]
    fn artwork_has_independent_nonemissive_pbr_maps() {
        let mut images = Assets::<Image>::default();
        let mut textures = std::collections::BTreeSet::new();
        for seed in 0..8 {
            let mut material = StandardMaterial::default();
            apply(seed, &mut material, &mut images);
            assert_eq!(material.metallic, 0.);
            assert_eq!(material.emissive, LinearRgba::BLACK);
            assert!(material.emissive_texture.is_none());
            assert_eq!(material.uv_transform, bevy::math::Affine2::IDENTITY);
            let data = images
                .get(material.metallic_roughness_texture.as_ref().unwrap())
                .unwrap();
            assert!(data.data.as_ref().unwrap()[..256 * 256 * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 255 && p[1] > 130 && p[2] == 0));
            let image = images
                .get(material.base_color_texture.as_ref().unwrap())
                .unwrap();
            textures.insert(image.data.clone());
        }
        assert_eq!(textures.len(), 8);
    }
    #[test]
    #[ignore = "local visual content review sheet; no GPU or published artifacts"]
    fn visual_content_contact_sheet() {
        let mut sheet = image::RgbaImage::new(1024, 1152);
        let mut index = 0;
        for (name, count, function) in [
            (
                "SCREEN",
                16,
                super::super::screens::pixels as fn(u64) -> Vec<u8>,
            ),
            ("PHONE", 8, super::super::screens::mobile_pixels),
            ("TV", 8, super::super::screens::television_pixels),
            ("BOARD", 16, super::super::boards::pixels),
            ("ART", 16, pixels),
        ] {
            for seed in 0..count {
                let texture = image::RgbaImage::from_raw(256, 256, function(seed)).unwrap();
                let texture = image::imageops::resize(
                    &texture,
                    128,
                    128,
                    image::imageops::FilterType::Lanczos3,
                );
                let x = index % 8 * 128;
                let y = index / 8 * 144;
                image::imageops::overlay(&mut sheet, &texture, x, y + 16);
                let mut label = Canvas::new([239, 239, 235]);
                label.text(name, 4, 3, 1, [35, 44, 52]);
                label.digit(seed as u32 / 10, 48, 2, 1, [35, 44, 52]);
                label.digit(seed as u32 % 10, 55, 2, 1, [35, 44, 52]);
                let label = image::RgbaImage::from_raw(256, 256, label.0).unwrap();
                let label = image::imageops::crop_imm(&label, 0, 0, 128, 16).to_image();
                image::imageops::overlay(&mut sheet, &label, x, y);
                index += 1;
            }
        }
        std::fs::create_dir_all("out/visual-content-review").unwrap();
        sheet
            .save("out/visual-content-review/programs.png")
            .unwrap();
    }
}
