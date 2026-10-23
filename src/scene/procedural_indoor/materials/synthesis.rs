//! CPU substrate evaluation and GPU-bake selection. No asset installation.
use super::*;

pub(super) fn prepare_map(
    scene: &IndoorManifest,
    definition: &Definition,
    gpu: bool,
) -> Option<gpu::Maps> {
    let (surface, _, roughness, _, _) = *definition;
    if !surface.has_procedural_maps() {
        return None;
    }
    let recipe = scene
        .program
        .as_ref()
        .map(|p| &p.materials[surface as usize]);
    if let Some(recipe) = recipe {
        return Some(gpu::Maps::prepare(recipe, scene.floor_style, gpu));
    }
    let maps = texture_maps(
        surface,
        scene.floor_style,
        scene.material_seed(),
        roughness,
        recipe,
    );
    Some(gpu::Maps::Cpu(Box::new(mapped_images(
        maps,
        recipe.map_or(256, |r| r.map_size(scene.floor_style)),
        recipe,
    ))))
}

pub(super) fn mapped_images(
    maps: TextureMaps,
    size: u32,
    recipe: Option<&program::MaterialRecipe>,
) -> [Image; 3] {
    let mut images = filter::images(maps, size);
    if recipe
        .is_some_and(|r| r.leaf.is_some() || r.carpet.as_ref().is_some_and(|c| c.rug.is_some()))
    {
        for image in &mut images {
            clamp_sampler(image);
        }
    }
    images
}

pub(super) fn clamp_sampler(image: &mut Image) {
    let mut descriptor = match &image.sampler {
        ImageSampler::Descriptor(d) => d.clone(),
        _ => ImageSamplerDescriptor::default(),
    };
    descriptor.address_mode_u = ImageAddressMode::ClampToEdge;
    descriptor.address_mode_v = ImageAddressMode::ClampToEdge;
    image.sampler = ImageSampler::Descriptor(descriptor);
}

pub(super) fn hash(x: u32, y: u32, seed: u64) -> f32 {
    let mut v = x.wrapping_mul(374761393)
        ^ y.wrapping_mul(668265263)
        ^ seed as u32
        ^ ((seed >> 32) as u32).wrapping_mul(0x9e3779b9);
    v = (v ^ (v >> 13)).wrapping_mul(1274126177);
    (v ^ (v >> 16)) as f32 / u32::MAX as f32
}

// Periodic value noise: continuous at tile edges, with independent longitudinal
// and transverse frequencies for wood fibres and large-scale surface variation.
pub(super) fn periodic_noise(u: f32, v: f32, nx: u32, ny: u32, seed: u64) -> f32 {
    let x = field::periodic_unit(u) * nx as f32;
    let y = field::periodic_unit(v) * ny as f32;
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let tx = smooth(x.fract());
    let ty = smooth(y.fract());
    let a = hash(ix % nx, iy % ny, seed);
    let b = hash((ix + 1) % nx, iy % ny, seed);
    let c = hash(ix % nx, (iy + 1) % ny, seed);
    let d = hash((ix + 1) % nx, (iy + 1) % ny, seed);
    (a + (b - a) * tx) * (1.0 - ty) + (c + (d - c) * tx) * ty
}

pub(super) type TextureMaps = (Vec<u8>, Vec<u8>, Vec<u8>);
pub(super) fn texture_maps(
    surface: Surface,
    floor_style: u32,
    seed: u64,
    roughness: f32,
    recipe: Option<&program::MaterialRecipe>,
) -> TextureMaps {
    texture_maps_with_preparation(surface, floor_style, seed, roughness, recipe, true)
}

// Keep the scalar evaluation path available to exact whole-map replay tests.
// This internal switch never changes recipe sampling, derivatives or encoding.
pub(super) fn texture_maps_with_preparation(
    surface: Surface,
    floor_style: u32,
    seed: u64,
    roughness: f32,
    recipe: Option<&program::MaterialRecipe>,
    prepare: bool,
) -> TextureMaps {
    // Older serialized recipes omit the new programs. Resolve their defaults
    // once per map, rather than initializing an RNG for each of 65,536 texels.
    let initialized = recipe.and_then(|r| {
        let textile = matches!(surface, Surface::Fabric | Surface::FabricAlt)
            || surface == Surface::Floor && floor_style == 1;
        if textile && r.textile.is_none() || surface == Surface::Leather && r.leather.is_none() {
            let mut r = r.clone();
            if textile {
                r.textile = Some(textile::TextileRecipe::sample(r.seed));
            }
            if surface == Surface::Leather {
                r.leather = Some(leather::LeatherRecipe::sample(r.seed));
            }
            Some(r)
        } else {
            None
        }
    });
    let recipe = initialized.as_ref().or(recipe);
    let prepared = prepare
        .then(|| recipe.and_then(|r| r.prepare_texels(floor_style)))
        .flatten();
    let n = recipe.map_or(256, |r| r.map_size(floor_style)) as usize;
    let mut heights = vec![0.0; n * n];
    // Disjoint pre-sized bands keep each texel's original arithmetic while
    // permitting long atlases to share the existing native worker budget.
    let mut colors = vec![0; n * n * 4];
    let mut data = vec![0; n * n * 4];
    atlas::texels(
        n,
        prepare,
        &mut heights,
        &mut colors,
        &mut data,
        |first_y, heights, colors, data| {
            for local_y in 0..heights.len() / n {
                let y = first_y + local_y;
                for x in 0..n {
                    let u = x as f32 / n as f32;
                    let v = y as f32 / n as f32;
                    let fallback = || {
                        let noise = hash(x as u32, y as u32, seed);
                        let wood_surface = matches!(surface, Surface::Wood | Surface::WoodEdge)
                            || (surface == Surface::Floor && floor_style == 0);
                        let fibre = if wood_surface {
                            let warp = periodic_noise(u, v, 3, 3, seed.wrapping_add(17)) * 0.035;
                            0.65 * periodic_noise(u + warp, v, 80, 4, seed)
                                + 0.35 * periodic_noise(u + warp, v, 29, 2, seed.wrapping_add(61))
                        } else {
                            0.5
                        };
                        let (shade, h) = match surface {
                            Surface::Wood | Surface::WoodEdge => {
                                (0.81 + fibre * 0.15 + noise * 0.016, fibre * 0.001)
                            }
                            Surface::Floor if floor_style == 0 => {
                                let strip = (u * 10.0).floor() as u32;
                                let stagger = if strip.is_multiple_of(2) { 0.0 } else { 0.5 };
                                let seam = (u * 10.0).fract() < 0.022
                                    || (v * 2.0 + stagger).fract() < 0.008;
                                let shade = if seam {
                                    0.36
                                } else {
                                    0.75 + hash(strip, (v * 2.0 + stagger).floor() as u32, seed)
                                        * 0.19
                                        + (fibre - 0.5) * 0.09
                                };
                                (shade, if seam { -0.025 } else { fibre * 0.002 })
                            }
                            Surface::Floor if floor_style == 2 => {
                                let seam = (u * 4.0).fract() < 0.014 || (v * 4.0).fract() < 0.014;
                                (
                                    if seam { 0.6 } else { 0.89 + noise * 0.09 },
                                    if seam { -0.06 } else { noise * 0.003 },
                                )
                            }
                            Surface::Fabric | Surface::FabricAlt | Surface::Floor => {
                                let weave = ((x / 2 + y / 2) % 2) as f32;
                                (
                                    0.76 + noise * 0.17 + weave * 0.06,
                                    weave * 0.012 + noise * 0.009,
                                )
                            }
                            Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated => {
                                let vein = (u - 0.5).abs() < 0.015
                                    || ((v + (u - 0.5).abs() * 0.7) * 14.0).fract() < 0.028;
                                (
                                    if vein {
                                        1.0
                                    } else {
                                        if surface == Surface::LeafVariegated {
                                            0.53 + 0.24 * (v * 78.0 + (u * 30.0).sin()).sin().abs()
                                                + 0.21 * ((u - 0.5).abs() * 2.0).powi(5)
                                        } else {
                                            0.75 + 0.15 * (v * std::f32::consts::PI).sin()
                                                + noise * 0.07
                                        }
                                    },
                                    if vein { 0.013 } else { 0.0 },
                                )
                            }
                            Surface::Bark => {
                                let ridges = periodic_noise(u, v, 26, 3, seed);
                                (0.66 + ridges * 0.28 + noise * 0.06, ridges * 0.045)
                            }
                            Surface::Terracotta => (0.85 + noise * 0.12, noise * 0.02),
                            Surface::Soil => (0.50 + noise * 0.45, noise * 0.12),
                            Surface::Concrete => (0.84 + noise * 0.13, noise * 0.022),
                            Surface::Ceiling => {
                                (0.94 + noise * 0.04, if noise < 0.1 { -0.018 } else { 0.0 })
                            }
                            _ => (0.95 + noise * 0.04, noise * 0.008),
                        };
                        (shade, h * 0.001, roughness + (noise - 0.5) * 0.06)
                    };
                    let texel = if let Some(recipe) = recipe {
                        recipe.texel_prepared(u, v, floor_style, prepared.as_ref())
                    } else {
                        let (shade, h, roughness) = fallback();
                        program::Texel {
                            color: [shade; 3],
                            height: h,
                            roughness,
                            occlusion: 1.,
                        }
                    };
                    let index = local_y * n + x;
                    heights[index] = texel.height;
                    let rgb = texel.color.map(|c| (c.clamp(0., 1.) * 255.) as u8);
                    colors[index * 4..index * 4 + 4]
                        .copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                    data[index * 4..index * 4 + 4].copy_from_slice(&[
                        (texel.occlusion * 255.).round() as u8,
                        (texel.roughness.clamp(0.05, 1.0) * 255.0) as u8,
                        if surface == Surface::Chrome { 255 } else { 0 },
                        255,
                    ]);
                }
            }
        },
    );
    let mut normals = vec![0; n * n * 4];
    let period = recipe.map_or(Vec2::ONE, |r| {
        r.leaf
            .as_ref()
            .map_or_else(|| r.period_uv(), |l| Vec2::from_array(l.reference_size_m))
    });
    let slope = Vec2::splat(n as f32 * 0.5) / period;
    // All height writes are joined before a normal reads an adjacent band.
    atlas::normals(n, prepare, &mut normals, |first_y, normals| {
        for local_y in 0..normals.len() / (n * 4) {
            let y = first_y + local_y;
            for x in 0..n {
                let atlas = recipe.is_some_and(|r| r.leaf.is_some());
                let [left, right, bottom, top] = if atlas {
                    [
                        x.saturating_sub(1),
                        (x + 1).min(n - 1),
                        y.saturating_sub(1),
                        (y + 1).min(n - 1),
                    ]
                } else {
                    [(x + n - 1) % n, (x + 1) % n, (y + n - 1) % n, (y + 1) % n]
                };
                let dx = heights[y * n + right] - heights[y * n + left];
                let dy = heights[top * n + x] - heights[bottom * n + x];
                let slope = if atlas {
                    Vec2::new(
                        n as f32 / (right - left) as f32,
                        n as f32 / (top - bottom) as f32,
                    ) / period
                } else {
                    slope
                };
                // Tangent space follows increasing mesh U/V. Both slopes oppose the
                // height gradient; flipping only Y would invert relief in one axis.
                let normal = Vec3::new(-dx * slope.x, -dy * slope.y, 1.0).normalize();
                let index = (local_y * n + x) * 4;
                normals[index..index + 4].copy_from_slice(&filter::encode(normal));
            }
        }
    });
    (colors, normals, data)
}
