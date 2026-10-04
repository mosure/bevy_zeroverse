//! Role-specific PBR textures, generated in memory with repeat sampling and mip chains.
pub mod boards;
pub mod botanical;
pub mod ceramic;
pub mod coating;
pub mod concrete;
mod environment;
mod field;
mod filter;
pub mod glass;
mod human;
pub mod layers;
pub mod leather;
mod microstructure;
pub mod mineral;
pub mod paint;
mod paper;
pub mod program;
mod raster;
pub mod screens;
pub mod textile;
pub mod timber;
pub mod variants;
use super::layout::{IndoorManifest, LightingMood};
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[repr(usize)]
pub enum Surface {
    Paint,
    Accent,
    Wood,
    WoodEdge,
    Floor,
    Ceiling,
    Metal,
    Chrome,
    Plastic,
    Fabric,
    FabricAlt,
    Glass,
    Ceramic,
    Soil,
    Leaf,
    LeafLight,
    Paper,
    Screen,
    Ink,
    Light,
    Concrete,
    Art,
    Rubber,
    LeafVariegated,
    Terracotta,
    Bark,
    GlassInterior,
    ContainerGlass,
    Liquid,
    Drink,
    PhoneScreen,
    PrintedPaper,
    Leather,
    Whiteboard,
    Television,
}

pub struct IndoorMaterials {
    handles: Vec<Handle<StandardMaterial>>,
    pub cloth: Handle<StandardMaterial>,
    pub skin: Handle<StandardMaterial>,
    pub hair: Handle<StandardMaterial>,
    pub knit: Handle<StandardMaterial>,
    light_variants: Vec<Handle<StandardMaterial>>,
    pub(crate) variants: std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
    pub environment: EnvironmentMapLight,
}

impl IndoorMaterials {
    pub fn get(&self, surface: Surface) -> Handle<StandardMaterial> {
        self.handles[surface as usize].clone()
    }

    pub fn for_part(&self, surface: Surface, label: &str) -> Handle<StandardMaterial> {
        if let Some(material) = label
            .rsplit_once("#finish")
            .and_then(|(_, s)| s.parse::<usize>().ok())
            .and_then(|i| self.variants.get(&(surface, i)))
        {
            return material.clone();
        }
        if surface == Surface::Light {
            if let Some(material) = label
                .strip_prefix("lamp#")
                .and_then(|s| s.parse::<usize>().ok())
                .and_then(|i| self.light_variants.get(i))
            {
                return material.clone();
            }
        }
        self.get(surface)
    }

    pub fn build(
        scene: &IndoorManifest,
        images: &mut impl super::preparation::AssetStore<Image>,
        materials: &mut impl super::preparation::AssetStore<StandardMaterial>,
    ) -> Self {
        Self::build_with_quality(scene, super::IndoorQuality::Auto, images, materials)
    }

    pub fn build_with_quality(
        scene: &IndoorManifest,
        quality: super::IndoorQuality,
        images: &mut impl super::preparation::AssetStore<Image>,
        materials: &mut impl super::preparation::AssetStore<StandardMaterial>,
    ) -> Self {
        Self::build_with_finishes(scene, quality, images, materials, None)
    }

    fn build_with_finishes(
        scene: &IndoorManifest,
        quality: super::IndoorQuality,
        images: &mut impl super::preparation::AssetStore<Image>,
        materials: &mut impl super::preparation::AssetStore<StandardMaterial>,
        finishes: Option<&std::collections::BTreeSet<(Surface, usize)>>,
    ) -> Self {
        let definitions = definitions(scene);
        // Generate immutable maps on a bounded pool; asset insertion remains ordered.
        // Wasm uses the identical serial function and produces the same pixels.
        let prepare = |definition: &Definition| prepare_map(scene, definition);
        #[cfg(not(target_arch = "wasm32"))]
        let prepared = {
            let pool = super::preparation::workers::pool();
            pool.scope(|scope| {
                for definition in &definitions {
                    let prepare = &prepare;
                    scope.spawn(async move { prepare(definition) });
                }
            })
        };
        #[cfg(target_arch = "wasm32")]
        let prepared: Vec<_> = definitions.iter().map(prepare).collect();
        Self::insert_prepared(
            scene,
            quality,
            images,
            materials,
            definitions,
            prepared,
            finishes,
        )
    }

    pub async fn build_async(
        scene: &IndoorManifest,
        quality: super::IndoorQuality,
        images: &mut impl super::preparation::AssetStore<Image>,
        materials: &mut impl super::preparation::AssetStore<StandardMaterial>,
    ) -> Self {
        Self::build_async_with_finishes(scene, quality, images, materials, None).await
    }

    pub(crate) async fn build_async_with_finishes(
        scene: &IndoorManifest,
        quality: super::IndoorQuality,
        images: &mut impl super::preparation::AssetStore<Image>,
        materials: &mut impl super::preparation::AssetStore<StandardMaterial>,
        finishes: Option<&std::collections::BTreeSet<(Surface, usize)>>,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::build_with_finishes(scene, quality, images, materials, finishes)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let definitions = definitions(scene);
            let mut maps = Vec::with_capacity(definitions.len());
            for definition in &definitions {
                super::preparation::cooperate().await;
                maps.push(prepare_map(scene, definition));
            }
            Self::insert_prepared(
                scene,
                quality,
                images,
                materials,
                definitions,
                maps,
                finishes,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_prepared(
        scene: &IndoorManifest,
        quality: super::IndoorQuality,
        images: &mut impl super::preparation::AssetStore<Image>,
        materials: &mut impl super::preparation::AssetStore<StandardMaterial>,
        definitions: [Definition; 35],
        prepared: Vec<Option<[Image; 3]>>,
        finishes: Option<&std::collections::BTreeSet<(Surface, usize)>>,
    ) -> Self {
        let mut handles = Vec::new();
        for ((surface, rgb, roughness, metallic, period), maps) in
            definitions.into_iter().zip(prepared)
        {
            let mut mat = StandardMaterial {
                base_color: Color::srgb(rgb[0], rgb[1], rgb[2]),
                perceptual_roughness: roughness,
                metallic,
                reflectance: 0.5,
                uv_transform: bevy::math::Affine2::from_scale(Vec2::splat(1.0 / period)),
                ..default()
            };
            if let Some(program) = &scene.program {
                if program.materials[surface as usize].absolute_color(scene.floor_style) {
                    mat.base_color = Color::WHITE;
                }
                mat.uv_transform = bevy::math::Affine2::from_scale(
                    program.materials[surface as usize].period_uv().recip(),
                );
                program.materials[surface as usize].apply_pbr(&mut mat);
                if surface == Surface::Floor && scene.floor_style == 0 {
                    if let Some(w) = &program.materials[surface as usize].wood {
                        w.apply(&mut mat);
                    }
                }
            }
            if let Some([color, normal, data]) = maps {
                mat.base_color_texture = Some(images.add(color));
                mat.normal_map_texture = Some(images.add(normal));
                let data = images.add(data);
                if scene.program.is_some()
                    || matches!(
                        surface,
                        Surface::Fabric | Surface::FabricAlt | Surface::Leather
                    )
                    || surface == Surface::Floor && scene.floor_style == 1
                {
                    mat.occlusion_texture = Some(data.clone());
                }
                mat.metallic_roughness_texture = Some(data);
                mat.perceptual_roughness = 1.0;
            }
            if matches!(
                surface,
                Surface::Fabric
                    | Surface::FabricAlt
                    | Surface::Leaf
                    | Surface::LeafLight
                    | Surface::LeafVariegated
            ) {
                mat.cull_mode = None;
                mat.double_sided = true;
            }
            match surface {
                Surface::Glass | Surface::GlassInterior => {
                    glass::GlassRecipe::sample(scene.material_seed(), surface)
                        .apply(&mut mat, quality);
                }
                Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated => {
                    if scene.program.is_none() {
                        mat.diffuse_transmission = 0.18;
                    }
                    // The blade atlas remains 0..1 despite its metric derivative reference.
                    mat.uv_transform = bevy::math::Affine2::IDENTITY;
                }
                Surface::Screen => {
                    screens::apply(scene.material_seed(), false, &mut mat, images);
                }
                Surface::Whiteboard => {
                    boards::apply(scene.material_seed(), &mut mat, images);
                }
                Surface::Television => {
                    screens::apply_tv(scene.material_seed(), &mut mat, images);
                }
                Surface::PhoneScreen => {
                    screens::apply(scene.material_seed(), true, &mut mat, images);
                }
                Surface::PrintedPaper => {
                    paper::apply(scene.material_seed(), &mut mat, images);
                }
                Surface::ContainerGlass | Surface::Liquid => {
                    mat.base_color = Color::srgb(0.94, 0.985, 0.99);
                    mat.ior = if surface == Surface::Liquid {
                        1.333
                    } else {
                        1.47
                    };
                    mat.perceptual_roughness = 0.06;
                    mat.thickness = if surface == Surface::Liquid {
                        0.025
                    } else {
                        0.0012
                    };
                    mat.attenuation_distance = 1.0;
                    mat.attenuation_color = Color::srgb(0.91, 0.98, 0.985);
                    if quality.specular_transmission() {
                        mat.specular_transmission = 0.94;
                    } else {
                        mat.alpha_mode = AlphaMode::Blend;
                        mat.base_color = mat.base_color.with_alpha(0.18);
                    }
                }
                Surface::Chrome => {
                    // Micron scratches belong in the scattering distribution.
                    // RGBA8 cannot resolve their tiny slopes: its two nearest
                    // codes instead introduce artificial orange-peel facets.
                    mat.normal_map_texture = None;
                    // Rougher finishes are brushed; polished chrome is nearly isotropic.
                    mat.anisotropy_strength = ((roughness - 0.08) * 3.0).clamp(0.0, 0.48);
                }
                Surface::Wood | Surface::WoodEdge | Surface::Ceramic => {
                    if scene.program.is_none() {
                        mat.clearcoat = if roughness < 0.46 { 0.32 } else { 0.08 };
                        mat.clearcoat_perceptual_roughness = (roughness * 0.55).max(0.09);
                    }
                }
                Surface::Leather => {
                    if scene.program.is_none() {
                        mat.clearcoat = 0.10;
                        mat.clearcoat_perceptual_roughness = 0.32;
                    }
                }
                Surface::Light => {
                    let c = kelvin_rgb(scene.light_kelvin);
                    let c = Color::srgb(c.x, c.y, c.z).to_linear();
                    let luminance = super::architecture::fixture_lumens(scene)
                        / (std::f32::consts::PI
                            * (super::architecture::fixture_size(scene).x - 0.06)
                            * (super::architecture::fixture_size(scene).z - 0.055));
                    mat.emissive = LinearRgba::new(
                        c.red * luminance,
                        c.green * luminance,
                        c.blue * luminance,
                        1.0,
                    );
                    mat.emissive_exposure_weight = 1.0;
                }
                _ => (),
            }
            handles.push(materials.add(mat));
        }
        let environment = EnvironmentMapLight {
            intensity: scene.domain().map_or_else(
                || match scene.lighting {
                    LightingMood::Daylight => 95.0,
                    LightingMood::Overcast => 80.0,
                    LightingMood::Evening => 45.0,
                },
                |d| d.photometry.environment_intensity,
            ),
            ..default()
        };
        // Keep the visible emitter/reference export consistent with each direct-light proxy.
        let area = (super::architecture::fixture_size(scene).x - 0.06)
            * (super::architecture::fixture_size(scene).z - 0.055);
        let light_variants = super::architecture::fixture_positions(scene)
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let (color, lumens) = super::architecture::fixture_photometry(scene, i);
                let c = Color::srgb(color.x, color.y, color.z).to_linear();
                let mut material = materials
                    .get(&handles[Surface::Light as usize])
                    .unwrap()
                    .clone();
                let luminance = lumens / (std::f32::consts::PI * area);
                material.emissive = LinearRgba::new(
                    c.red * luminance,
                    c.green * luminance,
                    c.blue * luminance,
                    1.0,
                );
                materials.add(material)
            })
            .collect();
        let variants = variants::build(scene, &handles, images, materials, finishes);
        let cloth_template = materials
            .get(&handles[Surface::Fabric as usize])
            .unwrap()
            .clone();
        let [cloth, skin, hair] =
            human::maps(scene.material_seed(), cloth_template, images, materials);
        let knit = if scene.humans.iter().any(|h| h.outfit.knitted()) {
            human::knit(scene.material_seed(), images, materials)
        } else {
            cloth.clone()
        };
        let mut result = Self {
            handles,
            light_variants,
            environment,
            cloth,
            skin,
            hair,
            knit,
            variants,
        };
        // The staged production path supplies its already-built geometry/BVH.
        // Standalone callers receive the same scene-derived environment here.
        if finishes.is_none() {
            let transport = super::gi::BakeScene::from_manifest(scene, &result, materials, images);
            result.build_environment(scene, &transport, images);
        }
        result
    }

    pub(crate) fn build_environment(
        &mut self,
        scene: &IndoorManifest,
        transport: &super::gi::BakeScene,
        images: &mut impl super::preparation::AssetStore<Image>,
    ) {
        let [diffuse, specular] = environment::build(scene, transport);
        self.environment = EnvironmentMapLight {
            diffuse_map: images.add(diffuse),
            specular_map: images.add(specular),
            // Maps store linear cd/m², rather than clipped RGB scaled by a prior.
            intensity: 1.,
            rotation: Quat::from_rotation_y(scene.world_yaw),
            ..default()
        };
    }
}

type Definition = (Surface, [f32; 3], f32, f32, f32);
fn definitions(scene: &IndoorManifest) -> [Definition; 35] {
    let (wood, fabric, accent) = match scene.palette {
        0 => ([0.58, 0.37, 0.19], [0.16, 0.25, 0.28], [0.22, 0.34, 0.32]),
        1 => ([0.76, 0.60, 0.40], [0.31, 0.34, 0.35], [0.42, 0.48, 0.52]),
        2 => ([0.31, 0.19, 0.11], [0.26, 0.32, 0.40], [0.22, 0.29, 0.40]),
        3 => ([0.68, 0.50, 0.31], [0.43, 0.24, 0.17], [0.54, 0.32, 0.22]),
        4 => ([0.72, 0.64, 0.48], [0.27, 0.36, 0.23], [0.39, 0.45, 0.31]),
        _ => ([0.45, 0.29, 0.18], [0.37, 0.27, 0.33], [0.40, 0.33, 0.39]),
    };
    let floor = match scene.floor_style {
        0 => [wood[0] * 0.82, wood[1] * 0.82, wood[2] * 0.82],
        1 => [0.36, 0.37, 0.36],
        _ => [0.65, 0.64, 0.59],
    };
    let mut definitions = [
        (Surface::Paint, [0.84, 0.83, 0.79], 0.83, 0.0, 0.5),
        (Surface::Accent, accent, 0.83, 0.0, 0.5),
        (Surface::Wood, wood, 0.38, 0.0, 0.70),
        (Surface::WoodEdge, wood, 0.43, 0.0, 0.70),
        (
            Surface::Floor,
            floor,
            if scene.floor_style == 1 { 0.94 } else { 0.49 },
            0.0,
            2.0,
        ),
        (Surface::Ceiling, [0.88, 0.88, 0.85], 0.9, 0.0, 0.6),
        // Painted/powder-coated furniture metal reflects as a dielectric. Bare
        // polished/brushed metal uses Chrome, with a conductive base layer.
        (Surface::Metal, [0.12, 0.135, 0.15], 0.34, 0.0, 1.0),
        (Surface::Chrome, [0.64, 0.66, 0.68], 0.23, 1.0, 1.0),
        (Surface::Plastic, [0.065, 0.074, 0.08], 0.48, 0.0, 1.0),
        (Surface::Fabric, fabric, 0.92, 0.0, 0.12),
        (Surface::FabricAlt, accent, 0.93, 0.0, 0.12),
        (Surface::Glass, [0.88, 0.96, 0.96], 0.08, 0.0, 1.0),
        (Surface::Ceramic, [0.77, 0.75, 0.69], 0.24, 0.0, 1.0),
        (Surface::Soil, [0.095, 0.055, 0.025], 0.98, 0.0, 0.2),
        (Surface::Leaf, [0.10, 0.25, 0.055], 0.50, 0.0, 1.0),
        (Surface::LeafLight, [0.20, 0.34, 0.09], 0.53, 0.0, 1.0),
        (Surface::Paper, [0.86, 0.86, 0.81], 0.88, 0.0, 1.0),
        (Surface::Screen, [0.055, 0.09, 0.12], 0.27, 0.0, 1.0),
        (Surface::Ink, [0.055, 0.12, 0.19], 0.7, 0.0, 1.0),
        (Surface::Light, [0.90, 0.90, 0.84], 0.4, 0.0, 1.0),
        (Surface::Concrete, [0.52, 0.51, 0.47], 0.86, 0.0, 0.75),
        (Surface::Art, accent, 0.83, 0.0, 1.0),
        (Surface::Rubber, [0.033, 0.035, 0.038], 0.91, 0.0, 1.0),
        (Surface::LeafVariegated, [0.30, 0.42, 0.13], 0.43, 0.0, 1.0),
        (Surface::Terracotta, [0.53, 0.27, 0.16], 0.84, 0.0, 0.3),
        (Surface::Bark, [0.22, 0.16, 0.085], 0.91, 0.0, 0.2),
        (Surface::GlassInterior, [1.0; 3], 0.06, 0.0, 1.0),
        (Surface::ContainerGlass, [1.0; 3], 0.06, 0.0, 1.0),
        (Surface::Liquid, [1.0; 3], 0.04, 0.0, 1.0),
        (Surface::Drink, [0.09, 0.038, 0.016], 0.17, 0.0, 1.0),
        (Surface::PhoneScreen, [1.0; 3], 0.24, 0.0, 1.0),
        (Surface::PrintedPaper, [1.0; 3], 0.9, 0.0, 1.0),
        (Surface::Leather, [0.22, 0.12, 0.07], 0.42, 0.0, 0.14),
        (Surface::Whiteboard, [1.0; 3], 0.22, 0.0, 1.0),
        (Surface::Television, [1.0; 3], 0.24, 0.0, 1.0),
    ];
    if let Some(program) = &scene.program {
        for (surface, rgb, roughness, _, period) in &mut definitions {
            let Some(recipe) = program.materials.get(*surface as usize) else {
                continue;
            };
            if recipe.color != [1.0; 3] {
                *rgb = recipe.color;
            }
            *roughness = recipe.roughness;
            *period = recipe.period_m;
        }
    }
    definitions
}
fn prepare_map(scene: &IndoorManifest, definition: &Definition) -> Option<[Image; 3]> {
    let (surface, _, roughness, _, _) = *definition;
    if !matches!(
        surface,
        Surface::Wood
            | Surface::WoodEdge
            | Surface::Floor
            | Surface::Paint
            | Surface::Accent
            | Surface::Ceiling
            | Surface::Fabric
            | Surface::FabricAlt
            | Surface::Bark
            | Surface::Terracotta
            | Surface::Soil
            | Surface::Concrete
            | Surface::Ceramic
            | Surface::Leaf
            | Surface::LeafLight
            | Surface::LeafVariegated
            | Surface::Metal
            | Surface::Chrome
            | Surface::Plastic
            | Surface::Rubber
            | Surface::Paper
            | Surface::Art
            | Surface::Leather
    ) {
        return None;
    }
    let recipe = scene
        .program
        .as_ref()
        .map(|p| &p.materials[surface as usize]);
    let maps = texture_maps(
        surface,
        scene.floor_style,
        scene.material_seed(),
        roughness,
        recipe,
    );
    Some(mapped_images(
        maps,
        recipe.map_or(256, |r| r.map_size(scene.floor_style)),
        recipe,
    ))
}

fn mapped_images(
    maps: TextureMaps,
    size: u32,
    recipe: Option<&program::MaterialRecipe>,
) -> [Image; 3] {
    let mut images = filter::images(maps, size);
    if recipe.is_some_and(|r| r.leaf.is_some()) {
        for image in &mut images {
            let mut descriptor = match &image.sampler {
                ImageSampler::Descriptor(d) => d.clone(),
                _ => ImageSamplerDescriptor::default(),
            };
            descriptor.address_mode_u = ImageAddressMode::ClampToEdge;
            descriptor.address_mode_v = ImageAddressMode::ClampToEdge;
            image.sampler = ImageSampler::Descriptor(descriptor);
        }
    }
    images
}

fn hash(x: u32, y: u32, seed: u64) -> f32 {
    let mut v = x.wrapping_mul(374761393)
        ^ y.wrapping_mul(668265263)
        ^ seed as u32
        ^ ((seed >> 32) as u32).wrapping_mul(0x9e3779b9);
    v = (v ^ (v >> 13)).wrapping_mul(1274126177);
    (v ^ (v >> 16)) as f32 / u32::MAX as f32
}

// Periodic value noise: continuous at tile edges, with independent longitudinal
// and transverse frequencies for wood fibres and large-scale surface variation.
fn periodic_noise(u: f32, v: f32, nx: u32, ny: u32, seed: u64) -> f32 {
    let x = u.rem_euclid(1.0) * nx as f32;
    let y = v.rem_euclid(1.0) * ny as f32;
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

type TextureMaps = (Vec<u8>, Vec<u8>, Vec<u8>);
fn texture_maps(
    surface: Surface,
    floor_style: u32,
    seed: u64,
    roughness: f32,
    recipe: Option<&program::MaterialRecipe>,
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
    let n = recipe.map_or(256, |r| r.map_size(floor_style)) as usize;
    let mut heights = vec![0.0; n * n];
    let mut colors = Vec::with_capacity(n * n * 4);
    let mut data = Vec::with_capacity(n * n * 4);
    for y in 0..n {
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
                        let seam =
                            (u * 10.0).fract() < 0.022 || (v * 2.0 + stagger).fract() < 0.008;
                        let shade = if seam {
                            0.36
                        } else {
                            0.75 + hash(strip, (v * 2.0 + stagger).floor() as u32, seed) * 0.19
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
                                    0.75 + 0.15 * (v * std::f32::consts::PI).sin() + noise * 0.07
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
                recipe.texel(u, v, floor_style)
            } else {
                let (shade, h, roughness) = fallback();
                program::Texel {
                    color: [shade; 3],
                    height: h,
                    roughness,
                    occlusion: 1.,
                }
            };
            heights[y * n + x] = texel.height;
            let rgb = texel.color.map(|c| (c.clamp(0., 1.) * 255.) as u8);
            colors.extend([rgb[0], rgb[1], rgb[2], 255]);
            data.extend([
                (texel.occlusion * 255.).round() as u8,
                (texel.roughness.clamp(0.05, 1.0) * 255.0) as u8,
                if surface == Surface::Chrome { 255 } else { 0 },
                255,
            ]);
        }
    }
    let mut normals = Vec::with_capacity(n * n * 4);
    let period = recipe.map_or(Vec2::ONE, |r| {
        r.leaf
            .as_ref()
            .map_or_else(|| r.period_uv(), |l| Vec2::from_array(l.reference_size_m))
    });
    let slope = Vec2::splat(n as f32 * 0.5) / period;
    for y in 0..n {
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
            normals.extend(filter::encode(normal));
        }
    }
    (colors, normals, data)
}

#[derive(Clone, Copy)]
enum MapType {
    Color,
    Normal,
    Data,
}

fn mip_image(base: Vec<u8>, size: u32, kind: MapType) -> Image {
    // Input texels are bytes: evaluate exactly the same transfer function once
    // per possible input instead of millions of powf calls per room. Filtering
    // and the floating-point accumulation order remain unchanged.
    let linear = srgb8_table();
    let mut bytes = base.clone();
    let mut prev = base;
    let mut n = size as usize;
    while n > 1 {
        let next_n = n / 2;
        let mut next = Vec::with_capacity(next_n * next_n * 4);
        for y in 0..next_n {
            for x in 0..next_n {
                let mut sum = Vec3::ZERO;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let i = ((y * 2 + dy) * n + x * 2 + dx) * 4;
                    sum += match kind {
                        MapType::Color => Vec3::new(
                            linear[prev[i] as usize],
                            linear[prev[i + 1] as usize],
                            linear[prev[i + 2] as usize],
                        ),
                        MapType::Normal => {
                            Vec3::new(prev[i] as f32, prev[i + 1] as f32, prev[i + 2] as f32)
                                / 255.0
                                * 2.0
                                - Vec3::ONE
                        }
                        MapType::Data => {
                            Vec3::new(prev[i] as f32, prev[i + 1] as f32, prev[i + 2] as f32)
                                / 255.0
                        }
                    };
                }
                let value = match kind {
                    MapType::Color => (sum * 0.25).map(linear_to_srgb),
                    MapType::Normal => sum.normalize_or_zero() * 0.5 + Vec3::splat(0.5),
                    MapType::Data => sum * 0.25,
                };
                next.extend([
                    (value.x * 255.0) as u8,
                    (value.y * 255.0) as u8,
                    (value.z * 255.0) as u8,
                    255,
                ]);
            }
        }
        bytes.extend_from_slice(&next);
        prev = next;
        n = next_n;
    }
    mip_chain_image(bytes, size, kind)
}

fn mip_chain_image(bytes: Vec<u8>, size: u32, kind: MapType) -> Image {
    // Image::new validates base-level size, so append mip data afterwards.
    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0; (size * size * 4) as usize],
        if matches!(kind, MapType::Color) {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        },
        RenderAssetUsages::default(),
    );
    image.data = Some(bytes);
    image.texture_descriptor.mip_level_count = size.ilog2() + 1;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 4,
        ..default()
    });
    image
}

fn srgb8_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear((Vec3::splat(i as f32) / 255.0).x)))
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub fn kelvin_rgb(k: f32) -> Vec3 {
    // Smooth warm tungsten through cool sky RGB prior; not a spectral model.
    if k < 4500.0 {
        Vec3::new(1.0, 0.46, 0.18).lerp(
            Vec3::new(1.0, 0.90, 0.78),
            ((k - 1800.0) / 2700.0).clamp(0.0, 1.0),
        )
    } else {
        Vec3::new(1.0, 0.90, 0.78).lerp(
            Vec3::new(0.68, 0.82, 1.0),
            ((k - 4500.0) / 5500.0).clamp(0.0, 1.0),
        )
    }
}

#[cfg(test)]
mod surface_tests {
    use super::*;

    #[test]
    fn substrate_fields_consume_the_upper_seed_bits() {
        for (x, y) in [(3, 7), (79, 19), (12, 53)] {
            assert_ne!(hash(x, y, 73), hash(x, y, 73 + (1u64 << 32)));
        }
    }

    #[test]
    fn leaf_maps_clamp_at_atlas_edges_and_bind_absolute_color() {
        let scene = IndoorManifest::generate_with_humans(
            31,
            super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        let (mut images, mut materials) = (Assets::default(), Assets::default());
        let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
        for s in [Surface::Leaf, Surface::LeafLight, Surface::LeafVariegated] {
            let m = materials.get(&set.get(s)).unwrap();
            assert_eq!(m.base_color, Color::WHITE);
            assert_eq!(m.uv_transform, bevy::math::Affine2::IDENTITY);
            assert!(m.diffuse_transmission > 0. && m.diffuse_transmission < 0.5);
            for h in [
                &m.base_color_texture,
                &m.normal_map_texture,
                &m.metallic_roughness_texture,
            ] {
                let ImageSampler::Descriptor(d) = &images.get(h.as_ref().unwrap()).unwrap().sampler
                else {
                    panic!("leaf sampler")
                };
                assert_eq!(d.address_mode_u, ImageAddressMode::ClampToEdge);
                assert_eq!(d.address_mode_v, ImageAddressMode::ClampToEdge);
            }
        }
    }

    #[test]
    fn byte_color_lookup_preserves_every_original_transfer_value() {
        for (byte, &cached) in srgb8_table().iter().enumerate() {
            let original = (Vec3::splat(byte as f32) / 255.).map(srgb_to_linear).x;
            assert_eq!(cached.to_bits(), original.to_bits(), "byte {byte}");
        }
    }

    #[test]
    fn used_finish_palette_preserves_materials_and_texture_pixels() {
        let scene = IndoorManifest::generate_with_humans(
            44,
            super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.0,
        )
        .unwrap();
        let mut assemblies = vec![super::super::architecture::architecture(&scene)];
        assemblies.extend(
            scene
                .objects
                .iter()
                .map(super::super::objects::build_object),
        );
        let keys: std::collections::BTreeSet<_> = assemblies
            .iter()
            .flat_map(|a| a.parts.keys())
            .filter_map(|(surface, label)| {
                label
                    .rsplit_once("#finish")
                    .and_then(|(_, s)| s.parse::<usize>().ok())
                    .map(|i| (*surface, i))
            })
            .collect();
        let (mut all_images, mut all_materials) = (Assets::default(), Assets::default());
        let all = IndoorMaterials::build(&scene, &mut all_images, &mut all_materials);
        let (mut images, mut materials) = (Assets::default(), Assets::default());
        let used = IndoorMaterials::build_with_finishes(
            &scene,
            super::super::IndoorQuality::Auto,
            &mut images,
            &mut materials,
            Some(&keys),
        );
        assert!(!used.variants.is_empty());
        assert!(used.variants.len() < all.variants.len());
        assert!(images.len() < all_images.len());
        for key in keys.into_iter().filter(|k| variants::supports(k.0)) {
            let a = all_materials.get(&all.variants[&key]).unwrap();
            let b = materials.get(&used.variants[&key]).unwrap();
            assert_eq!(a.base_color, b.base_color);
            assert_eq!(a.emissive, b.emissive);
            assert_eq!(a.perceptual_roughness, b.perceptual_roughness);
            assert_eq!(a.metallic, b.metallic);
            assert_eq!(a.uv_transform, b.uv_transform);
            assert_eq!(a.clearcoat, b.clearcoat);
            for (ah, bh) in [
                (&a.base_color_texture, &b.base_color_texture),
                (&a.emissive_texture, &b.emissive_texture),
                (&a.normal_map_texture, &b.normal_map_texture),
                (&a.metallic_roughness_texture, &b.metallic_roughness_texture),
            ] {
                assert_eq!(ah.is_some(), bh.is_some());
                if let (Some(ah), Some(bh)) = (ah, bh) {
                    let ai = all_images.get(ah).unwrap();
                    let bi = images.get(bh).unwrap();
                    assert_eq!(ai.texture_descriptor, bi.texture_descriptor);
                    assert_eq!(ai.data, bi.data, "finish {key:?}");
                }
            }
        }
    }

    #[test]
    fn ceramic_items_share_three_glazes_with_independent_pigment() {
        let scene = IndoorManifest::generate_with_humans(
            81,
            super::super::layout::IndoorLayout::Conference,
            0.5,
            0,
            0.,
        )
        .unwrap();
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
        let mut structures = std::collections::BTreeSet::new();
        let mut pigments = std::collections::BTreeSet::new();
        for slot in 0..variants::COUNT {
            let mat = materials
                .get(&set.variants[&(Surface::Ceramic, slot)])
                .unwrap();
            let r =
                scene.program.as_ref().unwrap().materials[Surface::Ceramic as usize].variant(slot);
            let c = r.coating.as_ref().unwrap();
            assert_eq!(mat.clearcoat, c.clearcoat);
            assert_eq!(mat.clearcoat_perceptual_roughness, c.coat_roughness);
            assert_eq!(mat.metallic, 0.);
            assert_eq!(mat.occlusion_texture, mat.metallic_roughness_texture);
            structures.insert(mat.normal_map_texture.as_ref().unwrap().id());
            pigments.insert(mat.base_color.to_srgba().to_u8_array());
        }
        assert_eq!(structures.len(), 3);
        assert_eq!(pigments.len(), variants::COUNT);
        let size = images.len();
        for slot in 0..1000 {
            set.for_part(
                Surface::Ceramic,
                &format!("mug#finish{}", slot % variants::COUNT),
            );
        }
        assert_eq!(images.len(), size);
    }

    #[test]
    fn wardrobe_reuses_bounded_structure_maps_with_independent_tint() {
        use super::super::humans::{cloth_finish, person_material, HumanSurface};
        let scene = IndoorManifest::generate_with_humans(
            44,
            super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.8,
        )
        .unwrap();
        assert!(!scene.humans.is_empty());
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let indoor = IndoorMaterials::build(&scene, &mut images, &mut materials);
        let before = images.len();
        let mut structures = std::collections::BTreeSet::new();
        for person in &scene.humans {
            for surface in [
                HumanSurface::Shoes,
                HumanSurface::Sole,
                HumanSurface::ShoeDetail,
            ] {
                let material = person_material(person, surface, &indoor, &materials);
                assert_eq!(material.base_color, person.material_color(surface));
                assert!(material.base_color_texture.is_some());
                assert!(material.normal_map_texture.is_some());
                assert!(material.metallic_roughness_texture.is_some());
                assert_eq!(material.metallic, 0.);
                assert_eq!(material.clearcoat, 0.);
            }
            for surface in [HumanSurface::Top, HumanSurface::Trousers] {
                let m = person_material(person, surface, &indoor, &materials);
                let base = if person.outfit.knitted() && surface == HumanSurface::Top {
                    materials.get(&indoor.knit).unwrap()
                } else {
                    materials
                        .get(&indoor.variants[&cloth_finish(person, surface)])
                        .unwrap()
                };
                assert_eq!(m.base_color, person.material_color(surface));
                assert_eq!(m.base_color_texture, base.base_color_texture);
                assert_eq!(m.normal_map_texture, base.normal_map_texture);
                assert_eq!(
                    m.metallic_roughness_texture,
                    base.metallic_roughness_texture
                );
                assert_eq!(m.occlusion_texture, m.metallic_roughness_texture);
                assert_eq!(m.metallic, 0.);
                assert_eq!(m.clearcoat, 0.);
                structures.insert(m.normal_map_texture.unwrap().id());
                if let Some(a) = &person.appearance {
                    assert!(
                        (m.anisotropy_rotation - base.anisotropy_rotation + a.weave_rotation).abs()
                            < 1e-5
                    );
                }
            }
        }
        assert!(structures.len() <= 7);
        assert_eq!(images.len(), before, "wardrobe allocated per-person maps");
    }

    #[test]
    fn hard_surface_maps_vary_normals_and_roughness_without_losing_metalness() {
        let recipes = program::sample(81);
        for surface in [
            Surface::Metal,
            Surface::Chrome,
            Surface::Plastic,
            Surface::Rubber,
            Surface::Paper,
        ] {
            let recipe = &recipes[surface as usize];
            let (_, normals, data) = texture_maps(surface, 0, 81, recipe.roughness, Some(recipe));
            assert!(
                normals
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|p| p[0] != 128 || p[1] != 128),
                "{surface:?} has no resolved relief"
            );
            let roughness: std::collections::BTreeSet<_> =
                data.as_chunks::<4>().0.iter().map(|p| p[1]).collect();
            assert!(roughness.len() > 8, "{surface:?} has flat roughness");
            assert!(data
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[2] == if surface == Surface::Chrome { 255 } else { 0 }));
        }
    }

    #[test]
    fn pbr_maps_have_correct_encodings_mips_and_finite_unit_normals() {
        for surface in [
            Surface::Wood,
            Surface::Floor,
            Surface::Fabric,
            Surface::Leaf,
            Surface::Concrete,
        ] {
            let (color, normals, data) = texture_maps(surface, 0, 11, 0.7, None);
            assert_ne!(color, texture_maps(surface, 0, 12, 0.7, None).0);
            for pixel in normals.as_chunks::<4>().0.iter() {
                let normal = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32) / 127.5
                    - Vec3::ONE;
                assert!((normal.length() - 1.0).abs() < 0.018);
                assert!(normal.z > 0.0);
            }
            for pixel in data.as_chunks::<4>().0.iter() {
                assert_eq!(pixel[2], 0, "dielectrics must not become metallic");
            }
            let color_image = mip_image(color, 256, MapType::Color);
            let normal_image = mip_image(normals, 256, MapType::Normal);
            let data_image = mip_image(data, 256, MapType::Data);
            assert_eq!(
                color_image.texture_descriptor.format,
                TextureFormat::Rgba8UnormSrgb
            );
            assert_eq!(
                normal_image.texture_descriptor.format,
                TextureFormat::Rgba8Unorm
            );
            assert_eq!(
                data_image.texture_descriptor.format,
                TextureFormat::Rgba8Unorm
            );
            assert_eq!(color_image.texture_descriptor.mip_level_count, 9);
            assert_eq!(
                color_image.data.as_ref().unwrap().len(),
                (0..9)
                    .map(|m| ((256 >> m) as usize).pow(2) * 4)
                    .sum::<usize>()
            );
        }
    }

    #[test]
    fn sampled_metric_recipes_produce_valid_pbr_maps() {
        for seed in [0, 6, 115] {
            let recipes = program::sample(seed);
            for surface in [
                Surface::Paint,
                Surface::Wood,
                Surface::Fabric,
                Surface::Floor,
            ] {
                let recipe = &recipes[surface as usize];
                let (colors, normals, data) = texture_maps(
                    surface,
                    seed as u32 % 3,
                    seed,
                    recipe.roughness,
                    Some(recipe),
                );
                let size = recipe.map_size(seed as u32 % 3) as usize;
                assert_eq!(colors.len(), size * size * 4);
                for pixel in normals.as_chunks::<4>().0.iter() {
                    let n = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32) / 127.5
                        - Vec3::ONE;
                    assert!((n.length() - 1.0).abs() < 0.018 && n.z > 0.0);
                }
                assert!(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| p[1] >= 30 && p[2] == 0));
            }
        }
    }

    #[test]
    fn portable_glazing_avoids_screen_space_refraction() {
        let scene = IndoorManifest::generate(0, super::super::layout::IndoorLayout::Mixed, 0.65, 1)
            .unwrap();
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let set = IndoorMaterials::build_with_quality(
            &scene,
            super::super::IndoorQuality::Portable,
            &mut images,
            &mut materials,
        );
        let glass = materials.get(&set.get(Surface::Glass)).unwrap();
        assert_eq!(glass.alpha_mode, AlphaMode::Blend);
        assert_eq!(glass.specular_transmission, 0.0);
    }
}
