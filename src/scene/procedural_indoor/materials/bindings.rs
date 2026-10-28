//! Material selection, asset ownership and native/cooperative installation.
use super::*;

pub struct IndoorMaterials {
    pub(super) glazing: std::collections::BTreeMap<u64, Handle<StandardMaterial>>,
    pub reflections: Vec<reflections::ReflectionVolume>,
    pub(crate) gpu_bakes: gpu::Requests,
    pub(super) gpu_environment: bool,
    pub(super) handles: Vec<Handle<StandardMaterial>>,
    pub cloth: Handle<StandardMaterial>,
    pub skin: Handle<StandardMaterial>,
    pub hair: Handle<StandardMaterial>,
    pub knit: Handle<StandardMaterial>,
    pub(super) light_variants: Vec<Handle<StandardMaterial>>,
    pub(super) object_emitters:
        std::collections::BTreeMap<(usize, usize), Handle<StandardMaterial>>,
    pub(super) sky: Handle<StandardMaterial>,
    pub(crate) variants: std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
    pub environment: EnvironmentMapLight,
}

/// Texture parents and finish slots referenced by a prepared scene. Keep all
/// base material handles addressable; only synthesis of unused images is skipped.
pub(crate) struct MaterialSelection {
    pub surfaces: std::collections::BTreeSet<Surface>,
    pub finishes: std::collections::BTreeSet<(Surface, usize)>,
    /// Parts which resolve to the base slot, plus templates cloned by people.
    pub direct_surfaces: std::collections::BTreeSet<Surface>,
}

impl MaterialSelection {
    pub(super) fn omits_parent_maps(&self, scene: &IndoorManifest, surface: Surface) -> bool {
        if scene.program.is_none()
            || !self.surfaces.contains(&surface)
            || self.direct_surfaces.contains(&surface)
            || !variants::supports(surface)
            || variants::structure_count(surface) <= 1
        {
            return false;
        }
        let mut replacements = false;
        for &(s, slot) in &self.finishes {
            if s != surface {
                continue;
            }
            // Unresolved slots fall back to the base material. Structure zero
            // also inherits its maps; only nonzero groups replace all four.
            if slot >= variants::COUNT || slot % variants::structure_count(surface) == 0 {
                return false;
            }
            replacements = true;
        }
        replacements
    }

    pub(super) fn needs_maps(&self, scene: &IndoorManifest, surface: Surface) -> bool {
        self.surfaces.contains(&surface) && !self.omits_parent_maps(scene, surface)
    }
}

impl IndoorMaterials {
    /// Build just the material programs referenced by `(surface, part label)`.
    /// Useful for object/material studies. Whole-scene generation already uses
    /// the equivalent selection from its prepared geometry, including people.
    /// Handles for unused base surfaces remain addressable but have no maps.
    pub fn build_for_parts<'a>(
        scene: &IndoorManifest,
        images: &mut impl super::super::preparation::AssetStore<Image>,
        materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
        parts: impl IntoIterator<Item = (Surface, &'a str)>,
    ) -> Self {
        let mut selection = MaterialSelection {
            surfaces: Default::default(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        for (surface, label) in parts {
            selection.surfaces.insert(surface);
            if let Some(slot) =
                finish_slot(surface, label).filter(|&slot| registered_finish(surface, slot))
            {
                selection.finishes.insert((surface, slot));
            } else {
                selection.direct_surfaces.insert(surface);
            }
        }
        let mut result = Self::build_with_selection(
            scene,
            super::super::IndoorQuality::Auto,
            images,
            materials,
            Some(&selection),
            false,
        );
        // Internal staged preparation supplies its transport/environment later.
        // This public standalone helper must return usable cubemaps, like build;
        // the default image handles resolve to D2 textures and cannot be bound
        // as an EnvironmentMapLight on a camera.
        let transport =
            super::super::gi::BakeScene::from_manifest(scene, &result, materials, images);
        result.build_environment(scene, &transport, images);
        result
    }

    pub fn get(&self, surface: Surface) -> Handle<StandardMaterial> {
        self.handles[surface as usize].clone()
    }

    pub fn for_part(&self, surface: Surface, label: &str) -> Handle<StandardMaterial> {
        if let Some(handle) = label
            .rsplit_once("#glazing")
            .and_then(|(_, key)| key.parse::<u64>().ok())
            .and_then(|key| self.glazing.get(&key))
        {
            return handle.clone();
        }
        if label == "sky#exterior" {
            return self.sky.clone();
        }
        if let Some(material) =
            finish_slot(surface, label).and_then(|i| self.variants.get(&(surface, i)))
        {
            return material.clone();
        }
        if surface == Surface::Light {
            if let Some(material) =
                emitters::source_key(label).and_then(|id| self.object_emitters.get(&id))
            {
                return material.clone();
            }
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
        images: &mut impl super::super::preparation::AssetStore<Image>,
        materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
    ) -> Self {
        Self::build_with_quality(scene, super::super::IndoorQuality::Auto, images, materials)
    }

    pub fn build_with_quality(
        scene: &IndoorManifest,
        quality: super::super::IndoorQuality,
        images: &mut impl super::super::preparation::AssetStore<Image>,
        materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
    ) -> Self {
        Self::build_with_selection(scene, quality, images, materials, None, false)
    }

    pub(super) fn build_with_selection(
        scene: &IndoorManifest,
        quality: super::super::IndoorQuality,
        images: &mut impl super::super::preparation::AssetStore<Image>,
        materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
        selection: Option<&MaterialSelection>,
        gpu: bool,
    ) -> Self {
        let definitions = definitions(scene);
        // Generate immutable maps on a bounded pool; asset insertion remains ordered.
        // Wasm uses the identical serial function and produces the same pixels.
        let prepare = |definition: &Definition| {
            selection
                .is_none_or(|s| s.needs_maps(scene, definition.0))
                .then(|| prepare_map(scene, definition, gpu))
                .flatten()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let prepared = {
            super::super::preparation::workers::pool().scope(|scope| {
                for definition in &definitions {
                    let prepare = &prepare;
                    scope.spawn(async move { prepare(definition) });
                }
            })
        };
        #[cfg(target_arch = "wasm32")]
        let prepared: Vec<_> = definitions.iter().map(prepare).collect();
        // This compatibility API remains synchronous on every target. Only
        // the staged async browser path below opts into event-loop yields.
        bevy::tasks::block_on(Self::insert_prepared(
            scene,
            quality,
            images,
            materials,
            definitions,
            prepared,
            selection,
            gpu,
            false,
        ))
    }

    pub async fn build_async(
        scene: &IndoorManifest,
        quality: super::super::IndoorQuality,
        images: &mut impl super::super::preparation::AssetStore<Image>,
        materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
    ) -> Self {
        Self::build_async_with_selection(scene, quality, images, materials, None, false).await
    }

    pub(crate) async fn build_async_with_selection(
        scene: &IndoorManifest,
        quality: super::super::IndoorQuality,
        images: &mut impl super::super::preparation::AssetStore<Image>,
        materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
        selection: Option<&MaterialSelection>,
        gpu: bool,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::build_with_selection(scene, quality, images, materials, selection, gpu)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let definitions = definitions(scene);
            let mut maps = Vec::with_capacity(definitions.len());
            for definition in &definitions {
                super::super::preparation::cooperate().await;
                maps.push(
                    selection
                        .is_none_or(|s| s.needs_maps(scene, definition.0))
                        .then(|| prepare_map(scene, definition, gpu))
                        .flatten(),
                );
            }
            Self::insert_prepared(
                scene,
                quality,
                images,
                materials,
                definitions,
                maps,
                selection,
                gpu,
                true,
            )
            .await
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn insert_prepared(
        scene: &IndoorManifest,
        quality: super::super::IndoorQuality,
        images: &mut impl super::super::preparation::AssetStore<Image>,
        materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
        definitions: [Definition; 36],
        prepared: Vec<Option<gpu::Maps>>,
        selection: Option<&MaterialSelection>,
        gpu: bool,
        cooperative: bool,
    ) -> Self {
        let mut handles = Vec::new();
        let mut gpu_bakes = gpu::Requests::default();
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
            if let Some(maps) = maps {
                let [color, normal, data] = maps.insert(images, &mut gpu_bakes);
                mat.base_color_texture = Some(color);
                mat.normal_map_texture = Some(normal);
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
            } else if selection.is_some_and(|s| s.omits_parent_maps(scene, surface)) {
                // Nonzero structure variants replace every texture, but clone
                // the parent's PBR scalars. Retain the map insertion's scalar
                // effect without synthesizing images which no part samples.
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
                        mat.diffuse_transmission = 0.06;
                    }
                    // The blade atlas remains 0..1 despite its metric derivative reference.
                    mat.uv_transform = bevy::math::Affine2::IDENTITY;
                }
                Surface::Screen if selection.is_none_or(|s| s.surfaces.contains(&surface)) => {
                    screens::apply(scene.material_seed(), false, &mut mat, images);
                }
                Surface::Whiteboard if selection.is_none_or(|s| s.surfaces.contains(&surface)) => {
                    boards::apply(scene.material_seed(), &mut mat, images);
                }
                Surface::Art if selection.is_none_or(|s| s.surfaces.contains(&surface)) => {
                    art::apply(scene.material_seed(), &mut mat, images);
                }
                Surface::Television if selection.is_none_or(|s| s.surfaces.contains(&surface)) => {
                    screens::apply_tv(scene.material_seed(), &mut mat, images);
                }
                Surface::PhoneScreen if selection.is_none_or(|s| s.surfaces.contains(&surface)) => {
                    screens::apply(scene.material_seed(), true, &mut mat, images);
                }
                Surface::PrintedPaper
                    if selection.is_none_or(|s| s.surfaces.contains(&surface)) =>
                {
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
                    let luminance = super::super::architecture::fixture_lumens(scene)
                        / (std::f32::consts::PI
                            * super::super::architecture::fixtures::emitting_area(scene));
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
            response::conformal_coat(&mut mat);
            handles.push(materials.add(mat));
            if cooperative {
                super::super::preparation::cooperate().await;
            }
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
        let fixture_positions = super::super::architecture::fixture_positions(scene);
        let light_variants = fixture_positions
            .iter()
            .enumerate()
            .map(|(i, &position)| {
                let area = super::super::architecture::fixtures::emitting_area_at(scene, position);
                let (color, lumens) =
                    super::super::architecture::fixture_photometry_at(scene, i, &fixture_positions);
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
        let object_emitters = emitters::build(scene, &handles[Surface::Light as usize], materials);
        let mut variants = variants::build(
            scene,
            &handles,
            images,
            materials,
            selection.map(|s| &s.finishes),
            gpu,
            &mut gpu_bakes,
            cooperative,
        )
        .await;
        variants.extend(kitchen::build(
            scene,
            &handles,
            images,
            materials,
            selection.map(|s| &s.finishes),
        ));
        variants.extend(exterior::build(
            scene,
            &handles,
            materials,
            selection.map(|s| &s.finishes),
        ));
        variants.extend(signals::build(scene, materials, selection));
        variants.extend(supplies::build(scene, &handles, materials, selection));
        let cloth_template = materials
            .get(&handles[Surface::Fabric as usize])
            .unwrap()
            .clone();
        let [cloth, skin, hair] = if selection.is_none() || !scene.humans.is_empty() {
            human::maps(scene.material_seed(), cloth_template, images, materials)
        } else {
            // No actor refers to these templates. Aliases preserve the handle
            // contract without paying for skin and hair atlases in empty rooms.
            std::array::from_fn(|_| handles[Surface::Fabric as usize].clone())
        };
        if cooperative {
            super::super::preparation::cooperate().await;
        }
        let knit = if scene.humans.iter().any(|h| h.outfit.knitted()) {
            human::knit(
                scene.material_seed(),
                images,
                materials,
                gpu,
                &mut gpu_bakes,
            )
        } else {
            cloth.clone()
        };
        for (slot, pair) in rugs::build(
            scene,
            &handles[Surface::FabricAlt as usize],
            images,
            materials,
            gpu,
            &mut gpu_bakes,
            cooperative,
        )
        .await
        {
            for (binding, handle) in pair.into_iter().enumerate() {
                variants.insert(
                    (Surface::FabricAlt, rugs::finish_slot(slot, binding != 0)),
                    handle,
                );
            }
        }
        let mut result = Self {
            glazing: glazing::build(scene, quality, images, materials),
            reflections: Vec::new(),
            sky: materials.add(super::super::architecture::background::material(scene)),
            handles,
            gpu_bakes,
            gpu_environment: gpu,
            light_variants,
            object_emitters,
            environment,
            cloth,
            skin,
            hair,
            knit,
            variants,
        };
        // The staged production path supplies its already-built geometry/BVH.
        // Standalone callers receive the same scene-derived environment here.
        if selection.is_none() {
            let transport =
                super::super::gi::BakeScene::from_manifest(scene, &result, materials, images);
            #[cfg(not(target_arch = "wasm32"))]
            result.build_environment(scene, &transport, images);
            #[cfg(target_arch = "wasm32")]
            if cooperative {
                result
                    .build_environment_async(scene, &transport, images)
                    .await;
            } else {
                result.build_environment(scene, &transport, images);
            }
        }
        result
    }

    pub(crate) fn build_environment(
        &mut self,
        scene: &IndoorManifest,
        transport: &super::super::gi::BakeScene,
        images: &mut impl super::super::preparation::AssetStore<Image>,
    ) {
        let plans = reflections::plans(scene, transport);
        let origins: Vec<_> = std::iter::once(transport.reflection_origin(scene))
            .chain(plans.iter().map(|p| p.translation))
            .collect();
        let mut maps = environment::insert_many(
            scene,
            transport,
            &origins,
            images,
            &mut self.gpu_bakes,
            self.gpu_environment,
        )
        .into_iter();
        let [diffuse, specular] = maps.next().expect("global reflection map");
        self.environment = EnvironmentMapLight {
            diffuse_map: diffuse,
            specular_map: specular,
            // Maps store linear cd/m², rather than clipped RGB scaled by a prior.
            intensity: 1.,
            rotation: Quat::from_rotation_y(scene.world_yaw),
            ..default()
        };
        self.reflections = plans
            .into_iter()
            .zip(maps)
            .map(|(plan, maps)| reflections::bind(plan, maps))
            .collect();
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn build_environment_async(
        &mut self,
        scene: &IndoorManifest,
        transport: &super::super::gi::BakeScene,
        images: &mut impl super::super::preparation::AssetStore<Image>,
    ) {
        let [diffuse, specular] = if self.gpu_environment {
            let pixels = environment::cooperative::trace_async(scene, transport).await;
            environment::gpu::prepare(pixels, images, &mut self.gpu_bakes)
        } else {
            environment::build_async(scene, transport)
                .await
                .map(|image| images.add(image))
        };
        self.environment = EnvironmentMapLight {
            diffuse_map: diffuse,
            specular_map: specular,
            intensity: 1.,
            rotation: Quat::from_rotation_y(scene.world_yaw),
            ..default()
        };
        // The browser uses the global probe, without binding-array volumes.
        self.reflections.clear();
    }
}

/// Whether a numbered finish is supported for a surface, including dedicated
/// kitchen, exterior and safety-sign bindings. This is the same registry used
/// by consumed-map selection and transport; callers need not duplicate limits.
pub fn registered_finish(surface: Surface, slot: usize) -> bool {
    (variants::supports(surface) && slot < variants::count(surface))
        || exterior::supports(surface, slot)
        || signals::supports(surface, slot)
        || supplies::supports(surface, slot)
        || matches!(
            (surface, slot),
            (Surface::Ceramic, 100..=102) | (Surface::Terracotta, 103) | (Surface::Paper, 104)
        )
}
/// Shared by raster material resolution and symbolic diffuse-transport binding.
pub(crate) fn finish_slot(surface: Surface, label: &str) -> Option<usize> {
    if surface == Surface::FabricAlt {
        if let Some((slot, binding)) = rugs::key(label) {
            return Some(rugs::finish_slot(slot, binding));
        }
    }
    label
        .rsplit_once("#finish")
        .and_then(|(_, s)| s.split('#').next()?.parse::<usize>().ok())
}
