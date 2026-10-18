//! Render-only image aliases keep Bevy 0.20's bindless allocator balanced.
//!
//! Its allocation candidate counts repeated resources before inserting any of
//! them, but freeing counts only the last reference. A packed PBR image or an
//! emissive screen used in two slots can therefore strand slab capacity forever.
//! Give image slots distinct Bevy identities, retaining the SAME WGPU views.
//! Share samplers per slot when their complete descriptions match. No texture
//! upload/copy, filtering change, shader change or main-world edit is involved.
use bevy::{
    asset::{uuid::Uuid, VisitAssetDependencies},
    prelude::*,
    reflect::ReflectMut,
    render::{
        erased_render_asset::{prepare_erased_assets, ExtractedAssets},
        render_asset::{prepare_assets, RenderAssets},
        render_resource::{AsBindGroup, Sampler, SamplerId, TextureView, TextureViewId},
        renderer::RenderDevice,
        texture::GpuImage,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use std::collections::{HashMap, HashSet};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_tests;
mod samplers;

#[derive(Resource, Default)]
struct Sources(HashSet<AssetId<Image>>);

struct Alias {
    handle: Handle<Image>,
    uploaded: Option<(TextureViewId, SamplerId)>,
}

#[derive(Resource, Default)]
struct Aliases {
    images: HashMap<(AssetId<Image>, usize), Alias>,
    // Only slots within one material need distinct identities. Preserve
    // sharing across images and materials with identical filtering settings.
    samplers: HashMap<(samplers::Key, usize), Sampler>,
    device: Option<RenderDevice>,
}

pub(super) fn install(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    let Some(device) = render.world().get_resource::<RenderDevice>() else {
        return;
    };
    if !StandardMaterial::bindless_supported(device) {
        return;
    }
    render
        .init_resource::<Sources>()
        .init_resource::<Aliases>()
        .init_resource::<samplers::Recipes>();
    render.add_systems(ExtractSchedule, extract_sources);
    render.add_systems(
        Render,
        samplers::remember
            .in_set(RenderSystems::PrepareAssets)
            .before(prepare_assets::<GpuImage>),
    );
    render.add_systems(
        Render,
        prepare
            .in_set(RenderSystems::PrepareAssets)
            .after(prepare_assets::<GpuImage>)
            .before(prepare_erased_assets::<MeshMaterial3d<StandardMaterial>>),
    );
}

fn extract_sources(
    images: Extract<Res<Assets<Image>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    mut sources: ResMut<Sources>,
) {
    sources.0.clear();
    sources.0.extend(images.ids());
    // An asynchronously loading image need not exist in either asset world
    // yet. Keep its mapping while a live source material references it, so a
    // material already in Bevy's retry queue resolves when the upload arrives.
    for (_, material) in materials.iter() {
        sources.add_material(material);
    }
}

impl Sources {
    fn add_material(&mut self, material: &StandardMaterial) {
        material.visit_dependencies(&mut |id| {
            if let Ok(image) = id.try_typed::<Image>() {
                self.0.insert(image);
            }
        });
    }
}

impl Aliases {
    fn material(&mut self, material: &mut StandardMaterial) {
        // Reflection includes feature-gated PBR image slots enabled by a caller's
        // dependency graph, without maintaining an incomplete parallel field list.
        let ReflectMut::Struct(fields) = material.reflect_mut() else {
            unreachable!("StandardMaterial is a reflected struct")
        };
        for slot in 0..fields.field_len() {
            let Some(Some(image)) = fields
                .field_at_mut(slot)
                .and_then(|f| f.try_downcast_mut::<Option<Handle<Image>>>())
            else {
                continue;
            };
            let alias = self
                .images
                .entry((image.id(), slot))
                .or_insert_with(|| Alias {
                    handle: Uuid::new_v4().into(),
                    uploaded: None,
                });
            *image = alias.handle.clone();
        }
    }

    fn refresh(
        &mut self,
        sources: &Sources,
        recipes: &samplers::Recipes,
        images: &mut RenderAssets<GpuImage>,
    ) {
        let mut used_samplers = HashSet::new();
        self.images.retain(|(source, slot), alias| {
            // RENDER_WORLD-only images intentionally leave Assets<Image> after
            // extraction. Their GPU image remains authoritative until Bevy
            // observes the last source handle being dropped.
            if !sources.0.contains(source) && images.get(*source).is_none() {
                images.remove(alias.handle.id());
                return false;
            }
            let Some(original) = images.get(*source) else {
                // The material can already be in Bevy's retry queue. Its alias
                // becomes ready only after the original image upload is ready.
                images.remove(alias.handle.id());
                alias.uploaded = None;
                return true;
            };
            let stamp = (original.texture_view.id(), original.sampler.id());
            let sampler_key = (recipes.key(*source, stamp.1), *slot);
            used_samplers.insert(sampler_key);
            if alias.uploaded != Some(stamp) {
                let mut image = original.clone();
                image.texture_view = TextureView::from((*original.texture_view).clone());
                image.sampler = self
                    .samplers
                    .entry(sampler_key)
                    .or_insert_with(|| Sampler::from((*original.sampler).clone()))
                    .clone();
                images.insert(alias.handle.id(), image);
                alias.uploaded = Some(stamp);
            }
            true
        });
        self.samplers.retain(|key, _| used_samplers.contains(key));
    }
}

fn prepare(
    mut extracted: ResMut<ExtractedAssets<MeshMaterial3d<StandardMaterial>>>,
    sources: Res<Sources>,
    mut recipes: ResMut<samplers::Recipes>,
    device: Res<RenderDevice>,
    mut images: ResMut<RenderAssets<GpuImage>>,
    mut aliases: ResMut<Aliases>,
) {
    // Descriptor equality must never reuse an object from a lost device.
    if aliases.device.as_ref().map(RenderDevice::wgpu_device) != Some(device.wgpu_device()) {
        aliases.samplers.clear();
        aliases.device = Some((*device).clone());
    }
    recipes.observe_uploads(&images);
    for (_, material) in &mut extracted.extracted {
        aliases.material(material);
    }
    aliases.refresh(&sources, &recipes, &mut images);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_maps_are_distinct_per_slot_and_reused_across_materials() {
        let image: Handle<Image> = Uuid::new_v4().into();
        let source = StandardMaterial {
            base_color_texture: Some(image.clone()),
            emissive_texture: Some(image.clone()),
            metallic_roughness_texture: Some(image.clone()),
            occlusion_texture: Some(image.clone()),
            ..default()
        };
        let mut aliases = Aliases::default();
        let mut a = source.clone();
        let mut b = source.clone();
        aliases.material(&mut a);
        aliases.material(&mut b);
        assert_eq!(aliases.images.len(), 4);
        assert_ne!(a.base_color_texture, a.emissive_texture);
        assert_ne!(a.metallic_roughness_texture, a.occlusion_texture);
        assert_eq!(a.base_color_texture, b.base_color_texture);
        assert_eq!(source.base_color_texture, source.emissive_texture);
        assert_eq!(source.base_color_texture.as_ref().unwrap().id(), image.id());

        // An image waiting for upload remains queued; removing the source also
        // removes an alias that never became ready, bounding failed loads.
        let mut images = RenderAssets::<GpuImage>::default();
        let mut pending = Sources::default();
        pending.add_material(&source);
        aliases.refresh(&pending, &samplers::Recipes::default(), &mut images);
        assert_eq!(aliases.images.len(), 4);
        aliases.refresh(
            &Sources::default(),
            &samplers::Recipes::default(),
            &mut images,
        );
        assert!(aliases.images.is_empty());
    }
}
