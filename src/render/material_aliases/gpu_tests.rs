use super::*;
use bevy::render::{
    material_bind_groups::MaterialBindGroupAllocator,
    pipelined_rendering::PipelinedRenderingPlugin, render_resource::*,
};

/// Exercise Bevy's actual allocator, including repeated creation, replacement
/// and removal. This catches capacity stranded by duplicate PBR map bindings.
#[test]
#[ignore = "requires a native bindless GPU"]
fn shared_texture_churn_releases_every_binding_without_copying_gpu_objects() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::render::RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    );
    app.finish();
    app.cleanup();
    app.update();
    let render = app.sub_app(RenderApp).world();
    let device = render.resource::<RenderDevice>().clone();
    let queue = render
        .resource::<bevy::render::renderer::RenderQueue>()
        .clone();
    assert!(StandardMaterial::bindless_supported(&device));
    let original = render
        .resource::<RenderAssets<GpuImage>>()
        .iter()
        .next()
        .expect("default image uploaded")
        .1
        .clone();
    let resources = vec![
        BindlessResourceType::Texture2d,
        BindlessResourceType::SamplerFiltering,
        BindlessResourceType::Texture2d,
        BindlessResourceType::SamplerFiltering,
    ];
    let layout = BindGroupLayoutDescriptor::new(
        "alias regression",
        &create_bindless_bind_group_layout_entries(4, 16, BindingNumber(0), &resources),
    );
    let mut allocator = MaterialBindGroupAllocator::new(
        &device,
        "alias regression",
        Some(BindlessDescriptor {
            resources: resources.into(),
            buffers: Vec::new().into(),
            index_tables: vec![BindlessIndexTableDescriptor {
                indices: BindlessIndex(0)..BindlessIndex(4),
                binding_number: BindingNumber(0),
            }]
            .into(),
        }),
        layout.clone(),
        Some(BindlessSlabResourceLimit::Custom(16)),
    );
    let descriptor = bevy::image::ImageSamplerDescriptor::linear();
    let explicit_samplers =
        std::array::from_fn::<_, 2, _>(|_| device.create_sampler(&descriptor.as_wgpu()));
    assert_ne!(explicit_samplers[0].id(), explicit_samplers[1].id());
    let mut aliases = Aliases::default();
    let mut images = RenderAssets::<GpuImage>::default();
    for cycle in 0..2048 {
        let handles: [Handle<Image>; 2] = std::array::from_fn(|_| Uuid::new_v4().into());
        let mut materials = handles.each_ref().map(|handle| StandardMaterial {
            base_color_texture: Some(handle.clone()),
            emissive_texture: Some(handle.clone()),
            ..default()
        });
        let mut recipes = samplers::Recipes::default();
        let mut extracted = bevy::render::render_asset::ExtractedAssets::<GpuImage>::default();
        for handle in &handles {
            let mut source = Image::default();
            if cycle % 2 != 0 {
                source.sampler = bevy::image::ImageSampler::Descriptor(descriptor.clone());
            }
            extracted.extracted.push((handle.id(), source));
        }
        recipes.capture(&extracted);
        let mut pending = Sources::default();
        for material in &mut materials {
            pending.add_material(material);
            aliases.material(material);
        }
        aliases.refresh(&pending, &recipes, &mut images);
        assert_eq!(aliases.images.len(), 4);
        assert_eq!(images.iter().count(), 0);
        // A render-only source has no remaining CPU Assets<Image> entry.
        for (i, handle) in handles.iter().enumerate() {
            let mut image = original.clone();
            if cycle % 2 != 0 {
                image.sampler = explicit_samplers[i].clone();
            }
            images.insert(handle.id(), image);
        }
        recipes.observe_uploads(&images);
        aliases.refresh(&Sources::default(), &recipes, &mut images);
        assert_eq!(aliases.samplers.len(), 2);
        let sampler = |handle: &Option<Handle<Image>>| {
            images
                .get(handle.as_ref().unwrap().id())
                .unwrap()
                .sampler
                .id()
        };
        assert_eq!(
            sampler(&materials[0].base_color_texture),
            sampler(&materials[1].base_color_texture)
        );
        assert_ne!(
            sampler(&materials[0].base_color_texture),
            sampler(&materials[0].emissive_texture)
        );
        let bindings = materials.map(|material| {
            let mut bindings = BindGroupBuilder::default();
            for (slot, image) in [material.base_color_texture, material.emissive_texture]
                .into_iter()
                .enumerate()
            {
                let alias = images.get(image.unwrap().id()).unwrap();
                assert_eq!(&*alias.texture_view, &*original.texture_view);
                if cycle % 2 == 0 {
                    assert_eq!(&*alias.sampler, &*original.sampler);
                } else {
                    assert!(explicit_samplers.iter().any(|s| **s == *alias.sampler));
                }
                assert_ne!(alias.texture_view.id(), original.texture_view.id());
                bindings.push((
                    slot as u32 * 2,
                    UnpreparedBindingResource::TextureView(
                        TextureViewDimension::D2,
                        alias.texture_view.clone(),
                    ),
                ));
                bindings.push((
                    slot as u32 * 2 + 1,
                    UnpreparedBindingResource::Sampler(
                        SamplerBindingType::Filtering,
                        alias.sampler.clone(),
                    ),
                ));
            }
            allocator.allocate_unprepared(&mut bindings, &layout)
        });
        // Four distinct image bindings share just two slot-specific samplers.
        assert_eq!(allocator.allocations(), 6);
        allocator.free(bindings[0]);
        assert_eq!(allocator.allocations(), 4);
        allocator.free(bindings[1]);
        assert_eq!(allocator.allocations(), 0, "cycle {cycle}");
        assert_eq!(allocator.slab_count(), 1, "cycle {cycle}");
        if cycle % 2 != 0 {
            // A caller may replace a GPU image directly. Without a matching
            // source extraction, never apply an older sampler description.
            let replacement =
                device.create_sampler(&bevy::image::ImageSamplerDescriptor::nearest().as_wgpu());
            let mut changed = original.clone();
            changed.sampler = replacement.clone();
            images.insert(handles[0].id(), changed);
            recipes.observe_uploads(&images);
            aliases.refresh(&Sources::default(), &recipes, &mut images);
            assert!(aliases
                .images
                .iter()
                .filter(|((id, _), _)| *id == handles[0].id())
                .all(|(_, alias)| *images.get(alias.handle.id()).unwrap().sampler == *replacement));
        }
        let saved = handles
            .each_ref()
            .map(|handle| images.remove(handle.id()).unwrap());
        aliases.refresh(&pending, &recipes, &mut images);
        assert_eq!(aliases.images.len(), 4);
        assert_eq!(
            images.iter().count(),
            0,
            "pending sources cannot expose stale aliases"
        );
        assert!(aliases.samplers.is_empty());
        // Even restoring the identical GPU objects must restore the aliases.
        for (handle, image) in handles.iter().zip(saved) {
            images.insert(handle.id(), image);
        }
        aliases.refresh(&Sources::default(), &recipes, &mut images);
        assert_eq!(images.iter().count(), 6);
        for handle in &handles {
            images.remove(handle.id());
        }
        aliases.refresh(&Sources::default(), &recipes, &mut images);
        assert!(aliases.images.is_empty());
        assert!(aliases.samplers.is_empty());
        assert_eq!(images.iter().count(), 0);
    }
    queue.submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    drop(allocator);
    drop(images);
    drop(original);
    drop(app);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
