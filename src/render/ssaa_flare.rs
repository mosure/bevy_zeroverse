//! Reuse exact source highlights within one logical exposure. All cameras share
//! one scratch texture. Queue ordering keeps writes and reads paired, including
//! when the source changes at the next instant.
use super::*;

#[derive(Default)]
pub(super) struct Cache {
    pipelines: Option<(
        CachedRenderPipelineId,
        CachedRenderPipelineId,
        BindGroupLayout,
    )>,
    image: Option<ImageBinding>,
    #[cfg(test)]
    pub(super) draws: std::sync::atomic::AtomicUsize,
}

struct ImageBinding {
    size: UVec2,
    view: TextureView,
    group: BindGroup,
}

impl Cache {
    pub(super) fn release_image(&mut self) {
        self.image = None;
    }

    pub(super) fn prepare(&mut self, size: UVec2, device: &RenderDevice, cache: &PipelineCache) {
        if size == UVec2::ZERO
            || self
                .image
                .as_ref()
                .is_some_and(|i| size.cmple(i.size).all())
        {
            return;
        }
        let (_, _, layout) = self.pipelines.get_or_insert_with(|| {
            let grid = BindGroupLayoutDescriptor::new(
                "rgb_flare_grid",
                &BindGroupLayoutEntries::single(
                    ShaderStages::FRAGMENT,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            );
            let source = source_layout();
            let fill = queue_fragment(
                cache,
                "rgb_flare_highlights",
                SENSOR_SHADER,
                "flare_cells",
                vec![source.clone()],
            );
            let resolve = queue_fragment(
                cache,
                "rgb_sensor_from_highlights",
                SENSOR_SHADER,
                "cached_sensor_fragment",
                vec![source, grid.clone()],
            );
            (fill, resolve, cache.get_bind_group_layout(&grid))
        });
        let size = self.image.as_ref().map_or(size, |i| i.size.max(size));
        let view = device
            .create_texture(&TextureDescriptor {
                label: Some("rgb_flare_grid"),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&default());
        let group =
            device.create_bind_group("rgb_flare_grid", layout, &BindGroupEntries::single(&view));
        self.image = Some(ImageBinding { size, view, group });
    }

    pub(super) fn encode<'a>(
        &'a self,
        encoder: &mut CommandEncoder,
        cache: &'a PipelineCache,
        source: &BindGroup,
        size: UVec2,
    ) -> Option<(&'a RenderPipeline, &'a BindGroup)> {
        let (fill, resolve, _) = self.pipelines.as_ref()?;
        let image = self.image.as_ref()?;
        let fill = cache.get_render_pipeline(*fill)?;
        let resolve = cache.get_render_pipeline(*resolve)?;
        draw(
            encoder,
            "rgb_flare_highlights",
            &image.view,
            fill,
            source,
            None,
            Some(size),
        );
        #[cfg(test)]
        self.draws.fetch_add(1, Ordering::Relaxed);
        Some((resolve, &image.group))
    }
}
