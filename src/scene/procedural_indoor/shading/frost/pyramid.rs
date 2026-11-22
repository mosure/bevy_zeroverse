use bevy::render::render_resource::{Texture, TextureView, TextureViewId};
use wgpu as gpu;

/// Mip zero is filled by Bevy's ordinary transmission copy. Higher levels
/// cover the opaque scene that immediately precedes that copy.
pub(super) struct Pyramid {
    pub texture: Texture,
    pub all: TextureView,
    targets: Vec<TextureView>,
    reductions: Vec<gpu::BindGroup>,
}

pub(super) struct Filters {
    first_depth: gpu::RenderPipeline,
    first_color: gpu::RenderPipeline,
    next: gpu::RenderPipeline,
}

impl Filters {
    pub fn new(device: &gpu::Device) -> Self {
        let shader = device.create_shader_module(gpu::ShaderModuleDescriptor {
            label: Some("Indoor frost reduction"),
            source: gpu::ShaderSource::Wgsl(include_str!("reduce.wgsl").into()),
        });
        let create = |entry_point| {
            device.create_render_pipeline(&gpu::RenderPipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                vertex: gpu::VertexState {
                    module: &shader,
                    entry_point: Some("fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(gpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    targets: &[Some(gpu::TextureFormat::Rgba16Float.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            first_depth: create("opaque_depth"),
            first_color: create("opaque_color"),
            next: create("previous_level"),
        }
    }
    fn first(&self, depth: bool) -> &gpu::RenderPipeline {
        if depth {
            &self.first_depth
        } else {
            &self.first_color
        }
    }
}

fn sources(
    device: &gpu::Device,
    pipeline: &gpu::RenderPipeline,
    source: &TextureView,
    depth: Option<&TextureView>,
) -> gpu::BindGroup {
    let color = gpu::BindGroupEntry {
        binding: 0,
        resource: gpu::BindingResource::TextureView(source),
    };
    let depth = depth.map(|v| gpu::BindGroupEntry {
        binding: 1,
        resource: gpu::BindingResource::TextureView(v),
    });
    let layout = pipeline.get_bind_group_layout(0);
    match depth {
        Some(d) => device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("Indoor frost source and depth"),
            layout: &layout,
            entries: &[color, d],
        }),
        None => device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("Indoor frost source"),
            layout: &layout,
            entries: &[color],
        }),
    }
}

pub(super) struct Source {
    key: (TextureViewId, Option<TextureViewId>),
    group: gpu::BindGroup,
}

impl Pyramid {
    pub fn new(device: &gpu::Device, filters: &Filters, width: u32, height: u32) -> Self {
        let levels = (width.max(height).ilog2() + 1).min(7);
        let texture: Texture = device
            .create_texture(&gpu::TextureDescriptor {
                label: Some("Indoor frost radiance and depth bounds"),
                size: gpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: gpu::TextureDimension::D2,
                format: gpu::TextureFormat::Rgba16Float,
                usage: {
                    let usage = gpu::TextureUsages::TEXTURE_BINDING
                        | gpu::TextureUsages::COPY_DST
                        | gpu::TextureUsages::RENDER_ATTACHMENT;
                    #[cfg(test)]
                    let usage = usage | gpu::TextureUsages::COPY_SRC;
                    usage
                },
                view_formats: &[],
            })
            .into();
        let all = texture.create_view(&Default::default());
        let targets: Vec<_> = (1..levels)
            .map(|level| {
                texture.create_view(&gpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let reductions = targets
            .iter()
            .take(targets.len().saturating_sub(1))
            .map(|view| sources(device, &filters.next, view, None))
            .collect();
        Self {
            texture,
            all,
            targets,
            reductions,
        }
    }
    pub fn encode(
        &self,
        device: &gpu::Device,
        filters: &Filters,
        encoder: &mut gpu::CommandEncoder,
        source: &TextureView,
        depth: Option<&TextureView>,
        cached: &mut Option<Source>,
    ) {
        let key = (source.id(), depth.map(TextureView::id));
        let first = filters.first(depth.is_some());
        if cached.as_ref().is_none_or(|s| s.key != key) {
            *cached = Some(Source {
                key,
                group: sources(device, first, source, depth),
            });
        }
        let bindings = std::iter::once((first, &cached.as_ref().unwrap().group))
            .chain(self.reductions.iter().map(|group| (&filters.next, group)));
        for (target, (pipeline, group)) in self.targets.iter().zip(bindings) {
            let mut pass = encoder.begin_render_pass(&gpu::RenderPassDescriptor {
                label: Some("Filter indoor transmission"),
                color_attachments: &[Some(gpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: gpu::Operations {
                        load: gpu::LoadOp::Clear(gpu::Color::TRANSPARENT),
                        store: gpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
