use wgpu::{self as gpu, TextureFormat};

/// A single cached float attachment is needed for surfaces that expect
/// encoded pixels. Only its image and bind group change when the window resizes.
pub(super) struct Transfer {
    pipeline: gpu::RenderPipeline,
    pub view: bevy::render::render_resource::TextureView,
    group: gpu::BindGroup,
}

impl Transfer {
    pub fn new(device: &gpu::Device, size: [u32; 2], format: TextureFormat, entry: &str) -> Self {
        let shader = device.create_shader_module(gpu::ShaderModuleDescriptor {
            label: Some("Studio HDR display transfer"),
            source: gpu::ShaderSource::Wgsl(include_str!("transfer.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&gpu::RenderPipelineDescriptor {
            label: Some("Studio HDR display transfer"),
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
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let (view, group) = Self::target(device, size, &pipeline);
        Self {
            pipeline,
            view,
            group,
        }
    }

    fn target(
        device: &gpu::Device,
        [width, height]: [u32; 2],
        pipeline: &gpu::RenderPipeline,
    ) -> (bevy::render::render_resource::TextureView, gpu::BindGroup) {
        let texture = device.create_texture(&gpu::TextureDescriptor {
            label: Some("Studio linear HDR composite"),
            size: gpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: gpu::TextureUsages::RENDER_ATTACHMENT
                | gpu::TextureUsages::TEXTURE_BINDING
                | gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let group = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("Studio HDR composite"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[gpu::BindGroupEntry {
                binding: 0,
                resource: gpu::BindingResource::TextureView(&view),
            }],
        });
        (view.into(), group)
    }

    pub fn resize(&mut self, device: &gpu::Device, size: [u32; 2]) {
        (self.view, self.group) = Self::target(device, size, &self.pipeline);
    }

    pub fn encode(&self, encoder: &mut gpu::CommandEncoder, output: &gpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&gpu::RenderPassDescriptor {
            label: Some("Encode Studio HDR window"),
            color_attachments: &[Some(gpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: gpu::Operations {
                    load: gpu::LoadOp::Clear(gpu::Color::BLACK),
                    store: gpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a wgpu adapter; runs on software Vulkan"]
    fn hdr_surface_transfers_preserve_color_and_highlights_on_gpu() {
        let instance = gpu::Instance::new(gpu::InstanceDescriptor {
            backends: gpu::Backends::from_env().unwrap_or(gpu::Backends::PRIMARY),
            ..gpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        for (format, entry) in [
            (TextureFormat::Rgba16Float, "extended_srgb"),
            (TextureFormat::Rgb10a2Unorm, "bt2100_pq"),
        ] {
            let transfer = Transfer::new(&device, [3, 2], format, entry);
            let size = gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            };
            let output = device.create_texture(&gpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: gpu::TextureDimension::D2,
                format,
                usage: gpu::TextureUsages::RENDER_ATTACHMENT | gpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let read = device.create_buffer(&gpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: gpu::BufferUsages::COPY_DST | gpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            drop(encoder.begin_render_pass(&gpu::RenderPassDescriptor {
                color_attachments: &[Some(gpu::RenderPassColorAttachment {
                    view: &transfer.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: gpu::Operations {
                        load: gpu::LoadOp::Clear(gpu::Color {
                            r: 0.18,
                            g: 1.,
                            b: 4.,
                            a: 1.,
                        }),
                        store: gpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            }));
            transfer.encode(&mut encoder, &output.create_view(&Default::default()));
            encoder.copy_texture_to_buffer(
                output.as_image_copy(),
                gpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: gpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                size,
            );
            queue.submit([encoder.finish()]);
            super::super::screenshot::verify_readback(&device, &queue, transfer.view.texture());
            let (tx, rx) = std::sync::mpsc::channel();
            read.slice(..)
                .map_async(gpu::MapMode::Read, move |result| tx.send(result).unwrap());
            device.poll(gpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = read.slice(..).get_mapped_range().unwrap();
            if format == TextureFormat::Rgba16Float {
                let rgb: Vec<_> = bytes[..6]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| half::f16::from_le_bytes(*v).to_f32())
                    .collect();
                for (measured, expected) in rgb.iter().zip([0.46136, 1., 1.82480]) {
                    assert!((measured - expected).abs() < 0.002, "{rgb:?}");
                }
            } else {
                let packed = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                let actual = [packed & 1023, (packed >> 10) & 1023, (packed >> 20) & 1023];
                // Independent ST 2084 reference evaluated at the three output channel luminances.
                for (i, coefficients) in [
                    [0.627404, 0.329283, 0.043313],
                    [0.069097, 0.919540, 0.011363],
                    [0.016391, 0.088013, 0.895596],
                ]
                .iter()
                .enumerate()
                {
                    let nits: f64 =
                        80. * (0.18 * coefficients[0] + coefficients[1] + 4. * coefficients[2]);
                    let p = (nits / 10000.).powf(2610. / 16384.);
                    let expected = ((3424. / 4096. + 2413. / 128. * p) / (1. + 2392. / 128. * p))
                        .powf(2523. / 32.);
                    assert!(
                        (f64::from(actual[i]) / 1023. - expected).abs() < 0.002,
                        "{actual:?}"
                    );
                }
            }
        }
    }
}
