// TODO: bevy/multi_threaded support - see: https://github.com/bevyengine/bevy/pull/13006/files

/// Derived from: https://github.com/bevyengine/bevy/pull/5550
pub mod image_copy {
    use std::sync::Arc;

    use bevy::prelude::*;
    use bevy::render::render_asset::RenderAssets;
    use bevy::render::render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel};
    use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
    use bevy::render::{Extract, RenderApp};
    use bevy::render::texture::GpuImage;
    use bevy::render::render_resource::TextureFormat;

    use bevy::render::render_resource::{
        Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d,
        ImageCopyBuffer, ImageDataLayout, MapMode,
    };
    use pollster::FutureExt;
    use wgpu::Maintain;

    use std::sync::atomic::{AtomicBool, Ordering};

    pub fn receive_images(
        image_copiers: Query<&ImageCopier>,
        mut images: ResMut<Assets<Image>>,
        render_device: Res<RenderDevice>,
    ) {
        for image_copier in image_copiers.iter() {
            if !image_copier.enabled() {
                continue;
            }

            // Derived from: https://sotrh.github.io/learn-wgpu/showcase/windowless/#a-triangle-without-a-window
            // We need to scope the mapping variables so that we can
            // unmap the buffer
            async {
                let buffer_slice = image_copier.buffer.slice(..);

                // NOTE: We have to create the mapping THEN device.poll() before await
                // the future. Otherwise the application will freeze.
                let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
                render_device.poll(Maintain::Wait);
                rx.receive().await.unwrap().unwrap();
                if let Some(image) = images.get_mut(&image_copier.dst_image) {
                    image.data = buffer_slice.get_mapped_range().to_vec();
                }

                image_copier.buffer.unmap();
            }
            .block_on();
        }
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub struct ImageCopyLabel;

    pub struct ImageCopyPlugin;
    impl Plugin for ImageCopyPlugin {
        fn build(&self, app: &mut App) {
            let render_app = app
                .add_systems(Update, receive_images)
                .sub_app_mut(RenderApp);

            render_app.add_systems(ExtractSchedule, image_copy_extract);

            let image_copy_node = ImageCopyDriver::from_world(render_app.world_mut());

            let mut graph = render_app.world_mut().get_resource_mut::<RenderGraph>().unwrap();

            graph.add_node(ImageCopyLabel, image_copy_node);
            graph.add_node_edge(ImageCopyLabel, bevy::render::graph::CameraDriverLabel);
        }
    }

    #[derive(Clone, Default, Resource, Deref, DerefMut)]
    pub struct ImageCopiers(pub Vec<ImageCopier>);

    // TODO: struct GpuImageCopier to avoid optional buffer
    #[derive(Clone, Component)]
    pub struct ImageCopier {
        buffer: Buffer,
        enabled: Arc<AtomicBool>,
        pub src_image: Handle<Image>,
        pub dst_image: Handle<Image>,
    }

    impl ImageCopier {
        pub fn new(
            src_image: Handle<Image>,
            dst_image: Handle<Image>,
            size: Extent3d,
            texture_format: TextureFormat,
            render_device: &RenderDevice,
        ) -> ImageCopier {
            let block_dimensions = texture_format.block_dimensions();
            let block_size = texture_format.block_copy_size(None).unwrap();

            let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                (size.width as usize / block_dimensions.0 as usize)
                    * block_size as usize,
            );
            let buffer_size = padded_bytes_per_row as u64 * size.height as u64;

            let cpu_buffer = render_device.create_buffer(&BufferDescriptor {
                label: "image_copier_cpu_buffer".into(),
                size: buffer_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            ImageCopier {
                buffer: cpu_buffer,
                src_image,
                dst_image,
                enabled: Arc::new(AtomicBool::new(true)),
            }
        }

        pub fn enabled(&self) -> bool {
            self.enabled.load(Ordering::Relaxed)
        }
    }

    pub fn image_copy_extract(
        mut commands: Commands,
        image_copiers: Extract<Query<&ImageCopier>>,
    ) {
        commands.insert_resource(ImageCopiers(
            image_copiers.iter().cloned().collect::<Vec<ImageCopier>>(),
        ));
    }

    #[derive(Default)]
    pub struct ImageCopyDriver;

    impl render_graph::Node for ImageCopyDriver {
        fn run(
            &self,
            _graph: &mut RenderGraphContext,
            render_context: &mut RenderContext,
            world: &World,
        ) -> Result<(), NodeRunError> {
            let image_copiers = world.get_resource::<ImageCopiers>().unwrap();
            let gpu_images = world.get_resource::<RenderAssets<GpuImage>>().unwrap();

            for image_copier in image_copiers.iter() {
                if !image_copier.enabled() {
                    continue;
                }

                let src_image = gpu_images.get(&image_copier.src_image).unwrap();

                let mut encoder = render_context
                    .render_device()
                    .create_command_encoder(&CommandEncoderDescriptor::default());

                let block_dimensions = src_image.texture_format.block_dimensions();
                let block_size = src_image.texture_format.block_copy_size(None).unwrap();

                let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                    (src_image.size.x as usize / block_dimensions.0 as usize)
                        * block_size as usize,
                );

                let texture_extent = Extent3d {
                    width: src_image.size.x,
                    height: src_image.size.y,
                    depth_or_array_layers: 1,
                };

                encoder.copy_texture_to_buffer(
                    src_image.texture.as_image_copy(),
                    ImageCopyBuffer {
                        buffer: &image_copier.buffer,
                        layout: ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(
                                std::num::NonZeroU32::new(padded_bytes_per_row as u32)
                                    .unwrap()
                                    .into(),
                            ),
                            rows_per_image: None,
                        },
                    },
                    texture_extent,
                );

                let render_queue = world.get_resource::<RenderQueue>().unwrap();
                render_queue.submit(std::iter::once(encoder.finish()));
            }

            Ok(())
        }
    }
}
