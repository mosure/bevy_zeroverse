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
        MapMode,
    };
    use pollster::FutureExt;
    use wgpu::{
        Maintain,
        TexelCopyBufferInfo,
        TexelCopyBufferLayout,
    };

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
                    image.data = buffer_slice
                        .get_mapped_range()
                        .to_vec()
                        .into();
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

    #[derive(Component, Clone, Default, Resource, Deref, DerefMut)]
    pub struct ImageCopiers(pub Vec<ImageCopier>);

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
        _image_copier_bundles: Extract<Query<&ImageCopiers>>,
    ) {
        // TODO: merge bundles and singletons
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
                    (src_image.size.width as usize / block_dimensions.0 as usize)
                        * block_size as usize,
                );

                let texture_extent = Extent3d {
                    width: src_image.size.width,
                    height: src_image.size.height,
                    depth_or_array_layers: 1,
                };

                encoder.copy_texture_to_buffer(
                    src_image.texture.as_image_copy(),
                    TexelCopyBufferInfo {
                        buffer: &image_copier.buffer,
                        layout: TexelCopyBufferLayout {
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


pub mod prepass_copy {
    use std::sync::Arc;

    use bevy::prelude::*;
    use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
    use bevy::core_pipeline::prepass::ViewPrepassTextures;
    use bevy::ecs::query::QueryItem;
    use bevy::render::render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner};
    use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
    use bevy::render::{Extract, RenderApp};
    use bevy::render::render_resource::TextureFormat;
    use bevy::render::sync_world::RenderEntity;

    use bevy::render::render_resource::{
        Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d,
        MapMode,
    };
    use pollster::FutureExt;
    use wgpu::{
        Maintain,
        TexelCopyBufferInfo,
        TexelCopyBufferLayout,
    };

    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::render::RenderMode;

    pub fn receive_images(
        prepass_copiers: Query<&PrepassCopier>,
        mut images: ResMut<Assets<Image>>,
        render_device: Res<RenderDevice>,
    ) {
        for prepass_copier in prepass_copiers.iter() {
            if !prepass_copier.enabled() {
                continue;
            }

            // Derived from: https://sotrh.github.io/learn-wgpu/showcase/windowless/#a-triangle-without-a-window
            // We need to scope the mapping variables so that we can
            // unmap the buffer
            async {
                let buffer_slice = prepass_copier.buffer.slice(..);

                // NOTE: We have to create the mapping THEN device.poll() before await
                // the future. Otherwise the application will freeze.
                let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
                render_device.poll(Maintain::Wait);
                rx.receive().await.unwrap().unwrap();
                if let Some(image) = images.get_mut(&prepass_copier.dst_image) {
                    image.data = buffer_slice
                        .get_mapped_range()
                        .to_vec()
                        .into();
                }

                prepass_copier.buffer.unmap();
            }
            .block_on();
        }
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub struct PrepassCopyLabel;

    pub struct PrepassCopyPlugin;
    impl Plugin for PrepassCopyPlugin {
        fn build(&self, app: &mut App) {
            let render_app = app
                .add_systems(Update, receive_images)
                .sub_app_mut(RenderApp);

            render_app.add_systems(ExtractSchedule, prepass_copy_extract);

            render_app.add_render_graph_node::<ViewNodeRunner<PrepassCopyDriver>>(
                Core3d,
                PrepassCopyLabel,
            );
            render_app.add_render_graph_edge(
                Core3d,
                Node3d::EndMainPass,
                PrepassCopyLabel,
            );
        }
    }

    #[derive(Component, Clone, Default, Resource, Deref, DerefMut)]
    pub struct PrepassCopiers(pub Vec<PrepassCopier>);

    #[derive(Clone, Component)]
    pub struct PrepassCopier {
        buffer: Buffer,
        enabled: Arc<AtomicBool>,
        pub src_mode: RenderMode,
        pub dst_image: Handle<Image>,
    }

    impl PrepassCopier {
        pub fn new(
            src_mode: RenderMode,
            dst_image: Handle<Image>,
            size: Extent3d,
            texture_format: TextureFormat,
            render_device: &RenderDevice,
        ) -> PrepassCopier {
            let block_dimensions = texture_format.block_dimensions();
            let block_size = texture_format.block_copy_size(None).unwrap();

            let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                (size.width as usize / block_dimensions.0 as usize)
                    * block_size as usize,
            );
            let buffer_size = padded_bytes_per_row as u64 * size.height as u64;

            let cpu_buffer = render_device.create_buffer(&BufferDescriptor {
                label: "prepass_copier_cpu_buffer".into(),
                size: buffer_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            PrepassCopier {
                buffer: cpu_buffer,
                src_mode,
                dst_image,
                enabled: Arc::new(AtomicBool::new(true)),
            }
        }

        pub fn enabled(&self) -> bool {
            self.enabled.load(Ordering::Relaxed)
        }
    }

    pub fn prepass_copy_extract(
        mut commands: Commands,
        prepass_copier_bundles: Extract<Query<(
            RenderEntity,
            &PrepassCopiers,
        )>>,
    ) {
        for (
            entity,
            prepass_copiers,
        ) in prepass_copier_bundles.iter() {
            commands
                .entity(entity)
                .insert(prepass_copiers.clone());
        }
    }

    #[derive(Default)]
    pub struct PrepassCopyDriver;

    impl ViewNode for PrepassCopyDriver {
        type ViewQuery = (
            &'static PrepassCopiers,
            &'static ViewPrepassTextures,
        );

        fn run(
            &self,
            _graph: &mut RenderGraphContext,
            render_context: &mut RenderContext,
            (
                prepass_copiers,
                prepass_texture,
            ): QueryItem<
                Self::ViewQuery,
            >,
            world: &World,
        ) -> Result<(), NodeRunError> {
            let render_queue = world.get_resource::<RenderQueue>().unwrap();
            let mut encoder = render_context
                .render_device()
                .create_command_encoder(&CommandEncoderDescriptor::default());

            for prepass_copier in prepass_copiers.iter() {
                if !prepass_copier.enabled() {
                    continue;
                }

                let src_texture = match &prepass_copier.src_mode {
                    RenderMode::Depth => &prepass_texture.depth.as_ref().unwrap().texture,
                    RenderMode::Normal => &prepass_texture.normal.as_ref().unwrap().texture,
                    RenderMode::MotionVectors => &prepass_texture.motion_vectors.as_ref().unwrap().texture,
                    _ => panic!("unsupported prepass src_mode"),
                };

                let format = src_texture.texture.format();
                let size = src_texture.texture.size();

                let block_dimensions = format.block_dimensions();
                let block_size = format.block_copy_size(None).unwrap();

                let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                    (size.width as usize / block_dimensions.0 as usize)
                        * block_size as usize,
                );

                // TODO: image as a compute node target, single sample
                encoder.copy_texture_to_buffer(
                    src_texture.texture.as_image_copy(),
                    TexelCopyBufferInfo {
                        buffer: &prepass_copier.buffer,
                        layout: TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(
                                std::num::NonZeroU32::new(padded_bytes_per_row as u32)
                                    .unwrap()
                                    .into(),
                            ),
                            rows_per_image: None,
                        },
                    },
                    size,
                );
            }

            render_queue.submit(std::iter::once(encoder.finish()));

            Ok(())
        }
    }
}
