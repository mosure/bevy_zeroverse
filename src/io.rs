// TODO: bevy/multi_threaded support - see: https://github.com/bevyengine/bevy/pull/13006/files

pub mod channels {
    use std::sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    };

    use once_cell::sync::OnceCell;

    use crate::sample::Sample;

    pub static APP_FRAME_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
    pub static APP_FRAME_SENDER: OnceCell<Sender<()>> = OnceCell::new();

    pub static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<Sample>>>> = OnceCell::new();
    pub static SAMPLE_SENDER: OnceCell<Sender<Sample>> = OnceCell::new();

    pub fn channels_initialized() -> bool {
        APP_FRAME_RECEIVER.get().is_some()
    }

    pub fn init_channels() {
        if channels_initialized() {
            return;
        }

        let (app_sender, app_receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(app_receiver));
        let _ = APP_FRAME_RECEIVER.set(receiver);
        let _ = APP_FRAME_SENDER.set(app_sender);

        let (sample_sender, sample_receiver) = mpsc::channel();
        let sample_receiver = Arc::new(Mutex::new(sample_receiver));
        let _ = SAMPLE_RECEIVER.set(sample_receiver);
        let _ = SAMPLE_SENDER.set(sample_sender);
    }

    pub fn app_frame_sender() -> &'static Sender<()> {
        APP_FRAME_SENDER
            .get()
            .expect("app frame sender not initialized")
    }

    pub fn app_frame_receiver() -> Option<&'static Arc<Mutex<Receiver<()>>>> {
        APP_FRAME_RECEIVER.get()
    }

    pub fn sample_sender() -> &'static Sender<Sample> {
        SAMPLE_SENDER.get().expect("sample sender not initialized")
    }

    pub fn sample_receiver() -> Option<&'static Arc<Mutex<Receiver<Sample>>>> {
        SAMPLE_RECEIVER.get()
    }
}

/// Derived from: https://github.com/bevyengine/bevy/pull/5550
pub mod image_copy {
    use std::sync::{Arc, Mutex};

    use bevy::prelude::*;
    use bevy::render::render_asset::RenderAssets;
    use bevy::render::render_graph::{
        self, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
    };
    use bevy::render::render_resource::TextureFormat;
    use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
    use bevy::render::texture::GpuImage;
    use bevy::render::{Extract, Render, RenderApp, RenderSystems};

    use bevy::render::render_resource::{
        Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
    };
    use pollster::FutureExt;
    use wgpu::{PollType, TexelCopyBufferInfo, TexelCopyBufferLayout};

    use std::sync::atomic::{AtomicBool, Ordering};

    pub fn receive_images(
        image_copiers: Res<ImageCopiers>,
        images: Option<ResMut<Assets<Image>>>,
    ) {
        let mut images = images;
        for image_copier in image_copiers.iter() {
            if !image_copier.enabled() {
                continue;
            }
            if let Some(images) = images.as_mut() {
                if let Ok(cpu_data) = image_copier.cpu_data.lock() {
                    if cpu_data.is_empty() {
                        continue;
                    }
                    if let Some(image) = images.get_mut(&image_copier.dst_image) {
                        image.data = cpu_data.clone().into();
                    }
                }
            }
        }
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub struct ImageCopyLabel;

    pub struct ImageCopyPlugin;
    impl Plugin for ImageCopyPlugin {
        fn build(&self, app: &mut App) {
            let render_app = app.sub_app_mut(RenderApp);
            render_app.add_systems(Render, receive_images.in_set(RenderSystems::Cleanup));

            render_app.add_systems(ExtractSchedule, image_copy_extract);

            let image_copy_node = ImageCopyDriver::from_world(render_app.world_mut());

            let mut graph = render_app
                .world_mut()
                .get_resource_mut::<RenderGraph>()
                .unwrap();

            graph.add_node(ImageCopyLabel, image_copy_node);
            graph.add_node_edge(bevy::render::graph::CameraDriverLabel, ImageCopyLabel);
        }
    }

    #[derive(Component, Clone, Default, Resource, Deref, DerefMut)]
    pub struct ImageCopiers(pub Vec<ImageCopier>);

    #[derive(Clone, Component)]
    pub struct ImageCopier {
        buffer: Buffer,
        enabled: Arc<AtomicBool>,
        mapped: Arc<AtomicBool>,
        pub cpu_data: Arc<Mutex<Vec<u8>>>,
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
                (size.width as usize / block_dimensions.0 as usize) * block_size as usize,
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
                mapped: Arc::new(AtomicBool::new(false)),
                cpu_data: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub fn enabled(&self) -> bool {
            self.enabled.load(Ordering::Relaxed)
        }

        pub fn try_begin_read(&self) -> bool {
            self.mapped
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        }

        pub fn finish_read(&self) {
            self.mapped.store(false, Ordering::Release);
        }

        pub fn is_mapped(&self) -> bool {
            self.mapped.load(Ordering::Acquire)
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
                if image_copier.is_mapped() {
                    continue;
                }
                if !image_copier.try_begin_read() {
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

                // Map the buffer immediately so main-world consumers can read the latest frame.
                async {
                    let buffer_slice = image_copier.buffer.slice(..);
                    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                    buffer_slice.map_async(MapMode::Read, move |result| {
                        tx.send(result).unwrap();
                    });
                    let _ = render_context.render_device().poll(PollType::Wait);
                    rx.receive().await.unwrap().unwrap();
                    let mapped = buffer_slice.get_mapped_range().to_vec();
                    if let Ok(mut cpu_data) = image_copier.cpu_data.lock() {
                        *cpu_data = mapped;
                    }
                    image_copier.buffer.unmap();
                }
                .block_on();

                image_copier.finish_read();
            }

            Ok(())
        }
    }
}

pub mod prepass_copy {
    use std::sync::Arc;

    use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
    use bevy::core_pipeline::prepass::ViewPrepassTextures;
    use bevy::ecs::query::QueryItem;
    use bevy::prelude::*;
    use bevy::render::render_graph::{
        NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
    };
    use bevy::render::render_resource::TextureFormat;
    use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
    use bevy::render::sync_world::RenderEntity;
    use bevy::render::{Extract, Render, RenderApp, RenderSystems};

    use bevy::render::render_resource::{
        Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
    };
    use pollster::FutureExt;
    use wgpu::{PollType, TexelCopyBufferInfo, TexelCopyBufferLayout};

    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::render::RenderMode;

    pub fn receive_images(
        prepass_copiers: Query<&PrepassCopier>,
        images: Option<ResMut<Assets<Image>>>,
        render_device: Res<RenderDevice>,
    ) {
        let Some(mut images) = images else {
            return;
        };
        for prepass_copier in prepass_copiers.iter() {
            if !prepass_copier.enabled() {
                continue;
            }
            if !prepass_copier.try_begin_read() {
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
                let _ = render_device.poll(PollType::Wait);
                rx.receive().await.unwrap().unwrap();
                if let Some(image) = images.get_mut(&prepass_copier.dst_image) {
                    image.data = buffer_slice.get_mapped_range().to_vec().into();
                }

                prepass_copier.buffer.unmap();
            }
            .block_on();

            prepass_copier.finish_read();
        }
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub struct PrepassCopyLabel;

    pub struct PrepassCopyPlugin;
    impl Plugin for PrepassCopyPlugin {
        fn build(&self, app: &mut App) {
            let render_app = app.sub_app_mut(RenderApp);
            render_app.add_systems(Render, receive_images.in_set(RenderSystems::Cleanup));

            render_app.add_systems(ExtractSchedule, prepass_copy_extract);

            render_app.add_render_graph_node::<ViewNodeRunner<PrepassCopyDriver>>(
                Core3d,
                PrepassCopyLabel,
            );
            render_app.add_render_graph_edge(Core3d, Node3d::EndMainPass, PrepassCopyLabel);
        }
    }

    #[derive(Component, Clone, Default, Resource, Deref, DerefMut)]
    pub struct PrepassCopiers(pub Vec<PrepassCopier>);

    #[derive(Clone, Component)]
    pub struct PrepassCopier {
        buffer: Buffer,
        enabled: Arc<AtomicBool>,
        mapped: Arc<AtomicBool>,
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
                (size.width as usize / block_dimensions.0 as usize) * block_size as usize,
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
                mapped: Arc::new(AtomicBool::new(false)),
            }
        }

        pub fn enabled(&self) -> bool {
            self.enabled.load(Ordering::Relaxed)
        }

        pub fn try_begin_read(&self) -> bool {
            self.mapped
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        }

        pub fn finish_read(&self) {
            self.mapped.store(false, Ordering::Release);
        }

        pub fn is_mapped(&self) -> bool {
            self.mapped.load(Ordering::Acquire)
        }
    }

    pub fn prepass_copy_extract(
        mut commands: Commands,
        prepass_copier_bundles: Extract<Query<(RenderEntity, &PrepassCopiers)>>,
    ) {
        for (entity, prepass_copiers) in prepass_copier_bundles.iter() {
            commands.entity(entity).insert(prepass_copiers.clone());
        }
    }

    #[derive(Default)]
    pub struct PrepassCopyDriver;

    impl ViewNode for PrepassCopyDriver {
        type ViewQuery = (&'static PrepassCopiers, &'static ViewPrepassTextures);

        fn run(
            &self,
            _graph: &mut RenderGraphContext,
            render_context: &mut RenderContext,
            (prepass_copiers, prepass_texture): QueryItem<Self::ViewQuery>,
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
                if prepass_copier.is_mapped() {
                    continue;
                }

                let src_texture = match &prepass_copier.src_mode {
                    RenderMode::Depth => &prepass_texture.depth.as_ref().unwrap().texture,
                    RenderMode::Normal => &prepass_texture.normal.as_ref().unwrap().texture,
                    RenderMode::MotionVectors => {
                        &prepass_texture.motion_vectors.as_ref().unwrap().texture
                    }
                    _ => panic!("unsupported prepass src_mode"),
                };

                let format = src_texture.texture.format();
                let size = src_texture.texture.size();

                let block_dimensions = format.block_dimensions();
                let block_size = format.block_copy_size(None).unwrap();

                let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                    (size.width as usize / block_dimensions.0 as usize) * block_size as usize,
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
