// TODO: bevy/multi_threaded support - see: https://github.com/bevyengine/bevy/pull/13006/files

pub mod channels {
    use std::sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    };

    use once_cell::sync::OnceCell;

    use crate::sample::Sample;

    /// An indexed indoor request regenerates exactly this seed before capture.
    /// A default request advances the existing scene stream.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct AppFrameRequest {
        pub indoor_seed: Option<u64>,
    }

    pub static APP_FRAME_RECEIVER: OnceCell<Arc<Mutex<Receiver<AppFrameRequest>>>> =
        OnceCell::new();
    pub static APP_FRAME_SENDER: OnceCell<Sender<AppFrameRequest>> = OnceCell::new();

    pub static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<Sample>>>> = OnceCell::new();
    pub static SAMPLE_SENDER: OnceCell<Sender<Sample>> = OnceCell::new();
    static CAPTURE_ERROR: Mutex<Option<String>> = Mutex::new(None);

    pub fn report_capture_failure(message: String) {
        *CAPTURE_ERROR.lock().expect("capture error lock poisoned") = Some(message);
    }

    pub fn take_capture_failure() -> Option<String> {
        CAPTURE_ERROR
            .lock()
            .expect("capture error lock poisoned")
            .take()
    }

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

    pub fn app_frame_sender() -> &'static Sender<AppFrameRequest> {
        APP_FRAME_SENDER
            .get()
            .expect("app frame sender not initialized")
    }

    pub fn app_frame_receiver() -> Option<&'static Arc<Mutex<Receiver<AppFrameRequest>>>> {
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
/// Remove WebGPU's row alignment before exposing tightly packed image pixels.
fn packed_readback_rows(
    mapped: &[u8],
    row_bytes: usize,
    padded_row_bytes: usize,
    rows: usize,
) -> Vec<u8> {
    let mut packed = Vec::with_capacity(row_bytes * rows);
    for row in 0..rows {
        let offset = row * padded_row_bytes;
        packed.extend_from_slice(&mapped[offset..offset + row_bytes]);
    }
    packed
}

#[cfg(test)]
mod readback_tests {
    #[test]
    fn gpu_padding_is_not_exposed_as_pixels() {
        let mapped = [1, 2, 3, 99, 4, 5, 6, 99];
        assert_eq!(
            super::packed_readback_rows(&mapped, 3, 4, 2),
            [1, 2, 3, 4, 5, 6]
        );
        assert_eq!(super::packed_readback_rows(&mapped, 4, 4, 2), mapped);
    }
}

pub mod image_copy {
    use bevy::render::diagnostic::RecordDiagnostics;
    use bevy::{
        prelude::*,
        render::{
            render_asset::RenderAssets,
            render_resource::{
                Buffer, BufferDescriptor, BufferUsages, Extent3d, MapMode, TextureFormat,
            },
            renderer::{
                render_system, RenderContext, RenderDevice, RenderGraph, RenderGraphSystems,
            },
            texture::GpuImage,
            Extract, Render, RenderApp, RenderSystems,
        },
    };
    use std::sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    };
    use wgpu::{PollType, TexelCopyBufferInfo, TexelCopyBufferLayout};

    #[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
    pub struct ImageCopyLabel;
    pub struct ImageCopyPlugin;

    /// A capture is published only after every attachment from the same request completes.
    #[derive(Debug)]
    pub struct CapturedImages {
        pub request_id: u64,
        pub planes: Vec<Vec<u8>>,
    }

    #[derive(Default)]
    struct CaptureState {
        requested: AtomicU64,
        busy: AtomicBool,
        submitted: AtomicU64,
        completed: Mutex<Option<CapturedImages>>,
        failure: Mutex<Option<String>>,
    }

    /// One bounded staging allocation per attachment. Nothing is copied during warmup
    /// or while the sampler is idle. Mapping never waits for the GPU on the render thread.
    #[derive(Clone, Component)]
    pub struct ImageCopier {
        sources: Vec<Handle<Image>>,
        buffers: Vec<Buffer>,
        state: Arc<CaptureState>,
        row_bytes: usize,
        padded_row_bytes: usize,
        rows: usize,
    }

    impl ImageCopier {
        /// Compatibility constructor for a single color attachment. The CPU image is no
        /// longer uploaded back to the GPU; consumers take the completed packet directly.
        pub fn new(
            src: Handle<Image>,
            _dst: Handle<Image>,
            size: Extent3d,
            format: TextureFormat,
            device: &RenderDevice,
        ) -> Self {
            Self::for_targets(vec![src], size, format, device)
        }
        pub fn for_targets(
            sources: Vec<Handle<Image>>,
            size: Extent3d,
            format: TextureFormat,
            device: &RenderDevice,
        ) -> Self {
            let block = format.block_dimensions();
            let row_bytes = size.width as usize / block.0 as usize
                * format
                    .block_copy_size(None)
                    .expect("copyable capture format") as usize;
            let padded_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);
            let rows = size.height as usize / block.1 as usize;
            let buffers = sources
                .iter()
                .map(|_| {
                    device.create_buffer(&BufferDescriptor {
                        label: Some("zeroverse_async_readback"),
                        size: (padded_row_bytes * rows) as u64,
                        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            Self {
                sources,
                buffers,
                state: Arc::default(),
                row_bytes,
                padded_row_bytes,
                rows,
            }
        }
        pub fn request(&self, id: u64) {
            assert_ne!(id, 0, "zero is reserved for idle readback");
            if id == self.requested_id() {
                return;
            }
            if self.state.busy.swap(true, Ordering::AcqRel) {
                *self.state.failure.lock().unwrap() =
                    Some("capture requested before consuming the preceding packet".into());
                return;
            }
            self.state.requested.store(id, Ordering::Release);
        }
        pub fn requested_id(&self) -> u64 {
            self.state.requested.load(Ordering::Acquire)
        }
        /// Last request whose copies have been encoded into the render graph.
        pub fn submitted_id(&self) -> u64 {
            self.state.submitted.load(Ordering::Acquire)
        }
        pub fn ready(&self, id: u64) -> bool {
            self.state
                .completed
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|p| p.request_id == id)
        }
        pub fn take(&self, id: u64) -> Option<CapturedImages> {
            let mut packet = self.state.completed.lock().unwrap();
            if packet.as_ref().is_some_and(|p| p.request_id == id) {
                self.state.busy.store(false, Ordering::Release);
                packet.take()
            } else {
                None
            }
        }
        pub fn failure(&self) -> Option<String> {
            self.state.failure.lock().unwrap().clone()
        }
        pub fn attachment_count(&self) -> usize {
            self.sources.len()
        }
        pub fn staging_bytes(&self) -> usize {
            self.sources.len() * self.padded_row_bytes * self.rows
        }
    }

    #[derive(Resource, Default)]
    struct ImageCopiers(Vec<ExtractedCopier>);
    struct ExtractedCopier {
        copier: ImageCopier,
        #[cfg(not(target_arch = "wasm32"))]
        ground_truth: Option<crate::render::ground_truth::GroundTruthCamera>,
    }

    #[derive(Resource, Clone, Default)]
    pub struct CapturePipelineReadiness {
        ready: Arc<AtomicBool>,
        failure: Arc<Mutex<Option<String>>>,
        pipeline_count: Arc<AtomicU64>,
    }
    impl CapturePipelineReadiness {
        pub fn pipeline_count(&self) -> u64 {
            self.pipeline_count.load(Ordering::Acquire)
        }
        pub fn ready(&self) -> bool {
            self.ready.load(Ordering::Acquire)
        }
        pub fn failure(&self) -> Option<String> {
            self.failure.lock().unwrap().clone()
        }
    }
    fn update_capture_readiness(
        cache: Res<bevy::render::render_resource::PipelineCache>,
        readiness: Res<CapturePipelineReadiness>,
    ) {
        use bevy::render::render_resource::CachedPipelineState;
        use bevy::shader::ShaderCacheError;
        let failure = cache
            .pipelines()
            .find_map(|pipeline| match &pipeline.state {
                CachedPipelineState::Err(
                    ShaderCacheError::ShaderNotLoaded(_)
                    | ShaderCacheError::ShaderImportNotYetAvailable,
                ) => None,
                CachedPipelineState::Err(error) => Some(error.to_string()),
                _ => None,
            });
        readiness.ready.store(
            failure.is_none() && cache.waiting_pipelines().next().is_none(),
            Ordering::Release,
        );
        *readiness.failure.lock().unwrap() = failure;
        readiness
            .pipeline_count
            .store(cache.pipelines().count() as u64, Ordering::Release);
    }

    // The render graph appends copies to its own encoder. Only map after render_system
    // has submitted that encoder; submitting a separate encoder here can read stale GT.
    struct MapJob {
        copier: ImageCopier,
        request_id: u64,
    }
    #[derive(Resource, Default)]
    struct PendingMaps(Mutex<Vec<MapJob>>);
    fn map_submitted(
        pending: Res<PendingMaps>,
        device: Res<RenderDevice>,
        copiers: Res<ImageCopiers>,
    ) {
        for job in pending.0.lock().unwrap().drain(..) {
            let count = job.copier.buffers.len();
            let planes = Arc::new(Mutex::new((
                0usize,
                (0..count).map(|_| Vec::new()).collect::<Vec<_>>(),
            )));
            for (index, buffer) in job.copier.buffers.iter().enumerate() {
                let buffer = buffer.clone();
                let callback_buffer = buffer.clone();
                let state = job.copier.state.clone();
                let planes = planes.clone();
                let (row, padded, rows) = (
                    job.copier.row_bytes,
                    job.copier.padded_row_bytes,
                    job.copier.rows,
                );
                let id = job.request_id;
                buffer.slice(..).map_async(MapMode::Read, move |result| {
                    if let Err(error) = result {
                        *state.failure.lock().unwrap() =
                            Some(format!("capture {id} GPU map failed: {error}"));
                        return;
                    }
                    let view = callback_buffer.slice(..).get_mapped_range();
                    let data = super::packed_readback_rows(&view, row, padded, rows);
                    drop(view);
                    callback_buffer.unmap();
                    let mut completed = planes.lock().unwrap();
                    completed.1[index] = data;
                    completed.0 += 1;
                    if completed.0 == count {
                        *state.completed.lock().unwrap() = Some(CapturedImages {
                            request_id: id,
                            planes: std::mem::take(&mut completed.1),
                        });
                    }
                });
            }
        }
        if let Err(error) = device.poll(PollType::Poll) {
            for extracted in &copiers.0 {
                *extracted.copier.state.failure.lock().unwrap() =
                    Some(format!("capture device poll failed: {error}"));
            }
        }
    }

    impl Plugin for ImageCopyPlugin {
        fn build(&self, app: &mut App) {
            let readiness = CapturePipelineReadiness::default();
            app.insert_resource(readiness.clone());
            #[cfg(not(target_arch = "wasm32"))]
            app.add_systems(Last, stamp_ground_truth);
            let render_app = app.sub_app_mut(RenderApp);
            render_app
                .insert_resource(readiness)
                .init_resource::<PendingMaps>();
            render_app.add_systems(
                Render,
                update_capture_readiness.in_set(RenderSystems::Cleanup),
            );
            render_app.add_systems(
                Render,
                map_submitted
                    .after(render_system)
                    .in_set(RenderSystems::Render),
            );
            render_app.add_systems(ExtractSchedule, image_copy_extract);
            let copy = copy_images
                .after(bevy::core_pipeline::schedule::camera_driver)
                .in_set(ImageCopyLabel)
                .in_set(RenderGraphSystems::Render);
            #[cfg(not(target_arch = "wasm32"))]
            let copy = copy.after(crate::render::ground_truth::GroundTruthLabel);
            render_app.add_systems(RenderGraph, copy);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn stamp_ground_truth(
        mut cameras: Query<(
            &ImageCopier,
            &Projection,
            &mut crate::render::ground_truth::GroundTruthCamera,
        )>,
    ) {
        for (copier, projection, mut gt) in &mut cameras {
            gt.frame_id = copier.requested_id();
            if let Projection::Perspective(p) = projection {
                gt.near = p.near;
                gt.far = p.far;
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn image_copy_extract(
        mut commands: Commands,
        cameras: Extract<
            Query<(
                &ImageCopier,
                Option<&crate::render::ground_truth::GroundTruthCamera>,
            )>,
        >,
    ) {
        commands.insert_resource(ImageCopiers(
            cameras
                .iter()
                .map(|(copier, gt)| ExtractedCopier {
                    copier: copier.clone(),
                    ground_truth: gt.cloned(),
                })
                .collect(),
        ));
    }
    #[cfg(target_arch = "wasm32")]
    fn image_copy_extract(mut commands: Commands, cameras: Extract<Query<&ImageCopier>>) {
        commands.insert_resource(ImageCopiers(
            cameras
                .iter()
                .map(|copier| ExtractedCopier {
                    copier: copier.clone(),
                })
                .collect(),
        ));
    }

    fn copy_images(world: &World, mut context: RenderContext) {
        let copiers = world.resource::<ImageCopiers>();
        let images = world.resource::<RenderAssets<GpuImage>>();
        let pending = world.resource::<PendingMaps>();
        let diagnostics = context.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let mut copy_span = None;
        for extracted in &copiers.0 {
            let copier = &extracted.copier;
            let id = copier.requested_id();
            if id == 0 || id == copier.state.submitted.load(Ordering::Acquire) {
                continue;
            }
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(gt) = &extracted.ground_truth {
                if let Some(error) = gt.failure() {
                    *copier.state.failure.lock().unwrap() = Some(error);
                }
                if gt.rendered_frame() != Some(id) {
                    continue;
                }
            }
            if copier
                .sources
                .iter()
                .any(|handle| images.get(handle).is_none())
            {
                continue;
            }
            if copy_span.is_none() {
                copy_span =
                    Some(diagnostics.time_span(context.command_encoder(), "dataset_readback_copy"));
            }
            for (source, buffer) in copier.sources.iter().zip(&copier.buffers) {
                let image = images.get(source).unwrap();
                context.command_encoder().copy_texture_to_buffer(
                    image.texture.as_image_copy(),
                    TexelCopyBufferInfo {
                        buffer,
                        layout: TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(copier.padded_row_bytes as u32),
                            rows_per_image: None,
                        },
                    },
                    image.texture_descriptor.size,
                );
            }
            copier.state.submitted.store(id, Ordering::Release);
            pending.0.lock().unwrap().push(MapJob {
                copier: copier.clone(),
                request_id: id,
            });
        }
        if let Some(span) = copy_span {
            span.end(context.command_encoder());
        }
    }
}

pub mod prepass_copy {
    use std::sync::Arc;

    use bevy::core_pipeline::prepass::ViewPrepassTextures;
    use bevy::core_pipeline::{schedule::Core3d, Core3dSystems};
    use bevy::prelude::*;
    use bevy::render::render_resource::TextureFormat;
    use bevy::render::renderer::{RenderContext, RenderDevice, ViewQuery};
    use bevy::render::sync_world::RenderEntity;
    use bevy::render::{Extract, Render, RenderApp, RenderSystems};

    use bevy::render::render_resource::{
        Buffer, BufferDescriptor, BufferUsages, Extent3d, MapMode,
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
                let _ = render_device.poll(PollType::wait_indefinitely());
                rx.receive().await.unwrap().unwrap();
                if let Some(mut image) = images.get_mut(&prepass_copier.dst_image) {
                    let format = image.texture_descriptor.format;
                    let blocks = format.block_dimensions();
                    let row_bytes = image.width() as usize / blocks.0 as usize
                        * format.block_copy_size(None).unwrap() as usize;
                    image.data = Some(super::packed_readback_rows(
                        &buffer_slice.get_mapped_range(),
                        row_bytes,
                        RenderDevice::align_copy_bytes_per_row(row_bytes),
                        image.height() as usize / blocks.1 as usize,
                    ));
                }

                prepass_copier.buffer.unmap();
            }
            .block_on();

            prepass_copier.finish_read();
        }
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
    pub struct PrepassCopyLabel;

    pub struct PrepassCopyPlugin;
    impl Plugin for PrepassCopyPlugin {
        fn build(&self, app: &mut App) {
            let render_app = app.sub_app_mut(RenderApp);
            render_app.add_systems(Render, receive_images.in_set(RenderSystems::Cleanup));

            render_app.add_systems(ExtractSchedule, prepass_copy_extract);

            render_app.add_systems(
                Core3d,
                copy_prepass
                    .after(Core3dSystems::MainPass)
                    .before(Core3dSystems::PostProcess)
                    .in_set(PrepassCopyLabel),
            );
        }
    }

    #[derive(Component, Clone, Default, Deref, DerefMut)]
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

    fn copy_prepass(
        view: ViewQuery<(&PrepassCopiers, &ViewPrepassTextures)>,
        mut context: RenderContext,
    ) {
        let (prepass_copiers, prepass_texture) = view.into_inner();
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
            context.command_encoder().copy_texture_to_buffer(
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
    }
}
