//! One persistent renderer, one bounded future sample. Neither scheduling nor
//! tensor storage copies require downstream tuning or an additional GPU device.
use super::*;
use burn::data::dataset::{Dataset, DatasetError};
use std::{
    sync::{Mutex, mpsc},
    thread,
};

enum Request {
    Capture { seed: u64, indexed: bool },
    Shutdown,
}

/// Full-quality indoor JIT capture on the training device. Construction and
/// rendering overlap training automatically, with at most one future sample.
/// Retained tensors remain owned by their consumer, across subsequent captures.
pub struct GpuLiveDataset {
    device: Device,
    next_seed: u64,
    pending: Option<u64>,
    requests: mpsc::SyncSender<Request>,
    samples: mpsc::Receiver<Result<(GpuSample, GpuCaptureTimings)>>,
    worker: Option<thread::JoinHandle<()>>,
    last_capture_timings: Option<GpuCaptureTimings>,
}

impl GpuLiveDataset {
    /// Three 512x512 views, one timestep, full native lighting and annotations.
    pub fn indoor() -> Result<Self> {
        Self::new(Self::indoor_config())
    }

    /// The same full-quality indoor preset as [`Self::indoor`], delivered to
    /// CUDA through external GPU memory. Requires Linux and `cuda_interop`.
    /// See [`Self::new_cuda`] for the copy and synchronization contract.
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    pub fn indoor_cuda() -> Result<Self> {
        Self::new_cuda(Self::indoor_config())
    }

    fn indoor_config() -> BevyZeroverseConfig {
        BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            num_cameras: 3,
            width: 512.,
            height: 512.,
            playback_steps: 1,
            ..Default::default()
        }
    }

    pub fn new(config: BevyZeroverseConfig) -> Result<Self> {
        Self::with_transport(config, Transport::Wgpu)
    }

    /// Experimental Linux Vulkan/CUDA capture on the same physical GPU.
    /// Uses external device memory, never a host image round trip. One additional
    /// GPU copy and a producer-side completion fence make returned Burn tensors
    /// safe on any consumer thread/stream. Requires `cuda_interop`; no patches.
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    pub fn new_cuda(config: BevyZeroverseConfig) -> Result<Self> {
        Self::with_transport(config, Transport::Cuda)
    }

    fn with_transport(mut config: BevyZeroverseConfig, transport: Transport) -> Result<Self> {
        let next_seed = *config.indoor_seed.get_or_insert_with(rand::random);
        let (requests, incoming) = mpsc::sync_channel(1);
        let (outgoing, samples) = mpsc::sync_channel(1);
        let (ready, initialized) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("zeroverse-gpu-capture".into())
            .spawn(move || {
                // App is created and destroyed on its own thread. No native
                // windows or winit event loop are initialized here. Motion
                // models remain lazy and load only when the config requests them.
                let mut renderer = match Renderer::with_transport(config, transport) {
                    Ok(renderer) => renderer,
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                if ready.send(Ok(renderer.device().clone())).is_err() {
                    return;
                }
                while let Ok(Request::Capture { seed, indexed }) = incoming.recv() {
                    let reset = indexed || renderer.queued_seed != Some(seed);
                    let sample = renderer.capture(seed, reset).map(|sample| {
                        (
                            sample,
                            renderer
                                .last_capture_timings
                                .take()
                                .expect("completed GPU capture timings"),
                        )
                    });
                    if sample.is_err() {
                        renderer.queued_seed = None;
                    }
                    if outgoing.send(sample).is_err() {
                        break;
                    }
                }
            })?;
        let device = initialized
            .recv()
            .context("GPU renderer initialization failed")??;
        Ok(Self {
            device,
            next_seed,
            pending: None,
            requests,
            samples,
            worker: Some(worker),
            last_capture_timings: None,
        })
    }

    /// Construct the Burn model and optimizer here. WGPU shares Bevy's device
    /// and queue; the opt-in CUDA device matches Bevy's physical GPU by UUID.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Timings for the last successfully received sample, never a queued future
    /// room. These CPU wall intervals add no GPU fence/readback and are kept
    /// outside deterministic scene metadata. Measure completed training steps
    /// separately; a WGPU submission interval is not GPU completion time.
    pub fn last_capture_timings(&self) -> Option<&GpuCaptureTimings> {
        self.last_capture_timings.as_ref()
    }

    /// Adapt the live stream to Burn's data loader. Indices bound each epoch;
    /// they do not select a cached room. Every successful get owns a new sample.
    pub fn into_dataset(self, samples_per_epoch: usize) -> GpuDataset {
        GpuDataset {
            device: self.device.clone(),
            renderer: Mutex::new(self),
            samples_per_epoch,
        }
    }

    /// Consume the next consecutive seed and prepare one future room while the
    /// model trains. WGPU returns after submission; CUDA fences its transfer
    /// stream before returning. Neither transport maps image data to the host.
    pub fn next_sample(&mut self) -> Result<GpuSample> {
        let seed = self.next_seed;
        let next = seed.checked_add(1).context("indoor seed overflow")?;
        if self.pending.is_none() {
            self.request(seed, false)?;
        }
        let sample = self.receive()?;
        ensure!(
            sample.metadata.indoor.as_ref().map(|m| m.seed) == Some(seed),
            "GPU stream seed mismatch"
        );
        self.next_seed = next;
        self.request(next, false)?;
        Ok(sample)
    }

    /// Capture a requested seed without advancing the consecutive stream's
    /// cursor. Any already prepared lookahead is drained first; no stale room
    /// can be mistaken for the indexed result.
    pub fn sample_seed(&mut self, seed: u64) -> Result<GpuSample> {
        if self.pending.is_some() {
            let _ = self.receive();
        }
        self.request(seed, true)?;
        self.receive()
    }

    fn request(&mut self, seed: u64, indexed: bool) -> Result<()> {
        ensure!(
            self.pending.is_none(),
            "GPU capture request is already pending"
        );
        self.requests
            .send(Request::Capture { seed, indexed })
            .context("GPU renderer stopped")?;
        self.pending = Some(seed);
        Ok(())
    }

    fn receive(&mut self) -> Result<GpuSample> {
        self.last_capture_timings = None;
        let seed = self.pending.take().context("no GPU capture requested")?;
        let (sample, timings) = self
            .samples
            .recv()
            .context("GPU renderer stopped during capture")??;
        ensure!(
            sample.metadata.indoor.as_ref().map(|m| m.seed) == Some(seed),
            "GPU capture seed mismatch"
        );
        ensure!(timings.seed == seed, "GPU capture timing seed mismatch");
        self.last_capture_timings = Some(timings);
        Ok(sample)
    }
}

/// Thread-safe live dataset adapter with one automatically scheduled renderer.
pub struct GpuDataset {
    device: Device,
    renderer: Mutex<GpuLiveDataset>,
    samples_per_epoch: usize,
}
impl GpuDataset {
    pub fn device(&self) -> &Device {
        &self.device
    }
}
impl Dataset<GpuSample> for GpuDataset {
    fn len(&self) -> usize {
        self.samples_per_epoch
    }
    fn get(&self, index: usize) -> Result<GpuSample, DatasetError> {
        assert!(index < self.len(), "GPU live dataset index out of bounds");
        let mut renderer = self
            .renderer
            .lock()
            .map_err(|_| DatasetError::new(std::io::Error::other("GPU renderer lock poisoned")))?;
        renderer
            .next_sample()
            .map_err(|e| DatasetError::new(std::io::Error::other(e.to_string())))
    }
}

impl Drop for GpuLiveDataset {
    fn drop(&mut self) {
        let _ = self.requests.send(Request::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
