//! Bounded, explicit room requests. Control-plane receive never drives the GPU.
use super::*;
use crate::generator::admission::{CaptureControl, CaptureRequest};
use burn::data::dataset::{Dataset, DatasetError};
use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex, mpsc},
    thread,
};

#[derive(Default)]
struct Requests {
    queued: VecDeque<CaptureRequest>,
    planned: Vec<CaptureRequest>,
    stopped: bool,
}

/// Every accepted request has one completion, including capture failures.
/// Completion order is local to this lane; consumers retain their training order.
pub struct GpuCompletion {
    pub request: CaptureRequest,
    pub result: Result<GpuSample>,
    pub timings: Option<GpuCaptureTimings>,
}

/// Explicit-demand native session. Planning seeds permits CPU lookahead only;
/// submitting a request plus a CaptureControl credit permits GPU capture.
/// Tensor ownership is independent of session/renderer lifetime.
pub struct GpuCaptureSession {
    device: Device,
    queue: Arc<(Mutex<Requests>, Condvar)>,
    control: Arc<CaptureControl>,
    completions: mpsc::Receiver<GpuCompletion>,
    outstanding: Vec<CaptureRequest>,
    last_sequence: Option<usize>,
    capacity: usize,
    worker: Option<thread::JoinHandle<()>>,
}

pub struct GpuCaptureBuilder {
    spec: ResolvedCaptureSpec,
    runtime: CaptureRuntimeOptions,
    transport: Transport,
    control: Option<Arc<CaptureControl>>,
}
impl GpuCaptureBuilder {
    pub fn new(spec: ResolvedCaptureSpec) -> Self {
        Self {
            spec,
            runtime: Default::default(),
            transport: Transport::Wgpu,
            control: None,
        }
    }
    pub fn runtime(mut self, runtime: CaptureRuntimeOptions) -> Self {
        self.runtime = runtime;
        self
    }
    pub fn control(mut self, control: Arc<CaptureControl>) -> Self {
        self.control = Some(control);
        self
    }
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    pub fn cuda(mut self) -> Self {
        self.transport = Transport::Cuda;
        self
    }
    /// The CUDA receiver must use this device; UUID/context checks fail closed.
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    pub fn cuda_on(mut self, device: Device) -> Self {
        self.transport = Transport::CudaOn(device);
        self
    }
    pub fn build_session(self) -> Result<GpuCaptureSession> {
        GpuCaptureSession::build(self)
    }
    pub fn build(self) -> Result<GpuLiveDataset> {
        let next_seed = self.spec.scene().indoor_seed.unwrap_or_else(rand::random);
        let receive_timeout = self.runtime.capture_timeout;
        Ok(GpuLiveDataset {
            session: self.build_session()?,
            next_seed,
            pending: None,
            sequence: 0,
            last_capture_timings: None,
            receive_timeout,
        })
    }
}

impl GpuCaptureSession {
    fn build(builder: GpuCaptureBuilder) -> Result<Self> {
        builder.runtime.validate(&builder.spec)?;
        let capacity = builder.runtime.queue_capacity;
        let timeout = builder.runtime.initialization_timeout;
        let control = builder.control.unwrap_or_else(|| {
            let control = Arc::new(CaptureControl::default());
            control.grant_through(u64::MAX);
            control
        });
        control.start()?;
        let queue = Arc::new((Mutex::new(Requests::default()), Condvar::new()));
        let (outgoing, completions) = mpsc::sync_channel(capacity);
        let (ready, initialized) = mpsc::sync_channel(1);
        let worker_queue = queue.clone();
        let worker_control = control.clone();
        let worker = thread::Builder::new()
            .name("zeroverse-gpu-capture".into())
            .spawn(move || {
                let mut renderer =
                    match Renderer::configured(builder.spec, builder.runtime, builder.transport) {
                        Ok(renderer) => renderer,
                        Err(error) => {
                            let _ = ready.send(Err(error));
                            return;
                        }
                    };
                if ready.send(Ok(renderer.device().clone())).is_err() {
                    return;
                }
                loop {
                    let (request, seeds) = {
                        let (lock, wake) = &*worker_queue;
                        let mut queue = lock.lock().unwrap();
                        while queue.queued.is_empty() && !queue.stopped {
                            queue = wake.wait(queue).unwrap();
                        }
                        if queue.stopped {
                            break;
                        }
                        let request = queue.queued.pop_front().unwrap();
                        let seeds = queue
                            .planned
                            .iter()
                            .position(|r| *r == request)
                            .map(|i| queue.planned[i + 1..].iter().map(|r| r.seed).collect())
                            .unwrap_or_else(|| queue.queued.iter().map(|r| r.seed).collect());
                        (request, seeds)
                    };
                    let result = match worker_control.acquire(request) {
                        Ok(Some(mut permit)) => {
                            renderer
                                .app
                                .world_mut()
                                .resource_mut::<procedural_indoor::preparation::IndoorPrefetch>()
                                .explicit_seeds = Some(seeds);
                            let start = Instant::now();
                            // reset changes only the requested sequence cursor. The
                            // configuration-keyed lookahead retains matching jobs.
                            match renderer.capture(request.seed, true) {
                                Ok(sample) => {
                                    permit.captured(start.elapsed(), None);
                                    permit.delivered();
                                    Ok(sample)
                                }
                                Err(error) => {
                                    if let Some(
                                        bevy_zeroverse::sample::failure::CaptureError::Rejected {
                                            rejection,
                                            retired: true,
                                        },
                                    ) = error.downcast_ref()
                                    {
                                        let _ = permit.rejected(rejection.clone(), start.elapsed());
                                    } else {
                                        permit.failed(error.to_string(), start.elapsed());
                                        worker_control.fail(error.to_string());
                                    }
                                    Err(error)
                                }
                            }
                        }
                        Ok(None) => Err(anyhow::anyhow!("GPU capture cancelled")),
                        Err(error) => Err(error),
                    };
                    let timings = renderer.last_capture_timings.take();
                    if outgoing
                        .send(GpuCompletion {
                            request,
                            result,
                            timings,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;
        let device = match initialized.recv_timeout(timeout) {
            Ok(Ok(device)) => device,
            result => {
                control.cancel();
                queue.0.lock().unwrap().stopped = true;
                queue.1.notify_all();
                return Err(match result {
                    Ok(Err(e)) => e,
                    Err(e) => anyhow::anyhow!("GPU initialization did not complete: {e}"),
                    _ => unreachable!(),
                });
            }
        };
        Ok(Self {
            device,
            queue,
            control,
            completions,
            outstanding: Vec::new(),
            last_sequence: None,
            capacity,
            worker: Some(worker),
        })
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn outstanding_rooms(&self) -> usize {
        self.outstanding.len()
    }
    pub fn control(&self) -> &Arc<CaptureControl> {
        &self.control
    }
    /// Ordered exact seeds, including the current request. Maximum 4 future jobs.
    /// This does not admit rendering and never changes the requested scene policy.
    pub fn plan_requests(&self, requests: &[CaptureRequest]) -> Result<()> {
        ensure!(
            requests.len() <= 5,
            "preparation plans contain at most five requests"
        );
        ensure!(
            requests.windows(2).all(|w| w[0].sequence < w[1].sequence),
            "request sequences must increase"
        );
        self.queue.0.lock().unwrap().planned = requests.to_vec();
        Ok(())
    }
    /// False means bounded backpressure; no request/credit was consumed.
    pub fn try_submit(&mut self, request: CaptureRequest) -> Result<bool> {
        ensure!(!self.control.snapshot().cancelled, "GPU session cancelled");
        ensure!(
            self.last_sequence
                .is_none_or(|last| request.sequence > last),
            "duplicate/stale GPU request sequence"
        );
        if self.outstanding.len() == self.capacity {
            return Ok(false);
        }
        let mut queue = self.queue.0.lock().unwrap();
        ensure!(!queue.stopped, "GPU session stopped");
        queue.queued.push_back(request);
        self.outstanding.push(request);
        self.last_sequence = Some(request.sequence);
        self.queue.1.notify_one();
        Ok(true)
    }
    fn accept(&mut self, completion: GpuCompletion) -> Result<GpuCompletion> {
        let index = self
            .outstanding
            .iter()
            .position(|r| *r == completion.request)
            .context("unknown/duplicate GPU completion")?;
        self.outstanding.remove(index);
        if let Ok(sample) = &completion.result {
            ensure!(
                sample.metadata.indoor.as_ref().map(|m| m.seed) == Some(completion.request.seed),
                "GPU completion seed mismatch"
            );
        }
        Ok(completion)
    }
    pub fn try_receive(&mut self) -> Result<Option<GpuCompletion>> {
        match self.completions.try_recv() {
            Ok(completion) => self.accept(completion).map(Some),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => anyhow::bail!("GPU worker disconnected"),
        }
    }
    /// A timeout leaves the request outstanding; retry receive, never resubmit it.
    pub fn receive_timeout(&mut self, timeout: Duration) -> Result<Option<GpuCompletion>> {
        match self.completions.recv_timeout(timeout) {
            Ok(completion) => self.accept(completion).map(Some),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => anyhow::bail!("GPU worker disconnected"),
        }
    }
    pub fn cancel(&self) {
        self.control.cancel();
    }
    /// Bounded shutdown. False means a driver call is still running; only an
    /// external process supervisor can kill that renderer safely.
    pub fn shutdown(&mut self, timeout: Duration) -> bool {
        self.stop();
        let deadline = Instant::now() + timeout;
        while self.worker.as_ref().is_some_and(|w| !w.is_finished()) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        if self
            .worker
            .as_ref()
            .is_some_and(thread::JoinHandle::is_finished)
        {
            let _ = self.worker.take().unwrap().join();
            true
        } else {
            self.worker.is_none()
        }
    }
    fn stop(&self) {
        self.control.cancel();
        self.queue.0.lock().unwrap().stopped = true;
        self.queue.1.notify_all();
    }
}
impl Drop for GpuCaptureSession {
    fn drop(&mut self) {
        self.stop();
        if self
            .worker
            .as_ref()
            .is_some_and(thread::JoinHandle::is_finished)
        {
            let _ = self.worker.take().unwrap().join();
        }
        // Never block the training thread in a destructor on an unkillable GPU
        // driver call. The worker owns all in-flight leases until it finishes.
    }
}

/// Convenience consecutive stream, backed by the same explicit-demand session.
pub struct GpuLiveDataset {
    session: GpuCaptureSession,
    next_seed: u64,
    pending: Option<u64>,
    sequence: usize,
    last_capture_timings: Option<GpuCaptureTimings>,
    receive_timeout: Duration,
}
impl GpuLiveDataset {
    pub fn builder(spec: ResolvedCaptureSpec) -> GpuCaptureBuilder {
        GpuCaptureBuilder::new(spec)
    }
    pub fn indoor() -> Result<Self> {
        Self::new(Self::indoor_config())
    }
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    pub fn indoor_cuda() -> Result<Self> {
        Self::new_cuda(Self::indoor_config())
    }
    fn indoor_config() -> BevyZeroverseConfig {
        crate::generator::GenConfig::indoor("", 0).scene_config()
    }
    pub fn new(config: BevyZeroverseConfig) -> Result<Self> {
        Self::builder(CaptureSpec::from_scene(config).resolve()?).build()
    }
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    pub fn new_cuda(config: BevyZeroverseConfig) -> Result<Self> {
        Self::builder(CaptureSpec::from_scene(config).resolve()?)
            .cuda()
            .build()
    }
    pub fn device(&self) -> &Device {
        self.session.device()
    }
    pub fn last_capture_timings(&self) -> Option<&GpuCaptureTimings> {
        self.last_capture_timings.as_ref()
    }
    pub fn into_dataset(self, samples_per_epoch: usize) -> GpuDataset {
        GpuDataset {
            device: self.device().clone(),
            renderer: Mutex::new(self),
            samples_per_epoch,
        }
    }
    pub fn next_sample(&mut self) -> Result<GpuSample> {
        let seed = self.next_seed;
        let next = seed.checked_add(1).context("indoor seed overflow")?;
        if self.pending.is_none() {
            self.request(seed)?;
        }
        let result = self.receive();
        // A consumed terminal rejection must not replay the same physical room
        // on the next call. Timeouts retain the outstanding request and cursor.
        if self.pending.is_none() {
            self.next_seed = next;
        }
        let sample = result?;
        ensure!(
            sample.metadata.indoor.as_ref().map(|m| m.seed) == Some(seed),
            "GPU stream seed mismatch"
        );
        self.request(next)?;
        Ok(sample)
    }
    pub fn sample_seed(&mut self, seed: u64) -> Result<GpuSample> {
        if self.pending.is_some() {
            let _ = self.receive()?;
        }
        self.request(seed)?;
        self.receive()
    }
    fn request(&mut self, seed: u64) -> Result<()> {
        ensure!(self.pending.is_none(), "GPU request already pending");
        let request = CaptureRequest {
            sequence: self.sequence,
            seed,
        };
        let future: Vec<_> = (0..4)
            .filter_map(|i| {
                Some(CaptureRequest {
                    sequence: self.sequence.checked_add(i)?,
                    seed: seed.checked_add(i as u64)?,
                })
            })
            .collect();
        self.session.plan_requests(&future)?;
        ensure!(self.session.try_submit(request)?, "GPU request queue full");
        self.sequence = self
            .sequence
            .checked_add(1)
            .context("GPU sequence overflow")?;
        self.pending = Some(seed);
        Ok(())
    }
    fn receive(&mut self) -> Result<GpuSample> {
        let _ = self.pending.context("no GPU request pending")?;
        let completion = self
            .session
            .receive_timeout(self.receive_timeout)?
            .context("GPU capture deadline exceeded; request remains outstanding")?;
        self.pending = None;
        self.last_capture_timings = completion.timings;
        completion.result
    }
}

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
        assert!(index < self.len(), "GPU dataset index out of bounds");
        self.renderer
            .lock()
            .map_err(|_| DatasetError::new(std::io::Error::other("GPU renderer lock poisoned")))?
            .next_sample()
            .map_err(|e| DatasetError::new(std::io::Error::other(e.to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_zeroverse::sample::failure::{CaptureError, SceneRejection, SceneRejectionKind};

    // Only channels and a device identifier: never initialize a GPU or renderer.
    fn pending_stream() -> (GpuLiveDataset, mpsc::Sender<GpuCompletion>) {
        let request = CaptureRequest {
            sequence: 0,
            seed: 42,
        };
        let (sender, completions) = mpsc::channel();
        let session = GpuCaptureSession {
            device: Device::default(),
            queue: Arc::new((Mutex::new(Requests::default()), Condvar::new())),
            control: Arc::new(CaptureControl::default()),
            completions,
            outstanding: vec![request],
            last_sequence: Some(0),
            capacity: 2,
            worker: None,
        };
        (
            GpuLiveDataset {
                session,
                next_seed: 42,
                pending: Some(42),
                sequence: 1,
                last_capture_timings: None,
                receive_timeout: Duration::ZERO,
            },
            sender,
        )
    }

    #[test]
    fn stream_rejection_advances_seed_but_timeout_preserves_outstanding_request() {
        let (mut stream, sender) = pending_stream();
        assert!(stream.next_sample().is_err());
        assert_eq!(stream.next_seed, 42);
        assert_eq!(stream.pending, Some(42));
        assert_eq!(stream.session.outstanding.len(), 1);
        sender
            .send(GpuCompletion {
                request: CaptureRequest {
                    sequence: 0,
                    seed: 42,
                },
                result: Err(CaptureError::Rejected {
                    rejection: SceneRejection {
                        seed: 42,
                        kind: SceneRejectionKind::CameraPlacement,
                        reason: "fixture camera rejection".into(),
                    },
                    retired: true,
                }
                .into()),
                timings: None,
            })
            .unwrap();
        let error = stream
            .next_sample()
            .err()
            .expect("rejection returned to caller");
        assert!(matches!(
            error.downcast_ref::<CaptureError>(),
            Some(CaptureError::Rejected { retired: true, .. })
        ));
        assert_eq!(stream.next_seed, 43);
        assert_eq!(stream.pending, None);
        assert!(stream.session.outstanding.is_empty());
        assert!(stream.session.queue.0.lock().unwrap().queued.is_empty());
    }
}
