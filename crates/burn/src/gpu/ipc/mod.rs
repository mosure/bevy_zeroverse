//! Opt-in Linux process transport. Downstream owns children, retries and training
//! order. No pixels cross the socket; CUDA imports a registered Vulkan arena and
//! copies into owned Burn storage before acknowledging reuse.
use super::*;
use crate::generator::admission::{CaptureControl, CaptureRequest, Permit};
use bevy_zeroverse_capture::gpu::wire;
use burn::cubecl::wgpu::wgpu;
use serde::{Deserialize, Serialize};
use std::{
    os::fd::OwnedFd,
    os::unix::net::UnixStream,
    sync::{
        Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};
mod import;
pub(super) mod pool;
mod socket;
use socket::Socket;

#[derive(Serialize, Deserialize)]
enum Message {
    Hello {
        protocol: u32,
        session: wire::SessionId,
        policy: String,
        device_uuid: [u8; 16],
        allocation: wire::Allocation,
        generator: u32,
        core_version: String,
        adapter_version: String,
        provenance: serde_json::Value,
        adapter: serde_json::Value,
    },
    Imported,
    Capture {
        request: wire::RequestId,
        lookahead: Vec<u64>,
    },
    Ready {
        lease: wire::LeaseId,
        ranges: Vec<wire::ViewRange>,
        metadata: Box<bevy_zeroverse::sample::Sample>,
        timings: GpuCaptureTimings,
    },
    Release {
        lease: wire::LeaseId,
    },
    Rejected {
        lease: wire::LeaseId,
        rejection: bevy_zeroverse::sample::failure::SceneRejection,
    },
    Failed {
        request: wire::RequestId,
        error: String,
    },
    Stop,
}

mod server;
pub use server::GpuCaptureServer;

/// A connected, locally supervised renderer. The receiver does not spawn or kill it.
pub struct CudaLane {
    pub lane: u32,
    pub connection: UnixStream,
}
pub struct CudaCompletion {
    pub lane: u32,
    pub request: CaptureRequest,
    pub result: Result<GpuSample>,
    pub timings: Option<GpuCaptureTimings>,
}
struct Pending {
    request: CaptureRequest,
    permit: Permit,
}
#[derive(Default)]
struct Inbox {
    pending: Mutex<Option<Pending>>,
    wake: Condvar,
    closed: AtomicBool,
}
impl Inbox {
    fn wait(&self) -> bool {
        let mut pending = self.pending.lock().unwrap();
        while pending.is_none() && !self.closed.load(Ordering::Acquire) {
            pending = self.wake.wait(pending).unwrap();
        }
        !self.closed.load(Ordering::Acquire)
    }
    fn close(&self) {
        // Serialize the predicate change with wait, avoiding a lost wakeup.
        let _pending = self.pending.lock().unwrap();
        self.closed.store(true, Ordering::Release);
        self.wake.notify_all();
    }
}
struct Lane {
    socket: Mutex<Socket>,
    inbox: Arc<Inbox>,
    last_sequence: Option<usize>,
    plan: Vec<CaptureRequest>,
    worker: Option<thread::JoinHandle<()>>,
    registered_bytes: u64,
}

/// One independent ingress thread per lane. An idle/slow renderer cannot block
/// completion from another. All receive calls are nonblocking or deadline bounded.
/// Admission and training order remain controlled by the caller's input ledger.
pub struct CudaCaptureReceiver {
    lanes: HashMap<u32, Lane>,
    completions: mpsc::Receiver<CudaCompletion>,
    control: Arc<CaptureControl>,
    device: Device,
    spec: ResolvedCaptureSpec,
    runtime: CaptureRuntimeOptions,
    outgoing: mpsc::SyncSender<CudaCompletion>,
    outstanding: std::collections::HashSet<u32>,
    capacity: usize,
}
impl CudaCaptureReceiver {
    pub fn attach(
        device: Device,
        spec: ResolvedCaptureSpec,
        connections: Vec<CudaLane>,
        runtime: CaptureRuntimeOptions,
        control: Arc<CaptureControl>,
    ) -> Result<Self> {
        runtime.validate(&spec)?;
        ensure!(
            !connections.is_empty() && connections.len() <= 16,
            "CUDA receiver supports 1..=16 lanes"
        );
        control.start()?;
        let capacity = connections.len();
        let (outgoing, completions) = mpsc::sync_channel(capacity);
        let mut receiver = Self {
            lanes: HashMap::new(),
            completions,
            control,
            device,
            spec,
            runtime,
            outgoing,
            outstanding: Default::default(),
            capacity,
        };
        for connection in connections {
            receiver.connect_lane(connection)?;
        }
        Ok(receiver)
    }
    fn connect_lane(&mut self, connection: CudaLane) -> Result<()> {
        let device = &self.device;
        let spec = &self.spec;
        let runtime = &self.runtime;
        let outgoing = &self.outgoing;
        ensure!(
            !self.lanes.contains_key(&connection.lane),
            "duplicate CUDA lane"
        );
        let lane_id = connection.lane;
        let mut socket = Socket(connection.connection);
        socket.authenticate()?;
        socket
            .0
            .set_read_timeout(Some(runtime.initialization_timeout))?;
        socket.0.set_write_timeout(Some(runtime.capture_timeout))?;
        let (message, fds) = socket.receive()?;
        let (session, uuid, allocation) = match message {
            Message::Hello {
                protocol,
                session,
                policy,
                device_uuid,
                allocation,
                generator,
                core_version,
                adapter_version,
                provenance,
                ..
            } => {
                ensure!(
                    protocol == wire::PROTOCOL_VERSION
                        && policy == spec.fingerprint
                        && session.lane == lane_id
                        && generator == bevy_zeroverse_capture::GENERATOR_VERSION
                        && core_version == bevy_zeroverse::CRATE_VERSION
                        && adapter_version == env!("CARGO_PKG_VERSION")
                        && runtime
                            .expected_generator_source
                            .as_ref()
                            .is_none_or(
                                |sha| provenance["source_sha256"].as_str() == Some(sha.as_str())
                            ),
                    "CUDA handshake policy/version/lane mismatch"
                );
                (session, device_uuid, allocation)
            }
            _ => anyhow::bail!("missing CUDA producer handshake"),
        };
        allocation
            .validate(runtime.max_capture_bytes)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            allocation.bytes == spec.room_bytes * 2 && fds.len() == 2,
            "CUDA arena/FD count mismatch"
        );
        let registered_bytes = allocation.allocation_bytes;
        let previous = self.lanes.values().try_fold(0_u64, |n, lane| {
            n.checked_add(lane.registered_bytes)
                .context("import budget overflow")
        })?;
        ensure!(
            previous
                .checked_add(registered_bytes)
                .is_some_and(|n| n <= runtime.max_imported_bytes),
            "aggregate imported CUDA arena budget exceeded"
        );
        let mut importer = import::Importer::new(device.clone(), uuid, allocation, fds)?;
        socket.send(&Message::Imported, &[])?;
        socket.0.set_read_timeout(Some(runtime.capture_timeout))?;
        let writer = Socket(socket.0.try_clone()?);
        let inbox = Arc::new(Inbox::default());
        let worker_inbox = inbox.clone();
        let worker_outgoing = outgoing.clone();
        let spec = spec.clone();
        let timeout = runtime.capture_timeout;
        let worker = thread::Builder::new()
            .name(format!("zeroverse-cuda-ingress-{lane_id}"))
            .spawn(move || {
                let mut expected_generation = 1_u64;
                while worker_inbox.wait() {
                    let received = socket.receive();
                    let result = (|| -> Result<(wire::LeaseId, GpuSample, GpuCaptureTimings)> {
                        let (message, fds) = received?;
                        ensure!(fds.is_empty(), "unexpected ready FDs");
                        let (lease, ranges, metadata, mut timings) = match message {
                            Message::Ready {
                                lease,
                                ranges,
                                metadata,
                                timings,
                            } => (lease, ranges, metadata, timings),
                            Message::Rejected { lease, rejection } => {
                                ensure!(
                                    lease.session == session
                                        && lease.generation == expected_generation
                                        && rejection.seed == lease.request.seed,
                                    "stale/mismatched rejection"
                                );
                                let pending = worker_inbox.pending.lock().unwrap();
                                ensure!(
                                    pending.as_ref().is_some_and(|p| p.request.sequence as u64
                                        == lease.request.sequence
                                        && p.request.seed == lease.request.seed),
                                    "rejection request mismatch"
                                );
                                expected_generation = expected_generation
                                    .checked_add(1)
                                    .context("generation overflow")?;
                                return Err(
                                    bevy_zeroverse::sample::failure::CaptureError::Rejected {
                                        rejection,
                                        retired: true,
                                    }
                                    .into(),
                                );
                            }
                            Message::Failed { request, error } => {
                                let pending = worker_inbox.pending.lock().unwrap();
                                ensure!(
                                    pending.as_ref().is_some_and(|p| p.request.sequence as u64
                                        == request.sequence
                                        && p.request.seed == request.seed),
                                    "failure request mismatch"
                                );
                                anyhow::bail!(error)
                            }
                            _ => anyhow::bail!("unexpected CUDA producer message"),
                        };
                        ensure!(
                            lease.session == session && lease.generation == expected_generation,
                            "stale/repeated CUDA lease"
                        );
                        {
                            let pending = worker_inbox.pending.lock().unwrap();
                            ensure!(
                                pending.as_ref().is_some_and(|p| p.request.sequence as u64
                                    == lease.request.sequence
                                    && p.request.seed == lease.request.seed),
                                "unknown CUDA request"
                            );
                        }
                        ensure!(
                            metadata.indoor.as_ref().map(|m| m.seed) == Some(lease.request.seed)
                                && timings.seed == lease.request.seed,
                            "CUDA metadata seed mismatch"
                        );
                        Duration::try_from_secs_f64(timings.producer_seconds)
                            .context("invalid producer duration")?;
                        let (sample, import_timings) =
                            importer.copy(&spec, *metadata, &ranges, timeout)?;
                        timings.cuda_import = Some(import_timings);
                        socket.send(&Message::Release { lease }, &[])?;
                        expected_generation = expected_generation
                            .checked_add(1)
                            .context("generation overflow")?;
                        Ok((lease, sample, timings))
                    })();
                    let Some(pending) = worker_inbox.pending.lock().unwrap().take() else {
                        break;
                    };
                    let mut failed = result.is_err();
                    let (result, timings) = match result {
                        Ok((_, sample, timings)) => {
                            let mut permit = pending.permit;
                            permit
                                .captured(Duration::from_secs_f64(timings.producer_seconds), None);
                            permit.delivered();
                            (Ok(sample), Some(timings))
                        }
                        Err(error) => {
                            if let Some(bevy_zeroverse::sample::failure::CaptureError::Rejected {
                                rejection,
                                retired: true,
                            }) = error.downcast_ref()
                            {
                                failed = !pending
                                    .permit
                                    .rejected(rejection.clone(), Duration::ZERO)
                                    .unwrap_or(false);
                            } else {
                                pending.permit.failed(error.to_string(), Duration::ZERO);
                            }
                            (Err(error), None)
                        }
                    };
                    if failed {
                        // Mark retirement before publishing its completion: the
                        // caller may immediately replace or try to re-admit it.
                        worker_inbox.close();
                    }
                    if worker_outgoing
                        .send(CudaCompletion {
                            lane: lane_id,
                            request: pending.request,
                            result,
                            timings,
                        })
                        .is_err()
                        || failed
                    {
                        break;
                    }
                }
                worker_inbox.close();
            })?;
        self.lanes.insert(
            lane_id,
            Lane {
                socket: Mutex::new(writer),
                inbox,
                last_sequence: None,
                plan: Vec::new(),
                worker: Some(worker),
                registered_bytes,
            },
        );
        Ok(())
    }
    /// Replace one drained/failed lane without resetting other lanes or the
    /// cumulative request ledger. Consume its completion before replacement.
    /// Resolution/policy transitions require a new receiver after draining.
    pub fn replace_lane(&mut self, connection: CudaLane, timeout: Duration) -> Result<()> {
        let id = connection.lane;
        ensure!(
            !self.outstanding.contains(&id),
            "consume the lane completion before replacement"
        );
        let old = self.lanes.get_mut(&id).context("unknown CUDA lane")?;
        ensure!(
            old.inbox.pending.lock().unwrap().is_none(),
            "lane still has admitted work"
        );
        old.inbox.close();
        let _ = old
            .socket
            .lock()
            .unwrap()
            .0
            .shutdown(std::net::Shutdown::Both);
        let deadline = Instant::now() + timeout;
        while old.worker.as_ref().is_some_and(|w| !w.is_finished()) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        ensure!(
            old.worker
                .as_ref()
                .is_none_or(thread::JoinHandle::is_finished),
            "old CUDA imports have not retired; replacement refused"
        );
        if let Some(worker) = old.worker.take() {
            let _ = worker.join();
        }
        let last_sequence = old.last_sequence;
        let mut retired = self.lanes.remove(&id).unwrap();
        retired.registered_bytes = 0;
        if let Err(error) = self.connect_lane(connection) {
            // Preserve the identity/sequence ledger so a failed handshake can be
            // retried, without counting already retired imports against budget.
            self.lanes.insert(id, retired);
            return Err(error);
        }
        self.lanes.get_mut(&id).unwrap().last_sequence = last_sequence;
        Ok(())
    }
    pub fn plan_requests(&mut self, lane: u32, requests: &[CaptureRequest]) -> Result<()> {
        ensure!(
            requests.len() <= 5 && requests.windows(2).all(|w| w[0].sequence < w[1].sequence),
            "invalid bounded preparation plan"
        );
        self.lanes.get_mut(&lane).context("unknown CUDA lane")?.plan = requests.to_vec();
        Ok(())
    }
    /// Nonblocking admission. False means a hold, no credit, or a leased lane.
    pub fn try_admit(&mut self, lane: u32, request: CaptureRequest) -> Result<bool> {
        if self.outstanding.len() == self.capacity || self.outstanding.contains(&lane) {
            return Ok(false);
        }
        let lane_id = lane;
        let lane = self.lanes.get_mut(&lane).context("unknown CUDA lane")?;
        ensure!(
            lane.last_sequence.is_none_or(|s| request.sequence > s),
            "duplicate/stale CUDA request"
        );
        ensure!(
            !lane.inbox.closed.load(Ordering::Acquire),
            "CUDA lane retired; replace before admitting"
        );
        if lane.inbox.pending.lock().unwrap().is_some() {
            return Ok(false);
        }
        let Some(permit) = self.control.try_acquire(request)? else {
            return Ok(false);
        };
        let mut pending = lane.inbox.pending.lock().unwrap();
        let lookahead = lane
            .plan
            .iter()
            .position(|r| *r == request)
            .map(|i| lane.plan[i + 1..].iter().map(|r| r.seed).collect())
            .unwrap_or_default();
        let message = Message::Capture {
            request: wire::RequestId {
                sequence: request.sequence as u64,
                seed: request.seed,
            },
            lookahead,
        };
        let sent = lane.socket.lock().unwrap().send(&message, &[]);
        if let Err(error) = sent {
            drop(pending);
            lane.inbox.close();
            permit.failed(error.to_string(), Duration::ZERO);
            return Err(error);
        }
        *pending = Some(Pending { request, permit });
        lane.last_sequence = Some(request.sequence);
        self.outstanding.insert(lane_id);
        lane.inbox.wake.notify_one();
        Ok(true)
    }
    pub fn try_receive(&mut self) -> Result<Option<CudaCompletion>> {
        match self.completions.try_recv() {
            Ok(result) => {
                ensure!(
                    self.outstanding.remove(&result.lane),
                    "unknown/duplicate lane completion"
                );
                Ok(Some(result))
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    pub fn receive_timeout(&mut self, timeout: Duration) -> Result<Option<CudaCompletion>> {
        match self.completions.recv_timeout(timeout) {
            Ok(result) => {
                ensure!(
                    self.outstanding.remove(&result.lane),
                    "unknown/duplicate lane completion"
                );
                Ok(Some(result))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    /// Reserved imported bytes, including drained lanes until removal/replacement.
    pub fn registered_bytes(&self) -> u64 {
        self.lanes.values().map(|lane| lane.registered_bytes).sum()
    }
    pub fn outstanding_rooms(&self) -> usize {
        self.outstanding.len()
    }
    pub fn control(&self) -> &Arc<CaptureControl> {
        &self.control
    }
    pub fn shutdown(&mut self, timeout: Duration) -> bool {
        self.close();
        let deadline = Instant::now() + timeout;
        while self
            .lanes
            .values()
            .any(|lane| lane.worker.as_ref().is_some_and(|w| !w.is_finished()))
            && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(2));
        }
        if self.lanes.values().all(|lane| {
            lane.worker
                .as_ref()
                .is_none_or(thread::JoinHandle::is_finished)
        }) {
            for lane in self.lanes.values_mut() {
                if let Some(worker) = lane.worker.take() {
                    let _ = worker.join();
                }
            }
            true
        } else {
            false
        }
    }
    fn close(&self) {
        self.control.cancel();
        for lane in self.lanes.values() {
            lane.inbox.close();
            if let Ok(mut socket) = lane.socket.lock() {
                if lane.inbox.pending.lock().is_ok_and(|p| p.is_none()) {
                    let _ = socket.0.set_write_timeout(Some(Duration::from_millis(20)));
                    let _ = socket.send(&Message::Stop, &[]);
                }
                let _ = socket.0.shutdown(std::net::Shutdown::Both);
            }
        }
    }
}
impl Drop for CudaCaptureReceiver {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idle_ingress_wait_is_interruptible_without_lost_wakeup() {
        for _ in 0..16 {
            let inbox = Arc::new(Inbox::default());
            let waiting = inbox.clone();
            let (sent, received) = mpsc::channel();
            let worker = thread::spawn(move || sent.send(waiting.wait()).unwrap());
            inbox.close();
            assert!(!received.recv_timeout(Duration::from_secs(1)).unwrap());
            worker.join().unwrap();
        }
    }
    #[test]
    fn admitted_ingress_wakes_and_capture_read_has_a_deadline() {
        let control = Arc::new(CaptureControl::default());
        control.start().unwrap();
        control.grant_through(1);
        let inbox = Arc::new(Inbox::default());
        let waiting = inbox.clone();
        let (sent, received) = mpsc::channel();
        let worker = thread::spawn(move || sent.send(waiting.wait()).unwrap());
        let request = CaptureRequest {
            sequence: 4,
            seed: 9,
        };
        *inbox.pending.lock().unwrap() = Some(Pending {
            request,
            permit: control.try_acquire(request).unwrap().unwrap(),
        });
        inbox.wake.notify_one();
        assert!(received.recv_timeout(Duration::from_secs(1)).unwrap());
        worker.join().unwrap();
        inbox
            .pending
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .permit
            .delivered();
        let (_producer, receiver) = UnixStream::pair().unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_millis(20)))
            .unwrap();
        let start = Instant::now();
        assert!(Socket(receiver).receive().is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}

#[cfg(test)]
mod rejection_wire_tests {
    use super::*;
    #[test]
    fn placement_rejection_preserves_identity_and_allows_the_next_lease() {
        let (producer, consumer) = UnixStream::pair().unwrap();
        let mut producer = Socket(producer);
        let mut consumer = Socket(consumer);
        let session = wire::SessionId {
            nonce: [1; 16],
            lane: 3,
            incarnation: [2; 16],
        };
        let mut state = wire::LeaseState::new(session);
        let request = wire::RequestId {
            sequence: 17,
            seed: 123,
        };
        let lease = state.begin(request).unwrap();
        producer
            .send(
                &Message::Rejected {
                    lease,
                    rejection: bevy_zeroverse::sample::failure::SceneRejection {
                        seed: 123,
                        kind:
                            bevy_zeroverse::sample::failure::SceneRejectionKind::PlacementClearance,
                        reason: "floor furniture floats, sinks or intersects ceiling fixtures"
                            .into(),
                    },
                },
                &[],
            )
            .unwrap();
        let (message, fds) = consumer.receive().unwrap();
        assert!(fds.is_empty());
        let Message::Rejected {
            lease: received,
            rejection,
        } = message
        else {
            panic!("expected structured rejection")
        };
        assert_eq!(received, lease);
        assert_eq!(rejection.seed, request.seed);
        assert_eq!(
            rejection.kind,
            bevy_zeroverse::sample::failure::SceneRejectionKind::PlacementClearance
        );
        state.retire(received).unwrap();
        let next = state
            .begin(wire::RequestId {
                sequence: 18,
                seed: 124,
            })
            .unwrap();
        assert_eq!(next.generation, lease.generation + 1);
    }
}
