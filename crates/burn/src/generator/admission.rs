//! Optional room-level backpressure. The controller owns inventory policy;
//! Zeroverse owns capture validity, bounded preparation and atomic export.
mod rejection;
use bevy_zeroverse::sample::failure::SceneRejection;
pub use rejection::{PublicationReceipt, RejectionPolicy};
use serde::Serialize;
use std::{
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CaptureRequest {
    /// Absolute attempted capture index, including `sample_offset`; distinct from archive index after a rejection.
    pub sequence: usize,
    /// Exact indoor seed; all views/timesteps belong to this one physical room.
    pub seed: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CaptureStage {
    WaitingForDemand,
    Capturing,
    Exporting,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum CaptureOutcome {
    Published,
    /// GPU-owned data delivered without implying durable archive publication.
    Delivered,
    Rejected {
        rejection: SceneRejection,
        continued: bool,
    },
    Cancelled,
    Failed(String),
}
#[derive(Clone, Debug, Serialize)]
pub enum CaptureEvent {
    Stage {
        request: CaptureRequest,
        stage: CaptureStage,
    },
    Completed {
        request: CaptureRequest,
        admitted: bool,
        publication: Option<PublicationReceipt>,
        outcome: CaptureOutcome,
        demand_wait_seconds: f64,
        /// Capture including readiness/render/readback, excluding admission/export.
        capture_seconds: f64,
        export_seconds: f64,
        /// Preparation service/wall times and cumulative prefetch counters.
        telemetry: Option<Box<bevy_zeroverse::io::channels::CaptureTelemetry>>,
    },
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct AdmissionSnapshot {
    pub credits: u64,
    pub admitted: u64,
    pub completed: u64,
    /// Rejected, fully retired attempts whose unpublished reservation was returned.
    pub returned: u64,
    pub held: bool,
    pub cancelled: bool,
    pub waiting: Option<CaptureRequest>,
}
#[derive(Default)]
struct State {
    snapshot: AdmissionSnapshot,
    used: bool,
    failure: Option<String>,
    consecutive_rejections: u64,
}
type Observer = dyn Fn(CaptureEvent) + Send + Sync;

/// Cumulative room credits, with interruptible hold/resume/cancel. One control
/// belongs to one generator run; credits count rooms, not views or timesteps.
/// Default is zero credits. Callbacks must be short, nonblocking and not panic.
/// No callback runs under a capture, admission or archive-writer mutex.
/// Cancellation wakes admission immediately; an already admitted room finishes
/// under its normal watchdog and is exported before shutdown (credits cannot be
/// revoked). No new idle GPU work is started by controlled generation.
#[derive(Default)]
pub struct CaptureControl {
    state: Mutex<State>,
    wake: Condvar,
    observer: Option<Arc<Observer>>,
    rejection_policy: RejectionPolicy,
}
impl CaptureControl {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_observer(observer: impl Fn(CaptureEvent) + Send + Sync + 'static) -> Self {
        Self {
            observer: Some(Arc::new(observer)),
            ..Self::default()
        }
    }
    /// Select bounded, typed recovery before this control is started.
    pub fn with_rejection_policy(mut self, policy: RejectionPolicy) -> anyhow::Result<Self> {
        policy.validate()?;
        anyhow::ensure!(
            !self.state.get_mut().unwrap().used,
            "configure rejection policy before starting"
        );
        self.rejection_policy = policy;
        Ok(self)
    }
    pub(crate) fn recovers_rejections(&self) -> bool {
        self.rejection_policy.enabled()
    }
    /// Idempotent cumulative grants. A stale grant cannot revoke credits.
    pub fn grant_through(&self, credits: u64) {
        let mut state = self.state.lock().unwrap();
        state.snapshot.credits = state.snapshot.credits.max(credits);
        self.wake.notify_all();
    }
    pub fn hold(&self, held: bool) {
        self.state.lock().unwrap().snapshot.held = held;
        self.wake.notify_all();
    }
    pub fn cancel(&self) {
        self.state.lock().unwrap().snapshot.cancelled = true;
        self.wake.notify_all();
    }
    pub fn snapshot(&self) -> AdmissionSnapshot {
        self.state.lock().unwrap().snapshot.clone()
    }
    pub(crate) fn start(&self) -> anyhow::Result<()> {
        let mut state = self.state.lock().unwrap();
        anyhow::ensure!(!state.used, "CaptureControl belongs to one generator run");
        state.used = true;
        Ok(())
    }
    fn emit(&self, event: CaptureEvent) {
        if let Some(observer) = &self.observer {
            observer(event);
        }
    }
    pub(crate) fn fail(&self, message: String) {
        let mut state = self.state.lock().unwrap();
        state.failure.get_or_insert(message);
        state.snapshot.cancelled = true;
        self.wake.notify_all();
    }
    #[cfg(any(test, all(feature = "cuda_ipc", target_os = "linux")))]
    /// Nonblocking variant used by multiplexed GPU receivers. No credit is
    /// consumed on a hold or empty budget; callbacks run outside the mutex.
    pub(crate) fn try_acquire(
        self: &Arc<Self>,
        request: CaptureRequest,
    ) -> anyhow::Result<Option<Permit>> {
        let mut state = self.state.lock().unwrap();
        anyhow::ensure!(!state.snapshot.cancelled, "capture admission cancelled");
        if state.snapshot.held
            || state.snapshot.admitted - state.snapshot.returned >= state.snapshot.credits
        {
            return Ok(None);
        }
        state.snapshot.admitted += 1;
        drop(state);
        let permit = Permit {
            control: self.clone(),
            request,
            admitted: true,
            wait: Duration::ZERO,
            capture: Duration::ZERO,
            export: Duration::ZERO,
            telemetry: None,
            publication: None,
            completed: false,
        };
        self.emit(CaptureEvent::Stage {
            request,
            stage: CaptureStage::Capturing,
        });
        Ok(Some(permit))
    }
    pub(crate) fn acquire(
        self: &Arc<Self>,
        request: CaptureRequest,
    ) -> anyhow::Result<Option<Permit>> {
        let started = Instant::now();
        self.emit(CaptureEvent::Stage {
            request,
            stage: CaptureStage::WaitingForDemand,
        });
        let mut state = self.state.lock().unwrap();
        state.snapshot.waiting = Some(request);
        while !state.snapshot.cancelled
            && (state.snapshot.held
                || state.snapshot.admitted - state.snapshot.returned >= state.snapshot.credits)
        {
            state = self.wake.wait(state).unwrap();
        }
        state.snapshot.waiting = None;
        let failure = state.failure.clone();
        let admitted = !state.snapshot.cancelled;
        if admitted {
            state.snapshot.admitted += 1;
        }
        drop(state);
        let mut permit = Permit {
            control: self.clone(),
            request,
            admitted,
            wait: started.elapsed(),
            capture: Duration::ZERO,
            export: Duration::ZERO,
            telemetry: None,
            publication: None,
            completed: false,
        };
        if !admitted {
            permit.complete(failure.as_ref().map_or(CaptureOutcome::Cancelled, |e| {
                CaptureOutcome::Failed(e.clone())
            }));
            return match failure {
                Some(e) => Err(anyhow::anyhow!(e)),
                None => Ok(None),
            };
        }
        self.emit(CaptureEvent::Stage {
            request,
            stage: CaptureStage::Capturing,
        });
        Ok(Some(permit))
    }
}
/// Ownership travels with the room through the existing bounded writer channel.
/// Every path (validation error, disconnected writer, panic, cancellation) drops
/// or completes exactly one token, including partial batches.
pub(crate) struct Permit {
    control: Arc<CaptureControl>,
    request: CaptureRequest,
    admitted: bool,
    wait: Duration,
    capture: Duration,
    export: Duration,
    telemetry: Option<bevy_zeroverse::io::channels::CaptureTelemetry>,
    publication: Option<PublicationReceipt>,
    completed: bool,
}
impl Permit {
    pub(crate) fn captured(
        &mut self,
        elapsed: Duration,
        telemetry: Option<bevy_zeroverse::io::channels::CaptureTelemetry>,
    ) {
        self.capture = elapsed;
        self.control.state.lock().unwrap().consecutive_rejections = 0;
        self.telemetry = telemetry.filter(|value| value.seed == Some(self.request.seed));
    }
    pub(crate) fn exporting(&self) {
        self.control.emit(CaptureEvent::Stage {
            request: self.request,
            stage: CaptureStage::Exporting,
        });
    }
    pub(crate) fn failed(mut self, message: String, elapsed: Duration) {
        self.capture = elapsed;
        self.complete(CaptureOutcome::Failed(message));
    }
    pub(crate) fn rejected(
        mut self,
        rejection: SceneRejection,
        elapsed: Duration,
    ) -> anyhow::Result<bool> {
        self.capture = elapsed;
        let (allowed, cancelled) = {
            let mut state = self.control.state.lock().unwrap();
            state.snapshot.returned += 1;
            state.consecutive_rejections += 1;
            let allowed = self.control.rejection_policy.permits(
                rejection.kind,
                state.snapshot.returned,
                state.consecutive_rejections,
            );
            self.control.wake.notify_all();
            (allowed, state.snapshot.cancelled)
        };
        let continued = allowed && !cancelled;
        let reason = format!(
            "scene rejection policy exhausted or category not allowed for seed {}: {}",
            rejection.seed, rejection.reason
        );
        self.complete(CaptureOutcome::Rejected {
            rejection,
            continued,
        });
        if !allowed {
            self.control.fail(reason.clone());
            anyhow::bail!(reason);
        }
        Ok(continued)
    }
    pub(crate) fn finish(
        mut self,
        elapsed: Duration,
        error: Option<&str>,
        publication: Option<PublicationReceipt>,
    ) {
        if error.is_none() {
            self.publication = publication;
        }
        self.export = elapsed;
        self.complete(error.map_or(CaptureOutcome::Published, |e| {
            CaptureOutcome::Failed(e.into())
        }));
    }
    #[cfg(any(test, all(feature = "gpu_tensor", not(target_arch = "wasm32"))))]
    pub(crate) fn delivered(mut self) {
        self.complete(CaptureOutcome::Delivered);
    }
    fn complete(&mut self, outcome: CaptureOutcome) {
        if self.completed {
            return;
        }
        self.completed = true;
        self.control.state.lock().unwrap().snapshot.completed += 1;
        self.control.emit(CaptureEvent::Completed {
            request: self.request,
            admitted: self.admitted,
            publication: self.publication.take(),
            outcome,
            demand_wait_seconds: self.wait.as_secs_f64(),
            capture_seconds: self.capture.as_secs_f64(),
            export_seconds: self.export.as_secs_f64(),
            telemetry: self.telemetry.take().map(Box::new),
        });
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.complete(CaptureOutcome::Failed(
            "room transaction ended before atomic publication".into(),
        ));
    }
}

/// Also wakes a producer waiting for demand if the writer unwinds.
pub(crate) struct WriterGuard(pub Option<Arc<CaptureControl>>);
impl Drop for WriterGuard {
    fn drop(&mut self) {
        if let Some(control) = &self.0 {
            control.fail("archive writer exited unexpectedly".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, thread};
    fn request(sequence: usize) -> CaptureRequest {
        CaptureRequest {
            sequence,
            seed: 500 + sequence as u64,
        }
    }
    #[test]
    fn gpu_admission_is_nonblocking_and_delivery_is_not_publication() {
        let control = Arc::new(CaptureControl::default());
        control.start().unwrap();
        let request = CaptureRequest {
            sequence: 9,
            seed: 77,
        };
        assert!(control.try_acquire(request).unwrap().is_none());
        control.grant_through(1);
        control.hold(true);
        assert!(control.try_acquire(request).unwrap().is_none());
        assert_eq!(control.snapshot().admitted, 0);
        control.hold(false);
        control.try_acquire(request).unwrap().unwrap().delivered();
        assert_eq!(control.snapshot().completed, 1);
        assert!(control.try_acquire(request).unwrap().is_none());
        control.cancel();
        assert!(control.try_acquire(request).is_err());
    }
    #[test]
    fn zero_credit_hold_resume_cancel_and_nonrevocable_permits() {
        let (tx, rx) = mpsc::channel();
        let control = Arc::new(CaptureControl::with_observer(move |event| {
            tx.send(event).unwrap();
        }));
        control.start().unwrap();
        assert!(control.start().is_err());
        let worker = {
            let c = control.clone();
            thread::spawn(move || c.acquire(request(0)).unwrap())
        };
        assert!(matches!(
            rx.recv().unwrap(),
            CaptureEvent::Stage {
                stage: CaptureStage::WaitingForDemand,
                ..
            }
        ));
        assert_eq!(control.snapshot().admitted, 0);
        control.hold(true);
        control.grant_through(1);
        assert!(rx.recv_timeout(Duration::from_millis(20)).is_err());
        control.hold(false);
        let permit = worker.join().unwrap().unwrap();
        control.grant_through(0);
        assert_eq!(control.snapshot().credits, 1);
        control.cancel();
        permit.finish(Duration::ZERO, None, None); // accepted credits cannot be revoked
        assert!(control.acquire(request(1)).unwrap().is_none());
        let outcomes: Vec<_> = rx
            .try_iter()
            .filter_map(|e| {
                if let CaptureEvent::Completed {
                    request, outcome, ..
                } = e
                {
                    Some((request.sequence, outcome))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            outcomes,
            [
                (0, CaptureOutcome::Published),
                (1, CaptureOutcome::Cancelled)
            ]
        );
        assert_eq!(control.snapshot().completed, 2);
    }
    #[test]
    fn writer_failure_wakes_waiter_and_each_token_completes_once() {
        let (tx, rx) = mpsc::channel();
        let control = Arc::new(CaptureControl::with_observer(move |event| {
            tx.send(event).unwrap();
        }));
        control.grant_through(1);
        let permit = control.acquire(request(0)).unwrap().unwrap();
        let worker = {
            let c = control.clone();
            thread::spawn(move || c.acquire(request(1)))
        };
        control.fail("writer disk full".into());
        permit.finish(Duration::ZERO, Some("writer disk full"), None);
        assert!(worker.join().unwrap().is_err());
        let terminal = rx
            .try_iter()
            .filter(|e| {
                matches!(
                    e,
                    CaptureEvent::Completed {
                        outcome: CaptureOutcome::Failed(_),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(terminal, 2);
        assert_eq!(control.snapshot().completed, 2);
    }
    #[test]
    fn cancelled_before_startup_needs_no_renderer_or_credits() {
        let control = Arc::new(CaptureControl::new());
        control.cancel();
        control.start().unwrap();
        assert!(control.acquire(request(3)).unwrap().is_none());
        assert_eq!(control.snapshot().admitted, 0);
        assert_eq!(control.snapshot().completed, 1);
    }
}

#[cfg(test)]
mod rejection_tests;
