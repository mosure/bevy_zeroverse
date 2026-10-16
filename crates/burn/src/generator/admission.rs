//! Optional room-level backpressure. The controller owns inventory policy;
//! Zeroverse owns capture validity, bounded preparation and atomic export.
use serde::Serialize;
use std::{
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CaptureRequest {
    /// Absolute output sample index, including `sample_offset`.
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
    pub held: bool,
    pub cancelled: bool,
    pub waiting: Option<CaptureRequest>,
}
#[derive(Default)]
struct State {
    snapshot: AdmissionSnapshot,
    used: bool,
    failure: Option<String>,
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
    pub(super) fn start(&self) -> anyhow::Result<()> {
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
    pub(super) fn fail(&self, message: String) {
        let mut state = self.state.lock().unwrap();
        state.failure.get_or_insert(message);
        state.snapshot.cancelled = true;
        self.wake.notify_all();
    }
    pub(super) fn acquire(
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
            && (state.snapshot.held || state.snapshot.admitted >= state.snapshot.credits)
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
pub(super) struct Permit {
    control: Arc<CaptureControl>,
    request: CaptureRequest,
    admitted: bool,
    wait: Duration,
    capture: Duration,
    export: Duration,
    telemetry: Option<bevy_zeroverse::io::channels::CaptureTelemetry>,
    completed: bool,
}
impl Permit {
    pub(super) fn captured(
        &mut self,
        elapsed: Duration,
        telemetry: Option<bevy_zeroverse::io::channels::CaptureTelemetry>,
    ) {
        self.capture = elapsed;
        self.telemetry = telemetry.filter(|value| value.seed == Some(self.request.seed));
    }
    pub(super) fn exporting(&self) {
        self.control.emit(CaptureEvent::Stage {
            request: self.request,
            stage: CaptureStage::Exporting,
        });
    }
    pub(super) fn finish(mut self, elapsed: Duration, error: Option<&str>) {
        self.export = elapsed;
        self.complete(error.map_or(CaptureOutcome::Published, |e| {
            CaptureOutcome::Failed(e.into())
        }));
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
pub(super) struct WriterGuard(pub Option<Arc<CaptureControl>>);
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
        permit.finish(Duration::ZERO, None); // accepted credits cannot be revoked
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
        permit.finish(Duration::ZERO, Some("writer disk full"));
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
