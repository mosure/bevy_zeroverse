use super::*;
use bevy_zeroverse::sample::failure::{CaptureError, SceneRejectionKind};

fn request(sequence: usize) -> CaptureRequest {
    CaptureRequest {
        sequence,
        seed: 100 + sequence as u64,
    }
}
fn rejection(sequence: usize) -> SceneRejection {
    SceneRejection {
        seed: request(sequence).seed,
        kind: SceneRejectionKind::CameraPlacement,
        reason: "bounded search exhausted".into(),
    }
}
fn policy(total: u64, consecutive: u64) -> RejectionPolicy {
    RejectionPolicy::Continue {
        kinds: vec![SceneRejectionKind::CameraPlacement],
        max_total: total,
        max_consecutive: consecutive,
    }
}

#[test]
fn rejections_return_credits_without_recycling_attempts_or_archive_indices() {
    let events = Arc::new(Mutex::new(vec![]));
    let sink = events.clone();
    let c = Arc::new(
        CaptureControl::with_observer(move |event| sink.lock().unwrap().push(event))
            .with_rejection_policy(policy(3, 2))
            .unwrap(),
    );
    c.start().unwrap();
    c.grant_through(1);
    c.acquire(request(7))
        .unwrap()
        .unwrap()
        .rejected(rejection(7), Duration::ZERO)
        .unwrap();
    let mut p = c.acquire(request(8)).unwrap().unwrap();
    p.captured(Duration::ZERO, None);
    p.finish(
        Duration::ZERO,
        None,
        Some(PublicationReceipt {
            index: 3,
            path: "out/000003.tar".into(),
        }),
    );
    let s = c.snapshot();
    assert_eq!(
        (s.credits, s.admitted, s.returned, s.completed),
        (1, 2, 1, 2)
    );
    let c2 = c.clone();
    let waiter = std::thread::spawn(move || c2.acquire(request(9)).unwrap());
    c.cancel();
    assert!(waiter.join().unwrap().is_none());
    let events = events.lock().unwrap();
    let completed: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            CaptureEvent::Completed {
                request,
                outcome,
                publication,
                ..
            } => Some((request.sequence, outcome, publication)),
            _ => None,
        })
        .collect();
    assert_eq!(completed.len(), 3);
    assert!(matches!(
        completed[0].1,
        CaptureOutcome::Rejected {
            continued: true,
            ..
        }
    ));
    assert!(completed[0].2.is_none());
    assert_eq!(completed[1].2.as_ref().unwrap().index, 3);
    assert_eq!(completed[1].0, 8);
    assert_eq!(completed[2].1, &CaptureOutcome::Cancelled);
}

#[test]
fn rejection_limits_and_categories_stop_without_dropping_terminal_receipts() {
    for (limit, accepted_between) in [(policy(3, 1), false), (policy(1, 1), true)] {
        let c = Arc::new(CaptureControl::new().with_rejection_policy(limit).unwrap());
        c.grant_through(2);
        c.acquire(request(0))
            .unwrap()
            .unwrap()
            .rejected(rejection(0), Duration::ZERO)
            .unwrap();
        if accepted_between {
            let mut p = c.acquire(request(1)).unwrap().unwrap();
            p.captured(Duration::ZERO, None);
            p.finish(Duration::ZERO, None, None);
        }
        assert!(
            c.acquire(request(2))
                .unwrap()
                .unwrap()
                .rejected(rejection(2), Duration::ZERO)
                .is_err()
        );
        assert!(c.snapshot().cancelled);
        assert_eq!(c.snapshot().returned, 2);
        assert_eq!(c.snapshot().completed, 2 + u64::from(accepted_between));
    }
    let c = Arc::new(
        CaptureControl::new()
            .with_rejection_policy(policy(3, 3))
            .unwrap(),
    );
    c.grant_through(1);
    let mut wrong_kind = rejection(0);
    wrong_kind.kind = SceneRejectionKind::MissingProgramAnchor;
    assert!(
        c.acquire(request(0))
            .unwrap()
            .unwrap()
            .rejected(wrong_kind, Duration::ZERO)
            .is_err()
    );
    assert!(c.snapshot().cancelled);
}

#[test]
fn configuration_and_unknown_errors_are_never_inferred_as_rejections() {
    for invalid in [
        policy(0, 0),
        policy(1, 2),
        RejectionPolicy::Continue {
            kinds: vec![],
            max_total: 1,
            max_consecutive: 1,
        },
    ] {
        assert!(
            CaptureControl::new()
                .with_rejection_policy(invalid)
                .is_err()
        );
    }
    assert!(!CaptureControl::new().recovers_rejections());
    assert!(matches!(
        CaptureError::from("seed 123: missing required Elevator"),
        CaptureError::Failed(_)
    ));
    assert!(matches!(
        CaptureError::from("GPU device lost"),
        CaptureError::Failed(_)
    ));
}

#[test]
fn cancellation_during_rejected_capture_retires_it_without_admitting_another() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let log = events.clone();
    let control = Arc::new(
        CaptureControl::with_observer(move |event| log.lock().unwrap().push(event))
            .with_rejection_policy(policy(2, 2))
            .unwrap(),
    );
    control.grant_through(1);
    let permit = control.acquire(request(0)).unwrap().unwrap();
    control.cancel();
    assert!(!permit.rejected(rejection(0), Duration::ZERO).unwrap());
    let snapshot = control.snapshot();
    assert_eq!(
        (snapshot.admitted, snapshot.returned, snapshot.completed),
        (1, 1, 1)
    );
    let events = events.lock().unwrap();
    let outcomes: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            CaptureEvent::Completed { outcome, .. } => Some(outcome),
            _ => None,
        })
        .collect();
    assert!(matches!(
        outcomes.as_slice(),
        [CaptureOutcome::Rejected {
            continued: false,
            ..
        }]
    ));
}

#[test]
fn placement_rejection_returns_gpu_credit_and_next_request_continues() {
    let control = Arc::new(
        CaptureControl::new()
            .with_rejection_policy(RejectionPolicy::Continue {
                kinds: vec![SceneRejectionKind::PlacementClearance],
                max_total: 2,
                max_consecutive: 1,
            })
            .unwrap(),
    );
    control.start().unwrap();
    control.grant_through(1);
    let first = request(10);
    let rejected = SceneRejection {
        seed: first.seed,
        kind: SceneRejectionKind::PlacementClearance,
        reason: "floor furniture floats, sinks or intersects ceiling fixtures".into(),
    };
    assert!(
        control
            .try_acquire(first)
            .unwrap()
            .unwrap()
            .rejected(rejected.clone(), Duration::ZERO)
            .unwrap()
    );
    let next = request(11);
    control.try_acquire(next).unwrap().unwrap().delivered();
    let state = control.snapshot();
    assert_eq!((state.admitted, state.completed, state.returned), (2, 2, 1));
    assert!(!state.cancelled);
    assert!(control.try_acquire(request(12)).unwrap().is_none());
    // Identical wording from an untyped renderer/transport source stays fatal.
    assert!(matches!(
        CaptureError::from(rejected.reason),
        CaptureError::Failed(_)
    ));
}
