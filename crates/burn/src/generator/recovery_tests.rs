use super::*;
use admission::{CaptureEvent, CaptureOutcome, RejectionPolicy};
use bevy_zeroverse::sample::failure::{SceneRejection, SceneRejectionKind};
use std::sync::Mutex;

#[test]
fn rejection_receipt_waits_for_previous_atomic_export_and_writer_errors_remain_fatal() {
    for writer_fails in [false, true] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let log = events.clone();
        let control = Arc::new(
            CaptureControl::with_observer(move |event| {
                if let CaptureEvent::Completed {
                    request, outcome, ..
                } = event
                {
                    log.lock().unwrap().push((request.sequence, outcome));
                }
            })
            .with_rejection_policy(RejectionPolicy::Continue {
                kinds: vec![SceneRejectionKind::CameraPlacement],
                max_total: 2,
                max_consecutive: 2,
            })
            .unwrap(),
        );
        control.grant_through(2);
        let previous = control
            .acquire(CaptureRequest {
                sequence: 0,
                seed: 100,
            })
            .unwrap()
            .unwrap();
        let rejected = control
            .acquire(CaptureRequest {
                sequence: 1,
                seed: 101,
            })
            .unwrap()
            .unwrap();
        let (tx, rx) = mpsc::sync_channel::<WriteJob>(0);
        let (release, wait) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let writer_control = control.clone();
        let writer = thread::spawn(move || {
            // A deliberately blocked earlier archive owns its permit until closed.
            wait.recv().unwrap();
            if writer_fails {
                writer_control.fail("disk full".into());
                previous.finish(Duration::ZERO, Some("disk full"), None);
            } else {
                previous.finish(
                    Duration::ZERO,
                    None,
                    Some(PublicationReceipt {
                        index: 5,
                        path: "000005.tar".into(),
                    }),
                );
                assert!(matches!(rx.recv().unwrap().data, WriteData::Fence));
            }
        });
        let capture = thread::spawn(move || {
            let result = complete_rejection_after_exports(
                &tx,
                rejected,
                SceneRejection {
                    seed: 101,
                    kind: SceneRejectionKind::CameraPlacement,
                    reason: "bounded search".into(),
                },
                Duration::ZERO,
            );
            done_tx.send(result).unwrap();
        });
        assert!(done_rx.try_recv().is_err());
        assert!(events.lock().unwrap().is_empty());
        release.send(()).unwrap();
        writer.join().unwrap();
        capture.join().unwrap();
        let result = done_rx.recv().unwrap();
        let events = events.lock().unwrap();
        assert_eq!(events.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(control.snapshot().completed, 2);
        if writer_fails {
            assert!(result.is_err());
            assert!(
                events
                    .iter()
                    .all(|(_, outcome)| matches!(outcome, CaptureOutcome::Failed(_)))
            );
            assert_eq!(control.snapshot().returned, 0);
        } else {
            assert!(result.unwrap());
            assert!(matches!(events[0].1, CaptureOutcome::Published));
            assert!(matches!(
                events[1].1,
                CaptureOutcome::Rejected {
                    continued: true,
                    ..
                }
            ));
            assert_eq!(control.snapshot().returned, 1);
        }
    }
}
