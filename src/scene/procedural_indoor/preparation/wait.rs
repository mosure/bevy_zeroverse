//! Bound CPU-task polling without extracting/submitting another idle frame.
//! Only the current requested room can wait; uploads and GPU/model readiness
//! retain their ordinary app updates after this CPU task completes.
use std::{
    future::Future,
    pin::Pin,
    sync::{mpsc, Arc},
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Default)]
pub(crate) struct WaitPolicy {
    pub threaded: bool,
    pub headless: bool,
    pub image_copiers: bool,
    pub editor: bool,
    pub sampler_enabled: bool,
    pub current_indoor: bool,
    pub request_matches: bool,
    pub readback_in_flight: bool,
}
impl WaitPolicy {
    fn enabled(self) -> bool {
        self.threaded
            && self.headless
            && self.image_copiers
            && !self.editor
            && self.sampler_enabled
            && self.current_indoor
            && self.request_matches
            && !self.readback_in_flight
    }
}

// Service cancellation, deadlines and other app systems periodically even if
// a CPU job is exceptionally slow. Completion wakes the worker immediately.
const MAX_WINDOW: Duration = Duration::from_millis(50);

struct TaskWake(mpsc::SyncSender<()>);
impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // One outstanding signal suffices. A late wake after timeout is benign.
        let _ = self.0.try_send(());
    }
}

pub(super) fn poll_current<F: Future + Unpin>(
    policy: WaitPolicy,
    task: &mut F,
) -> Option<F::Output> {
    poll_with_window(policy, task, MAX_WINDOW)
}

fn poll_with_window<F: Future + Unpin>(
    policy: WaitPolicy,
    task: &mut F,
    window: Duration,
) -> Option<F::Output> {
    let mut task = Pin::new(task);
    if let Poll::Ready(value) = task.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        return Some(value);
    }
    if !policy.enabled() {
        return None;
    }
    let deadline = Instant::now() + window;
    let (notify, notified) = mpsc::sync_channel(1);
    let waker = Waker::from(Arc::new(TaskWake(notify)));
    let mut cx = Context::from_waker(&waker);
    loop {
        // Register before waiting. A task that completed during registration
        // either returns Ready here or leaves a notification in the channel.
        if let Poll::Ready(value) = task.as_mut().poll(&mut cx) {
            return Some(value);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || notified.recv_timeout(remaining).is_err() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::preparation::pipeline::Request;
    use std::cell::Cell;

    fn eligible() -> WaitPolicy {
        WaitPolicy {
            threaded: true,
            headless: true,
            image_copiers: true,
            sampler_enabled: true,
            current_indoor: true,
            request_matches: true,
            ..Default::default()
        }
    }

    #[test]
    fn completed_tasks_and_errors_return_immediately() {
        for result in [Ok(17), Err("requested room rejected")] {
            assert_eq!(
                poll_current(eligible(), &mut std::future::ready(result)),
                Some(result)
            );
        }
    }

    #[test]
    fn wake_between_poll_and_wait_is_retained() {
        let polls = Cell::new(0);
        let mut task = std::future::poll_fn(|cx| {
            polls.set(polls.get() + 1);
            if polls.get() == 2 {
                cx.waker().wake_by_ref();
                cx.waker().wake_by_ref();
            }
            if polls.get() == 3 {
                Poll::Ready(31)
            } else {
                Poll::Pending
            }
        });
        assert_eq!(poll_current(eligible(), &mut task), Some(31));
        assert_eq!(polls.get(), 3);
    }

    #[test]
    fn unfinished_task_yields_at_deadline_without_periodic_polling() {
        let polls = Cell::new(0);
        let mut pending = std::future::poll_fn(|_| {
            polls.set(polls.get() + 1);
            Poll::<()>::Pending
        });
        assert_eq!(
            poll_with_window(eligible(), &mut pending, Duration::from_millis(5)),
            None
        );
        assert_eq!(polls.get(), 2);
    }

    #[test]
    fn completion_on_another_thread_wakes_the_waiter() {
        let (sender, receiver) = mpsc::channel::<Waker>();
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let done = &done;
            scope.spawn(move || {
                let waker = receiver.recv().unwrap();
                done.store(true, std::sync::atomic::Ordering::Release);
                waker.wake();
            });
            let mut polls = 0;
            let mut task = std::future::poll_fn(|cx| {
                polls += 1;
                if done.load(std::sync::atomic::Ordering::Acquire) {
                    return Poll::Ready(42);
                }
                // The first poll is the immediate, nonblocking check. Hand
                // the completing thread the waker registered for the wait.
                if polls == 2 {
                    sender.send(cx.waker().clone()).unwrap();
                }
                Poll::Pending
            });
            assert_eq!(
                poll_with_window(eligible(), &mut task, Duration::from_secs(2)),
                Some(42)
            );
        });
    }

    #[test]
    fn interactive_idle_stale_scene_and_readback_work_never_wait() {
        let request = Request::new(
            202,
            &Default::default(),
            &Default::default(),
            Default::default(),
        );
        let stale = request.successor();
        for policy in [
            WaitPolicy {
                threaded: false,
                ..eligible()
            },
            WaitPolicy {
                headless: false,
                ..eligible()
            },
            WaitPolicy {
                image_copiers: false,
                ..eligible()
            },
            WaitPolicy {
                editor: true,
                ..eligible()
            },
            WaitPolicy {
                sampler_enabled: false,
                ..eligible()
            },
            WaitPolicy {
                current_indoor: false,
                ..eligible()
            },
            WaitPolicy {
                request_matches: request == stale,
                ..eligible()
            },
            WaitPolicy {
                readback_in_flight: true,
                ..eligible()
            },
        ] {
            let polls = Cell::new(0);
            assert_eq!(
                poll_current(
                    policy,
                    &mut std::future::poll_fn(|_| {
                        polls.set(polls.get() + 1);
                        Poll::<()>::Pending
                    })
                ),
                None
            );
            assert_eq!(polls.get(), 1);
        }
    }
}
