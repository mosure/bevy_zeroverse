//! Bound CPU-task polling without extracting/submitting another idle frame.
//! Only the current requested room can wait; uploads and GPU/model readiness
//! retain their ordinary app updates after this CPU task completes.
use std::time::{Duration, Instant};

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

const MAX_WINDOW: Duration = Duration::from_millis(10);
const INTERVAL: Duration = Duration::from_millis(1);

pub(crate) fn poll_current<T>(policy: WaitPolicy, poll: impl FnMut() -> Option<T>) -> Option<T> {
    let start = Instant::now();
    poll_window(policy, poll, || start.elapsed(), std::thread::sleep)
}

fn poll_window<T>(
    policy: WaitPolicy,
    mut poll: impl FnMut() -> Option<T>,
    mut elapsed: impl FnMut() -> Duration,
    mut sleep: impl FnMut(Duration),
) -> Option<T> {
    // Match the ordinary once-per-update poll for every ineligible caller.
    let ready = poll();
    if ready.is_some() || !policy.enabled() {
        return ready;
    }
    loop {
        let remaining = MAX_WINDOW.saturating_sub(elapsed());
        if remaining.is_zero() {
            return None;
        }
        sleep(INTERVAL.min(remaining));
        if let Some(ready) = poll() {
            return Some(ready);
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
    fn completed_tasks_and_errors_return_without_sleeping() {
        for result in [Ok(17), Err("requested room rejected")] {
            assert_eq!(
                poll_window(
                    eligible(),
                    || Some(result),
                    || panic!("completed task needs no deadline"),
                    |_| panic!("completed task must not sleep"),
                ),
                Some(result)
            );
        }
    }

    #[test]
    fn current_task_finishing_during_wait_stops_immediately() {
        let elapsed = Cell::new(Duration::ZERO);
        let polls = Cell::new(0);
        let result = poll_window(
            eligible(),
            || {
                polls.set(polls.get() + 1);
                (polls.get() == 4).then_some(Ok::<_, String>(31))
            },
            || elapsed.get(),
            |duration| elapsed.set(elapsed.get() + duration),
        );
        assert_eq!(result, Some(Ok(31)));
        assert_eq!(elapsed.get(), Duration::from_millis(3));
        assert_eq!(polls.get(), 4);
    }

    #[test]
    fn unfinished_task_returns_at_deadline_and_last_sleep_is_truncated() {
        let elapsed = Cell::new(Duration::ZERO);
        let sleeps = Cell::new(0);
        let result = poll_window(
            eligible(),
            || {
                // Simulate 0.3 ms of polling work before every sleep.
                elapsed.set(elapsed.get() + Duration::from_micros(300));
                None::<()>
            },
            || elapsed.get(),
            |duration| {
                assert!(duration <= INTERVAL);
                assert!(elapsed.get() + duration <= MAX_WINDOW);
                sleeps.set(sleeps.get() + 1);
                elapsed.set(elapsed.get() + duration);
            },
        );
        assert_eq!(result, None);
        assert!(sleeps.get() > 0);
        assert!(elapsed.get() >= MAX_WINDOW);
        assert!(elapsed.get() <= MAX_WINDOW + Duration::from_micros(300));
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
                poll_window(
                    policy,
                    || {
                        polls.set(polls.get() + 1);
                        None::<()>
                    },
                    || panic!("ineligible request needs no deadline"),
                    |_| panic!("ineligible request must not sleep"),
                ),
                None
            );
            assert_eq!(polls.get(), 1);
        }
    }
}
