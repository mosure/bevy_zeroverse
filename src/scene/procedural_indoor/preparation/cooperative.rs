//! Browser work slices, shared across preparation stages. Cheap parts stay in
//! the same task; expensive work returns to the event loop on a time budget.
#[cfg(any(test, target_arch = "wasm32"))]
mod budget {
    use bevy::platform::time::Instant;
    use std::time::Duration;
    pub(super) const SLICE: Duration = Duration::from_millis(4);
    #[derive(Default)]
    pub(super) struct Budget(Option<Instant>);
    impl Budget {
        pub(super) fn due(&mut self, now: Instant) -> bool {
            now.duration_since(*self.0.get_or_insert(now)) >= SLICE
        }
        pub(super) fn resumed(&mut self, now: Instant) {
            self.0 = Some(now);
        }
    }
}
#[cfg(target_arch = "wasm32")]
std::thread_local! {
    static BUDGET: std::cell::RefCell<budget::Budget> = Default::default();
}
/// Native jobs already use the bounded preparation pool. WASM must yield to a
/// real browser task, but scheduling one timer per mesh adds a 4ms clamp for
/// hundreds of tiny parts. Yield only after useful work consumes the slice.
pub(crate) async fn cooperate() {
    #[cfg(target_arch = "wasm32")]
    {
        use bevy::platform::time::Instant;
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        use wasm_bindgen::{closure::Closure, JsCast};
        if !BUDGET.with_borrow_mut(|budget| budget.due(Instant::now())) {
            return;
        }
        let ready = Arc::new(AtomicBool::new(false));
        let mut scheduled = false;
        futures_lite::future::poll_fn(move |cx| {
            if ready.load(Ordering::Acquire) {
                return std::task::Poll::Ready(());
            }
            if !scheduled {
                scheduled = true;
                let ready = ready.clone();
                let wake = cx.waker().clone();
                let callback = Closure::once_into_js(move || {
                    BUDGET.with_borrow_mut(|budget| budget.resumed(Instant::now()));
                    ready.store(true, Ordering::Release);
                    wake.wake();
                });
                web_sys::window()
                    .expect("browser window")
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        callback.unchecked_ref(),
                        0,
                    )
                    .expect("schedule browser preparation");
            }
            std::task::Poll::Pending
        })
        .await;
    }
}
#[cfg(test)]
mod tests {
    use super::budget::*;
    use bevy::platform::time::Instant;
    use std::time::Duration;
    #[test]
    fn cheap_work_shares_a_bounded_slice_and_resumes_with_a_fresh_budget() {
        let now = Instant::now();
        let mut budget = Budget::default();
        for micros in 0..4000 {
            assert!(!budget.due(now + Duration::from_micros(micros)));
        }
        assert!(budget.due(now + SLICE));
        assert!(budget.due(now + SLICE * 2));
        let resumed = now + Duration::from_millis(25);
        budget.resumed(resumed);
        assert!(!budget.due(resumed));
        assert!(!budget.due(resumed + SLICE / 2));
        assert!(budget.due(resumed + SLICE));
    }
}
