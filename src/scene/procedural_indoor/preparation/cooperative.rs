//! Browser work slices, shared across preparation stages. Cheap parts stay in
//! the same task; expensive work returns to the event loop on a time budget.
#[cfg(target_arch = "wasm32")]
use web_sys::js_sys;

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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
export async function zeroversePreparationYield() {
    // A prioritized continuation avoids nested-timer clamping. Keep a real
    // event-loop yield on browsers without the scheduling API as well.
    if (typeof globalThis.scheduler?.yield === 'function') {
        try {
            await globalThis.scheduler.yield();
            return;
        } catch (_) {
            // An enclosing aborted postTask must not strand preparation.
        }
    }
    await new Promise(resolve => setTimeout(resolve, 0));
}
"#)]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = zeroversePreparationYield)]
    fn browser_yield() -> js_sys::Promise;
}

/// Native jobs already use the bounded preparation pool. WASM must yield to a
/// real browser task. Yield only after useful work consumes the slice; prefer
/// scheduler continuations to timers without requiring that API for correctness.
pub(crate) async fn cooperate() {
    #[cfg(target_arch = "wasm32")]
    {
        use bevy::platform::time::Instant;
        if !BUDGET.with_borrow_mut(|budget| budget.due(Instant::now())) {
            return;
        }
        wasm_bindgen_futures::JsFuture::from(browser_yield())
            .await
            .expect("schedule browser preparation");
        BUDGET.with_borrow_mut(|budget| budget.resumed(Instant::now()));
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
