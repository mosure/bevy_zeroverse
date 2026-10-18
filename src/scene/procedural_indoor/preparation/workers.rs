//! Process-wide bounded pool shared by rooms and their nested asset jobs.
use std::sync::{Mutex, OnceLock};
static POOL: OnceLock<bevy::tasks::TaskPool> = OnceLock::new();
static THREADS: Mutex<(Option<usize>, bool)> = Mutex::new((None, false));

/// Set a process-wide CPU preparation budget before the first indoor job.
/// Optional for multi-process applications; the default is at most eight,
/// shared by scene preparation and transport-tree construction.
/// Repeating the active budget is harmless; changing an initialized pool fails.
/// Wasm uses cooperative preparation and does not expose this native control.
pub fn configure_preparation_threads(threads: usize) -> Result<(), &'static str> {
    if !(1..=64).contains(&threads) {
        return Err("preparation threads must be in 1..=64");
    }
    let mut configured = THREADS.lock().unwrap();
    if configured.1 && configured.0 != Some(threads) {
        return Err("preparation pool already initialized; configure before generation");
    }
    configured.0 = Some(threads);
    Ok(())
}

pub(crate) fn pool() -> &'static bevy::tasks::TaskPool {
    POOL.get_or_init(|| {
        let mut configured = THREADS.lock().unwrap();
        let threads = *configured.0.get_or_insert_with(|| {
            std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
        });
        configured.1 = true;
        bevy::tasks::TaskPoolBuilder::new()
            .num_threads(threads)
            .thread_name("indoor-prepare".into())
            .build()
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn budget_rejects_unbounded_or_empty_pools() {
        assert!(super::configure_preparation_threads(0).is_err());
        assert!(super::configure_preparation_threads(65).is_err());
    }
}
