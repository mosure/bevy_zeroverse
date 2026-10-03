//! Shared native preparation budget. Different rooms/stages compete for this
//! bounded pool instead of each creating another four runnable worker threads.
//! Callers reserve output identities first and retain scoped submission order.

pub(crate) fn pool() -> &'static bevy::tasks::TaskPool {
    static POOL: std::sync::OnceLock<bevy::tasks::TaskPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        bevy::tasks::TaskPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(4)))
            .thread_name("indoor-prepare".into())
            .build()
    })
}
