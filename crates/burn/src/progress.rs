use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Debug)]
pub struct ProgressTracker {
    start: Instant,
    samples_done: AtomicUsize,
    chunks_done: AtomicUsize,
    last_update_ms: AtomicU64,
}

impl ProgressTracker {
    pub fn new() -> Self {
        let now = now_millis();
        Self {
            start: Instant::now(),
            samples_done: AtomicUsize::new(0),
            chunks_done: AtomicUsize::new(0),
            last_update_ms: AtomicU64::new(now),
        }
    }

    pub fn record_samples(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.samples_done.fetch_add(count, Ordering::Relaxed);
        self.touch();
    }

    pub fn record_chunks(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.chunks_done.fetch_add(count, Ordering::Relaxed);
        self.touch();
    }

    pub fn snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            samples_done: self.samples_done.load(Ordering::Relaxed),
            chunks_done: self.chunks_done.load(Ordering::Relaxed),
            elapsed: self.start.elapsed(),
            last_update_ms: self.last_update_ms.load(Ordering::Relaxed),
        }
    }

    fn touch(&self) {
        self.last_update_ms.store(now_millis(), Ordering::Relaxed);
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct ProgressSnapshot {
    pub samples_done: usize,
    pub chunks_done: usize,
    pub elapsed: Duration,
    pub last_update_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgressMessage {
    pub worker_id: usize,
    #[serde(default)]
    pub job_id: usize,
    pub samples_done: usize,
    pub chunks_done: usize,
    pub last_update_ms: u64,
    pub done: bool,
}

impl ProgressMessage {
    pub fn from_snapshot(worker_id: usize, snapshot: &ProgressSnapshot, done: bool) -> Self {
        Self {
            worker_id,
            job_id: 0,
            samples_done: snapshot.samples_done,
            chunks_done: snapshot.chunks_done,
            last_update_ms: snapshot.last_update_ms,
            done,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorkerSnapshot {
    pub worker_id: usize,
    pub samples_done: usize,
    pub chunks_done: usize,
    pub last_update_ms: u64,
    pub done: bool,
}

#[derive(Debug)]
struct SlotProgress {
    job_id: usize,
    completed_samples: usize,
    completed_chunks: usize,
    job_samples: usize,
    job_chunks: usize,
    snapshot: WorkerSnapshot,
}

#[derive(Debug)]
pub struct ProgressAggregator {
    start: Instant,
    workers: Mutex<HashMap<usize, SlotProgress>>,
}

impl ProgressAggregator {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            workers: Mutex::new(HashMap::new()),
        }
    }

    /// Register a replacement before accepting its UDP progress. Only one entry
    /// is retained per physical slot, regardless of the number of child jobs.
    pub fn start_job(&self, worker_id: usize, job_id: usize, samples: usize, chunks: usize) {
        let mut workers = self.workers.lock().unwrap();
        let previous = workers.get(&worker_id);
        let completed_samples = previous.map_or(0, |slot| slot.completed_samples);
        let completed_chunks = previous.map_or(0, |slot| slot.completed_chunks);
        workers.insert(
            worker_id,
            SlotProgress {
                job_id,
                completed_samples,
                completed_chunks,
                job_samples: samples,
                job_chunks: chunks,
                snapshot: WorkerSnapshot {
                    worker_id,
                    samples_done: completed_samples,
                    chunks_done: completed_chunks,
                    last_update_ms: now_millis(),
                    done: false,
                },
            },
        );
    }

    /// Successful process exit is authoritative even if its final UDP packet was
    /// dropped. Late packets from this or an older child cannot undo completion.
    pub fn finish_job(&self, worker_id: usize, job_id: usize) {
        let mut workers = self.workers.lock().unwrap();
        if let Some(slot) = workers.get_mut(&worker_id) {
            if slot.job_id != job_id || slot.snapshot.done {
                return;
            }
            slot.completed_samples += slot.job_samples;
            slot.completed_chunks += slot.job_chunks;
            slot.snapshot.samples_done = slot.completed_samples;
            slot.snapshot.chunks_done = slot.completed_chunks;
            slot.snapshot.last_update_ms = now_millis();
            slot.snapshot.done = true;
        }
    }

    pub fn apply_message(&self, message: ProgressMessage) {
        let mut workers = self.workers.lock().unwrap();
        let Some(slot) = workers.get_mut(&message.worker_id) else {
            return;
        };
        if slot.job_id != message.job_id
            || slot.snapshot.done
            || message.samples_done > slot.job_samples
            || message.chunks_done > slot.job_chunks
        {
            return;
        }
        slot.snapshot.samples_done = slot
            .snapshot
            .samples_done
            .max(slot.completed_samples + message.samples_done);
        slot.snapshot.chunks_done = slot
            .snapshot
            .chunks_done
            .max(slot.completed_chunks + message.chunks_done);
        slot.snapshot.last_update_ms = slot.snapshot.last_update_ms.max(message.last_update_ms);
        // A UDP done flag can precede a process failure; only finish_job commits
        // the epoch and its expected totals after the parent reaps a success.
    }

    pub fn snapshot(&self) -> AggregatedSnapshot {
        let workers = self.workers.lock().unwrap();
        let mut entries: Vec<WorkerSnapshot> =
            workers.values().map(|slot| slot.snapshot.clone()).collect();
        entries.sort_by_key(|w| w.worker_id);

        let samples_done = entries.iter().map(|w| w.samples_done).sum();
        let chunks_done = entries.iter().map(|w| w.chunks_done).sum();

        AggregatedSnapshot {
            samples_done,
            chunks_done,
            elapsed: self.start.elapsed(),
            workers: entries,
        }
    }
}

impl Default for ProgressAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct AggregatedSnapshot {
    pub samples_done: usize,
    pub chunks_done: usize,
    pub elapsed: Duration,
    pub workers: Vec<WorkerSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(job_id: usize, samples_done: usize, chunks_done: usize) -> ProgressMessage {
        ProgressMessage {
            worker_id: 0,
            job_id,
            samples_done,
            chunks_done,
            last_update_ms: now_millis(),
            done: true,
        }
    }

    #[test]
    fn reused_slot_keeps_cumulative_progress_and_rejects_stale_epochs() {
        let progress = ProgressAggregator::new();
        progress.start_job(0, 0, 3, 2);
        progress.apply_message(message(0, 2, 1));
        assert!(!progress.snapshot().workers[0].done);
        progress.finish_job(0, 0);
        progress.finish_job(0, 0); // Repeated completion must not double count.
        assert_eq!(progress.snapshot().samples_done, 3);
        progress.start_job(0, 1, 2, 1);
        progress.apply_message(message(1, 1, 0));
        progress.apply_message(message(0, 3, 2)); // Late UDP from the old process.
        progress.apply_message(message(1, 0, 0)); // Reordered current UDP.
        progress.apply_message(message(2, 2, 1)); // Unregistered future epoch.
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.samples_done, 4);
        assert_eq!(snapshot.chunks_done, 2);
        assert_eq!(snapshot.workers.len(), 1);
        progress.finish_job(0, 1); // Final UDP packet deliberately absent.
        assert_eq!(progress.snapshot().samples_done, 5);
        assert_eq!(progress.snapshot().chunks_done, 3);
        assert!(progress.snapshot().workers[0].done);
    }

    #[test]
    fn progress_storage_is_bounded_by_slots_and_planned_work() {
        let progress = ProgressAggregator::new();
        for job_id in 0..1000 {
            progress.start_job(0, job_id, 2, 1);
            progress.apply_message(message(job_id, 999, 99));
            assert_eq!(progress.snapshot().samples_done, job_id * 2);
            progress.finish_job(0, job_id);
        }
        assert_eq!(progress.snapshot().workers.len(), 1);
        assert_eq!(progress.snapshot().samples_done, 2000);
    }
}
