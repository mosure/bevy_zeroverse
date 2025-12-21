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
    pub samples_done: usize,
    pub chunks_done: usize,
    pub last_update_ms: u64,
    pub done: bool,
}

impl ProgressMessage {
    pub fn from_snapshot(worker_id: usize, snapshot: &ProgressSnapshot, done: bool) -> Self {
        Self {
            worker_id,
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
pub struct ProgressAggregator {
    start: Instant,
    workers: Mutex<HashMap<usize, WorkerSnapshot>>,
}

impl ProgressAggregator {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            workers: Mutex::new(HashMap::new()),
        }
    }

    pub fn apply_message(&self, message: ProgressMessage) {
        let mut workers = self.workers.lock().unwrap();
        workers.insert(
            message.worker_id,
            WorkerSnapshot {
                worker_id: message.worker_id,
                samples_done: message.samples_done,
                chunks_done: message.chunks_done,
                last_update_ms: message.last_update_ms,
                done: message.done,
            },
        );
    }

    pub fn snapshot(&self) -> AggregatedSnapshot {
        let workers = self.workers.lock().unwrap();
        let mut entries: Vec<WorkerSnapshot> = workers.values().cloned().collect();
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
