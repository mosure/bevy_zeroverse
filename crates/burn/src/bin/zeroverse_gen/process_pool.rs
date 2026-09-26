//! Bounded process lifetimes without changing a sample's global index or seed.
use std::{io, process::Child};

use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct WorkerJob {
    pub job_id: usize,
    pub sample_offset: usize,
    pub samples: usize,
    pub chunk_offset: usize,
    pub chunks: usize,
}

/// Constant-space partitioning. Short child lifetimes may produce partial chunks;
/// every job reserves its own exact chunk span before it is dispatched.
pub struct WorkerJobs {
    remaining: usize,
    job_samples: usize,
    chunk_size: usize,
    fs_output: bool,
    next: WorkerJob,
    total_chunks: usize,
}

impl WorkerJobs {
    pub fn new(
        samples: usize,
        workers: usize,
        max_scenes_per_process: usize,
        sample_offset: usize,
        chunk_offset: usize,
        chunk_size: usize,
        fs_output: bool,
    ) -> Result<Self> {
        ensure!(
            samples > 0 && workers > 0 && chunk_size > 0,
            "finite positive samples, workers and chunk size are required"
        );
        sample_offset
            .checked_add(samples)
            .context("sample index overflow")?;
        let per_worker = samples.div_ceil(workers);
        let job_samples = if max_scenes_per_process == 0 {
            per_worker
        } else {
            per_worker.min(max_scenes_per_process)
        };
        let count_chunks = |count: usize| {
            if fs_output {
                count
            } else {
                count.div_ceil(chunk_size)
            }
        };
        let full_jobs = samples / job_samples;
        let chunks = full_jobs
            .checked_mul(count_chunks(job_samples))
            .and_then(|full| full.checked_add(count_chunks(samples % job_samples)))
            .context("chunk count overflow")?;
        chunk_offset
            .checked_add(chunks)
            .context("chunk index overflow")?;
        Ok(Self {
            remaining: samples,
            job_samples,
            chunk_size,
            fs_output,
            next: WorkerJob {
                job_id: 0,
                sample_offset,
                samples: 0,
                chunk_offset,
                chunks: 0,
            },
            total_chunks: chunks,
        })
    }
}

impl WorkerJobs {
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }
}

impl Iterator for WorkerJobs {
    type Item = WorkerJob;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let samples = self.remaining.min(self.job_samples);
        let chunks = if self.fs_output {
            samples
        } else {
            samples.div_ceil(self.chunk_size)
        };
        let job = WorkerJob {
            samples,
            chunks,
            ..self.next
        };
        self.remaining -= samples;
        self.next.sample_offset += samples;
        self.next.chunk_offset += chunks;
        // With usize::MAX one-scene jobs the last increment would overflow,
        // even though every assigned sample/chunk index is representable.
        if self.remaining != 0 {
            self.next.job_id += 1;
        }
        Some(job)
    }
}

#[derive(Clone, Debug)]
pub struct ChildExit {
    pub success: bool,
    pub code: Option<i32>,
    pub description: String,
}

impl From<std::process::ExitStatus> for ChildExit {
    fn from(status: std::process::ExitStatus) -> Self {
        Self {
            success: status.success(),
            code: status.code(),
            description: status.to_string(),
        }
    }
}

pub trait WorkerProcess {
    fn id(&self) -> u32;
    fn try_wait(&mut self) -> io::Result<Option<ChildExit>>;
    fn kill(&mut self) -> io::Result<()>;
    fn wait(&mut self) -> io::Result<ChildExit>;
}

impl WorkerProcess for Child {
    fn id(&self) -> u32 {
        Child::id(self)
    }
    fn try_wait(&mut self) -> io::Result<Option<ChildExit>> {
        Child::try_wait(self).map(|status| status.map(Into::into))
    }
    fn kill(&mut self) -> io::Result<()> {
        Child::kill(self)
    }
    fn wait(&mut self) -> io::Result<ChildExit> {
        Child::wait(self).map(Into::into)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LifecycleEvent {
    pub event: &'static str,
    pub pid: u32,
    pub worker_id: usize,
    #[serde(flatten)]
    pub job: WorkerJob,
    pub success: Option<bool>,
    pub exit_code: Option<i32>,
}

struct Active<C: WorkerProcess> {
    child: C,
    job: WorkerJob,
}

/// Child::drop does not terminate/reap. This also protects unwinding paths.
struct Children<C: WorkerProcess>(Vec<Option<Active<C>>>);

impl<C: WorkerProcess> Drop for Children<C> {
    fn drop(&mut self) {
        for active in self.0.iter_mut().flatten() {
            let _ = active.child.kill();
        }
        for active in self.0.iter_mut().flatten() {
            let _ = active.child.wait();
        }
    }
}

/// Observe all exits before dispatching replacements; never wait behind the
/// longest-running child when a different slot has already failed.
pub fn run_pool<C: WorkerProcess>(
    mut jobs: impl Iterator<Item = WorkerJob>,
    workers: usize,
    mut spawn: impl FnMut(usize, WorkerJob) -> Result<C>,
    mut observe: impl FnMut(LifecycleEvent) -> Result<()>,
    mut pause: impl FnMut(),
) -> Result<()> {
    ensure!(workers > 0, "at least one worker is required");
    let mut children: Children<C> = Children((0..workers).map(|_| None).collect());
    let result = (|| -> Result<()> {
        loop {
            for (worker_id, slot) in children.0.iter_mut().enumerate() {
                let Some(active) = slot.as_mut() else {
                    continue;
                };
                let Some(exit) = active.child.try_wait().context("checking worker process")? else {
                    continue;
                };
                let active = slot.take().expect("completed active worker");
                observe(LifecycleEvent {
                    event: if exit.success { "completed" } else { "failed" },
                    pid: active.child.id(),
                    worker_id,
                    job: active.job,
                    success: Some(exit.success),
                    exit_code: exit.code,
                })?;
                ensure!(
                    exit.success,
                    "worker {} job {} exited with failure: {}",
                    worker_id,
                    active.job.job_id,
                    exit.description
                );
            }
            for (worker_id, slot) in children.0.iter_mut().enumerate() {
                if slot.is_some() {
                    continue;
                }
                let Some(job) = jobs.next() else { continue };
                let child = spawn(worker_id, job)
                    .with_context(|| format!("spawning worker {worker_id} job {}", job.job_id))?;
                let pid = child.id();
                *slot = Some(Active { child, job });
                observe(LifecycleEvent {
                    event: "started",
                    pid,
                    worker_id,
                    job,
                    success: None,
                    exit_code: None,
                })?;
            }
            if children.0.iter().all(Option::is_none) {
                break;
            }
            pause();
        }
        Ok(())
    })();
    if let Err(error) = result {
        // Kill all siblings before reaping any one of them.
        for active in children.0.iter_mut().flatten() {
            let _ = active.child.kill();
        }
        let mut cleanup_errors = Vec::new();
        for (worker_id, slot) in children.0.iter_mut().enumerate() {
            if let Some(mut active) = slot.take() {
                match active.child.wait() {
                    Ok(exit) => {
                        if let Err(error) = observe(LifecycleEvent {
                            event: "cancelled",
                            pid: active.child.id(),
                            worker_id,
                            job: active.job,
                            success: Some(exit.success),
                            exit_code: exit.code,
                        }) {
                            cleanup_errors.push(error.to_string());
                        }
                    }
                    Err(error) => {
                        cleanup_errors.push(format!("reaping worker {worker_id}: {error}"))
                    }
                }
            }
        }
        return if cleanup_errors.is_empty() {
            Err(error)
        } else {
            Err(error.context(format!(
                "worker cleanup errors: {}",
                cleanup_errors.join("; ")
            )))
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn partitions_cover_indices_and_partial_chunks_without_cap_rounding() {
        for samples in [1, 2, 7, 31, 257, 1025] {
            for workers in [1, 2, 8] {
                for cap in [0, 1, 2, 16, 256] {
                    for chunk_size in [1, 3, 16, 512] {
                        for fs in [false, true] {
                            let jobs: Vec<_> =
                                WorkerJobs::new(samples, workers, cap, 13, 9, chunk_size, fs)
                                    .unwrap()
                                    .collect();
                            let mut next_sample = 13;
                            let mut next_chunk = 9;
                            for (job_id, job) in jobs.iter().enumerate() {
                                assert_eq!(job.job_id, job_id);
                                assert_eq!(job.sample_offset, next_sample);
                                assert_eq!(job.chunk_offset, next_chunk);
                                assert!(job.samples > 0 && (cap == 0 || job.samples <= cap));
                                assert_eq!(
                                    job.chunks,
                                    if fs {
                                        job.samples
                                    } else {
                                        job.samples.div_ceil(chunk_size)
                                    }
                                );
                                next_sample += job.samples;
                                next_chunk += job.chunks;
                            }
                            assert_eq!(next_sample, samples + 13);
                            if cap == 0 {
                                assert!(jobs.len() <= workers);
                            }
                        }
                    }
                }
            }
        }
        let jobs: Vec<_> = WorkerJobs::new(7, 2, 2, 0, 0, 3, false).unwrap().collect();
        assert_eq!(
            jobs.iter().map(|job| job.samples).collect::<Vec<_>>(),
            [2, 2, 2, 1]
        );
        assert_eq!(
            jobs.iter().map(|job| job.chunk_offset).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn rejects_unbounded_and_overflowing_assignments_before_spawning() {
        assert!(WorkerJobs::new(0, 1, 2, 0, 0, 1, false).is_err());
        assert!(WorkerJobs::new(2, 0, 2, 0, 0, 1, false).is_err());
        assert!(WorkerJobs::new(2, 1, 2, usize::MAX, 0, 1, false).is_err());
        assert!(WorkerJobs::new(2, 1, 2, 0, usize::MAX, 1, false).is_err());
    }

    #[derive(Default)]
    struct Counts {
        live: usize,
        peak: usize,
        spawned: usize,
        killed: usize,
        reaped: usize,
    }
    struct FakeChild {
        id: u32,
        polls: usize,
        failing: bool,
        stopped: bool,
        counts: Rc<RefCell<Counts>>,
    }
    impl WorkerProcess for FakeChild {
        fn id(&self) -> u32 {
            self.id
        }
        fn try_wait(&mut self) -> io::Result<Option<ChildExit>> {
            self.polls -= 1;
            if self.polls == 0 {
                self.stopped = true;
                let mut counts = self.counts.borrow_mut();
                counts.live -= 1;
                counts.reaped += 1;
                Ok(Some(ChildExit {
                    success: !self.failing,
                    code: Some(if self.failing { 1 } else { 0 }),
                    description: "test exit".into(),
                }))
            } else {
                Ok(None)
            }
        }
        fn kill(&mut self) -> io::Result<()> {
            self.counts.borrow_mut().killed += 1;
            Ok(())
        }
        fn wait(&mut self) -> io::Result<ChildExit> {
            if !self.stopped {
                self.stopped = true;
                let mut counts = self.counts.borrow_mut();
                counts.live -= 1;
                counts.reaped += 1;
            }
            Ok(ChildExit {
                success: false,
                code: None,
                description: "cancelled".into(),
            })
        }
    }
    fn fake_child(counts: &Rc<RefCell<Counts>>, job: WorkerJob, failing: bool) -> FakeChild {
        let mut state = counts.borrow_mut();
        state.spawned += 1;
        state.live += 1;
        state.peak = state.peak.max(state.live);
        FakeChild {
            id: state.spawned as u32,
            polls: if failing { 1 } else { 3 + job.job_id % 2 },
            failing,
            stopped: false,
            counts: counts.clone(),
        }
    }

    #[test]
    fn reuses_bounded_slots_and_drains_all_jobs() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let events = Rc::new(RefCell::new(Vec::new()));
        run_pool(
            WorkerJobs::new(37, 3, 2, 11, 7, 3, false).unwrap(),
            3,
            |_, job| Ok(fake_child(&counts, job, false)),
            |event| {
                events.borrow_mut().push(event);
                Ok(())
            },
            || {},
        )
        .unwrap();
        let counts = counts.borrow();
        assert_eq!(counts.peak, 3);
        assert_eq!(counts.spawned, 19);
        assert_eq!(counts.reaped, 19);
        assert_eq!(counts.live, 0);
        assert_eq!(counts.killed, 0);
        assert!(events.borrow().iter().all(|event| event.worker_id < 3));
        assert_eq!(
            events
                .borrow()
                .iter()
                .filter(|event| event.event == "completed")
                .count(),
            19
        );
    }

    #[test]
    fn failed_child_stops_dispatch_and_kills_and_reaps_siblings() {
        let counts = Rc::new(RefCell::new(Counts::default()));
        let result = run_pool(
            WorkerJobs::new(100, 3, 2, 0, 0, 3, false).unwrap(),
            3,
            |_, job| Ok(fake_child(&counts, job, job.job_id == 1)),
            |_| Ok(()),
            || {},
        );
        assert!(result.is_err());
        let counts = counts.borrow();
        assert_eq!(counts.spawned, 3);
        assert_eq!(counts.killed, 2);
        assert_eq!(counts.reaped, 3);
        assert_eq!(counts.live, 0);
    }

    #[test]
    fn spawn_and_observer_failures_reap_previously_started_children() {
        for fail_spawn in [false, true] {
            let counts = Rc::new(RefCell::new(Counts::default()));
            let result = run_pool(
                WorkerJobs::new(10, 2, 2, 0, 0, 3, false).unwrap(),
                2,
                |_, job| {
                    ensure!(!fail_spawn || job.job_id == 0, "injected spawn error");
                    Ok(fake_child(&counts, job, false))
                },
                |event| {
                    ensure!(
                        fail_spawn || event.event != "started",
                        "injected journal failure"
                    );
                    Ok(())
                },
                || {},
            );
            assert!(result.is_err());
            let counts = counts.borrow();
            assert_eq!(counts.spawned, 1);
            assert_eq!(counts.killed, 1);
            assert_eq!(counts.reaped, 1);
            assert_eq!(counts.live, 0);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn actual_spawn_failure_reaps_a_running_cpu_only_child() {
        let mut pid = None;
        let result = run_pool(
            WorkerJobs::new(4, 2, 1, 0, 0, 2, false).unwrap(),
            2,
            |_, job| {
                if job.job_id == 1 {
                    anyhow::bail!("injected next-child spawn failure");
                }
                let child = std::process::Command::new("/bin/sleep").arg("60").spawn()?;
                pid = Some(child.id());
                Ok(child)
            },
            |_| Ok(()),
            || {},
        );
        assert!(result.is_err());
        assert!(!std::path::Path::new(&format!("/proc/{}", pid.unwrap())).exists());
    }
}
