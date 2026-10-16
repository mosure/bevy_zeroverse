//! Persistent, bounded workers with deterministic output ordering. A slow room
//! permits at most 2*workers outstanding results, not an unbounded reorder queue.
use anyhow::{ensure, Result};
use std::{
    collections::BTreeMap,
    sync::{mpsc, Arc, Mutex},
};

pub fn ordered<T: Send>(
    count: u32,
    workers: usize,
    produce: impl Fn(u32) -> T + Sync,
    mut consume: impl FnMut(T) -> Result<()>,
) -> Result<()> {
    ensure!((1..=4).contains(&workers), "worker budget must be 1..4");
    std::thread::scope(|scope| -> Result<()> {
        let (send_jobs, jobs) = mpsc::channel::<u32>();
        let jobs = Arc::new(Mutex::new(jobs));
        let (results, receive) = mpsc::channel();
        for _ in 0..workers {
            let jobs = jobs.clone();
            let produce = &produce;
            let results = results.clone();
            scope.spawn(move || loop {
                let Ok(index) = jobs.lock().expect("worker queue").recv() else {
                    break;
                };
                if results.send((index, produce(index))).is_err() {
                    break;
                }
            });
        }
        drop(results);
        let (mut next, mut written) = (0, 0);
        let mut pending = BTreeMap::new();
        while written < count {
            while next < count && next - written < (2 * workers) as u32 {
                send_jobs.send(next)?;
                next += 1;
            }
            let (index, result) = receive.recv()?;
            pending.insert(index, result);
            while let Some(result) = pending.remove(&written) {
                consume(result)?;
                written += 1;
            }
        }
        drop(send_jobs);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordering_and_worker_count_do_not_change_observations() {
        for workers in 1..=4 {
            let mut output = Vec::new();
            ordered(
                35,
                workers,
                |i| {
                    if i == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    i.wrapping_mul(17)
                },
                |v| {
                    output.push(v);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(output, (0..35).map(|i| i * 17).collect::<Vec<_>>());
        }
    }
    #[test]
    fn failed_writer_cancels_queued_work_without_hanging() {
        assert!(ordered(128, 2, |i| i, |_| Err(anyhow::anyhow!("disk full"))).is_err());
    }
}
