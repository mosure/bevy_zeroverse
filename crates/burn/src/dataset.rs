use std::{
    env,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use burn::data::dataset::Dataset;

use crate::chunk::{discover_chunks, load_chunk};

pub type ZeroverseSample = bevy_zeroverse::sample::Sample;

pub struct LiveDatasetConfig {
    pub asset_root: Option<PathBuf>,
    pub num_samples: usize,
    pub timeout: Duration,
    pub zeroverse_config: bevy_zeroverse::app::BevyZeroverseConfig,
    pub app_on_main_thread: bool,
    pub app_ready: Option<Arc<AtomicBool>>,
}

impl Default for LiveDatasetConfig {
    fn default() -> Self {
        Self {
            asset_root: None,
            num_samples: 0,
            timeout: Duration::from_secs(120),
            zeroverse_config: bevy_zeroverse::app::BevyZeroverseConfig {
                headless: true,
                image_copiers: true,
                editor: false,
                press_esc_close: false,
                keybinds: false,
                ..Default::default()
            },
            app_on_main_thread: false,
            app_ready: None,
        }
    }
}

pub struct LiveDataset {
    config: LiveDatasetConfig,
    initialized: AtomicBool,
    completed: Mutex<usize>,
}

fn normalize_asset_root(root: &Path) -> PathBuf {
    let file_name_is_assets = root.file_name().map(|n| n == "assets").unwrap_or(false);
    if file_name_is_assets {
        root.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| root.to_path_buf())
    } else {
        root.to_path_buf()
    }
}

static APP_STARTED: OnceLock<AtomicBool> = OnceLock::new();
static APP_CONTRACT: OnceLock<serde_json::Value> = OnceLock::new();
// The process owns one renderer and one response channel. A request and its
// receive are one transaction, including across compatible LiveDataset handles.
static CAPTURE_TRANSACTION: Mutex<()> = Mutex::new(());

fn app_contract(config: &LiveDatasetConfig) -> Result<serde_json::Value> {
    let mut engine = config.zeroverse_config.clone();
    // Only indoor scenes consume this seed. Output index offsets for legacy
    // scenes must not look like a renderer configuration change.
    if engine.scene_type != bevy_zeroverse::scene::ZeroverseSceneType::ProceduralIndoor {
        engine.indoor_seed = None;
    }
    let root = config
        .asset_root
        .clone()
        .unwrap_or(std::env::current_dir()?);
    Ok(serde_json::json!({"engine": engine,
        "asset_root": normalize_asset_root(&root).canonicalize()?}))
}

impl LiveDataset {
    pub fn new(config: LiveDatasetConfig) -> Self {
        config
            .zeroverse_config
            .validate_ovoxel()
            .expect("invalid O-voxel dataset configuration");
        Self {
            config,
            initialized: AtomicBool::new(false),
            completed: Mutex::new(0),
        }
    }

    fn ensure_initialized(&self) -> Result<()> {
        if self.initialized.load(Ordering::Acquire) {
            return Ok(());
        }
        if env::var("BEVY_ZEROVERSE_FAKE_APP").is_err() {
            let requested = app_contract(&self.config)?;
            let active = APP_CONTRACT.get_or_init(|| requested.clone());
            anyhow::ensure!(
                active == &requested,
                "LiveDataset supports one renderer configuration per process; use a fresh process for different dimensions, timesteps, modes, assets or scene settings"
            );
        }
        if self
            .initialized
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }

        bevy_zeroverse::headless::setup_globals(
            self.config
                .asset_root
                .as_ref()
                .map(|p| normalize_asset_root(p).display().to_string()),
        );

        // Allow tests to bypass spawning the full Bevy app while still driving channel-based flows.
        if env::var("BEVY_ZEROVERSE_FAKE_APP").is_ok() {
            return Ok(());
        }

        let started = APP_STARTED.get_or_init(|| AtomicBool::new(false));
        if started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
            && !self.config.app_on_main_thread
        {
            bevy_zeroverse::headless::setup_and_run_app(
                true,
                Some(self.config.zeroverse_config.clone()),
            );
        }

        if self.config.app_on_main_thread
            && let Some(ready) = self.config.app_ready.as_ref()
        {
            while !ready.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(10));
            }
        }
        Ok(())
    }

    fn receive_sample(&self) -> Result<ZeroverseSample> {
        let receiver = bevy_zeroverse::io::channels::sample_receiver()
            .context("sample receiver not initialized")?
            .clone();

        let lock = receiver
            .lock()
            .map_err(|_| anyhow::anyhow!("sample receiver lock poisoned"))?;
        let started = std::time::Instant::now();
        loop {
            if let Some(error) = bevy_zeroverse::io::channels::take_capture_failure() {
                anyhow::bail!(error);
            }
            match lock.recv_timeout(Duration::from_millis(100)) {
                Ok(sample) => return Ok(sample),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                    if started.elapsed() < self.config.timeout => {}
                Err(err) => anyhow::bail!("failed to recv sample: {err:?}"),
            }
        }
    }

    fn request_next(&self, prefetch_indoor: usize) -> Result<()> {
        // Ensure the app channels are present even if initialization was skipped (e.g. in tests).
        if !bevy_zeroverse::io::channels::channels_initialized() {
            bevy_zeroverse::headless::setup_globals(
                self.config
                    .asset_root
                    .as_ref()
                    .map(|p| normalize_asset_root(p).display().to_string()),
            );
        }

        let sender = bevy_zeroverse::io::channels::app_frame_sender();
        sender
            .send(bevy_zeroverse::io::channels::AppFrameRequest {
                prefetch_indoor,
                ..Default::default()
            })
            .context("failed to signal zeroverse app for next sample")
    }

    pub fn mark_initialized_for_tests(&self) {
        self.initialized.store(true, Ordering::Release);
    }

    pub fn inject_sample_for_tests(&self, sample: ZeroverseSample) {
        bevy_zeroverse::io::channels::sample_sender()
            .send(sample)
            .expect("failed to inject sample");
    }
}

impl Dataset<ZeroverseSample> for LiveDataset {
    fn len(&self) -> usize {
        self.config.num_samples
    }

    fn get(&self, _index: usize) -> Option<ZeroverseSample> {
        self.next_sample()
            .map_err(|err| eprintln!("capture failed: {err:#}"))
            .ok()
    }
}

impl LiveDataset {
    /// Capture the next scene using the persistent renderer. Scheduling and
    /// bounded lookahead are automatic; no thread, upload or queue tuning is
    /// needed. Quality and requested annotations are never reduced for speed.
    ///
    /// Like the Burn `Dataset::get` adapter, this is a live stream, not random
    /// access. `num_samples` bounds each epoch's lookahead; one-sample datasets
    /// do no speculative work. Zero denotes an unbounded stream.
    pub fn next_sample(&self) -> Result<ZeroverseSample> {
        self.capture(None)
    }

    // Compatibility for the generator's existing diagnostic prefetch overrides.
    // Ordinary downstream callers use next_sample or Dataset::get.
    pub(crate) fn capture_next(&self, prefetch_indoor: usize) -> Result<ZeroverseSample> {
        self.capture(Some(prefetch_indoor))
    }

    fn capture(&self, lookahead_override: Option<usize>) -> Result<ZeroverseSample> {
        let _transaction = CAPTURE_TRANSACTION
            .lock()
            .map_err(|_| anyhow::anyhow!("capture transaction lock poisoned"))?;
        self.ensure_initialized()?;
        let mut completed = self
            .completed
            .lock()
            .map_err(|_| anyhow::anyhow!("capture sequence lock poisoned"))?;
        let lookahead = lookahead_override.unwrap_or_else(|| {
            if self.config.zeroverse_config.scene_type
                == bevy_zeroverse::scene::ZeroverseSceneType::ProceduralIndoor
            {
                automatic_lookahead(self.config.num_samples, *completed)
            } else {
                0
            }
        });
        self.request_next(lookahead)?;
        let sample = self.receive_sample()?;
        *completed = completed.wrapping_add(1);
        Ok(sample)
    }
}

fn automatic_lookahead(epoch_samples: usize, completed: usize) -> usize {
    const DEPTH: usize = 3;
    if epoch_samples == 0 {
        DEPTH
    } else {
        DEPTH.min(epoch_samples - completed % epoch_samples - 1)
    }
}

type ChunkCache = Vec<(usize, Vec<ZeroverseSample>)>;

pub struct ChunkDataset {
    chunks: Vec<(PathBuf, usize, usize)>, // (path, start_idx, len)
    total: usize,
    cache: Arc<Mutex<ChunkCache>>,
}

impl ChunkDataset {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut chunks = Vec::new();
        let mut total = 0usize;

        for path in discover_chunks(dir)? {
            let samples = load_chunk(&path)?;
            let len = samples.len();
            let start = total;
            total += len;
            chunks.push((path.clone(), start, len));
        }

        Ok(Self {
            chunks,
            total,
            cache: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn fetch_chunk(&self, idx: usize) -> Option<Vec<ZeroverseSample>> {
        if let Ok(cache) = self.cache.lock()
            && let Some((_, samples)) = cache.iter().find(|(chunk_start, samples)| {
                let chunk_end = chunk_start + samples.len();
                idx >= *chunk_start && idx < chunk_end
            })
        {
            return Some(samples.clone());
        }

        let (path, start, _) = self
            .chunks
            .iter()
            .find(|(_, start, len)| {
                let end = *start + *len;
                idx >= *start && idx < end
            })?
            .clone();

        let samples = load_chunk(&path).ok()?;

        if let Ok(mut cache) = self.cache.lock() {
            cache.push((start, samples.clone()));
            if cache.len() > 2 {
                cache.remove(0);
            }
        }

        Some(samples)
    }
}

impl Dataset<ZeroverseSample> for ChunkDataset {
    fn len(&self) -> usize {
        self.total
    }

    fn get(&self, index: usize) -> Option<ZeroverseSample> {
        if index >= self.total {
            return None;
        }

        let (path, start, _len) = self
            .chunks
            .iter()
            .find(|(_, start, len)| {
                let end = *start + *len;
                index >= *start && index < end
            })?
            .clone();

        let local_idx = index - start;
        if let Some(chunk) = self.fetch_chunk(index) {
            return chunk.get(local_idx).cloned();
        }

        load_chunk(path)
            .ok()
            .and_then(|chunk| chunk.get(local_idx).cloned())
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    use bevy_zeroverse::io::channels;

    #[test]
    fn public_live_stream_automatically_prefetches_and_owns_each_response() {
        channels::init_channels();
        let dataset = LiveDataset::new(LiveDatasetConfig {
            num_samples: 5,
            zeroverse_config: bevy_zeroverse::app::BevyZeroverseConfig {
                scene_type: bevy_zeroverse::scene::ZeroverseSceneType::ProceduralIndoor,
                ..Default::default()
            },
            ..Default::default()
        });
        dataset.mark_initialized_for_tests();
        let server = thread::spawn(|| {
            let rx = channels::app_frame_receiver().unwrap().lock().unwrap();
            for (index, depth) in [3, 3, 2, 1, 0, 3].into_iter().enumerate() {
                let request = rx.recv_timeout(Duration::from_secs(5)).unwrap();
                assert_eq!(request.prefetch_indoor, depth);
                channels::sample_sender()
                    .send(ZeroverseSample {
                        view_dim: index as u32,
                        ..Default::default()
                    })
                    .unwrap();
            }
        });
        for index in 0..6 {
            let sample = if index % 2 == 0 {
                dataset.next_sample().unwrap()
            } else {
                Dataset::get(&dataset, index).unwrap()
            };
            assert_eq!(sample.view_dim, index as u32);
        }
        server.join().unwrap();

        let mut clients = Vec::new();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        for _ in 0..2 {
            let client = LiveDataset::new(LiveDatasetConfig {
                num_samples: 1,
                ..Default::default()
            });
            client.mark_initialized_for_tests();
            let barrier = barrier.clone();
            clients.push(thread::spawn(move || {
                barrier.wait();
                client.next_sample().unwrap()
            }));
        }
        barrier.wait();
        let rx = channels::app_frame_receiver().unwrap().lock().unwrap();
        for index in 0..2 {
            let request = rx.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(request.prefetch_indoor, 0);
            // A second caller must not submit while another owns the reply.
            assert!(matches!(
                rx.recv_timeout(Duration::from_millis(20)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            channels::sample_sender()
                .send(ZeroverseSample {
                    view_dim: index,
                    ..Default::default()
                })
                .unwrap();
        }
        drop(rx);
        let mut ids: Vec<_> = clients
            .into_iter()
            .map(|client| client.join().unwrap().view_dim)
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, [0, 1]);
        assert_eq!(automatic_lookahead(1, 0), 0);
        assert_eq!(automatic_lookahead(0, 100), 3);
    }
}
