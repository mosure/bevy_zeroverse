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

impl LiveDataset {
    pub fn new(config: LiveDatasetConfig) -> Self {
        Self {
            config,
            initialized: AtomicBool::new(false),
        }
    }

    fn ensure_initialized(&self) {
        if self
            .initialized
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        bevy_zeroverse::headless::setup_globals(
            self.config
                .asset_root
                .as_ref()
                .map(|p| normalize_asset_root(p).display().to_string()),
        );

        // Allow tests to bypass spawning the full Bevy app while still driving channel-based flows.
        if env::var("BEVY_ZEROVERSE_FAKE_APP").is_ok() {
            return;
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
    }

    fn receive_sample(&self) -> Result<ZeroverseSample> {
        let receiver = bevy_zeroverse::io::channels::sample_receiver()
            .context("sample receiver not initialized")?
            .clone();

        let lock = receiver
            .lock()
            .map_err(|_| anyhow::anyhow!("sample receiver lock poisoned"))?;
        lock.recv_timeout(self.config.timeout)
            .map_err(|err| anyhow::anyhow!("failed to recv sample: {err:?}"))
    }

    fn request_next(&self) -> Result<()> {
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
            .send(())
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
        self.ensure_initialized();
        if let Err(err) = self.request_next() {
            eprintln!("failed to request sample: {err:?}");
            return None;
        }
        match self.receive_sample() {
            Ok(sample) => Some(sample),
            Err(err) => {
                eprintln!("failed to receive sample: {err:?}");
                None
            }
        }
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
