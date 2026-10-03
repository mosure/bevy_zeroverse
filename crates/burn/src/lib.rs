//! Persistent synthetic capture and bounded dataset export.
//!
//! For full-quality indoor multi-view datasets, start with
//! [`generator::GenConfig::indoor`]. It retains native shadows, diffuse GI,
//! material detail and float32 geometric annotations. Scheduling is automatic;
//! `Portable` quality is an explicit reduction in rendering effects, not a
//! throughput preset.
//!
//! ```no_run
//! use bevy_zeroverse_burn::generator::{GenConfig, run_chunk_generation};
//!
//! let config = GenConfig::indoor("out/rooms", 128);
//! run_chunk_generation(config)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! For in-memory consumption, retain a [`LiveDataset`] and call
//! [`LiveDataset::next_sample`]. Its renderer persists across calls and compatible
//! handles, and indoor lookahead is bounded automatically. The Burn `Dataset`
//! adapter uses the same scheduling. Calls produce consecutive samples rather
//! than random access; keep a renderer process alive across output shards.
//! Configure scene content, cameras and annotations once before the first call.
//! Different render configurations require separate processes.

mod calibration;
pub mod chunk;
mod co_visibility;
pub mod compression;
pub mod dataset;
mod flow;
pub mod fs;
pub mod generator;
pub mod progress;
pub mod sensor;
pub mod tui;

pub use dataset::{ChunkDataset, LiveDataset, LiveDatasetConfig, ZeroverseSample};
pub use fs::{FsDataset, load_sample_dir, save_sample_to_fs};
