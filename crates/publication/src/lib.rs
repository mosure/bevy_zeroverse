//! The canonical capture → validate → media/metrics → page → paper → attest
//! pipeline. CI generates current evidence, then independently verifies it.
pub mod automation;
pub mod camera_study;
pub mod capacity;
pub mod capacity_theory;
pub mod cohorts;
pub mod config;
pub mod dataset;
mod depth;
pub mod doors;
pub mod fisheye;
pub mod graphics;
pub mod io;
pub mod materials;
pub mod media;
pub mod objects;
pub mod paper;
pub mod pipeline;
pub mod programs;
pub mod qualification;
pub mod retained;
mod snapshot;
pub mod study;
pub mod timing;
pub mod visibility;
