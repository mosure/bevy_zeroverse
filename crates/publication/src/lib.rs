//! The canonical capture → validate → media/metrics → page → paper → attest
//! pipeline. Normal CI verifies the checked-in bundle without a GPU or Python.
pub mod config;
pub mod dataset;
pub mod graphics;
pub mod io;
pub mod media;
pub mod paper;
pub mod pipeline;
pub mod qualification;
pub mod visibility;
