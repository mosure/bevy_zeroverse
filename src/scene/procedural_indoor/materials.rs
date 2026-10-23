//! Material subsystem: stable surface slots, sampled finish programs, texture
//! synthesis/filtering, and asset bindings have separate ownership. Geometry
//! requests a surface/finish; only bindings install Bevy material/image assets.
//! Role-specific PBR textures, generated in memory with repeat sampling and mip chains.
pub mod art;
mod atlas;
pub mod boards;
pub mod botanical;
pub mod carpet;
pub mod ceramic;
pub mod coating;
pub mod composition;
pub mod concrete;
pub mod construction;
mod content;
pub mod display_finish;
pub(crate) mod emitters;
mod environment;
pub(crate) mod exterior;
mod field;
mod filter;
pub mod flooring;
pub mod glass;
pub mod glazing;
pub mod gpu;
mod human;
mod kitchen;
pub mod layers;
pub mod leather;
pub mod manufactured;
mod microstructure;
pub mod mineral;
mod mip_transfer;
pub mod paint;
mod paper;
pub mod program;
#[cfg(test)]
mod qualification;
mod raster;
mod reflections;
pub mod rug_design;
pub mod rugs;
pub mod screens;
mod signals;
pub mod textile;
pub mod timber;
pub mod variants;
pub mod wall;
use super::layout::{IndoorManifest, LightingMood};
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
};

mod bindings;
mod defaults;
mod mipmaps;
mod surface;
mod synthesis;
pub use bindings::IndoorMaterials;
pub(crate) use bindings::{finish_slot, registered_finish, MaterialSelection};
use defaults::{definitions, Definition};
use mipmaps::*;
pub use surface::Surface;
use synthesis::*;

pub fn kelvin_rgb(k: f32) -> Vec3 {
    // Smooth warm tungsten through cool sky RGB prior; not a spectral model.
    if k < 4500.0 {
        Vec3::new(1.0, 0.46, 0.18).lerp(
            Vec3::new(1.0, 0.90, 0.78),
            ((k - 1800.0) / 2700.0).clamp(0.0, 1.0),
        )
    } else {
        Vec3::new(1.0, 0.90, 0.78).lerp(
            Vec3::new(0.68, 0.82, 1.0),
            ((k - 4500.0) / 5500.0).clamp(0.0, 1.0),
        )
    }
}

#[cfg(test)]
mod surface_tests;

#[cfg(test)]
#[path = "materials/mipmap_replay_tests.rs"]
mod mipmap_replay_tests;

#[cfg(test)]
#[path = "materials/map_replay_tests.rs"]
mod map_replay_tests;
