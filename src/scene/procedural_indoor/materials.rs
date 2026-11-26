//! Material subsystem: stable surface slots, sampled finish programs, texture
//! synthesis/filtering, and asset bindings have separate ownership. Geometry
//! requests a surface/finish; only bindings install Bevy material/image assets.
//! Role-specific PBR textures, generated in memory with repeat sampling and mip chains.
pub(crate) mod acoustic_tile;
pub(crate) mod appliance_panel;
pub mod art;
mod atlas;
pub(crate) mod bin_film;
pub mod boards;
pub mod botanical;
pub(crate) mod broth;
#[cfg(target_arch = "wasm32")]
mod browser_maps;
pub mod carpet;
pub mod ceramic;
pub(crate) mod clock_face;
pub mod coating;
pub mod composition;
pub mod concrete;
pub mod construction;
mod content;
pub(crate) mod cooktop;
pub(crate) mod cork;
pub(crate) mod cup_board;
pub mod display_finish;
pub(crate) mod drinkware;
pub(crate) mod emitters;
mod environment;
pub(crate) mod exterior;
mod field;
mod filter;
pub mod flooring;
pub(crate) mod garment_stock;
pub mod glass;
pub mod glazing;
pub mod gpu;
pub(crate) mod handset;
mod human;
pub(crate) mod input_housing;
pub(crate) mod keycaps;
mod kitchen;
pub mod layers;
pub mod leather;
pub mod manufactured;
pub(crate) mod masonry;
mod microstructure;
pub mod mineral;
mod mip_transfer;
pub(crate) mod package_print;
pub mod paint;
pub mod paper;
pub(crate) mod point_of_sale;
mod potting;
pub(crate) mod printer_lcd;
pub(crate) mod produce_skin;
pub mod program;
#[cfg(test)]
mod qualification;
mod raster;
pub(crate) mod roast;
pub(crate) mod speaker;
pub(crate) mod sport;
pub(crate) mod strength;
pub(crate) use raster::letter_rows;
pub(crate) mod coated_stock;
pub(crate) mod projection_cloth;
mod reflections;
mod response;
pub mod rug_design;
pub mod rugs;
pub mod screens;
mod signals;
mod table_linen;
pub(crate) use signals::indicator_lens_finish;
pub(crate) mod sprinkler;
mod supplies;
pub mod textile;
pub mod timber;
pub(crate) mod tissue_stock;
pub mod variants;
pub mod wall;
pub mod window_textile;
pub(crate) mod worktop_finish;
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
mod sheet;
mod surface;
mod synthesis;
pub(crate) use bindings::{finish_slot, MaterialSelection};
pub use bindings::{registered_finish, IndoorMaterials};
use defaults::{definitions, Definition};
use mipmaps::*;
pub use surface::Surface;
use synthesis::*;

/// Encoded palette used by older indoor lighting manifests and decorative media.
/// For room illumination use the manifest's `photometry.illuminant_model`.
pub fn kelvin_rgb(k: f32) -> Vec3 {
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

#[cfg(test)]
mod content_diversity_tests;
