#![feature(cfg_eval)]

use bevy::prelude::*;

pub mod app;
pub mod asset;
pub mod camera;
pub mod io;
pub mod manifold;
pub mod material;
pub mod mesh;
// pub mod plucker;
pub mod primitive;
pub mod render;
pub mod scene;

#[cfg(not(target_family = "wasm"))]
pub mod util;


pub struct BevyZeroversePlugin;

impl Plugin for BevyZeroversePlugin {
    fn build(&self, app: &mut App) {
        info!("initializing BevyZeroversePlugin...");

        app.add_plugins((
            asset::ZeroverseAssetPlugin,
            camera::ZeroverseCameraPlugin,
            material::ZeroverseMaterialPlugin,
            mesh::ZeroverseMeshPlugin,
            primitive::ZeroversePrimitivePlugin,
            render::RenderPlugin,
            scene::ZeroverseScenePlugin,
        ));

        #[cfg(feature = "plucker")]
        app.add_plugins(plucker::PluckerPlugin);
    }
}
