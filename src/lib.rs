#![recursion_limit = "256"]

use bevy::prelude::*;

pub mod app;
pub mod asset;
pub mod camera;
pub use bevy_zeroverse_capture::calibration;
pub mod headless;
pub mod io;
pub mod manifold;
pub mod material;
pub mod mesh;
pub mod ovoxel_mesh;
// pub mod plucker;
pub mod annotation;
pub use annotation::ovoxel;
pub mod human_motion;
pub mod primitive;
pub mod procedural_human;
pub mod provenance;
pub mod render;
pub mod sample;
pub mod scene;

#[cfg(not(target_family = "wasm"))]
pub mod util;

/// Pixel/annotation contract identity. Dependency versions are pinned in Cargo.toml.
/// Geometry grammar version is independent; renderer upgrades invalidate capture resume.
pub const CAPTURE_ENGINE_IDENTITY: &str =
    "capture-v48;calibration=1;co_visibility=1;bevy=0.19.1;burn=0.21.0;burn_human=0.5.1;bevy_burn_human=0.6.1;burn_human_motion=0.1.1;burn_ardy=0.1.4;burn_llama=0.1.2;burn_human_inference=0.1.4;ardy_motion=9;surface_flow=1;indoor=29;multiview=5;ovoxel=3;glass=2;hair=5;wardrobe=4;footwear=1;bounds=4;morphology=2;position=3;tabletop=2;finishes=8;seating=3;activity=2;playback=2;exterior=2;envelope=3";

pub struct BevyZeroversePlugin;

impl Plugin for BevyZeroversePlugin {
    fn build(&self, app: &mut App) {
        info!("initializing BevyZeroversePlugin...");

        app.add_plugins((
            asset::ZeroverseAssetPlugin,
            camera::ZeroverseCameraPlugin,
            material::ZeroverseMaterialPlugin,
            mesh::ZeroverseMeshPlugin,
            procedural_human::ZeroverseBurnHumanPlugin,
            primitive::ZeroversePrimitivePlugin,
            render::RenderPlugin,
            annotation::obb::ZeroverseObbPlugin,
            annotation::pose::ZeroversePosePlugin,
            scene::ZeroverseScenePlugin,
            human_motion::HumanMotionPlugin,
            ovoxel::OvoxelPlugin,
        ));

        #[cfg(feature = "plucker")]
        app.add_plugins(plucker::PluckerPlugin);
    }
}
