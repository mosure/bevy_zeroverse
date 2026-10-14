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
    "capture-v61;calibration=1;co_visibility=1;bevy=0.20.0;burn=0.22.0;burn_human=0.6.0;bevy_burn_human=0.7.0;burn_human_motion=0.2.0;burn_ardy=0.2.0;burn_llama=0.2.0;burn_human_inference=0.2.0;ardy_motion=9;surface_flow=1;indoor=40;multiview=8;ovoxel=3;glass=4;hair=6;wardrobe=5;footwear=1;bounds=4;morphology=3;position=3;tabletop=3;finishes=11;seating=3;activity=2;playback=2;exterior=5;envelope=4;furnishings=5;botany=3;visual_content=3;equipment=5;palette=2;photometry=3;sun_shadow=2;material_compute=2;semantic=4;ceiling=2;floor_details=2;window_treatments=1";

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
