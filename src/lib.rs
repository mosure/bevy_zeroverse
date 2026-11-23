#![recursion_limit = "256"]

use bevy::prelude::*;

pub mod app;
pub mod asset;
pub mod camera;
pub use bevy_zeroverse_capture::calibration;
/// Core renderer package provenance for capture handshakes.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
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
    "capture-v203;rgb_artifact_eligibility=1;rgb_gizmos=1;sensor_optics=1;generated_rejection=1;ssaa=2;capacity=1;primary_region=3;indoor_motion=5;calibration=2;fisheye=5;co_visibility=1;bevy=0.20.0;burn=0.22.1;burn_human=0.6.0;bevy_burn_human=0.7.0;burn_human_motion=0.2.0;burn_ardy=0.2.0;burn_llama=0.2.0;burn_human_inference=0.2.0;ardy_motion=9;static_pose=5;pose_annotations=2;surface_flow=1;indoor=59;public_programs=17;kitchens=14;doors=3;multiview=27;camera_motion=2;ceiling_camera=9;lens_distribution=3;ovoxel=3;glass=7;hair=7;wardrobe=5;footwear=1;bounds=4;morphology=3;face=2;position=3;tabletop=23;finishes=35;seating=11;activity=3;playback=2;exterior=12;envelope=18;furnishings=26;botany=6;visual_content=9;equipment=36;palette=4;photometry=25;irradiance_domain=1;transport_sampler=3;sun_shadow=2;material_compute=5;semantic=12;ceiling=9;floor_details=8;window_treatments=4;window_sashes=1";

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
