use bevy::prelude::*;

pub mod app;
pub mod camera;
pub mod io;
pub mod manifold;
pub mod material;
pub mod mesh;
pub mod plucker;
pub mod primitive;
pub mod render;
pub mod scene;


pub struct BevyZeroversePlugin;

impl Plugin for BevyZeroversePlugin {
    fn build(&self, app: &mut App) {
        info!("initializing BevyZeroversePlugin...");

        app.add_plugins((
            camera::ZeroverseCameraPlugin,
            material::ZeroverseMaterialPlugin,
            primitive::ZeroversePrimitivePlugin,
            render::RenderPlugin,
            scene::ZeroverseScenePlugin,
        ));

        #[cfg(feature = "plucker")]
        app.add_plugins(plucker::PluckerPlugin);
    }
}
