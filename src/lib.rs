use bevy::prelude::*;

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
            material::ZeroverseMaterialPlugin,
            primitive::ZeroversePrimitivePlugin,
            render::RenderPlugin,
        ));

        #[cfg(feature = "plucker")]
        app.add_plugins(plucker::PluckerPlugin);
    }
}
