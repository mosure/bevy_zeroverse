use bevy::prelude::*;

pub mod manifold;
pub mod material;
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
            plucker::PluckerPlugin,
            primitive::ZeroversePrimitivePlugin,
            render::RenderPlugin,
        ));
    }
}
