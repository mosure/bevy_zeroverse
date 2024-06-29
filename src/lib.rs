use bevy::prelude::*;

use material::ZeroverseMaterialPlugin;

pub mod manifold;
pub mod material;


pub struct BevyZeroversePlugin;

impl Plugin for BevyZeroversePlugin {
    fn build(&self, app: &mut App) {
        info!("initializing BevyZeroversePlugin...");

        app.add_plugins(ZeroverseMaterialPlugin);
    }
}
