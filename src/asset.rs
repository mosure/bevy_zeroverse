use bevy::prelude::*;

pub struct ZeroverseAssetPlugin;
impl Plugin for ZeroverseAssetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaitForAssets>();

        app.add_systems(Update, clear_loaded_assets);
    }
}

#[derive(Resource, Debug, Default)]
pub struct WaitForAssets {
    pub handles: Vec<UntypedHandle>,
}

impl WaitForAssets {
    pub fn is_waiting(&self) -> bool {
        !self.handles.is_empty()
    }
}

fn clear_loaded_assets(
    asset_server: ResMut<AssetServer>,
    mut wait_for_assets: ResMut<WaitForAssets>,
) {
    wait_for_assets
        .handles
        .retain(|id| !asset_server.is_loaded(id));
}
