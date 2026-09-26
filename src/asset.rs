use bevy::prelude::*;

pub struct ZeroverseAssetPlugin;
impl Plugin for ZeroverseAssetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaitForAssets>();
        let deferred = app
            .world()
            .get_resource::<crate::app::BevyZeroverseConfig>()
            .is_some_and(|args| {
                args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor
            });
        app.insert_resource(DeferredCatalogs(deferred));

        app.add_systems(Update, (clear_loaded_assets, request_deferred_catalogs));
    }
}

#[derive(Resource)]
struct DeferredCatalogs(bool);

fn request_deferred_catalogs(
    args: Option<Res<crate::app::BevyZeroverseConfig>>,
    mut deferred: ResMut<DeferredCatalogs>,
    mut materials: MessageWriter<crate::material::ShuffleMaterialsEvent>,
    mut meshes: MessageWriter<crate::mesh::ShuffleMeshesEvent>,
) {
    if deferred.0
        && args.is_some_and(|args| {
            args.scene_type != crate::scene::ZeroverseSceneType::ProceduralIndoor
        })
    {
        deferred.0 = false;
        materials.write(crate::material::ShuffleMaterialsEvent);
        meshes.write(crate::mesh::ShuffleMeshesEvent);
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
