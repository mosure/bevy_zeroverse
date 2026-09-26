use bevy::prelude::*;

pub struct ZeroverseAssetPlugin;
impl Plugin for ZeroverseAssetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaitForAssets>();
        app.init_resource::<SceneAssetDemand>();
        app.add_systems(First, update_asset_demand.in_set(AssetDemandSet));
        app.add_systems(Update, clear_loaded_assets);
    }
}

/// Only request catalogs/models actually consumed by the current scene. Updated
/// before catalog polling and scene generation, including inspector scene switches.
#[derive(Resource, Default, Debug, PartialEq, Eq)]
pub struct SceneAssetDemand {
    pub materials: bool,
    pub mesh_categories: std::collections::BTreeSet<String>,
    pub humans: bool,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetDemandSet;

#[allow(clippy::too_many_arguments)]
fn update_asset_demand(
    args: Res<crate::app::BevyZeroverseConfig>,
    object: Res<crate::scene::object::ZeroverseObjectSceneSettings>,
    room: Res<crate::scene::room::ZeroverseRoomSettings>,
    semantic: Res<crate::scene::semantic_room::ZeroverseSemanticRoomSettings>,
    human: Res<crate::scene::human::ZeroverseHumanSceneSettings>,
    primitives: Query<&crate::primitive::ZeroversePrimitiveSettings>,
    mut demand: ResMut<SceneAssetDemand>,
) {
    use crate::{primitive::ZeroversePrimitives, scene::ZeroverseSceneType as Scene};
    let mut next = SceneAssetDemand {
        materials: args.material_grid || args.scene_type != Scene::ProceduralIndoor,
        humans: args.scene_type == Scene::ProceduralIndoor && args.indoor_human_density > 0.0,
        ..default()
    };
    let mut add = |settings: &crate::primitive::ZeroversePrimitiveSettings| {
        for kind in &settings.available_types {
            if let ZeroversePrimitives::Mesh(category) = kind {
                if category == "human" {
                    next.humans = true;
                } else {
                    next.mesh_categories.insert(category.clone());
                }
            }
        }
    };
    match args.scene_type {
        Scene::Object => add(&object.primitive),
        Scene::Room => {
            add(&room.center_primitive_settings);
            add(&room.wall_primitive_settings);
        }
        Scene::Human => add(&human.primitive),
        Scene::SemanticRoom => {
            for settings in [
                &semantic.chair_settings,
                &semantic.table_settings,
                &semantic.plant_settings,
                &semantic.door_settings,
                &semantic.human_settings,
            ] {
                add(settings);
            }
        }
        _ => {}
    }
    for settings in &primitives {
        add(settings);
    }
    if *demand != next {
        *demand = next;
    }
}

/// Catalog discovery runs on the IO pool; polling never waits on directory IO.
#[derive(Resource)]
pub(crate) struct CatalogTask<T: Send + Sync + 'static>(pub Option<bevy::tasks::Task<T>>);
impl<T: Send + Sync + 'static> Default for CatalogTask<T> {
    fn default() -> Self {
        Self(None)
    }
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn asset_root() -> std::path::PathBuf {
    let root = std::env::var_os("BEVY_ASSET_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current directory"));
    let root = if root.ends_with("assets") {
        root
    } else {
        root.join("assets")
    };
    crate::util::strip_extended_length_prefix(&root)
}

#[derive(Resource, Debug, Default)]
pub struct WaitForAssets {
    pub handles: Vec<UntypedHandle>,
    pub pending_catalogs: usize,
}

impl WaitForAssets {
    pub fn is_waiting(&self) -> bool {
        self.pending_catalogs != 0 || !self.handles.is_empty()
    }
}

fn clear_loaded_assets(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut wait_for_assets: ResMut<WaitForAssets>,
) {
    wait_for_assets.handles.retain(|id| {
        if let bevy::asset::LoadState::Failed(error) = asset_server.load_state(id.id()) {
            commands.insert_resource(crate::sample::CaptureFailure(Some(format!(
                "required scene asset {:?} failed to load: {error}",
                id.id()
            ))));
            return false;
        }
        !asset_server.is_loaded_with_dependencies(id.id())
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{app::BevyZeroverseConfig, scene::ZeroverseSceneType};

    #[test]
    fn scene_demand_avoids_unrelated_models_and_tracks_scene_switches() {
        let mut app = App::new();
        app.init_resource::<SceneAssetDemand>()
            .init_resource::<BevyZeroverseConfig>()
            .init_resource::<crate::scene::object::ZeroverseObjectSceneSettings>()
            .init_resource::<crate::scene::room::ZeroverseRoomSettings>()
            .init_resource::<crate::scene::semantic_room::ZeroverseSemanticRoomSettings>()
            .init_resource::<crate::scene::human::ZeroverseHumanSceneSettings>()
            .add_systems(Update, update_asset_demand);
        for scene in [
            ZeroverseSceneType::Object,
            ZeroverseSceneType::Room,
            ZeroverseSceneType::CornellCube,
        ] {
            app.world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .scene_type = scene;
            app.update();
            let demand = app.world().resource::<SceneAssetDemand>();
            assert!(demand.materials);
            assert!(!demand.humans);
            assert!(demand.mesh_categories.is_empty());
        }
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .scene_type = ZeroverseSceneType::SemanticRoom;
        app.update();
        let demand = app.world().resource::<SceneAssetDemand>();
        assert!(demand.humans);
        assert!(demand.mesh_categories.contains("chair"));
        assert!(demand.mesh_categories.contains("plant"));
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .scene_type = ZeroverseSceneType::ProceduralIndoor;
        app.update();
        let demand = app.world().resource::<SceneAssetDemand>();
        assert!(demand.humans);
        assert!(!demand.materials);
        assert!(demand.mesh_categories.is_empty());
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_human_density = 0.0;
        app.update();
        assert!(!app.world().resource::<SceneAssetDemand>().humans);
        // Explicit user-supplied primitives still request the model catalog.
        app.world_mut()
            .spawn(crate::primitive::ZeroversePrimitiveSettings {
                available_types: vec![crate::primitive::ZeroversePrimitives::Mesh("chair".into())],
                ..default()
            });
        app.update();
        assert!(app
            .world()
            .resource::<SceneAssetDemand>()
            .mesh_categories
            .contains("chair"));
    }
}
