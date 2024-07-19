use bevy::prelude::*;
use bevy_args::{
    Deserialize,
    Serialize,
    ValueEnum,
};

pub mod lighting;
pub mod object;
pub mod room;


pub struct ZeroverseScenePlugin;

impl Plugin for ZeroverseScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<RegenerateSceneEvent>();
        app.add_event::<SceneLoadedEvent>();

        app.init_resource::<ZeroverseSceneSettings>();
        app.register_type::<ZeroverseSceneSettings>();

        app.add_plugins((
            lighting::ZeroverseLightingPlugin,
            object::ZeroverseObjectPlugin,
            room::ZeroverseRoomPlugin,
        ));

        app.add_systems(PreUpdate, clear_old_scenes);
    }
}


#[derive(Component, Debug, Reflect)]
pub struct ZeroverseScene;

#[derive(Component, Debug, Reflect)]
pub struct ZeroverseSceneRoot;


#[derive(
    Debug,
    Default,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    Reflect,
    ValueEnum,
)]
pub enum ZeroverseSceneType {
    #[default]
    Object,
    Room,
}

#[derive(Resource, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct ZeroverseSceneSettings {
    pub num_cameras: usize,
    pub scene_type: ZeroverseSceneType,
}


#[derive(Event)]
pub struct RegenerateSceneEvent;


#[derive(Event)]
pub struct SceneLoadedEvent;


pub fn clear_old_scenes(
    mut commands: Commands,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
) {
    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    for entity in clear_zeroverse_scenes.iter() {
        commands.entity(entity).despawn_recursive();
    }
}
