use bevy::prelude::*;
use bevy_args::{
    Deserialize,
    Serialize,
    ValueEnum,
};
use pyo3::prelude::*;

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
#[pyclass(eq, eq_int)]
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
