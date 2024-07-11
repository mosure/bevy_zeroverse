use bevy::prelude::*;

pub mod room;


pub struct ZeroverseScenePlugin;

impl Plugin for ZeroverseScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            room::ZeroverseRoomPlugin,
        ));
    }
}
