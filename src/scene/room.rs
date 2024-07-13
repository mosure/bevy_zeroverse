use bevy::prelude::*;

use crate::{
    material::ZeroverseMaterials,
    primitive::PrimitiveSettings,
};


pub struct ZeroverseRoomPlugin;
impl Plugin for ZeroverseRoomPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<RoomSettings>();

        app.add_systems(Update, process_rooms);
    }
}


#[derive(Clone, Component, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct RoomSettings {
    pub base_settings: PrimitiveSettings,
}

impl Default for RoomSettings {
    fn default() -> RoomSettings {
        RoomSettings {
            base_settings: PrimitiveSettings::default(),
        }
    }
}


#[derive(Bundle, Default, Debug)]
pub struct RoomBundle {
    pub settings: RoomSettings,
    pub spatial: SpatialBundle,
}


#[derive(Clone, Component, Debug, Reflect)]
pub struct ZeroverseRoom;


fn build_room(
    _commands: &mut ChildBuilder,
    _settings: &RoomSettings,
    _meshes: &mut ResMut<Assets<Mesh>>,
    _zeroverse_materials: &Res<ZeroverseMaterials>,
) {
    let _rng = &mut rand::thread_rng();
}


fn process_rooms(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    zeroverse_materials: Res<ZeroverseMaterials>,
    primitives: Query<
        (
            Entity,
            &RoomSettings,
        ),
        Without<ZeroverseRoom>,
    >,
) {
    for (entity, settings) in primitives.iter() {
        commands.entity(entity)
            .insert(ZeroverseRoom)
            .with_children(|subcommands| {
                build_room(
                    subcommands,
                    settings,
                    &mut meshes,
                    &zeroverse_materials,
                );
            });
    }
}
