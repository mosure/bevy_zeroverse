use bevy::prelude::*;

pub mod depth;
pub mod normal;


#[derive(Default, Debug, Resource, Reflect)]
#[reflect(Resource)]
pub enum RenderMode {
    #[default]
    Color,
    Depth,
    Normal,
}


#[derive(Debug, Default)]
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderMode>();
        app.register_type::<RenderMode>();

        app.add_plugins(depth::DepthPlugin);
        app.add_plugins(normal::NormalPlugin);

        // TODO: add wireframe pbr disable
        app.add_systems(PreUpdate, auto_disable_pbr_material::<depth::Depth>);
        app.add_systems(PreUpdate, auto_disable_pbr_material::<normal::Normal>);

        app.add_systems(PreUpdate, apply_render_modes);
        app.add_systems(PreUpdate, enable_pbr_material);
    }
}


#[derive(Component, Default, Debug, Reflect)]
pub struct DisabledPbrMaterial {
    pub material: Handle<StandardMaterial>,
}

#[derive(Component, Default, Debug, Reflect)]
pub struct EnablePbrMaterial;


pub fn auto_disable_pbr_material<T: Component>(
    mut commands: Commands,
    mut disabled_materials: Query<
        (
            Entity,
            &Handle<StandardMaterial>,
        ),
        (With<T>, Without<DisabledPbrMaterial>),
    >,
) {
    for (
        entity,
        disabled_material,
    ) in disabled_materials.iter_mut() {
        commands.entity(entity)
            .insert(DisabledPbrMaterial {
                material: disabled_material.clone(),
            })
            .remove::<EnablePbrMaterial>()
            .remove::<Handle<StandardMaterial>>();
    }
}

fn enable_pbr_material(
    mut commands: Commands,
    mut enabled_materials: Query<
        (
            Entity,
            &DisabledPbrMaterial,
        ),
        With<EnablePbrMaterial>,
    >,
) {
    for (entity, disabled_material) in enabled_materials.iter_mut() {
        commands.entity(entity)
            .insert(disabled_material.material.clone())
            .remove::<DisabledPbrMaterial>()
            .remove::<EnablePbrMaterial>();
    }
}


fn apply_render_modes(
    mut commands: Commands,
    render_mode: Res<RenderMode>,
    meshes: Query<Entity, With<Handle<Mesh>>>,
) {
    if render_mode.is_changed() {
        for entity in meshes.iter() {
            commands.entity(entity)
                .remove::<depth::Depth>()
                .remove::<normal::Normal>();

            match *render_mode {
                RenderMode::Color => {
                    commands.entity(entity)
                        .insert(EnablePbrMaterial);
                }
                RenderMode::Depth => {
                    commands.entity(entity)
                        .insert(depth::Depth);
                }
                RenderMode::Normal => {
                    commands.entity(entity)
                        .insert(normal::Normal);
                }
            }
        }
    }
}
