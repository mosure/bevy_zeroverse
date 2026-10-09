//! Reflected resource diagnostics using retained Bevy Feathers controls.
use super::*;
use bevy::{
    feathers::{controls::*, theme::ThemedText},
    reflect::serde::{ReflectDeserializer, ReflectSerializer},
    text::{EditableText, TextEditChange},
    ui_widgets::Activate,
};
use serde::de::DeserializeSeed;

#[derive(Resource, Default)]
pub(super) struct State {
    root: Option<Entity>,
    scene: Option<ZeroverseSceneType>,
    draft: String,
    apply: bool,
    close: bool,
    error: Option<Entity>,
}
fn snapshot<T: Resource + Reflect>(world: &World) -> Result<String, String> {
    let registry = world.resource::<AppTypeRegistry>().read();
    serde_json::to_string_pretty(&ReflectSerializer::new(world.resource::<T>(), &registry))
        .map_err(|e| e.to_string())
}
fn apply<T: Resource<Mutability = bevy::ecs::component::Mutable> + Reflect + Clone>(
    world: &mut World,
    input: &str,
) -> Result<(), String> {
    let value = {
        let registry = world.resource::<AppTypeRegistry>().read();
        let mut json = serde_json::Deserializer::from_str(input);
        let value = ReflectDeserializer::new(&registry)
            .deserialize(&mut json)
            .map_err(|e| e.to_string())?;
        json.end().map_err(|e| e.to_string())?;
        value
    };
    if value.get_represented_type_info().map(|t| t.type_id()) != Some(std::any::TypeId::of::<T>()) {
        return Err("The JSON must describe the selected scene’s resource type.".into());
    }
    // Failed edits never modify the live resource.
    let mut updated = world.resource::<T>().clone();
    updated
        .try_apply(value.as_partial_reflect())
        .map_err(|e| e.to_string())?;
    *world.resource_mut::<T>() = updated;
    Ok(())
}
pub(super) fn panel(world: &mut World) {
    let debug = world.resource::<editor::model::EditorState>().debug;
    let mut state = world.remove_resource::<State>().unwrap_or_default();
    if !debug || state.close {
        if let Some(root) = state.root.take() {
            world.despawn(root);
        }
        world.resource_mut::<editor::model::EditorState>().debug = false;
        state.close = false;
        world.insert_resource(state);
        return;
    }
    let scene = world.resource::<BevyZeroverseConfig>().scene_type.clone();
    if state.scene.as_ref() != Some(&scene) {
        if let Some(root) = state.root.take() {
            world.despawn(root);
        }
        state.scene = Some(scene.clone());
        state.apply = false;
    }
    macro_rules! resource {
        ($operation:ident $(, $argument:expr)?) => {
            match scene {
                ZeroverseSceneType::Object => $operation::<crate::scene::object::ZeroverseObjectSceneSettings>(world $(, $argument)?),
                ZeroverseSceneType::Room => $operation::<ZeroverseRoomSettings>(world $(, $argument)?),
                ZeroverseSceneType::SemanticRoom => $operation::<ZeroverseSemanticRoomSettings>(world $(, $argument)?),
                _ => Err("Use Scene Studio advanced configuration for this scene.".into()),
            }
        };
    }
    if state.apply {
        let result = resource!(apply, &state.draft);
        if let Some(mut text) = state.error.and_then(|e| world.get_mut::<Text>(e)) {
            text.0 = result
                .err()
                .unwrap_or_else(|| "Saved. Press R to regenerate.".into());
        }
        state.apply = false;
    }
    if state.root.is_none() {
        state.draft = resource!(snapshot).unwrap_or_else(|e| e);
        let root = world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px(390),
                    top: px(65),
                    width: px(480),
                    max_height: percent(85),
                    padding: UiRect::all(px(16)),
                    flex_direction: FlexDirection::Column,
                    row_gap: px(10),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.055, 0.075, 0.095)),
                GlobalZIndex(300),
                Name::new("resource_diagnostics"),
            ))
            .id();
        state.root = Some(root);
        let text = |world: &mut World, value: &str| {
            world
                .spawn((Text::new(value), ThemedText, ChildOf(root)))
                .id()
        };
        text(world, &format!("{scene:?} geometry controls"));
        text(
            world,
            "Save geometry controls, then regenerate. These edits are not included in share URLs.",
        );
        let container = world.spawn_scene(bsn! {
            @FeathersTextInputContainer Node { width: percent(100), height: px(420) } ChildOf(root)
        }).expect("resource input container").id();
        world.spawn_scene(bsn! {
            @FeathersTextInput Node { width: percent(100), height: px(410) } ChildOf(container)
            on(|event: On<TextEditChange>, texts: Query<&EditableText>, mut state: ResMut<State>| {
                if let Ok(text) = texts.get(event.event_target()) { state.draft = text.value().to_string(); }
            })
        }).expect("resource input").insert(EditableText { allow_newlines: true, visible_lines: Some(20.), ..EditableText::new(state.draft.clone()) });
        state.error = Some(text(world, ""));
        world.spawn_scene(bsn! {
            @FeathersButton { @caption: bsn! { Text("Save geometry") ThemedText } } ChildOf(root)
            on(|_: On<Activate>, mut state: ResMut<State>| { state.apply = true; })
        }).expect("save resource");
        world
            .spawn_scene(bsn! {
                @FeathersButton { @caption: bsn! { Text("Close") ThemedText } } ChildOf(root)
                on(|_: On<Activate>, mut state: ResMut<State>| { state.close = true; })
            })
            .expect("close diagnostics");
    }
    world.insert_resource(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_geometry_resources_round_trip_through_controls() {
        let mut world = World::new();
        world.insert_resource(AppTypeRegistry::new_with_derived_types());
        fn round_trip<T>(world: &mut World)
        where
            T: Resource<Mutability = bevy::ecs::component::Mutable> + Reflect + Clone + Default,
        {
            world.insert_resource(T::default());
            let before = snapshot::<T>(world).unwrap();
            apply::<T>(world, &before).unwrap();
            assert_eq!(snapshot::<T>(world).unwrap(), before);
        }
        round_trip::<crate::scene::object::ZeroverseObjectSceneSettings>(&mut world);
        round_trip::<ZeroverseRoomSettings>(&mut world);
        round_trip::<ZeroverseSemanticRoomSettings>(&mut world);
    }
    #[derive(Resource, Reflect, Clone, Debug, PartialEq)]
    struct Geometry {
        count: u32,
        enabled: bool,
    }
    #[derive(Resource, Reflect, Clone)]
    struct DifferentGeometry {
        count: u32,
        enabled: bool,
    }
    #[test]
    fn reflected_geometry_edits_roundtrip_and_reject_invalid_input_atomically() {
        let mut app = App::new();
        app.register_type::<Geometry>()
            .register_type::<DifferentGeometry>()
            .insert_resource(Geometry {
                count: 2,
                enabled: true,
            })
            .insert_resource(DifferentGeometry {
                count: 9,
                enabled: false,
            });
        let draft = snapshot::<Geometry>(app.world()).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&draft).unwrap();
        let value = json.as_object_mut().unwrap().values_mut().next().unwrap();
        value["count"] = serde_json::json!(6);
        apply::<Geometry>(app.world_mut(), &json.to_string()).unwrap();
        assert_eq!(app.world().resource::<Geometry>().count, 6);
        let different = snapshot::<DifferentGeometry>(app.world()).unwrap();
        assert!(apply::<Geometry>(app.world_mut(), &different).is_err());
        assert!(apply::<Geometry>(app.world_mut(), &(draft + " trailing")).is_err());
        assert_eq!(
            *app.world().resource::<Geometry>(),
            Geometry {
                count: 6,
                enabled: true
            }
        );
    }
}
