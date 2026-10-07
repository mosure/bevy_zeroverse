//! Retained Feathers controls: values update in place, so dragging and typing keep focus.
use super::{
    fields::{Field, Kind},
    model::EditorState,
    Action, Actions,
};
use bevy::{
    feathers::{controls::*, theme::ThemedText},
    prelude::*,
    text::{EditableText, TextEditChange},
    ui::{Checked, InteractionDisabled},
    ui_widgets::{Activate, SliderPrecision, SliderStep, SliderValue, ValueChange},
};
use serde_json::json;

#[derive(Component, Clone)]
pub struct Binding(pub Field);
#[derive(Component)]
pub struct ChoiceCaption(pub String);

pub fn text(
    world: &mut World,
    parent: Entity,
    value: impl Into<String>,
    size: f32,
    color: Color,
) -> Entity {
    let font = world
        .resource::<AssetServer>()
        .load(bevy::feathers::constants::fonts::REGULAR);
    world
        .spawn((
            Text::new(value),
            TextFont {
                font: font.into(),
                font_size: size.into(),
                ..default()
            },
            TextColor(color),
            Node {
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(parent),
        ))
        .id()
}
pub fn column(world: &mut World, parent: Entity, gap: f32) -> Entity {
    world
        .spawn((
            Node {
                width: percent(100),
                flex_direction: FlexDirection::Column,
                row_gap: px(gap),
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(parent),
        ))
        .id()
}
pub fn button(
    world: &mut World,
    parent: Entity,
    label: &str,
    action: Action,
    primary: bool,
) -> Entity {
    let caption = label.to_owned();
    let variant = if primary {
        ButtonVariant::Primary
    } else {
        ButtonVariant::Normal
    };
    world
        .spawn_scene(bsn! {
            @FeathersButton { @variant:variant }
            Node { min_height:px(32), flex_shrink:0. }
            AccessibleLabel({caption.clone()})
            ChildOf(parent)
            on(move |_e:On<Activate>,mut actions:ResMut<Actions>| {actions.0.push(action.clone());})
            Children[(Text(caption) ThemedText)]
        })
        .expect("button scene")
        .id()
}
pub fn control(world: &mut World, parent: Entity, field: Field, state: &EditorState) {
    let row = column(world, parent, 5.);
    let value = super::value(state, &field.path);
    let label = field.label.clone();
    if !matches!(field.kind, Kind::Toggle) {
        text(world, row, &label, 14., super::INK);
    }
    match &field.kind {
        Kind::Number(min, max, step) => {
            let (min, max, step) = (*min, *max, *step);
            let val = value.as_f64().unwrap_or(min as f64) as f32;
            let entity = world.spawn_scene(bsn! {
                @FeathersSlider { @value:val, @min:min, @max:max }
                Node {height:px(29),flex_shrink:0.}
                SliderStep(step) SliderPrecision({if step>=1. {0}else if step>=0.01 {2}else{3}})
                AccessibleLabel(label) ChildOf(row)
                on(|e:On<ValueChange<f32>>,bindings:Query<&Binding>,mut actions:ResMut<Actions>| {

                    if let Ok(binding)=bindings.get(e.source) {

                        actions.0.push(Action::Set(binding.0.path.clone(),json!(e.value)));
                    }
                })
            }).expect("slider scene").insert(Binding(field.clone())).id();
            if field.path == "@editor_fov" && state.view.camera.is_none() {
                world.entity_mut(entity).insert(InteractionDisabled);
            }
        }
        Kind::Toggle => {
            let entity=world.spawn_scene(bsn! {
                @FeathersCheckbox { @caption:bsn!{Text({label.clone()}) ThemedText} }
                Node {min_height:px(28),flex_shrink:0.}
                AccessibleLabel(label) ChildOf(row)
                on(|e:On<ValueChange<bool>>,bindings:Query<&Binding>,mut actions:ResMut<Actions>| {if let Ok(b)=bindings.get(e.source) {actions.0.push(Action::Set(b.0.path.clone(),json!(e.value)));}})
            }).expect("checkbox scene").insert(Binding(field.clone())).id();
            if value == true {
                world.entity_mut(entity).insert(Checked);
            }
            if field.path == "@motion" && !cfg!(feature = "human_motion") {
                world.entity_mut(entity).insert(InteractionDisabled);
            }
        }
        Kind::Choice(options) => {
            let caption = options
                .iter()
                .find(|(_, v)| *v == value)
                .map(|(s, _)| s.clone())
                .unwrap_or_else(|| "Custom".into());
            let menu = world
                .spawn_scene(bsn! { @FeathersMenu ChildOf(row) })
                .expect("menu")
                .id();
            let trigger=world.spawn_scene(bsn! { @FeathersMenuButton Node {width:percent(100), min_height:px(30)} AccessibleLabel(label) ChildOf(menu) }).expect("menu button").id();
            world.spawn((
                Text::new(caption),
                ThemedText,
                ChoiceCaption(field.path.clone()),
                ChildOf(trigger),
            ));
            let popup = world
                .spawn_scene(bsn! { @FeathersMenuPopup ChildOf(menu) GlobalZIndex(200) })
                .expect("menu popup")
                .id();
            for (label, value) in options {
                let action = Action::Set(field.path.clone(), value.clone());
                world.spawn_scene(bsn! {
                    @FeathersMenuItem { @caption:bsn!{Text({label.clone()}) ThemedText} }
                    ChildOf(popup)
                    on(move |_e:On<Activate>,mut actions:ResMut<Actions>|{actions.0.push(action.clone());})
                }).expect("menu item");
            }
        }
        Kind::Text | Kind::Json => {
            let json = matches!(field.kind, Kind::Json);
            let height = if json { 112. } else { 32. };
            let initial = state
                .invalid_text
                .get(&field.path)
                .cloned()
                .unwrap_or_else(|| input_text(&value, json));
            let container=world.spawn_scene(bsn! { @FeathersTextInputContainer Node {height:px(height),min_height:px(height),width:percent(100)} ChildOf(row) }).expect("text container").id();
            let editable = editable(initial, json);
            world.spawn_scene(bsn! {
                @FeathersTextInput { @visible_width:40f32 }
                // Explicit width avoids a zero-width intrinsic input inside the flex container.
                Node {width:percent(100),height:px(height-8.),min_width:px(0),flex_grow:1.}
                AccessibleLabel(label) ChildOf(container)
                on(|e:On<TextEditChange>,inputs:Query<(&Binding,&EditableText)>,mut actions:ResMut<Actions>| {
                    if let Ok((b,t))=inputs.get(e.event_target()) {actions.0.push(Action::Input(b.0.path.clone(),t.value().to_string(),matches!(b.0.kind,Kind::Json)));}
                })
            }).expect("text input").insert((Binding(field.clone()),editable));
        }
    }
    if !field.help.is_empty() {
        text(world, row, &field.help, 11.5, super::MUTED);
    }
}
pub fn synchronize(world: &mut World, state: &EditorState) {
    let bindings: Vec<_> = world
        .query::<(Entity, &Binding)>()
        .iter(world)
        .map(|(e, b)| (e, b.0.clone()))
        .collect();
    for (entity, field) in bindings {
        let val = super::value(state, &field.path);
        match field.kind {
            Kind::Toggle => {
                let checked = world.get::<Checked>(entity).is_some();
                if val == true && !checked {
                    world.entity_mut(entity).insert(Checked);
                } else if val != true && checked {
                    world.entity_mut(entity).remove::<Checked>();
                }
            }
            Kind::Number(..) => {
                if field.path == "@editor_fov"
                    && state.view.camera.is_some()
                    && world.get::<InteractionDisabled>(entity).is_some()
                {
                    world.entity_mut(entity).remove::<InteractionDisabled>();
                }
                if let (Some(v), Some(slider)) = (val.as_f64(), world.get::<SliderValue>(entity)) {
                    if (slider.0 - v as f32).abs() > 1e-6 {
                        world.entity_mut(entity).insert(SliderValue(v as f32));
                    }
                }
            }
            Kind::Text | Kind::Json => {
                let focused = world
                    .get_resource::<bevy::input_focus::InputFocus>()
                    .and_then(bevy::input_focus::InputFocus::get)
                    == Some(entity);
                if !focused && !state.input_errors.contains_key(&field.path) {
                    let desired = input_text(&val, matches!(field.kind, Kind::Json));
                    if world
                        .get::<EditableText>(entity)
                        .is_some_and(|t| t.value() != desired.as_str())
                    {
                        world
                            .entity_mut(entity)
                            .insert(editable(desired, matches!(field.kind, Kind::Json)));
                    }
                }
            }
            _ => {}
        }
    }
    let captions: Vec<_> = world
        .query::<(Entity, &ChoiceCaption)>()
        .iter(world)
        .map(|(e, c)| (e, c.0.clone()))
        .collect();
    let fields = super::fields::items(state);
    for (e, path) in captions {
        for item in &fields {
            if let super::fields::Item::Field(Field {
                path: p,
                kind: Kind::Choice(options),
                ..
            }) = item
            {
                if *p == path {
                    let value = super::value(state, p);
                    let label = options
                        .iter()
                        .find(|(_, v)| *v == value)
                        .map(|(l, _)| l.as_str())
                        .unwrap_or("Custom");
                    if let Some(mut t) = world.get_mut::<Text>(e) {
                        if t.0 != label {
                            t.0 = label.to_owned();
                        }
                    }
                }
            }
        }
    }
}

fn input_text(value: &serde_json::Value, json: bool) -> String {
    if json {
        serde_json::to_string_pretty(value).unwrap()
    } else if value.is_null() {
        String::new()
    } else if let Some(s) = value.as_str() {
        s.to_owned()
    } else {
        value.to_string()
    }
}

fn editable(initial: String, json: bool) -> EditableText {
    let mut text = EditableText::new(initial);
    text.visible_width = Some(40.);
    text.visible_lines = Some(if json { 6. } else { 1. });
    text.allow_newlines = json;
    text.cursor_width = 0.3;
    text.pending_edits.clear();
    text.queue_edit(bevy::text::TextEdit::TextStart(false));
    text
}
