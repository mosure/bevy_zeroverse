//! Retained accordion navigation. Hiding a group never replaces its controls.
#[cfg(test)]
mod tests;
use super::{
    fields::{self, Item, Section},
    model::{EditorState, Page, ViewState},
    widgets, Action, Actions, INK, MUTED,
};
use bevy::{
    feathers::{controls::*, theme::ThemedText},
    prelude::*,
    ui_widgets::{Activate, ScrollArea},
};

pub(super) struct Group {
    pub section: Section,
    pub items: Vec<Item>,
    pub default_open: bool,
}
impl Group {
    pub fn is_open(&self, view: &ViewState) -> bool {
        is_open(self.section.id, self.default_open, view)
    }
}

pub(super) fn groups(state: &EditorState) -> Vec<Group> {
    let mut result: Vec<Group> = Vec::new();
    for item in fields::items(state) {
        if let Item::Section(section) = item {
            result.push(Group {
                section,
                items: Vec::new(),
                default_open: false,
            });
        } else if let Some(group) = result.last_mut() {
            group.items.push(item);
        } else {
            debug_assert!(false, "Every Studio control needs an owning section");
        }
    }
    result.retain(|group| !group.items.is_empty());
    if let Some(first) = result.first_mut() {
        first.default_open = true;
    }
    result
}

/// The three formerly collapsible headings remain readable in saved URLs/API actions.
fn canonical(key: &str) -> &str {
    match key {
        "ADVANCED RIG" => "cameras.overlap",
        "PROMPT DISTRIBUTION" => "people.prompts",
        "OPTICAL FLOW" => "view.flow",
        _ => key,
    }
}
fn is_open(id: &str, default_open: bool, view: &ViewState) -> bool {
    !view.closed_sections.contains(id)
        && (default_open || view.expanded.iter().any(|key| canonical(key) == id))
}
fn set_open(view: &mut ViewState, group: &Group, open: bool) {
    let id = group.section.id;
    view.expanded.retain(|key| canonical(key) != id);
    view.closed_sections.remove(id);
    if open && !group.default_open {
        view.expanded.insert(id.into());
    } else if !open && group.default_open {
        view.closed_sections.insert(id.into());
    }
}
pub(super) fn toggle(state: &mut EditorState, key: &str) {
    if let Some(group) = groups(state)
        .into_iter()
        .find(|g| g.section.id == canonical(key))
    {
        let open = !group.is_open(&state.view);
        set_open(&mut state.view, &group, open);
    }
}
pub(super) fn open(state: &mut EditorState, key: &str) -> bool {
    if let Some(group) = groups(state)
        .into_iter()
        .find(|g| g.section.id == canonical(key))
    {
        set_open(&mut state.view, &group, true);
        true
    } else {
        false
    }
}
pub(super) fn toggle_all(state: &mut EditorState) {
    let groups = groups(state);
    let open = !groups.iter().all(|g| g.is_open(&state.view));
    for group in groups {
        set_open(&mut state.view, &group, open);
    }
}

#[derive(Resource, Default)]
pub(super) struct Navigation {
    pub positions: std::collections::HashMap<Page, ScrollPosition>,
    pub jump: Option<String>,
}
#[derive(Component)]
pub(super) struct GroupUi {
    pub section: Section,
    pub default_open: bool,
    pub body: Entity,
    pub header: Entity,
    chevron: Entity,
}
#[derive(Component, Clone, Default)]
struct ExpandCaption;

pub(super) fn toolbar(world: &mut World, parent: Entity, groups: &[Group]) {
    let row = world
        .spawn((
            Node {
                width: percent(100),
                column_gap: px(6),
                align_items: AlignItems::Center,
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(parent),
        ))
        .id();
    let menu = world
        .spawn_scene(bsn! {
            @FeathersMenu Node { min_width:px(0), flex_grow:1., height:px(30) } ChildOf(row)
        })
        .expect("group navigation menu")
        .id();
    world
        .spawn_scene(bsn! {
            @FeathersMenuButton { @caption:bsn!{ Text("Jump to group…") ThemedText } }
            Node { width:percent(100), min_height:px(30), align_items:AlignItems::Center }
            AccessibleLabel("Jump to settings group") ChildOf(menu)
        })
        .expect("group navigation button");
    let popup = world
        .spawn_scene(bsn! {
            @FeathersMenuPopup ChildOf(menu) GlobalZIndex(200)
        })
        .expect("group navigation popup")
        .id();
    for group in groups {
        let id = group.section.id;
        let title = group.section.title;
        world
            .spawn_scene(bsn! {
                @FeathersMenuItem { @caption:bsn!{ Text(title) ThemedText } } ChildOf(popup)
                on(move |_:On<Activate>, mut actions:ResMut<Actions>| {
                    actions.0.push(Action::JumpToSection(id.into()));
                })
            })
            .expect("group navigation item");
    }
    world.spawn_scene(bsn! {
        @FeathersButton { @variant:ButtonVariant::Plain,
            @caption:bsn!{ Text("Expand all") ThemedText ExpandCaption }
        }
        Node { min_height:px(30), flex_shrink:0. } ChildOf(row)
        AccessibleLabel("Expand or collapse all settings groups")
        on(|_:On<Activate>, mut actions:ResMut<Actions>| { actions.0.push(Action::ExpandSections); })
    }).expect("expand groups button");
}

pub(super) fn card(
    world: &mut World,
    parent: Entity,
    group: &Group,
    state: &EditorState,
) -> Entity {
    let open = group.is_open(&state.view);
    let root = world
        .spawn((
            Node {
                width: percent(100),
                flex_direction: FlexDirection::Column,
                flex_shrink: 0.,
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(6)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.075, 0.096, 0.121)),
            BorderColor::all(Color::srgb(0.16, 0.20, 0.25)),
            Name::new(group.section.id),
            ChildOf(parent),
        ))
        .id();
    let header = widgets::button(
        world,
        root,
        group.section.title,
        Action::Section(group.section.id.into()),
        false,
    );
    world.entity_mut(header).insert((
        ButtonVariant::Plain,
        Node {
            width: percent(100),
            height: px(40),
            min_height: px(40),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Start,
            column_gap: px(5),
            padding: UiRect::horizontal(px(10)),
            flex_shrink: 0.,
            ..default()
        },
        AccessibleLabel(group.section.title.into()),
    ));
    let chevron = world
        .spawn_scene(bsn! {
            @bevy::feathers::display::icon(bevy::feathers::constants::icons::CHEVRON_RIGHT)
            Node { width:px(12), height:px(12), flex_shrink:0. }
            UiTransform { rotation:chevron_rotation(open) }
            Pickable::IGNORE ChildOf(header)
        })
        .expect("group chevron")
        .id();
    world.entity_mut(header).insert_children(0, &[chevron]);
    if let Some(mut node) = world.get_mut::<bevy::a11y::AccessibilityNode>(header) {
        node.set_expanded(open);
    }
    world.spawn((
        Node {
            flex_grow: 1.,
            ..default()
        },
        ChildOf(header),
    ));
    let count = group
        .items
        .iter()
        .filter(|item| matches!(item, Item::Field(_)))
        .count();
    if count > 0 {
        widgets::text(world, header, count.to_string(), 11., MUTED);
    }
    let body = widgets::column(world, root, 14.);
    world.get_mut::<Node>(body).unwrap().padding = UiRect::all(px(12));
    world.get_mut::<Node>(body).unwrap().display = if open { Display::Flex } else { Display::None };
    widgets::text(world, body, group.section.help, 11.5, MUTED);
    world.entity_mut(root).insert(GroupUi {
        section: group.section,
        default_open: group.default_open,
        body,
        header,
        chevron,
    });
    body
}
fn chevron_rotation(open: bool) -> bevy::math::Rot2 {
    bevy::math::Rot2::turn_fraction(if open { 0.25 } else { 0. })
}

pub(super) fn synchronize(world: &mut World, state: &EditorState) {
    let cards: Vec<_> = world
        .query::<&GroupUi>()
        .iter(world)
        .map(|g| (g.section, g.default_open, g.body, g.header, g.chevron))
        .collect();
    let mut all_open = !cards.is_empty();
    for (section, default_open, body, header, chevron) in cards {
        let open = is_open(section.id, default_open, &state.view);
        all_open &= open;
        let display = if open { Display::Flex } else { Display::None };
        if let Some(mut node) = world.get_mut::<Node>(body) {
            if node.display != display {
                node.display = display;
            }
        }
        if let Some(mut transform) = world.get_mut::<UiTransform>(chevron) {
            let rotation = chevron_rotation(open);
            if transform.rotation != rotation {
                transform.rotation = rotation;
            }
        }
        if let Some(mut node) = world.get_mut::<bevy::a11y::AccessibilityNode>(header) {
            if node.is_expanded() != Some(open) {
                node.set_expanded(open);
            }
        }
        // Keep focus out of hidden controls after programmatic collapse.
        if !open {
            let focused = world
                .get_resource::<bevy::input_focus::InputFocus>()
                .and_then(bevy::input_focus::InputFocus::get);
            if focused.is_some_and(|e| ancestors(world, e).any(|p| p == body)) {
                world
                    .resource_mut::<bevy::input_focus::InputFocus>()
                    .set(header, bevy::input_focus::FocusCause::Navigated);
            }
        }
    }
    let value = if all_open {
        "Collapse all"
    } else {
        "Expand all"
    };
    for mut text in world
        .query_filtered::<&mut Text, With<ExpandCaption>>()
        .iter_mut(world)
    {
        if text.0 != value {
            text.0 = value.into();
        }
    }
}

/// Run after layout: a previously collapsed group has no usable position before this point.
pub(super) fn scroll_to_group(world: &mut World) {
    let Some(key) = world
        .get_resource_mut::<Navigation>()
        .and_then(|mut n| n.jump.take())
    else {
        return;
    };
    let Some(header) = world
        .query::<&GroupUi>()
        .iter(world)
        .find(|g| g.section.id == canonical(&key))
        .map(|g| g.header)
    else {
        return;
    };
    let Some(scroll) = ancestors(world, header).find(|e| world.get::<ScrollArea>(*e).is_some())
    else {
        return;
    };
    let Some(target_node) = world.get::<ComputedNode>(header) else {
        return;
    };
    let Some(target_transform) = world.get::<UiGlobalTransform>(header) else {
        return;
    };
    let target_top = (target_transform.translation.y - target_node.size().y * 0.5)
        * target_node.inverse_scale_factor;
    let Some(scroll_node) = world.get::<ComputedNode>(scroll) else {
        return;
    };
    let Some(scroll_transform) = world.get::<UiGlobalTransform>(scroll) else {
        return;
    };
    let scroll_top = (scroll_transform.translation.y - scroll_node.size().y * 0.5)
        * scroll_node.inverse_scale_factor;
    let max_scroll = ((scroll_node.content_size().y - scroll_node.size().y)
        * scroll_node.inverse_scale_factor)
        .max(0.);
    if let Some(mut position) = world.get_mut::<ScrollPosition>(scroll) {
        position.0.y = (position.0.y + target_top - scroll_top).clamp(0., max_scroll);
    }
    if let Some(mut focus) = world.get_resource_mut::<bevy::input_focus::InputFocus>() {
        focus.set(header, bevy::input_focus::FocusCause::Navigated);
    }
}

pub(super) fn legend(world: &mut World, parent: Entity) {
    let indices = world
        .get_resource::<crate::render::co_visibility::CoVisibilityLegend>()
        .map(|l| l.camera_indices.clone())
        .unwrap_or_default();
    if indices.is_empty() {
        widgets::text(
            world,
            parent,
            "Available after capture cameras are ready.",
            12.,
            MUTED,
        );
    }
    for (bit, index) in indices.iter().enumerate().take(16) {
        let rgb = crate::render::co_visibility::camera_color(bit, indices.len().min(16));
        let row = world
            .spawn((
                Node {
                    width: percent(100),
                    column_gap: px(8),
                    align_items: AlignItems::Center,
                    ..default()
                },
                ChildOf(parent),
            ))
            .id();
        world.spawn((
            Node {
                width: px(20),
                height: px(12),
                ..default()
            },
            BackgroundColor(Color::srgb_u8(rgb[0], rgb[1], rgb[2])),
            ChildOf(row),
        ));
        widgets::text(
            world,
            row,
            format!("View {index} · bit 0x{:04X}", 1u16 << bit),
            12.,
            INK,
        );
    }
}

fn ancestors(world: &World, entity: Entity) -> impl Iterator<Item = Entity> + '_ {
    std::iter::successors(
        world.get::<ChildOf>(entity).map(ChildOf::parent),
        move |e| world.get::<ChildOf>(*e).map(ChildOf::parent),
    )
}
