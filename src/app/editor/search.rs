//! Cross-page navigation over the same context-sensitive catalog as the controls.
use super::{
    fields::{items_for, Item, Kind},
    model::{EditorState, Page},
    widgets, Action, Actions, MUTED,
};
use bevy::{
    feathers::controls::*,
    prelude::*,
    text::{EditableText, TextEditChange},
};

#[derive(Resource, Default)]
pub(super) struct Search(String);
#[derive(Component)]
struct Results;
#[derive(Component)]
pub(super) struct FieldRow(pub String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Hit {
    pub page: Page,
    pub section: &'static str,
    pub path: String,
    label: String,
}

fn normalized(text: &str) -> String {
    text.to_lowercase().replace(['-', '_', '/'], " ")
}

fn find(state: &EditorState, input: &str) -> Vec<Hit> {
    let words = normalized(input);
    let terms = words.split_whitespace().collect::<Vec<_>>();
    if terms.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for page in Page::ALL {
        let mut group = None;
        for item in items_for(state, page) {
            match item {
                Item::Section(s) => group = Some(s),
                Item::Field(f) => {
                    let Some(section) = group else { continue };
                    let label = normalized(&f.label);
                    let path = normalized(&f.path);
                    let mut extra = normalized(&format!("{} {}", page.label(), section.title));
                    if let Kind::Choice(choices) = f.kind {
                        for (option, _) in choices {
                            extra.push(' ');
                            extra.push_str(&normalized(&option));
                        }
                    }
                    let score: Option<usize> = terms
                        .iter()
                        .map(|term| {
                            if label.contains(term) {
                                Some(0)
                            } else if path.contains(term) {
                                Some(1)
                            } else if extra.contains(term) {
                                Some(2)
                            } else {
                                None
                            }
                        })
                        .sum();
                    if let Some(score) = score {
                        hits.push((
                            score,
                            Hit {
                                page,
                                section: section.id,
                                path: f.path,
                                label: f.label,
                            },
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    hits.sort_by_key(|(score, _)| *score);
    hits.truncate(4);
    hits.into_iter().map(|(_, hit)| hit).collect()
}

pub(super) fn bar(world: &mut World, panel: Entity, state: &EditorState) {
    world.init_resource::<Search>();
    let text = world.resource::<Search>().0.clone();
    let row = world
        .spawn((
            Node {
                width: percent(100),
                column_gap: px(8.),
                align_items: AlignItems::Center,
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(panel),
        ))
        .id();
    widgets::text(world, row, "Find setting", 12., MUTED);
    let container=world.spawn_scene(bsn!{
        @FeathersTextInputContainer Node {height:px(30), min_width:px(0), flex_grow:1.} ChildOf(row)
    }).expect("settings search container").id();
    world.spawn_scene(bsn!{
        @FeathersTextInput { @visible_width:24f32 }
        Node { width:percent(100), height:px(24), min_width:px(0), flex_grow:1. }
        AccessibleLabel("Find a setting across all tabs") ChildOf(container)
        on(|event:On<TextEditChange>, inputs:Query<&EditableText>, mut actions: ResMut<Actions>| {
            if let Ok(input)=inputs.get(event.event_target()) { actions.0.push(Action::Search(input.value().to_string())); }
        })
    }).expect("settings search").insert(widgets::editable(text.clone(),false));
    let results = widgets::column(world, panel, 2.);
    world.entity_mut(results).insert(Results);
    update(world, state, text);
}

/// Replace the results only. The text field, focus and scroll remain live while typing.
pub(super) fn update(world: &mut World, state: &EditorState, text: String) {
    world.init_resource::<Search>();
    world.resource_mut::<Search>().0 = text;
    let Ok(entity) = world
        .query_filtered::<Entity, With<Results>>()
        .single(world)
    else {
        return;
    };
    world.entity_mut(entity).despawn_related::<Children>();
    let text = &world.resource::<Search>().0;
    let active = !text.trim().is_empty();
    let hits = find(state, text);
    world.get_mut::<Node>(entity).unwrap().display =
        if active { Display::Flex } else { Display::None };
    if active && hits.is_empty() {
        widgets::text(
            world,
            entity,
            "No matching settings in this scene and viewport.",
            12.,
            MUTED,
        );
    }
    for hit in hits {
        let caption = format!("{}  ·  {}", hit.page.label(), hit.label);
        let button = widgets::button(
            world,
            entity,
            &caption,
            Action::FindSetting(hit.page, hit.section.into(), hit.path),
            false,
        );
        world.entity_mut(button).insert(ButtonVariant::Plain);
    }
}

pub(super) fn clear(world: &mut World) {
    world.init_resource::<Search>();
    world.resource_mut::<Search>().0.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typing_keeps_the_input_and_selecting_a_hit_opens_its_control() {
        use crate::app::editor::{tests::studio_app, update};
        let mut app = studio_app(Page::Scene);
        let world = app.world_mut();
        let input = world
            .query::<(Entity, &AccessibleLabel)>()
            .iter(world)
            .find(|(_, label)| label.0 == "Find a setting across all tabs")
            .unwrap()
            .0;
        let draft = world.resource::<EditorState>().draft.clone();
        for query in ["t", "ta", "taa"] {
            world
                .resource_mut::<Actions>()
                .0
                .push(Action::Search(query.into()));
            update(world);
            assert!(world.entity(input).contains::<EditableText>());
        }
        let hit = find(world.resource::<EditorState>(), "taa").remove(0);
        world.resource_mut::<Actions>().0.push(Action::FindSetting(
            hit.page,
            hit.section.into(),
            hit.path.clone(),
        ));
        update(world);
        assert!(world.resource::<Search>().0.is_empty());
        let state = world.resource::<EditorState>();
        assert_eq!(state.draft, draft);
        assert_eq!(state.view.page, Page::View);
        assert!(state.view.expanded.contains("view.rgb"));
        super::super::navigation::scroll_to_group(world);
        let focus = world
            .resource::<bevy::input_focus::InputFocus>()
            .get()
            .unwrap();
        assert!(world
            .entity(focus)
            .contains::<bevy::ui_widgets::MenuButton>());
        assert_eq!(
            world.get::<AccessibleLabel>(focus).unwrap().0,
            "Preview anti-aliasing"
        );
    }

    #[test]
    fn searches_visible_settings_across_tabs_and_choice_names() {
        let mut state = EditorState::new(&crate::app::BevyZeroverseConfig::default());
        state.view.page = Page::Scene;
        for query in ["TAA", "msaa 8", "anti aliasing", "preview_aa"] {
            let hits = find(&state, query);
            assert!(
                hits.iter().any(|h| h.page == Page::View
                    && h.section == "view.rgb"
                    && h.path == "/preview_aa"),
                "{query}"
            );
        }
        assert_eq!(find(&state, "ssaa")[0].path, "/ssaa_factor");
        state.draft["room_schematic"] = serde_json::json!(true);
        assert!(find(&state, "taa").is_empty());
        assert_eq!(state.view.page, Page::Scene);
    }
}
