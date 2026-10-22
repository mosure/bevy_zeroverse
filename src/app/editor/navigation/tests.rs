use super::*;
use crate::app::{
    editor::{edit, tests::studio_app, update},
    BevyZeroverseConfig,
};
use serde_json::json;

fn group_for<'a>(groups: &'a [Group], path: &str) -> &'a str {
    groups
        .iter()
        .find(|g| {
            g.items
                .iter()
                .any(|item| matches!(item, Item::Field(f) if f.path == path))
        })
        .unwrap_or_else(|| panic!("Missing control {path}"))
        .section
        .id
}

#[test]
fn control_catalog_has_unique_groups_and_task_specific_ownership() {
    let mut state = EditorState::new(&crate::app::web_viewer_defaults());
    edit(&mut state, "@motion", json!(true)).unwrap();
    edit(&mut state, "@indoor_motion", json!(true)).unwrap();
    edit(&mut state, "@handheld", json!(true)).unwrap();
    edit(&mut state, "@mixture", json!(true)).unwrap();
    for scene in [
        "ProceduralIndoor",
        "Object",
        "Room",
        "SemanticRoom",
        "CornellCube",
        "Human",
        "Custom",
    ] {
        state.draft["scene_type"] = json!(scene);
        for page in Page::ALL {
            state.view.page = page;
            let groups = groups(&state);
            let mut ids = std::collections::HashSet::new();
            let mut paths = std::collections::HashSet::new();
            for group in &groups {
                assert!(
                    ids.insert(group.section.id),
                    "duplicate {scene}/{page:?}: {}",
                    group.section.id
                );
                assert!(!group.items.is_empty());
                for item in &group.items {
                    if let Item::Field(field) = item {
                        assert!(paths.insert(&field.path));
                    }
                }
            }
            assert_eq!(groups.iter().filter(|g| g.default_open).count(), 1);
            let catalog_fields = fields::items(&state)
                .iter()
                .filter(|i| matches!(i, Item::Field(_)))
                .count();
            assert_eq!(paths.len(), catalog_fields);
        }
    }
    state.draft["scene_type"] = json!("ProceduralIndoor");
    for (page, path, expected) in [
        (Page::Scene, "/indoor_density", "scene.clutter"),
        (Page::Cameras, "@baseline", "cameras.rig"),
        (Page::Cameras, "@fisheye_probability", "cameras.fisheye"),
        (
            Page::Cameras,
            "/indoor_camera/lenses/circular_probability",
            "cameras.fisheye-shape",
        ),
        (Page::Cameras, "@camera_motion_percent", "cameras.motion"),
        (
            Page::Appearance,
            "/indoor_appearance/exposure_ev100_offset",
            "appearance.lighting",
        ),
        (Page::People, "@motion", "people.human-motion"),
        (
            Page::People,
            "/indoor_motion/prop_fraction",
            "people.motion-selection",
        ),
        (Page::People, "/indoor_motion/fan_turns/1", "people.fans"),
        (Page::View, "/annotation_glass", "view.geometry"),
        (Page::View, "/orbit_smoothness", "view.navigation"),
        (Page::Advanced, "/width", "advanced.capture"),
        (Page::Advanced, "@config", "advanced.config"),
    ] {
        state.view.page = page;
        assert_eq!(group_for(&groups(&state), path), expected);
    }
    if cfg!(feature = "human_motion") {
        state.view.page = Page::People;
        assert_eq!(
            group_for(&groups(&state), "/human_motion/diffusion_steps"),
            "people.model"
        );
        assert_eq!(
            group_for(&groups(&state), "/human_motion/fraction"),
            "people.human-motion"
        );
    }
    // These modes hide complete groups, never leaving orphan controls under another heading.
    state.view.page = Page::Cameras;
    edit(&mut state, "@ceiling_mount", json!(true)).unwrap();
    assert_eq!(
        group_for(&groups(&state), "/indoor_camera/duration_seconds"),
        "cameras.motion"
    );
    assert!(groups(&state)
        .iter()
        .any(|g| g.section.id == "cameras.motion"));
    assert_eq!(
        group_for(
            &groups(&state),
            "/indoor_camera/ceiling_mount/motion_fraction"
        ),
        "cameras.mount-motion"
    );
    state.view.page = Page::View;
    edit(&mut state, "@viewport", json!("schematic")).unwrap();
    assert!(!groups(&state).iter().any(|g| matches!(
        g.section.id,
        "view.overlays" | "view.geometry" | "view.flow" | "view.navigation"
    )));
}

#[test]
fn group_state_roundtrips_without_changing_generation_and_accepts_old_links() {
    let config = BevyZeroverseConfig {
        scene_type: crate::scene::ZeroverseSceneType::ProceduralIndoor,
        ..crate::app::web_viewer_defaults()
    };
    let mut state = EditorState::new(&config);
    state.view.page = Page::Cameras;
    let draft = state.draft.clone();
    toggle(&mut state, "cameras.rig");
    open(&mut state, "cameras.fisheye");
    open(&mut state, "ADVANCED RIG");
    assert!(state.view.closed_sections.contains("cameras.rig"));
    assert!(state.view.expanded.contains("cameras.overlap"));
    let view = serde_json::to_string(&state.view).unwrap();
    let restored = EditorState::new(&BevyZeroverseConfig {
        viewer_state: Some(view),
        ..config
    });
    assert_eq!(state.draft, draft);
    assert_eq!(restored.applied, draft);
    assert_eq!(restored.view.closed_sections, state.view.closed_sections);
    assert_eq!(restored.view.expanded, state.view.expanded);
    assert_eq!(
        groups(&restored)
            .iter()
            .filter(|g| g.is_open(&restored.view))
            .count(),
        2
    );
    let old = ViewState::parse(r#"{"page":"cameras","expanded":["ADVANCED RIG"]}"#).unwrap();
    assert!(is_open("cameras.overlap", false, &old));
    toggle_all(&mut state);
    assert!(groups(&state).iter().all(|g| g.is_open(&state.view)));
    toggle_all(&mut state);
    assert!(groups(&state).iter().all(|g| !g.is_open(&state.view)));
    assert!(!state.pending());
    assert!(!open(&mut state, "missing-group"));
}

#[test]
fn accordion_buttons_retain_widgets_and_focus_and_restore_each_page_scroll() {
    use bevy::input_focus::{FocusCause, InputFocus};
    let mut app = studio_app(Page::Cameras);
    let world = app.world_mut();
    let (header, body) = world
        .query::<&GroupUi>()
        .iter(world)
        .find(|g| g.section.id == "cameras.fisheye")
        .map(|g| (g.header, g.body))
        .unwrap();
    let slider = world
        .query::<(Entity, &widgets::Binding)>()
        .iter(world)
        .find(|(_, b)| b.0.path == "@fisheye_probability")
        .unwrap()
        .0;
    let scroll = world
        .query_filtered::<Entity, With<ScrollArea>>()
        .single(world)
        .unwrap();
    let revision = world.resource::<EditorState>().revision;
    let draft = world.resource::<EditorState>().draft.clone();
    assert_eq!(world.get::<Node>(body).unwrap().display, Display::None);
    // Trigger the actual Feathers header, just as pointer/keyboard activation does.
    world.trigger(Activate { entity: header });
    world.flush();
    update(world);
    assert_eq!(world.get::<Node>(body).unwrap().display, Display::Flex);
    world
        .resource_mut::<InputFocus>()
        .set(slider, FocusCause::Navigated);
    world.get_mut::<ScrollPosition>(scroll).unwrap().0.y = 350.;
    world.trigger(Activate { entity: header });
    world.flush();
    update(world);
    assert_eq!(world.resource::<InputFocus>().get(), Some(header));
    assert_eq!(world.get::<Node>(body).unwrap().display, Display::None);
    assert!(world.get::<widgets::Binding>(slider).is_some());
    assert_eq!(world.resource::<EditorState>().revision, revision);
    assert_eq!(world.resource::<EditorState>().draft, draft);
    assert_eq!(world.get::<ScrollPosition>(scroll).unwrap().0.y, 350.);
    world
        .resource_mut::<Actions>()
        .0
        .push(Action::Page(Page::Appearance));
    update(world);
    let other_scroll = world
        .query_filtered::<Entity, With<ScrollArea>>()
        .single(world)
        .unwrap();
    assert_eq!(world.get::<ScrollPosition>(other_scroll).unwrap().0.y, 0.);
    world.get_mut::<ScrollPosition>(other_scroll).unwrap().0.y = 100.;
    world
        .resource_mut::<Actions>()
        .0
        .push(Action::Page(Page::Cameras));
    update(world);
    assert_eq!(
        world
            .query_filtered::<&ScrollPosition, With<ScrollArea>>()
            .single(world)
            .unwrap()
            .0
            .y,
        350.
    );
    world
        .resource_mut::<Actions>()
        .0
        .push(Action::Page(Page::Appearance));
    update(world);
    assert_eq!(
        world
            .query_filtered::<&ScrollPosition, With<ScrollArea>>()
            .single(world)
            .unwrap()
            .0
            .y,
        100.
    );
}

#[test]
fn group_jump_uses_layout_coordinates_and_scale_then_clamps_to_content() {
    let mut app = studio_app(Page::Cameras);
    let world = app.world_mut();
    let header = world
        .query::<&GroupUi>()
        .iter(world)
        .find(|g| g.section.id == "cameras.fisheye")
        .unwrap()
        .header;
    let scroll = world
        .query_filtered::<Entity, With<ScrollArea>>()
        .single(world)
        .unwrap();
    world
        .resource_mut::<Actions>()
        .0
        .push(Action::JumpToSection("cameras.fisheye".into()));
    update(world);
    // Layout fixture at 2x physical pixels per logical pixel.
    world.entity_mut(header).insert((
        ComputedNode {
            size: Vec2::new(600., 80.),
            inverse_scale_factor: 0.5,
            ..default()
        },
        UiGlobalTransform::from_translation(Vec2::new(300., 1000.)),
    ));
    world.entity_mut(scroll).insert((
        ComputedNode {
            size: Vec2::new(640., 400.),
            content_size: Vec2::new(640., 1600.),
            inverse_scale_factor: 0.5,
            ..default()
        },
        UiGlobalTransform::from_translation(Vec2::new(320., 600.)),
    ));
    world.get_mut::<ScrollPosition>(scroll).unwrap().0.y = 100.;
    scroll_to_group(world);
    assert_eq!(world.get::<ScrollPosition>(scroll).unwrap().0.y, 380.);
    assert!(world.resource::<Navigation>().jump.is_none());
    assert_eq!(
        world.resource::<bevy::input_focus::InputFocus>().get(),
        Some(header)
    );
    world.resource_mut::<Navigation>().jump = Some("cameras.fisheye".into());
    world
        .entity_mut(header)
        .insert(UiGlobalTransform::from_translation(Vec2::new(300., 10000.)));
    scroll_to_group(world);
    assert_eq!(world.get::<ScrollPosition>(scroll).unwrap().0.y, 600.);
}
