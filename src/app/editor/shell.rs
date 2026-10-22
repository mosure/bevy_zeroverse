//! Responsive editor chrome and a reserved, unobscured scene viewport.
use super::*;
use bevy::{camera::Viewport, ui_widgets::ScrollArea};
#[derive(Component)]
pub(super) struct CapacityEvidence;

pub(super) fn synchronize_evidence(world: &mut World, state: &EditorState) {
    if state.view.page != Page::Advanced {
        return;
    }
    let value = state.capacity_status();
    for mut text in world
        .query_filtered::<&mut Text, With<CapacityEvidence>>()
        .iter_mut(world)
    {
        if text.0 != value {
            text.0.clone_from(&value);
        }
    }
}
#[derive(Resource)]
pub(super) struct Shell {
    root: Entity,
    scroll: Option<Entity>,
    page: Page,
    status: Entity,
    hint: Entity,
    summary: Entity,
    seed: Entity,
    error: Entity,
}
fn row(world: &mut World, parent: Entity) -> Entity {
    world
        .spawn((
            Node {
                width: percent(100),
                align_items: AlignItems::Center,
                column_gap: px(8),
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(parent),
        ))
        .id()
}
fn bar(world: &mut World, root: Entity, top: bool) -> Entity {
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: if top { px(0) } else { Val::Auto },
                bottom: if top { Val::Auto } else { px(0) },
                width: percent(100),
                height: px(if top { 54 } else { 42 }),
                padding: UiRect::horizontal(px(16)),
                align_items: AlignItems::Center,
                column_gap: px(12),
                ..default()
            },
            BackgroundColor(Color::srgb(0.045, 0.063, 0.083)),
            ChildOf(root),
        ))
        .id()
}
pub(super) fn build(world: &mut World, state: &EditorState) {
    world.init_resource::<navigation::Navigation>();
    if let Some(old) = world.remove_resource::<Shell>() {
        if let Some(position) = old
            .scroll
            .and_then(|e| world.get::<ScrollPosition>(e))
            .cloned()
        {
            world
                .resource_mut::<navigation::Navigation>()
                .positions
                .insert(old.page, position);
        }
        world.entity_mut(old.root).despawn();
    }
    let scroll_position = world
        .resource::<navigation::Navigation>()
        .positions
        .get(&state.view.page)
        .cloned()
        .unwrap_or_default();
    let root = world
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                position_type: PositionType::Absolute,
                ..default()
            },
            GlobalZIndex(50),
            bevy::input_focus::tab_navigation::TabGroup::default(),
            Name::new("zeroverse_editor"),
        ))
        .id();
    let top = bar(world, root, true);
    widgets::button(
        world,
        top,
        if state.view.collapsed {
            "Controls"
        } else {
            "Hide"
        },
        Action::Collapse,
        false,
    );
    widgets::text(world, top, "ZERO / VERSE", 19., INK);
    widgets::text(world, top, "SCENE STUDIO", 10., ACCENT);
    world.spawn((
        Node {
            flex_grow: 1.,
            ..default()
        },
        ChildOf(top),
    ));
    widgets::button(world, top, "|<", Action::Rewind, false);
    widgets::button(world, top, "Play / pause", Action::Play, false);
    widgets::button(world, top, "Next  [R]", Action::Next, true);
    let bottom = bar(world, root, false);
    let status = widgets::text(world, bottom, "Preparing scene…", 12., MUTED);
    world.spawn((
        Node {
            flex_grow: 1.,
            ..default()
        },
        ChildOf(bottom),
    ));
    let hint = widgets::text(
        world,
        bottom,
        "Drag: orbit   ·   Shift-drag: pan   ·   Wheel: zoom",
        11.,
        MUTED,
    );
    let mut summary = status;
    let mut seed = status;
    let mut error = status;
    let mut scroll_entity = None;
    if !state.view.collapsed {
        let panel = world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    top: px(54),
                    bottom: px(42),
                    left: px(0),
                    width: px(PANEL),
                    max_width: percent(100),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(px(12)),
                    row_gap: px(8),
                    border: UiRect::right(px(1)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.065, 0.082, 0.105)),
                BorderColor::all(Color::srgb(0.15, 0.19, 0.24)),
                ChildOf(root),
            ))
            .id();
        let seedrow = row(world, panel);
        seed = widgets::text(world, seedrow, "SEED", 12., ACCENT);
        let actions = row(world, panel);
        widgets::button(world, actions, "Apply changes", Action::Apply, true);
        widgets::button(world, actions, "Discard", Action::Reset, false);
        summary = widgets::text(world, panel, "All changes applied", 11., MUTED);
        let tabs = world
            .spawn((
                Node {
                    display: Display::Grid,
                    grid_template_columns: RepeatedGridTrack::flex(2, 1.),
                    row_gap: px(4),
                    column_gap: px(4),
                    width: percent(100),
                    flex_shrink: 0.,
                    ..default()
                },
                ChildOf(panel),
            ))
            .id();
        for page in Page::ALL {
            widgets::button(
                world,
                tabs,
                page.label(),
                Action::Page(page),
                page == state.view.page,
            );
        }
        widgets::text(world, panel, state.view.page.intro(), 11.5, MUTED);
        let groups = navigation::groups(state);
        navigation::toolbar(world, panel, &groups);
        error = widgets::text(world, panel, "", 12., Color::srgb(1., 0.58, 0.4));
        let scroll_row = world
            .spawn((
                Node {
                    width: percent(100),
                    flex_grow: 1.,
                    min_height: px(0),
                    column_gap: px(6),
                    ..default()
                },
                ChildOf(panel),
            ))
            .id();
        let scroll = world
            .spawn((
                Node {
                    width: percent(100),
                    min_width: px(0),
                    flex_grow: 1.,
                    min_height: px(0),
                    overflow: Overflow::scroll_y(),
                    flex_direction: FlexDirection::Column,
                    row_gap: px(8),
                    padding: UiRect::right(px(4)),
                    ..default()
                },
                ScrollArea,
                scroll_position,
                ChildOf(scroll_row),
            ))
            .id();
        scroll_entity = Some(scroll);
        world.spawn_scene(bsn!{
            @bevy::feathers::controls::FeathersScrollbar { @target:scroll, @orientation:bevy::ui_widgets::ControlOrientation::Vertical }
            Node {width:px(8),min_width:px(8),height:percent(100)}
            ChildOf(scroll_row)
        }).expect("panel scrollbar");
        for group in groups {
            let body = navigation::card(world, scroll, &group, state);
            for item in group.items {
                match item {
                    fields::Item::Section(_) => unreachable!("sections partitioned into groups"),
                    fields::Item::Field(field) => widgets::control(world, body, field, state),
                    fields::Item::Distribution(kind) => {
                        distribution_preview::build(world, body, kind)
                    }
                    fields::Item::Note(note) => {
                        widgets::text(world, body, note, 12., MUTED);
                    }
                    fields::Item::CapacityEvidence => {
                        let entity =
                            widgets::text(world, body, state.capacity_status(), 12., MUTED);
                        world.entity_mut(entity).insert(CapacityEvidence);
                    }
                    fields::Item::Action(label, action) => {
                        widgets::button(world, body, label, action, false);
                    }
                    fields::Item::CoVisibilityLegend => navigation::legend(world, body),
                }
            }
        }
    }

    world.insert_resource(Shell {
        root,
        scroll: scroll_entity,
        page: state.view.page,
        status,
        hint,
        summary,
        seed,
        error,
    });
}
fn set_text(world: &mut World, e: Entity, s: String) {
    if let Some(mut t) = world.get_mut::<Text>(e) {
        if t.0 != s {
            t.0 = s;
        }
    }
}
pub(super) fn status(world: &mut World, state: &EditorState) {
    let shell = world.resource::<Shell>();
    let (status, summary, seed, error, hint) = (
        shell.status,
        shell.summary,
        shell.seed,
        shell.error,
        shell.hint,
    );
    let hint_text = if state.applied["room_schematic"] == true {
        "Metric footprints · capture cameras · live joints"
    } else if state.applied["camera_grid"] == true {
        "Synchronized capture views"
    } else {
        "Drag: orbit   ·   Shift-drag: pan   ·   Wheel: zoom"
    };
    set_text(world, hint, hint_text.into());
    let pending = world.get_resource::<crate::scene::procedural_indoor::IndoorGenerationStatus>();
    let message = if let Some(crate::sample::CaptureFailure(Some(e))) =
        world.get_resource::<crate::sample::CaptureFailure>()
    {
        format!("Generation failed: {e}")
    } else if pending.is_some_and(|p| p.pending) {
        "Preparing scene geometry…".into()
    } else if pending.is_some_and(|p| p.lighting_pending) {
        "Refining indirect lighting…".into()
    } else if let Some(m) = world
        .get_resource::<crate::human_motion::HumanMotionReport>()
        .filter(|m| m.pending)
    {
        m.stage.clone()
    } else if let Some(m) =
        world.get_resource::<crate::scene::procedural_indoor::layout::IndoorManifest>()
    {
        format!(
            "Ready  ·  {:.1} × {:.1} m  ·  {} objects  ·  {} people  ·  {} views",
            m.room_size.x,
            m.room_size.z,
            m.objects.len(),
            m.humans.len(),
            m.cameras.len()
        )
    } else {
        "Ready  ·  Scene controls apply on request".into()
    };
    set_text(world, status, message);
    if state.view.collapsed {
        return;
    }
    set_text(
        world,
        seed,
        state
            .active_seed
            .map(|s| format!("ACTIVE SEED  {s}"))
            .unwrap_or_else(|| "SCENE CONTROLS".into()),
    );
    set_text(
        world,
        summary,
        if state.pending() {
            "Pending changes · Apply keeps this seed".into()
        } else {
            "All changes applied · Next explores a new seed".into()
        },
    );
    let message = state
        .input_errors
        .iter()
        .next()
        .map(|(path, error)| format!("{path}: {error}"))
        .or_else(|| state.error.clone())
        .unwrap_or_default();
    set_text(world, error, message.clone());
    if let Some(mut n) = world.get_mut::<Node>(error) {
        n.display = if message.is_empty() {
            Display::None
        } else {
            Display::Flex
        };
    }
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn viewport(
    state: Res<EditorState>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    capture: Res<EditorInputCapture>,
    mut editor: Query<(&mut Camera, &mut PanOrbitCamera), With<EditorCameraMarker>>,
    mut grids: Query<
        &mut Node,
        Or<(
            With<CameraGrid>,
            With<MaterialGrid>,
            With<schematic::SchematicView>,
        )>,
    >,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let left = if state.view.collapsed {
        0.
    } else {
        PANEL.min(window.width())
    };
    if window.physical_width() == 0 || window.physical_height() == 0 {
        return;
    }
    for (mut camera, mut pan) in &mut editor {
        let rect = bounded_viewport(window, left);
        if camera.viewport.as_ref().is_none_or(|v| {
            v.physical_position != rect.physical_position || v.physical_size != rect.physical_size
        }) {
            camera.viewport = Some(rect);
        }
        let enabled = orbit_enabled(&state, window, &capture);
        pan.enabled_motion = bevy::camera_controller::pan_orbit_camera::prelude::EnabledMotion {
            pan: enabled,
            orbit: enabled,
            zoom: enabled,
        };
    }
    for mut node in &mut grids {
        node.position_type = PositionType::Absolute;
        node.left = px(left);
        node.top = px(54);
        node.width = px((window.width() - left).max(1.));
        node.height = px((window.height() - 96.).max(1.));
    }
}

/// Evaluate the current pointer position, including the first drag frame.
pub(in crate::app) fn orbit_enabled(
    state: &EditorState,
    window: &Window,
    capture: &EditorInputCapture,
) -> bool {
    let left = if state.view.collapsed {
        0.
    } else {
        PANEL.min(window.width())
    };
    let over = window
        .cursor_position()
        .is_some_and(|p| p.x < left || p.y < 54. || p.y > window.height() - 42.);
    window.focused
        && !state.debug
        && !over
        && !capture.keyboard
        && state.applied["room_schematic"] != true
        && state.applied["camera_grid"] != true
}

/// A collapsed/narrow/minimized viewport must never extend beyond its render target.
pub(super) fn bounded_viewport(window: &Window, left: f32) -> Viewport {
    let width = window.physical_width().max(1);
    let height = window.physical_height().max(1);
    let scale = window.scale_factor();
    let x = ((left * scale) as u32).min(width - 1);
    let y = ((54. * scale) as u32).min(height - 1);
    Viewport {
        physical_position: UVec2::new(x, y),
        physical_size: UVec2::new(
            width - x,
            height
                .saturating_sub(y)
                .saturating_sub((42. * scale) as u32)
                .max(1),
        ),
        ..default()
    }
}
