//! Responsive editor chrome and a reserved, unobscured scene viewport.
use super::*;
use bevy::{camera::Viewport, ui_widgets::ScrollArea};
#[derive(Resource)]
pub(super) struct Shell {
    root: Entity,
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
    if let Some(old) = world.remove_resource::<Shell>() {
        world.entity_mut(old.root).despawn();
    }
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
                    padding: UiRect::all(px(16)),
                    row_gap: px(12),
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
        widgets::text(world, panel, state.view.page.intro(), 12., MUTED);
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
                    row_gap: px(14),
                    padding: UiRect::right(px(8)),
                    ..default()
                },
                ScrollArea,
                ChildOf(scroll_row),
            ))
            .id();
        world.spawn_scene(bsn!{
            @bevy::feathers::controls::FeathersScrollbar { @target:scroll, @orientation:bevy::ui_widgets::ControlOrientation::Vertical }
            Node {width:px(8),min_width:px(8),height:percent(100)}
            ChildOf(scroll_row)
        }).expect("panel scrollbar");
        let mut hidden_section = false;
        for item in fields::items(state) {
            if hidden_section && !matches!(item, fields::Item::Section(..)) {
                continue;
            }
            match item {
                fields::Item::Section(title, help) => {
                    let collapsible = matches!(
                        title,
                        "ADVANCED RIG" | "PROMPT DISTRIBUTION" | "OPTICAL FLOW"
                    );
                    hidden_section = collapsible && !state.view.expanded.contains(title);
                    if collapsible {
                        widgets::button(
                            world,
                            scroll,
                            &format!("{}  {}", if hidden_section { ">" } else { "v" }, title),
                            Action::Section(title.into()),
                            false,
                        );
                    }
                    if hidden_section {
                        continue;
                    }

                    let section = widgets::column(world, scroll, 5.);
                    world.entity_mut(section).insert(Node {
                        width: percent(100),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(5),
                        padding: UiRect::top(px(10)),
                        border: UiRect::top(px(1)),
                        flex_shrink: 0.,
                        ..default()
                    });
                    world
                        .entity_mut(section)
                        .insert(BorderColor::all(Color::srgb(0.16, 0.20, 0.25)));
                    widgets::text(world, section, title, 11., ACCENT);
                    widgets::text(world, section, help, 11.5, MUTED);
                }
                fields::Item::Field(field) => widgets::control(world, scroll, field, state),
                fields::Item::Note(note) => {
                    widgets::text(world, scroll, note, 12., MUTED);
                }
            }
        }
        if state.view.page == Page::Advanced
            || (state.view.page == Page::Scene && state.draft["scene_type"] != "ProceduralIndoor")
        {
            widgets::button(
                world,
                scroll,
                "Scene resource inspector",
                Action::Debug,
                false,
            );
        }
        if state.view.page == Page::People {
            widgets::button(
                world,
                scroll,
                "Play motion at model speed",
                Action::ModelSpeed,
                false,
            );
        }
        if state.view.page == Page::View && state.draft["room_schematic"] != true {
            if let Some(legend) =
                world.get_resource::<crate::render::co_visibility::CoVisibilityLegend>()
            {
                let indices = legend.camera_indices.clone();
                if !indices.is_empty() {
                    widgets::text(world, scroll, "CO-VISIBILITY MEMBERSHIP", 11., ACCENT);
                    for (bit, index) in indices.iter().enumerate().take(16) {
                        let rgb =
                            crate::render::co_visibility::camera_color(bit, indices.len().min(16));
                        let row = row(world, scroll);
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
            }
        }
    }
    world.insert_resource(Shell {
        root,
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
        let over = window
            .cursor_position()
            .is_some_and(|p| p.x < left || p.y < 54. || p.y > window.height() - 42.);
        pan.enabled = !over
            && !capture.keyboard
            && state.applied["room_schematic"] != true
            && state.applied["camera_grid"] != true;
    }
    for mut node in &mut grids {
        node.position_type = PositionType::Absolute;
        node.left = px(left);
        node.top = px(54);
        node.width = px((window.width() - left).max(1.));
        node.height = px((window.height() - 96.).max(1.));
    }
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
