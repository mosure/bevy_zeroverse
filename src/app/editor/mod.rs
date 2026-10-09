//! Shared native/WebGPU scene controller built with retained Bevy Feathers widgets.
mod fields;
pub mod model;
pub mod schematic;
mod share;
pub(super) mod shell;
#[cfg(test)]
mod tests;
mod widgets;
use super::*;
use bevy::{
    feathers::{dark_theme::create_dark_theme, theme::UiTheme, FeathersPlugins},
    input_focus::InputFocus,
    text::EditableText,
};
use model::{EditorState, Page};
use serde_json::{json, Value};

const PANEL: f32 = 380.;
const INK: Color = Color::srgb(0.91, 0.94, 0.97);
const MUTED: Color = Color::srgb(0.56, 0.64, 0.71);
const ACCENT: Color = Color::srgb(0.28, 0.85, 0.74);
#[derive(Clone)]
pub enum Action {
    Set(String, Value),
    Input(String, String, bool),
    Page(Page),
    Apply,
    Next,
    Reset,
    Collapse,
    Section(String),
    Play,
    Rewind,
    Debug,
    ModelSpeed,
    Loaded(u64),
}
#[derive(Resource, Default)]
pub struct Actions(pub Vec<Action>);
#[derive(Resource, Default)]
pub(super) struct EditorInputCapture {
    pub keyboard: bool,
}
pub struct EditorPlugin;
impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        let state = EditorState::new(app.world().resource::<BevyZeroverseConfig>());
        app.insert_resource(state)
            .init_resource::<Actions>()
            .init_resource::<EditorInputCapture>()
            .add_plugins(FeathersPlugins)
            .insert_resource(studio_theme())
            .add_systems(Update, (scene_loaded, update).chain())
            .add_systems(PostUpdate, shell::viewport.after(EditorCameraSetup))
            .add_systems(Last, share::synchronize)
            .init_resource::<schematic::Preview>()
            .init_resource::<schematic::Predictions>()
            .add_systems(
                PostUpdate,
                schematic::update
                    .after(bevy::transform::TransformSystems::Propagate)
                    .after(crate::annotation::pose::compute_human_poses)
                    .after(crate::scene::procedural_indoor::humans::update_human_poses),
            );
    }
}
fn studio_theme() -> UiTheme {
    use bevy::feathers::tokens;
    let mut theme = create_dark_theme();
    for token in [
        tokens::BUTTON_PRIMARY_BG,
        tokens::SLIDER_BAR,
        tokens::CHECKBOX_BG_CHECKED,
        tokens::SCROLLBAR_THUMB,
    ] {
        if let Some(semantic) = theme.token_assignments.get(&token).cloned() {
            theme
                .semantic_base
                .insert(semantic, Color::srgb(0.12, 0.39, 0.36));
        }
    }
    for token in [
        tokens::BUTTON_PRIMARY_BG_HOVER,
        tokens::SLIDER_BAR_HOVER,
        tokens::CHECKBOX_BG_CHECKED_HOVER,
        tokens::SCROLLBAR_THUMB_HOVER,
    ] {
        if let Some(semantic) = theme.token_assignments.get(&token).cloned() {
            theme
                .semantic_base
                .insert(semantic, Color::srgb(0.16, 0.49, 0.44));
        }
    }
    if let Some(semantic) = theme.token_assignments.get(&tokens::FOCUS_RING).cloned() {
        theme.semantic_base.insert(semantic, ACCENT.with_alpha(0.7));
    }
    UiTheme(theme)
}
fn scene_loaded(
    mut events: MessageReader<SceneLoadedEvent>,
    manifest: Option<Res<crate::scene::procedural_indoor::layout::IndoorManifest>>,
    config: Res<BevyZeroverseConfig>,
    mut actions: ResMut<Actions>,
) {
    if !events.is_empty() {
        events.clear();
        if config.scene_type == ZeroverseSceneType::ProceduralIndoor {
            if let Some(m) = manifest {
                actions.0.push(Action::Loaded(m.seed));
            }
        }
    }
}
pub fn value(state: &EditorState, path: &str) -> Value {
    match path {
        "@config" => state.draft.clone(),
        "@viewport" => json!(if state.draft["room_schematic"] == true {
            "schematic"
        } else if state.draft["camera_grid"] == true {
            "grid"
        } else {
            "editor"
        }),
        "@baseline" => serde_json::from_value::<
            crate::scene::procedural_indoor::cameras::multiview::MultiViewSettings,
        >(state.draft["indoor_camera"]["multiview"].clone())
        .ok()
        .map(|m| {
            json!(m
                .baseline()
                .unwrap_or(((m.min_reference_baseline - 0.06) / 2.30).clamp(0., 1.)))
        })
        .unwrap_or(json!(0.5)),
        "@multiview" => json!(!state.draft["indoor_camera"]["multiview"].is_null()),
        "@mixture" => json!(!state.draft["indoor_camera"]["overlap_mixture"].is_null()),
        "@handheld" => json!(!state.draft["indoor_camera"]["handheld"].is_null()),
        "@motion" => json!(!state.draft["human_motion"].is_null()),
        "@path_preset" => {
            let c = &state.draft["indoor_camera"];
            let min = c["path_length_min"].as_f64().unwrap_or(-1.);
            let max = c["path_length_max"].as_f64().unwrap_or(-1.);
            json!(if min == 0. && max == 0. {
                "static"
            } else if !c["handheld"].is_null() {
                "handheld"
            } else if (min - 0.5).abs() < 1e-5 && (max - 8.).abs() < 1e-5 {
                "explore"
            } else {
                "custom"
            })
        }
        "@progress" => json!(state.view.progress),
        "@editor_fov" => json!(state
            .view
            .camera
            .as_ref()
            .map_or(45., |c| c.fov.to_degrees())),
        "@flow_interval" => json!(state.view.flow_interval),
        "@flow_scale" => json!(state.view.flow_scale),
        _ => state.draft.pointer(path).cloned().unwrap_or(Value::Null),
    }
}
fn edit(state: &mut EditorState, path: &str, mut v: Value) -> Result<bool, String> {
    use crate::scene::procedural_indoor::cameras::{
        handheld::HandheldSettings, mixture::OverlapMixture, multiview::MultiViewSettings,
    };
    let mut rebuild = false;
    state.error = None;
    match path {
        "@viewport" => {
            let mode = v.as_str().ok_or("Expected a viewport")?;
            if !["editor", "grid", "schematic"].contains(&mode) {
                return Err("Unknown viewport".into());
            }
            state.draft["room_schematic"] = json!(mode == "schematic");
            state.draft["camera_grid"] = json!(mode == "grid");
            rebuild = true;
        }
        "@config" => {
            let config = model::validate(&v)?;
            state.draft = model::expand(&config);
            return Ok(true);
        }
        "@baseline" => {
            state.draft["indoor_camera"]["multiview"] = json!(MultiViewSettings::from_baseline(
                v.as_f64().ok_or("Expected a number")? as f32
            )?);
        }
        "@multiview" | "@mixture" | "@handheld" | "@motion" => {
            let enabled = v.as_bool().ok_or("Expected true or false")?;
            let (key, default) = match path {
                "@multiview" => ("multiview", json!(MultiViewSettings::default())),
                "@mixture" => ("overlap_mixture", json!(OverlapMixture::default())),
                "@handheld" => ("handheld", json!(HandheldSettings::default())),
                _ => (
                    "human_motion",
                    json!(crate::human_motion::HumanMotionConfig::default()),
                ),
            };
            let target = if path == "@motion" {
                &mut state.draft
            } else {
                &mut state.draft["indoor_camera"]
            };
            target[key] = if enabled { default } else { Value::Null };
            if path == "@multiview" && !enabled {
                target["overlap_mixture"] = Value::Null;
            }
            rebuild = true;
        }
        "@path_preset" => {
            let camera = &mut state.draft["indoor_camera"];
            match v.as_str().unwrap_or("") {
                "static" => {
                    camera["path_length_min"] = json!(0.);
                    camera["path_length_max"] = json!(0.);
                    camera["long_path_fraction"] = json!(0.);
                    camera["handheld"] = Value::Null;
                }
                "handheld" => {
                    camera["path_length_min"] = json!(0.01);
                    camera["path_length_max"] = json!(0.6);
                    camera["long_path_fraction"] = json!(0.);
                    camera["handheld"] = json!(HandheldSettings::default());
                }
                "explore" => {
                    camera["path_length_min"] = json!(0.5);
                    camera["path_length_max"] = json!(8.);
                    camera["long_path_fraction"] = json!(0.8);
                    camera["handheld"] = Value::Null;
                }
                _ => {}
            }
            rebuild = true;
        }
        "@progress" => {
            state.view.progress = v.as_f64().ok_or("Expected time")? as f32;
            state.draft["playback_mode"] = json!("Still");
        }
        "@editor_fov" => {
            if let Some(camera) = &mut state.view.camera {
                camera.fov =
                    v.as_f64().ok_or("Expected degrees")? as f32 * std::f32::consts::PI / 180.;
            }
        }
        "@flow_interval" => {
            state.view.flow_interval = v.as_f64().ok_or("Expected interval")? as f32;
        }
        "@flow_scale" => {
            state.view.flow_scale = v.as_f64().ok_or("Expected scale")? as f32;
        }
        _ => {
            // Sliders emit f32; preserve integer types in the public config contract.
            if state
                .draft
                .pointer(path)
                .is_some_and(|old| old.is_u64() || old.is_i64())
                && v.is_f64()
            {
                if let Some(n) = v.as_f64() {
                    v = json!(n.round() as i64);
                }
            }
            state.set(path, v)?;
            rebuild |= path == "/scene_type";
            if path == "/scene_type"
                && state.draft["scene_type"] == "ProceduralIndoor"
                && state.draft["num_cameras"] == 0
            {
                state.draft["num_cameras"] = json!(4);
            }
        }
    }
    Ok(rebuild)
}
fn apply(world: &mut World, state: &mut EditorState, next: bool) -> Result<(), String> {
    if !state.input_errors.is_empty() {
        return Err("Fix invalid text fields before applying".into());
    }
    let mut config = state.config()?;
    for key in [
        "editor",
        "headless",
        "image_copiers",
        "initialize_scene",
        "press_esc_close",
    ] {
        if state.draft[key] != state.applied[key] {
            return Err(format!(
                "{key} is a startup option. Set it in the launch command or URL and restart."
            ));
        }
    }
    if config.human_motion.is_some() && !cfg!(feature = "human_motion") {
        return Err("Rebuild with the human_motion feature to generate motion".into());
    }
    let seed = if config.scene_type == ZeroverseSceneType::ProceduralIndoor {
        let seed = if next {
            state
                .active_seed
                .or(config.indoor_seed)
                .map(|s| s.wrapping_add(1))
                .unwrap_or_else(rand::random)
        } else {
            config
                .indoor_seed
                .or(state.active_seed)
                .unwrap_or_else(rand::random)
        };
        config.indoor_seed = Some(seed);
        Some(seed)
    } else {
        None
    };
    config.viewer_state = Some(serde_json::to_string(&state.view).unwrap());
    world.insert_resource(crate::camera::DefaultZeroverseCamera {
        resolution: UVec2::new(config.width as u32, config.height as u32).into(),
    });
    if let Some(mut gi) =
        world.get_resource_mut::<crate::scene::procedural_indoor::gi::IndoorGiSettings>()
    {
        gi.bake.rays_per_probe = config.indoor_gi_rays;
    }
    world.insert_resource(config.clone());
    if let Some(seed) = seed {
        crate::scene::procedural_indoor::reset_indoor_sequence(world, seed);
    }
    state.draft = model::expand(&config);
    state.applied = state.draft.clone();
    state.error = None;
    world.write_message(RegenerateSceneEvent);
    Ok(())
}
fn live_update(world: &mut World, state: &mut EditorState, path: &str) -> Result<(), String> {
    if path == "@viewport" {
        let mut config = world.resource_mut::<BevyZeroverseConfig>();
        config.camera_grid = state.draft["camera_grid"] == true;
        config.room_schematic = state.draft["room_schematic"] == true;
        for key in ["camera_grid", "room_schematic"] {
            state.applied[key] = state.draft[key].clone();
        }
    }
    if model::live(path) || path == "@progress" {
        let mut config = serde_json::to_value(world.resource::<BevyZeroverseConfig>()).unwrap();
        let key = if path == "@progress" {
            "playback_mode"
        } else {
            &path[1..]
        };
        config[key] = state.draft[key].clone();
        let config: BevyZeroverseConfig =
            serde_json::from_value(config).map_err(|e| e.to_string())?;
        world.insert_resource(config);
        state.applied[key] = state.draft[key].clone();
    }
    if path == "@progress" {
        let mut p = world.resource_mut::<Playback>();
        p.progress = state.view.progress;
        p.mode = PlaybackMode::Still;
    }
    if path == "@editor_fov" {
        if let Some(camera) = &state.view.camera {
            for mut projection in world
                .query_filtered::<&mut Projection, With<EditorCameraMarker>>()
                .iter_mut(world)
            {
                if let Projection::Perspective(p) = &mut *projection {
                    p.fov = camera.fov;
                }
            }
        }
    }
    if path.starts_with("@flow_") {
        if let Some(mut f) =
            world.get_resource_mut::<crate::render::optical_flow::FlowPreviewSettings>()
        {
            f.interval_seconds = state.view.flow_interval;
            f.full_scale_pixels = state.view.flow_scale;
        }
    }
    Ok(())
}
pub(super) fn update(world: &mut World) {
    if !world.contains_resource::<AssetServer>() {
        return;
    }
    let mut state = world.remove_resource::<EditorState>().unwrap();
    let focus = world.get_resource::<InputFocus>().and_then(InputFocus::get);
    let typing = focus.is_some_and(|e| world.get::<EditableText>(e).is_some());
    world.resource_mut::<EditorInputCapture>().keyboard = typing;
    let mut actions = std::mem::take(&mut world.resource_mut::<Actions>().0);
    if world.resource::<BevyZeroverseConfig>().keybinds
        && focus.is_none()
        && !typing
        && world
            .resource::<ButtonInput<KeyCode>>()
            .just_pressed(KeyCode::Space)
    {
        actions.push(Action::Play);
    }
    let dirty = !actions.is_empty();
    let mut rebuild = false;
    for action in actions {
        let result: Result<(), String> = (|| {
            match action {
                Action::Set(path, v) => {
                    rebuild |= edit(&mut state, &path, v)?;
                    live_update(world, &mut state, &path)?;
                }
                Action::Input(path, input, is_json) => {
                    let parsed = if input.trim().is_empty() {
                        Ok(Value::Null)
                    } else {
                        serde_json::from_str::<Value>(&input).or_else(|e| {
                            if !is_json && state.draft.pointer(&path).is_some_and(Value::is_string)
                            {
                                Ok(json!(input))
                            } else {
                                Err(e)
                            }
                        })
                    };
                    match parsed {
                        Ok(v) => {
                            state.input_errors.remove(&path);
                            state.invalid_text.remove(&path);
                            if value(&state, &path) != v {
                                // JSON remains editable across keystrokes; rebuilding here loses focus.
                                if let Err(error) = edit(&mut state, &path, v) {
                                    state.invalid_text.insert(path.clone(), input.clone());
                                    state.input_errors.insert(path, error);
                                }
                            }
                        }
                        Err(e) => {
                            state.invalid_text.insert(path.clone(), input.clone());
                            state.input_errors.insert(path, e.to_string());
                        }
                    }
                }
                Action::Page(page) => {
                    state.view.page = page;
                    rebuild = true;
                }
                Action::Section(section) => {
                    if !state.view.expanded.remove(&section) {
                        state.view.expanded.insert(section);
                    }
                    rebuild = true;
                }
                Action::Collapse => {
                    state.view.collapsed = !state.view.collapsed;
                    rebuild = true;
                }
                Action::Apply => {
                    apply(world, &mut state, false)?;
                    rebuild = true;
                }
                Action::Next => {
                    apply(world, &mut state, true)?;
                    rebuild = true;
                }
                Action::Reset => {
                    state.draft = state.applied.clone();
                    state.error = None;
                    state.input_errors.clear();
                    state.invalid_text.clear();
                    rebuild = true;
                }
                Action::Debug => {
                    state.debug = !state.debug;
                }
                Action::Play => {
                    let playing = state.draft["playback_mode"] != "Still";
                    state.draft["playback_mode"] = json!(if playing { "Still" } else { "Loop" });
                    live_update(world, &mut state, "/playback_mode")?;
                }
                Action::Rewind => {
                    world.resource_mut::<Playback>().progress = 0.;
                    state.view.progress = 0.;
                }
                Action::ModelSpeed => {
                    let frames = world
                        .get_resource::<crate::human_motion::HumanMotionReport>()
                        .and_then(|r| r.accepted.first())
                        .map(|p| p.request.frames)
                        .unwrap_or(160);
                    state.draft["playback_mode"] = json!("Once");
                    state.draft["playback_speed"] =
                        json!(20. / frames.saturating_sub(1).max(1) as f32);
                    live_update(world, &mut state, "/playback_mode")?;
                    live_update(world, &mut state, "/playback_speed")?;
                    world.resource_mut::<Playback>().progress = 0.;
                }
                Action::Loaded(seed) => {
                    state.active_seed = Some(seed);
                    if state.draft["indoor_seed"] == state.applied["indoor_seed"] {
                        state.draft["indoor_seed"] = json!(seed);
                    }
                    state.applied["indoor_seed"] = json!(seed);
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            state.error = Some(e);
        }
    }
    if let Some(p) = world.get_resource::<Playback>() {
        state.view.progress = p.progress;
        state.view.direction = p.direction;
    }
    if !world.contains_resource::<shell::Shell>() || rebuild {
        shell::build(world, &state);
        state.revision += 1;
    }
    if dirty || rebuild || state.view.page == Page::View {
        widgets::synchronize(world, &state);
    }
    if dirty && state.error.is_none() {
        state.error = state.config().err();
    }
    shell::status(world, &state);
    world.insert_resource(state);
}
