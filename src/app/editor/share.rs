//! Versioned URL snapshots. Browser history is replaced, never reloaded or appended per edit.
use super::*;
use model::{CameraPose, ViewState};

pub fn query(config: &BevyZeroverseConfig, view: &ViewState) -> String {
    let current = model::expand(config);
    let defaults = model::expand(&BevyZeroverseConfig::default());
    let mut pairs = Vec::new();
    for (key, value) in current.as_object().unwrap() {
        if key == "viewer_state" || value == &defaults[key] {
            continue;
        }
        let value = if let Some(s) = value.as_str() {
            s.to_owned()
        } else {
            value.to_string()
        };
        pairs.push(format!("{}={}", encode(key), encode(&value)));
    }
    pairs.push(format!(
        "viewer_state={}",
        encode(&serde_json::to_string(view).unwrap())
    ));
    pairs.join("&")
}
fn encode(s: &str) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            write!(out, "%{b:02X}").unwrap();
        }
    }
    out
}
#[derive(Default)]
pub(super) struct ShareState {
    restored: bool,
    camera_restored: bool,
    last: String,
    elapsed: f32,
}
#[allow(clippy::too_many_arguments)]
pub(super) fn synchronize(
    mut state: ResMut<EditorState>,
    config: Res<BevyZeroverseConfig>,
    time: Res<Time>,
    mut local: Local<ShareState>,
    mut cameras: Query<(&mut PanOrbitCamera, &mut Projection), With<EditorCameraMarker>>,
    mut roots: Query<&mut Transform, With<ZeroverseSceneRoot>>,
    mut playback: ResMut<Playback>,
    mut flow: Option<ResMut<crate::render::optical_flow::FlowPreviewSettings>>,
    generation: Option<Res<crate::scene::procedural_indoor::IndoorGenerationStatus>>,
) {
    if !local.restored {
        if generation.is_some_and(|g| g.busy()) {
            return;
        }
        if config.scene_type == ZeroverseSceneType::ProceduralIndoor && state.active_seed.is_none()
        {
            return;
        }
        // Read the startup snapshot, since live playback already updates the UI's progress.
        if let Some(saved) = config
            .viewer_state
            .as_deref()
            .and_then(|s| ViewState::parse(s).ok())
        {
            playback.progress = saved.progress;
            playback.direction = saved.direction;
            if let Some(f) = &mut flow {
                f.interval_seconds = saved.flow_interval;
                f.full_scale_pixels = saved.flow_scale;
            }
        }
        if let Some(r) = state.view.scene_rotation {
            for mut root in &mut roots {
                root.rotation = Quat::from_array(r).normalize();
            }
        }
        local.restored = true;
    }
    // A grid-only startup creates the editor camera later. Restore its pose then,
    // independently of the scene/timeline snapshot and URL update cadence.
    if !local.camera_restored {
        if let Ok((mut pan, mut projection)) = cameras.single_mut() {
            if let Some(c) = &state.view.camera {
                pan.focus = Vec3::from_array(c.focus);
                pan.target_focus = pan.focus;
                pan.yaw = Some(c.yaw);
                pan.target_yaw = c.yaw;
                pan.pitch = Some(c.pitch);
                pan.target_pitch = c.pitch;
                pan.radius = Some(c.radius);
                pan.target_radius = c.radius;
                pan.force_update = true;
                if let Projection::Perspective(p) = &mut *projection {
                    p.fov = c.fov;
                }
            }
            local.camera_restored = true;
        }
    }
    if let Ok((pan, projection)) = cameras.single_mut() {
        if let Projection::Perspective(p) = &*projection {
            state.view.camera = Some(CameraPose {
                focus: pan.focus.to_array(),
                yaw: pan.yaw.unwrap_or(pan.target_yaw),
                pitch: pan.pitch.unwrap_or(pan.target_pitch),
                radius: pan.radius.unwrap_or(pan.target_radius),
                fov: p.fov,
            });
        }
    }
    state.view.scene_rotation = roots.iter().next().map(|r| r.rotation.to_array());
    state.view.progress = playback.progress;
    state.view.direction = playback.direction;
    local.elapsed += time.delta_secs();
    if local.elapsed < 0.5 {
        return;
    }
    local.elapsed = 0.;
    if !state.input_errors.is_empty() {
        return;
    }
    if state.config().is_err() {
        return;
    }
    let Ok((config, view)) = snapshot(&state) else {
        return;
    };
    let next = query(&config, &view);
    if next == local.last {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let result = (|| -> Result<(), wasm_bindgen::JsValue> {
            let window = web_sys::window()
                .ok_or_else(|| wasm_bindgen::JsValue::from_str("window unavailable"))?;
            let location = window.location();
            let url = format!("{}?{}{}", location.pathname()?, next, location.hash()?);
            window
                .history()?
                .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url))
        })();
        if let Err(e) = result {
            state.error = Some(format!("Could not update share URL: {e:?}"));
            return;
        }
    }
    local.last = next;
}

/// Share the rendered configuration plus staged edits, so reloading never silently applies a draft.
pub(super) fn snapshot(state: &EditorState) -> Result<(BevyZeroverseConfig, ViewState), String> {
    let config = model::validate(&state.applied)?;
    let mut view = state.view.clone();
    let pending: serde_json::Map<String, Value> = state
        .draft
        .as_object()
        .unwrap()
        .iter()
        .filter(|(key, value)| *value != &state.applied[*key])
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    view.pending = (!pending.is_empty()).then_some(Value::Object(pending));
    Ok((config, view))
}
