//! Versioned URL snapshots. Browser history is replaced, never reloaded or appended per edit.
use super::*;
use model::{CameraPose, ViewState};

pub fn query(config: &BevyZeroverseConfig, view: &ViewState) -> String {
    let current = model::expand(config);
    // A shared link is opened by the browser entry point. Comparing against
    // library defaults adds unmodified CLI/browser differences to every URL.
    let defaults = model::expand(&super::super::web_viewer_defaults());
    let serde_defaults = model::expand(&BevyZeroverseConfig::default());
    let mut pairs = Vec::new();
    for (key, value) in current.as_object().unwrap() {
        if key == "viewer_state" || value == &defaults[key] {
            continue;
        }
        // A nested query parameter replaces its whole typed policy. Compact
        // against that type's serde defaults, not the browser's optional preset.
        let value = compact(value, &serde_defaults[key], &format!("/{key}"));
        let value = if let Some(s) = value.as_str() {
            s.to_owned()
        } else {
            value.to_string()
        };
        pairs.push(format!("{}={}", encode(key), encode(&value)));
    }
    if let Some(view) = delta::between(&json!(view), &json!(ViewState::default())) {
        pairs.push(format!("viewer_state={}", encode(&view.to_string())));
    }
    pairs.join("&")
}

fn compact(value: &Value, defaults: &Value, path: &str) -> Value {
    use crate::scene::procedural_indoor::cameras::{
        handheld::HandheldSettings, mixture::OverlapMixture,
    };
    // Optional programs must retain `{}` even when every parameter is default:
    // removing the object would disable them. Explicit null still disables one.
    let optional = defaults.is_null().then(|| match path {
        "/indoor_camera/lenses" => json!(crate::camera::lens::LensSampling::default()),
        "/indoor_camera/handheld" => json!(HandheldSettings::default()),
        "/indoor_camera/overlap_mixture" => json!(OverlapMixture::default()),
        "/human_motion" => json!(crate::human_motion::HumanMotionConfig::default()),
        "/indoor_motion" => {
            json!(crate::scene::procedural_indoor::motion::IndoorMotionConfig::default())
        }
        _ => Value::Null,
    });
    let defaults = optional.as_ref().unwrap_or(defaults);
    if let (Some(object), Some(base)) = (value.as_object(), defaults.as_object()) {
        Value::Object(
            object
                .iter()
                .filter_map(|(key, value)| {
                    let default = base.get(key).unwrap_or(&Value::Null);
                    (value != default).then(|| {
                        (
                            key.clone(),
                            compact(value, default, &format!("{path}/{key}")),
                        )
                    })
                })
                .collect(),
        )
    } else {
        value.clone()
    }
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
    last: Option<String>,
    elapsed: f32,
}
#[allow(clippy::too_many_arguments)]
pub(super) fn synchronize(
    mut state: ResMut<EditorState>,
    config: Res<BevyZeroverseConfig>,
    time: Res<Time>,
    mut local: Local<ShareState>,
    mut cameras: Query<
        (&mut PanOrbitCamera, &mut Transform, &mut Projection),
        With<EditorCameraMarker>,
    >,
    mut roots: Query<&mut Transform, (With<ZeroverseSceneRoot>, Without<EditorCameraMarker>)>,
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
        if let Ok((mut pan, mut transform, mut projection)) = cameras.single_mut() {
            if let Some(c) = &state.view.camera {
                *transform = orbit::transform(Vec3::from_array(c.focus), c.yaw, c.pitch, c.radius);
                orbit::reset(&mut pan, c.radius);
                if let Projection::Perspective(p) = &mut *projection {
                    p.fov = c.fov;
                }
            }
            local.camera_restored = true;
        }
    }
    if let Ok((pan, transform, projection)) = cameras.single_mut() {
        if let Projection::Perspective(p) = &*projection {
            state.view.camera = Some(CameraPose {
                focus: (transform.translation
                    + transform.forward().as_vec3() * pan.last_anchor_depth().abs() as f32)
                    .to_array(),
                yaw: transform.rotation.to_euler(EulerRot::YXZ).0,
                pitch: -transform.rotation.to_euler(EulerRot::YXZ).1,
                radius: pan.last_anchor_depth().abs() as f32,
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
    // Also compact an initially all-default URL on the first synchronization.
    if local.last.as_deref() == Some(next.as_str()) {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let result = (|| -> Result<(), wasm_bindgen::JsValue> {
            let window = web_sys::window()
                .ok_or_else(|| wasm_bindgen::JsValue::from_str("window unavailable"))?;
            let location = window.location();
            let search = if next.is_empty() {
                String::new()
            } else {
                format!("?{next}")
            };
            let url = format!("{}{}{}", location.pathname()?, search, location.hash()?);
            window
                .history()?
                .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url))
        })();
        if let Err(e) = result {
            state.error = Some(format!("Could not update share URL: {e:?}"));
            return;
        }
    }
    local.last = Some(next);
}

/// Share the rendered configuration plus staged edits, so reloading never silently applies a draft.
pub(super) fn snapshot(state: &EditorState) -> Result<(BevyZeroverseConfig, ViewState), String> {
    let config = model::validate(&state.applied)?;
    let mut view = state.view.clone();
    view.pending = delta::between(&state.draft, &state.applied);
    Ok((config, view))
}
