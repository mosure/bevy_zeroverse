//! Shared native/browser URL codec. Dotted policy fields, sparse defaults,
//! deterministic ordering, and backwards-compatible whole-policy JSON links.
use super::*;
use crate::scene::procedural_indoor::{appearance::AppearanceSettings, cameras::CameraSettings};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn root(name: &str) -> &str {
    match name {
        "camera" => "indoor_camera",
        "appearance" => "indoor_appearance",
        "clutter" => "indoor_clutter",
        "object_motion" => "indoor_motion",
        "view" => "viewer_state",
        _ => name,
    }
}
fn short(name: &str) -> &str {
    match name {
        "indoor_camera" => "camera",
        "indoor_appearance" => "appearance",
        "indoor_clutter" => "clutter",
        "indoor_motion" => "object_motion",
        "viewer_state" => "view",
        _ => name,
    }
}

/// Default for a nullable typed program. `{}` explicitly enables it.
fn optional(path: &str) -> Option<Value> {
    use crate::scene::procedural_indoor::{cameras, motion};
    Some(match path {
        "indoor_camera.lenses" | "indoor_camera.ceiling_mount.lenses" => {
            json!(crate::camera::lens::LensSampling::default())
        }
        "indoor_camera.ceiling_mount" => json!(cameras::ceiling::CeilingCameraSettings::default()),
        "indoor_camera.multiview" => json!(cameras::multiview::MultiViewSettings::default()),
        "indoor_camera.handheld" => json!(cameras::handheld::HandheldSettings::default()),
        "indoor_camera.overlap_mixture" => json!(cameras::mixture::OverlapMixture::default()),
        "human_motion" => json!(crate::human_motion::HumanMotionConfig::default()),
        "indoor_motion" => json!(motion::IndoorMotionConfig::default()),
        "viewer_state" => json!(super::view_state::ViewState::default()),
        _ => return None,
    })
}

/// Pairs are decoded by URLSearchParams. Reject ambiguous duplicates rather than
/// making replay depend on browser query ordering. Dotted fields override legacy
/// whole-policy values, independently of their order in the URL.
pub fn parse(
    defaults: BevyZeroverseConfig,
    pairs: impl IntoIterator<Item = (String, String)>,
) -> Result<BevyZeroverseConfig, String> {
    let mut old = BTreeMap::new();
    let mut dotted = BTreeMap::new();
    let mut bytes = 0;
    for (key, value) in pairs {
        bytes += key.len() + value.len();
        if bytes > 262_144 || old.len() + dotted.len() >= 4096 {
            return Err("Viewer URL exceeds size limit".into());
        }
        let key = key.replace('-', "_");
        let (head, tail) = key.split_once('.').unwrap_or((&key, ""));
        let head = root(head);
        let key = if tail.is_empty() {
            head.to_owned()
        } else {
            format!("{head}.{tail}")
        };
        let map = if !tail.is_empty() {
            &mut dotted
        } else {
            &mut old
        };
        if map.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate viewer setting: {key}"));
        }
    }
    let mut disabled: Vec<String> = old
        .iter()
        .filter(|(_, v)| v.trim() == "null")
        .map(|(k, _)| k.clone())
        .collect();
    let config = legacy(defaults, old)?;
    let mut value = expand(&config);
    value["viewer_state"] = config
        .viewer_state
        .as_deref()
        .map(super::view_state::ViewState::parse)
        .transpose()?
        .map(|s| json!(s))
        .unwrap_or_else(|| json!(super::view_state::ViewState::default()));
    let mut paths: Vec<_> = dotted.into_iter().collect();
    paths.sort_by(|(a, _), (b, _)| {
        a.split('.')
            .count()
            .cmp(&b.split('.').count())
            .then_with(|| {
                // Numeric path segments must apply in numeric order when growing arrays.
                a.split('.')
                    .zip(b.split('.'))
                    .map(|(a, b)| match (a.parse::<usize>(), b.parse::<usize>()) {
                        (Ok(a), Ok(b)) => a.cmp(&b),
                        _ => a.cmp(b),
                    })
                    .find(|o| !o.is_eq())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    for (path, raw) in paths {
        if disabled.iter().any(|p| path.starts_with(&format!("{p}."))) {
            return Err(format!(
                "setting inside explicitly disabled program: {path}"
            ));
        }
        let parsed = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
        if parsed.is_null() {
            disabled.push(path.clone());
        }
        set(&mut value, &path.split('.').collect::<Vec<_>>(), "", parsed)?;
    }
    let view: super::view_state::ViewState =
        serde_json::from_value(value["viewer_state"].clone()).map_err(|e| e.to_string())?;
    view.validate()?;
    value["viewer_state"] = json!(view).to_string().into();
    serde_json::from_value(value).map_err(|e| e.to_string())
}

fn set(value: &mut Value, keys: &[&str], path: &str, replacement: Value) -> Result<(), String> {
    if keys.is_empty() {
        *value = if replacement == json!({}) {
            optional(path).unwrap_or(replacement)
        } else {
            replacement
        };
        return Ok(());
    }
    if keys.len() > 32 || keys[0].is_empty() {
        return Err("Invalid viewer setting path".into());
    }
    if value.is_null() {
        *value = optional(path).unwrap_or_else(|| {
            if keys[0].parse::<usize>().is_ok() {
                json!([])
            } else {
                json!({})
            }
        });
    }
    let next = if path.is_empty() {
        keys[0].to_owned()
    } else {
        format!("{path}.{}", keys[0])
    };
    if let Some(array) = value.as_array_mut() {
        let index: usize = keys[0]
            .parse()
            .map_err(|_| format!("expected array index: {next}"))?;
        if index > array.len() || index >= 4096 {
            return Err(format!("non-contiguous or excessive index: {next}"));
        }
        if index == array.len() {
            array.push(Value::Null);
        }
        set(&mut array[index], &keys[1..], &next, replacement)
    } else if let Some(object) = value.as_object_mut() {
        if path.is_empty() && !object.contains_key(keys[0]) {
            return Err(format!("unknown viewer setting: {next}"));
        }
        // Final typed deserialization rejects unknown policy fields. Dynamic
        // pending edits/actor waypoint records are validated by their owner.
        set(
            object.entry(keys[0]).or_insert(Value::Null),
            &keys[1..],
            &next,
            replacement,
        )
    } else {
        Err(format!("setting is not a group: {path}"))
    }
}

/// Sparse, readable URL fields. Arrays use numeric path segments; only empty
/// arrays/objects need two-character markers, never an encoded JSON document.
pub fn encode(config: &BevyZeroverseConfig, view: &super::view_state::ViewState) -> String {
    let mut current = expand(config);
    let mut defaults = expand(&super::web_viewer_defaults());
    current["viewer_state"] = json!(view);
    defaults["viewer_state"] = json!(super::view_state::ViewState::default());
    let mut pairs = Vec::new();
    for (key, value) in current.as_object().unwrap() {
        flatten(value, &defaults[key], key, &mut pairs);
    }
    pairs
        .into_iter()
        .map(|(path, value)| {
            let (head, tail) = path.split_once('.').unwrap_or((&path, ""));
            let path = if tail.is_empty() {
                short(head).to_owned()
            } else {
                format!("{}.{tail}", short(head))
            };
            let value = if let Some(s) = value
                .as_str()
                .filter(|s| tail.is_empty() || serde_json::from_str::<Value>(s).is_err())
            {
                s.to_owned()
            } else if let Some(n) = value.as_f64().filter(|n| {
                // Expanded typed f32 policies are stored as f64 JSON numbers.
                // Use their shortest round-tripping f32 spelling instead of
                // exposing conversion tails such as 0.6000000238418579.
                value.is_f64() && (*n as f32) as f64 == *n
            }) {
                serde_json::to_string(&(n as f32)).expect("finite JSON number")
            } else {
                value.to_string()
            };
            format!("{}={}", escape(&path), escape(&value))
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn flatten(value: &Value, base: &Value, path: &str, pairs: &mut Vec<(String, Value)>) {
    if value == base {
        return;
    }
    match value {
        Value::Object(object) => {
            let enabled = if !base.is_object() {
                pairs.push((path.into(), json!({})));
                optional(path).unwrap_or(json!({}))
            } else {
                base.clone()
            };
            for (k, v) in object {
                if v.is_null() && enabled.get(k).is_none() {
                    // In a sparse pending patch, null is an explicit deletion.
                    pairs.push((format!("{path}.{k}"), Value::Null));
                } else {
                    flatten(v, &enabled[k], &format!("{path}.{k}"), pairs);
                }
            }
        }
        Value::Array(array) => {
            let reset = base.as_array().is_none_or(|a| a.len() != array.len());
            if reset {
                pairs.push((path.into(), json!([])));
            }
            for (i, v) in array.iter().enumerate() {
                if reset && v.is_null() {
                    pairs.push((format!("{path}.{i}"), Value::Null));
                    continue;
                }
                flatten(
                    v,
                    if reset { &Value::Null } else { &base[i] },
                    &format!("{path}.{i}"),
                    pairs,
                );
            }
        }
        _ => pairs.push((path.into(), value.clone())),
    }
}
fn escape(s: &str) -> String {
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

pub fn expand(config: &BevyZeroverseConfig) -> Value {
    let mut v = serde_json::to_value(config).expect("serializable config");
    v["indoor_camera"] = json!(config
        .indoor_camera
        .as_deref()
        .and_then(|s| CameraSettings::parse(s).ok())
        .unwrap_or_default());
    for key in [
        "duration_seconds",
        "handheld",
        "overlap_mixture",
        "lenses",
        "ceiling_mount",
    ] {
        v["indoor_camera"]
            .as_object_mut()
            .unwrap()
            .entry(key)
            .or_insert(Value::Null);
    }
    v["indoor_appearance"] = json!(config
        .indoor_appearance
        .as_deref()
        .and_then(|s| AppearanceSettings::parse(s).ok())
        .unwrap_or_default());
    v["indoor_clutter"] = json!(
        crate::scene::procedural_indoor::items::distribution::ClutterSettings::parse(
            config.indoor_clutter.as_deref().unwrap_or("{}")
        )
        .unwrap_or_default()
    );
    v["human_motion"] = config
        .human_motion
        .as_deref()
        .and_then(|s| crate::human_motion::HumanMotionConfig::parse(s).ok())
        .map(|v| json!(v))
        .unwrap_or(Value::Null);
    v["indoor_motion"] = config
        .indoor_motion
        .as_deref()
        .and_then(|s| crate::scene::procedural_indoor::motion::IndoorMotionConfig::parse(s).ok())
        .map(|v| json!(v))
        .unwrap_or(Value::Null);
    v["viewer_state"] = Value::Null;
    v
}

/// Active optical prior; mounting and free-camera policies remain independently editable.
fn legacy(
    defaults: BevyZeroverseConfig,
    pairs: impl IntoIterator<Item = (String, String)>,
) -> Result<BevyZeroverseConfig, String> {
    let mut json = serde_json::to_value(defaults).map_err(|e| e.to_string())?;
    for (key, value) in pairs {
        let key = key.replace('-', "_");
        let field = json
            .get_mut(&key)
            .ok_or_else(|| format!("unknown viewer setting: {key}"))?;
        let parsed = serde_json::from_str(&value)
            .unwrap_or_else(|_| serde_json::Value::String(value.clone()));
        *field = match key.as_str() {
            "scene_type" => ZeroverseSceneType::from_str(&value, true)
                .ok()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(parsed),
            "indoor_layout" => IndoorLayout::from_str(&value, true)
                .ok()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(parsed),
            "indoor_quality" => {
                crate::scene::procedural_indoor::IndoorQuality::from_str(&value, true)
                    .ok()
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|e| e.to_string())?
                    .unwrap_or(parsed)
            }
            "render_mode" => RenderMode::from_str(&value, true)
                .ok()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(parsed),
            _ if field.is_string() => serde_json::Value::String(value),
            _ => parsed,
        };
    }
    serde_json::from_value(json).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decode(query: &str) -> Vec<(String, String)> {
        fn unescape(s: &str) -> String {
            let mut bytes = Vec::new();
            let mut input = s.bytes();
            while let Some(b) = input.next() {
                if b == b'%' {
                    let hex: String = [input.next().unwrap(), input.next().unwrap()]
                        .into_iter()
                        .map(char::from)
                        .collect();
                    bytes.push(u8::from_str_radix(&hex, 16).unwrap());
                } else {
                    bytes.push(b);
                }
            }
            String::from_utf8(bytes).unwrap()
        }
        query
            .split('&')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let (a, b) = s.split_once('=').unwrap();
                (unescape(a), unescape(b))
            })
            .collect()
    }
    #[test]
    fn dotted_policies_are_sparse_order_independent_and_reject_malformed_paths() {
        let base = web_viewer_defaults();
        let pairs = [
            ("clutter.probability", "0.35"),
            ("clutter.bias", "0.7"),
            ("camera.lenses.fov_degrees.1", "210"),
            ("camera.lenses.probability", "0.8"),
            ("indoor_seed", "18446744073709551615"),
        ]
        .map(|(a, b)| (a.into(), b.into()));
        let config = parse(base.clone(), pairs.clone()).unwrap();
        let view = super::super::view_state::ViewState::default();
        let query = encode(&config, &view);
        assert!(query.contains("clutter.probability="));
        assert!(query.contains("clutter.probability=0.35"), "{query}");
        assert!(!query.contains("0.349999994"), "{query}");
        assert!(!query.contains("%7B") && !query.contains("%5B"), "{query}");
        assert_eq!(
            expand(&config),
            expand(&parse(base.clone(), pairs.into_iter().rev()).unwrap())
        );
        assert_eq!(
            expand(&config),
            expand(&parse(base.clone(), decode(&query)).unwrap())
        );
        for pairs in [
            vec![("clutter.typo", "1")],
            vec![("camera.lenses.probability", "2")],
            vec![("camera.lenses.fov_degrees.200", "90")],
            vec![("unknown.foo", "1")],
            vec![
                ("clutter.probability", "0.4"),
                ("indoor_clutter.probability", "0.6"),
            ],
            vec![
                ("camera.lenses", "null"),
                ("camera.lenses.probability", "0.5"),
            ],
        ] {
            assert!(parse(
                base.clone(),
                pairs.into_iter().map(|(a, b)| (a.into(), b.into()))
            )
            .is_err());
        }
    }
    #[test]
    fn scalar_looking_window_names_roundtrip_as_strings() {
        let base = web_viewer_defaults();
        for name in ["123", "true", "null", "\"quoted\"", "Room & café + 1"] {
            let config = BevyZeroverseConfig {
                name: name.into(),
                ..base.clone()
            };
            let text = encode(&config, &Default::default());
            assert_eq!(parse(base.clone(), decode(&text)).unwrap().name, name);
        }
    }
    #[test]
    fn old_json_links_and_dynamic_view_arrays_survive_roundtrip() {
        let base = web_viewer_defaults();
        let config = parse(
            base.clone(),
            vec![
                (
                    "indoor_camera".into(),
                    r#"{"handheld":{},"lenses":null}"#.into(),
                ),
                ("indoor_clutter".into(), r#"{"probability":0}"#.into()),
            ],
        )
        .unwrap();
        let view = super::super::view_state::ViewState {
            expanded: (0..12)
                .map(|i| format!("Section {i} • & + = 日本語"))
                .collect(),
            pending: Some(
                json!({"indoor_camera":{"intrinsics":{"ultra_wide_probability":null}},"indoor_clutter":{"bias":0.6}}),
            ),
            ..Default::default()
        };
        let query = encode(&config, &view);
        let restored = parse(base, decode(&query)).unwrap();
        assert_eq!(expand(&config), expand(&restored));
        let recovered =
            super::super::view_state::ViewState::parse(restored.viewer_state.as_deref().unwrap())
                .unwrap();
        assert_eq!(json!(view), json!(recovered));
    }
}
