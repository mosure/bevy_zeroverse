//! Small, lossless object deltas for URLs and unapplied controls. Arrays and
//! explicit nulls are atomic; an absent key never means deleting a setting.
use serde_json::Value;

pub(super) fn between(value: &Value, baseline: &Value) -> Option<Value> {
    if value == baseline {
        return None;
    }
    match (value, baseline) {
        (Value::Object(current), Value::Object(base)) => Some(Value::Object(
            current
                .iter()
                .filter_map(|(key, value)| {
                    between(value, base.get(key).unwrap_or(&Value::Null))
                        .map(|value| (key.clone(), value))
                })
                .chain(
                    base.keys()
                        .filter(|key| !current.contains_key(*key))
                        .map(|key| (key.clone(), Value::Null)),
                )
                .collect(),
        )),
        _ => Some(value.clone()),
    }
}

pub(super) fn merge(target: &mut Value, patch: &Value) {
    if let (Some(target), Some(patch)) = (target.as_object_mut(), patch.as_object()) {
        for (key, value) in patch {
            merge(target.entry(key.clone()).or_insert(Value::Null), value);
        }
    } else {
        *target = patch.clone();
    }
}
