use std::collections::{BTreeMap, BTreeSet};

use crate::hooks::{HookSnapshot, HookTensor};

#[derive(Debug, Clone, Copy, Default)]
pub struct TensorDiffMetrics {
    pub mean_abs: f32,
    pub max_abs: f32,
    pub rmse: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookDiffStatus {
    Match,
    MissingInActual,
    ShapeMismatch,
}

#[derive(Debug, Clone)]
pub struct HookDiffEntry {
    pub key: String,
    pub status: HookDiffStatus,
    pub reference_shape: Vec<usize>,
    pub actual_shape: Option<Vec<usize>>,
    pub metrics: Option<TensorDiffMetrics>,
}

#[derive(Debug, Clone, Default)]
pub struct HookDiffReport {
    pub entries: Vec<HookDiffEntry>,
    pub extra_in_actual: Vec<String>,
}

impl HookDiffReport {
    pub fn failing_entries(&self) -> impl Iterator<Item = &HookDiffEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.status != HookDiffStatus::Match)
    }

    pub fn max_abs(&self) -> f32 {
        self.entries
            .iter()
            .filter_map(|entry| entry.metrics)
            .map(|metrics| metrics.max_abs)
            .fold(0.0f32, f32::max)
    }

    pub fn max_rmse(&self) -> f32 {
        self.entries
            .iter()
            .filter_map(|entry| entry.metrics)
            .map(|metrics| metrics.rmse)
            .fold(0.0f32, f32::max)
    }
}

pub fn compare_hook_snapshots(
    reference: &HookSnapshot,
    actual: &HookSnapshot,
    prefix: Option<&str>,
) -> HookDiffReport {
    compare_maps(&reference.tensors, &actual.tensors, prefix)
}

pub fn compare_maps(
    reference: &BTreeMap<String, HookTensor>,
    actual: &BTreeMap<String, HookTensor>,
    prefix: Option<&str>,
) -> HookDiffReport {
    let mut keys = BTreeSet::new();
    for key in reference.keys() {
        if prefix.is_none_or(|value| key.starts_with(value)) {
            keys.insert(key.clone());
        }
    }

    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        let Some(reference_tensor) = reference.get(&key) else {
            continue;
        };
        match actual.get(&key) {
            None => entries.push(HookDiffEntry {
                key,
                status: HookDiffStatus::MissingInActual,
                reference_shape: reference_tensor.shape.clone(),
                actual_shape: None,
                metrics: None,
            }),
            Some(actual_tensor) if actual_tensor.shape != reference_tensor.shape => {
                entries.push(HookDiffEntry {
                    key,
                    status: HookDiffStatus::ShapeMismatch,
                    reference_shape: reference_tensor.shape.clone(),
                    actual_shape: Some(actual_tensor.shape.clone()),
                    metrics: None,
                })
            }
            Some(actual_tensor) => entries.push(HookDiffEntry {
                key,
                status: HookDiffStatus::Match,
                reference_shape: reference_tensor.shape.clone(),
                actual_shape: Some(actual_tensor.shape.clone()),
                metrics: Some(diff(&actual_tensor.data, &reference_tensor.data)),
            }),
        }
    }

    let mut extra_in_actual: Vec<String> = actual
        .keys()
        .filter(|key| !reference.contains_key(*key))
        .filter(|key| prefix.is_none_or(|value| key.starts_with(value)))
        .cloned()
        .collect();
    extra_in_actual.sort();

    HookDiffReport {
        entries,
        extra_in_actual,
    }
}

pub fn diff(actual: &[f32], reference: &[f32]) -> TensorDiffMetrics {
    let len = actual.len().min(reference.len());
    if len == 0 {
        return TensorDiffMetrics::default();
    }

    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    for i in 0..len {
        let delta = actual[i] - reference[i];
        let abs = delta.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        sum_sq += delta * delta;
    }

    let n = len as f32;
    TensorDiffMetrics {
        mean_abs: sum_abs / n,
        max_abs,
        rmse: (sum_sq / n).sqrt(),
    }
}

pub fn assert_parity_with_thresholds(
    report: &HookDiffReport,
    max_allowed_abs: f32,
    max_allowed_rmse: f32,
) -> Result<(), String> {
    let failing_status = report
        .failing_entries()
        .map(|entry| {
            format!(
                "{}: status={:?} ref_shape={:?} actual_shape={:?}",
                entry.key, entry.status, entry.reference_shape, entry.actual_shape
            )
        })
        .collect::<Vec<_>>();
    if !failing_status.is_empty() {
        return Err(format!(
            "hook parity failed due to missing/shape mismatch entries: {}",
            failing_status.join("; ")
        ));
    }
    if !report.extra_in_actual.is_empty() {
        return Err(format!(
            "hook parity failed due to extra tensors in actual: {}",
            report.extra_in_actual.join(", ")
        ));
    }

    let max_abs = report.max_abs();
    let max_rmse = report.max_rmse();
    if max_abs > max_allowed_abs || max_rmse > max_allowed_rmse {
        return Err(format!(
            "hook parity exceeded thresholds: max_abs={max_abs:.6e} (limit {max_allowed_abs:.6e}), max_rmse={max_rmse:.6e} (limit {max_allowed_rmse:.6e})"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{HookDiffStatus, compare_maps, diff};
    use crate::hooks::HookTensor;

    #[test]
    fn diff_metrics_are_stable() {
        let metrics = diff(&[1.0, 2.0, 3.0], &[1.5, 1.0, 3.0]);
        assert!((metrics.mean_abs - 0.5).abs() < 1e-6);
        assert!((metrics.max_abs - 1.0).abs() < 1e-6);
        assert!((metrics.rmse - (1.25f32 / 3.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn compare_maps_reports_missing_and_shape_mismatch() {
        let mut reference = BTreeMap::new();
        reference.insert(
            "a".to_string(),
            HookTensor {
                shape: vec![2],
                data: vec![0.0, 1.0],
            },
        );
        reference.insert(
            "b".to_string(),
            HookTensor {
                shape: vec![1, 2],
                data: vec![0.0, 1.0],
            },
        );

        let mut actual = BTreeMap::new();
        actual.insert(
            "a".to_string(),
            HookTensor {
                shape: vec![2],
                data: vec![0.0, 2.0],
            },
        );
        actual.insert(
            "b".to_string(),
            HookTensor {
                shape: vec![2, 1],
                data: vec![0.0, 1.0],
            },
        );
        actual.insert(
            "extra".to_string(),
            HookTensor {
                shape: vec![1],
                data: vec![0.0],
            },
        );

        let report = compare_maps(&reference, &actual, None);
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.entries[0].status, HookDiffStatus::Match);
        assert_eq!(report.entries[1].status, HookDiffStatus::ShapeMismatch);
        assert_eq!(report.extra_in_actual, vec!["extra".to_string()]);
    }
}
