//! Measurements from the actual captured planes. These never filter samples.
use super::View;
use bevy::prelude::*;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct DirectedOverlap {
    pub step_index: usize,
    pub source_camera: usize,
    pub target_camera: usize,
    pub valid_source_pixels: u64,
    pub shared_pixels: u64,
    pub shared_fraction_valid: Option<f64>,
    pub baseline_m: f32,
    pub mean_triangulation_degrees: Option<f64>,
    pub triangulation_sample_count: u64,
}

pub fn rendered_overlap(
    views: &[View],
    cameras: usize,
    aabb: [[f32; 3]; 2],
) -> Vec<DirectedOverlap> {
    if !(2..=crate::render::co_visibility::MAX_CAMERAS).contains(&cameras) {
        return Vec::new();
    }
    let mut result = Vec::new();
    let lo = Vec3::from_array(aabb[0]);
    let range = Vec3::from_array(aabb[1]) - lo;
    for (step, frame) in views.chunks_exact(cameras).enumerate() {
        let origins: Vec<_> = frame
            .iter()
            .map(|v| {
                Mat4::from_cols_array_2d(&v.world_from_view)
                    .w_axis
                    .truncate()
            })
            .collect();
        for (source, view) in frame.iter().enumerate() {
            if view.co_visibility.is_empty() {
                continue;
            }
            let mut valid = 0u64;
            let mut shared = vec![0u64; cameras];
            let mut angles = vec![0f64; cameras];
            let mut angle_counts = vec![0u64; cameras];
            let stride = (view.co_visibility.len() / 16).div_ceil(4096).max(1);
            for (pixel, bytes) in view.co_visibility.as_chunks::<16>().0.iter().enumerate() {
                let fields: [f32; 4] = std::array::from_fn(|i| {
                    f32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
                });
                if fields[2] < 0.5 {
                    continue;
                }
                valid += 1;
                let mut mask = fields[0] as u16;
                let point = (pixel % stride == 0
                    && view.position.len() == view.co_visibility.len())
                .then(|| {
                    let p = &view.position[pixel * 16..pixel * 16 + 16];
                    let p: [f32; 4] = std::array::from_fn(|i| {
                        f32::from_ne_bytes(p[i * 4..i * 4 + 4].try_into().unwrap())
                    });
                    (p[3] >= 0.5).then(|| lo + Vec3::new(p[0], p[1], p[2]) * range)
                })
                .flatten();
                while mask != 0 {
                    let target = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    if target >= cameras || target == source {
                        continue;
                    }
                    shared[target] += 1;
                    if let Some(point) = point {
                        let a = (point - origins[source]).try_normalize();
                        let b = (point - origins[target]).try_normalize();
                        if let (Some(a), Some(b)) = (a, b) {
                            angles[target] += a.dot(b).clamp(-1., 1.).acos().to_degrees() as f64;
                            angle_counts[target] += 1;
                        }
                    }
                }
            }
            for target in 0..cameras {
                if target == source {
                    continue;
                }
                result.push(DirectedOverlap {
                    step_index: step,
                    source_camera: source,
                    target_camera: target,
                    valid_source_pixels: valid,
                    shared_pixels: shared[target],
                    shared_fraction_valid: (valid != 0)
                        .then(|| shared[target] as f64 / valid as f64),
                    baseline_m: origins[source].distance(origins[target]),
                    mean_triangulation_degrees: (angle_counts[target] != 0)
                        .then(|| angles[target] / angle_counts[target] as f64),
                    triangulation_sample_count: angle_counts[target],
                });
            }
        }
    }
    result
}

pub fn metadata(
    scene: &crate::scene::procedural_indoor::layout::IndoorManifest,
    views: &[View],
    cameras: usize,
    aabb: [[f32; 3]; 2],
) -> serde_json::Value {
    use rand::Rng;
    serde_json::json!({
        "schema_version":1,
        "scene_family": {"id":format!("procedural_indoor/{}/{:016x}",scene.generator_version,scene.seed),
            "split_bucket_10000":crate::scene::procedural_indoor::layout::stream(scene.seed,347).random_range(0u32..10000),
            "rule":"group all frames, cameras and appearance/sensor variants of a generator-version/seed family before assigning dataset splits"},
        "requested_pairs":scene.camera_pair_targets(),
        "placement_estimate":(views.iter().any(|v| !v.co_visibility.is_empty()) || scene.camera_settings.overlap_mixture.is_some()).then(|| scene.camera_overlap()),
        "placement_denominator":"all 13x9 source proxy rays, including misses; at five normalized trajectory parameters; not measured for RGB-only captures without overlap_mixture",
        "rendered_overlap":rendered_overlap(views,cameras,aabb),
        "rendered_denominator":"valid source pixels; exact production same-time first-surface membership, including glass",
        "triangulation":"mean angle of shared rendered world positions on a regular grid of at most 4096 source pixels/view; null when position was not exported or no shared samples",
        "physical_time":"time_seconds = normalized playback progress * explicitly configured duration_seconds; null when unassigned; motion clips are retimed to this timeline",
        "selection":"requested strata are fixed before placement retries; no rendered or model-score filtering",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }
    #[test]
    fn directed_overlap_distinguishes_no_shared_geometry_from_missing_measurement() {
        let a = View {
            world_from_view: Mat4::IDENTITY.to_cols_array_2d(),
            co_visibility: bytes(&[2., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.]),
            ..default()
        };
        let b = View {
            world_from_view: Mat4::from_translation(Vec3::X).to_cols_array_2d(),
            co_visibility: bytes(&[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            ..default()
        };
        let result = rendered_overlap(&[a, b], 2, [[0.; 3], [1.; 3]]);
        assert_eq!(result[0].shared_fraction_valid, Some(0.5));
        assert_eq!(result[1].shared_fraction_valid, Some(0.0));
        assert_eq!(result[0].baseline_m, 1.0);
        assert_eq!(result[0].mean_triangulation_degrees, None);
        assert!(
            rendered_overlap(&[View::default(), View::default()], 2, [[0.; 3], [1.; 3]]).is_empty()
        );
    }
}
