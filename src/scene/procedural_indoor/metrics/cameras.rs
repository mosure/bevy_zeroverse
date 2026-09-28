//! Camera-group diagnostics share the sampler's time convention and geometry.
use super::super::{
    cameras::diversity::PATH_SAMPLES, layout::IndoorManifest, metrics_sort::NumericCollector,
};
use std::io::{self, Write};

pub(super) fn record(
    scene: &IndoorManifest,
    numeric: &mut NumericCollector,
    paths: &mut impl Write,
    groups: &mut impl Write,
) -> io::Result<()> {
    if let Some(group) = scene.camera_group_geometry() {
        writeln!(
            groups,
            "{},{},{},{},{}",
            scene.seed,
            group.min_pairwise_baseline_m,
            group.max_reference_baseline_m,
            group
                .min_horizontal_spread
                .map(|v| v.to_string())
                .unwrap_or_default(),
            group
                .min_relative_motion
                .map(|v| v.to_string())
                .unwrap_or_default()
        )?;
        for (name, value) in [
            (
                "camera_group_min_pairwise_baseline_m",
                Some(group.min_pairwise_baseline_m),
            ),
            (
                "camera_group_max_reference_baseline_m",
                Some(group.max_reference_baseline_m),
            ),
            (
                "camera_group_min_horizontal_spread",
                group.min_horizontal_spread,
            ),
            (
                "camera_group_min_relative_motion",
                group.min_relative_motion,
            ),
        ] {
            if let Some(value) = value {
                numeric.push(name, value as f64)?;
            }
        }
    }
    for (i, camera) in scene.cameras.iter().enumerate() {
        let mut path = camera.runtime_trajectory();
        for step in 0..PATH_SAMPLES {
            let time = step as f32 / (PATH_SAMPLES - 1) as f32;
            let p = path.sample(time).translation;
            writeln!(
                paths,
                "{},{},{},{},{},{}",
                scene.seed, i, time, p.x, p.y, p.z
            )?;
        }
    }
    Ok(())
}
