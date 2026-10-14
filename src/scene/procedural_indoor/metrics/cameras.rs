//! Camera-group diagnostics share the sampler's time convention and geometry.
use super::super::{
    cameras::diversity::PATH_SAMPLES, layout::IndoorManifest, metrics_sort::NumericCollector,
};
use bevy::prelude::*;
use std::io::{self, Write};

pub(super) fn record(
    scene: &IndoorManifest,
    numeric: &mut NumericCollector,
    paths: &mut impl Write,
    groups: &mut impl Write,
) -> io::Result<()> {
    let d = &scene.camera_diagnostics;
    for (name, n) in [
        ("reference_proposals", d.reference_proposals),
        ("long_reference_requests", d.long_reference_requests),
        ("short_reference_recoveries", d.short_reference_recoveries),
        ("group_restarts", d.group_restarts),
        ("peer_proposals", d.peer_proposals),
        ("endpoint_adjustments", d.endpoint_adjustments),
    ] {
        numeric.push(&format!("camera_search_{name}"), n as f64)?;
    }
    for (name, n) in ["clearance", "view", "endpoint", "curve", "coverage"]
        .into_iter()
        .zip(d.reference_rejections)
    {
        numeric.push(&format!("camera_reference_rejected_{name}"), n as f64)?;
    }
    for (name, n) in ["clearance", "rig", "length", "curve", "coverage", "overlap"]
        .into_iter()
        .zip(d.peer_rejections)
    {
        numeric.push(&format!("camera_peer_rejected_{name}"), n as f64)?;
    }
    if let Some(group) = scene.camera_group_geometry() {
        writeln!(
            groups,
            "{},{},{},{},{},{}",
            scene.seed,
            group.min_pairwise_baseline_m,
            group.min_reference_baseline_m,
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
                "camera_group_min_reference_baseline_m",
                Some(group.min_reference_baseline_m),
            ),
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
        let mut near_horizontal = 0usize;
        for step in 0..PATH_SAMPLES {
            let time = step as f32 / (PATH_SAMPLES - 1) as f32;
            let pose = path.sample(time);
            let p = pose.translation;
            let pitch = (pose.rotation * Vec3::NEG_Z)
                .y
                .clamp(-1., 1.)
                .asin()
                .to_degrees();
            near_horizontal += usize::from(pitch.abs() <= 5.);
            numeric.push("camera_path_pitch_degrees", pitch as f64)?;
            numeric.push(
                "camera_path_eye_height_m",
                (p.y - scene.floor_height(p.xz())) as f64,
            )?;
            writeln!(
                paths,
                "{},{},{},{},{},{}",
                scene.seed, i, time, p.x, p.y, p.z
            )?;
        }
        numeric.push(
            "camera_near_horizontal_path_fraction",
            near_horizontal as f64 / PATH_SAMPLES as f64,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        cameras::CameraMotion,
        layout::{IndoorCamera, IndoorLayout},
    };

    #[test]
    fn path_metrics_use_rendered_orientation_and_floor_relative_height() {
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.camera_settings.multiview = None;
        let position = Vec3::new(0., scene.floor_height(Vec2::ZERO) + 1.4, 0.);
        scene.cameras = vec![IndoorCamera {
            principal_point_offset: None,
            start: position,
            end: position,
            // Deliberately different from explicit orientations, as in handheld captures.
            target: position + Vec3::new(0., -2., -2.),
            fov_degrees: 60.,
            motion: Some(CameraMotion {
                orientations: Some([Quat::from_rotation_x(2_f32.to_radians()); 2]),
                control: [position; 2],
                target_end: position + Vec3::NEG_Z * 3.,
                roll: [0.; 2],
                route: Vec::new(),
            }),
        }];
        let directory = tempfile::tempdir().unwrap();
        let mut numeric = NumericCollector::new(directory.path()).unwrap();
        record(&scene, &mut numeric, &mut Vec::new(), &mut Vec::new()).unwrap();
        let values = numeric.finish().unwrap();
        assert_eq!(values["camera_path_pitch_degrees"].count, PATH_SAMPLES);
        assert!((values["camera_path_pitch_degrees"].mean - 2.).abs() < 1e-4);
        assert!((values["camera_path_eye_height_m"].mean - 1.4).abs() < 1e-5);
        assert_eq!(values["camera_near_horizontal_path_fraction"].mean, 1.);
    }
}
