//! Accepted-manifest shortcut-risk diagnostics. These are descriptive statistics,
//! not a score for learning utility: repeated manufacture and alignment can be real.
//! Collection is opt-in with the existing metrics export, never on the render path.
use super::super::{
    layout::{IndoorManifest, IndoorObject},
    metrics_sort::NumericCollector,
};
use bevy::prelude::*;
use serde::Serialize;
use std::{collections::BTreeMap, io};

pub const POLICY: &str = "Main-room objects only (neighbor=false), by kind, after admission. Dimensions are object-local metres; repeated-size ordered pairs use rounded 1 cm XYZ bins and exclude self pairs. Axis alignment means yaw within 2 degrees of a room X/Z axis. Support coordinates divide local X/Z by the actual support width/depth; centered means both within 0.02. Per-kind scene fractions omit undefined denominators; count metrics state the denominator. These are regularity diagnostics, not mesh-duplicate or learning-utility estimates.";

#[derive(Debug, Serialize)]
pub(super) struct KindStats {
    pub objects: usize,
    pub equal_size_pairs_1cm: usize,
    pub size_pairs: usize,
    pub axis_aligned_2deg: usize,
    pub supported: usize,
    pub support_centered_2pct: usize,
}

pub(super) fn support_coordinates(scene: &IndoorManifest, o: &IndoorObject) -> Option<Vec3> {
    let support = scene.objects.get(o.support?)?;
    let local = support
        .transform()
        .compute_affine()
        .inverse()
        .transform_point3(o.position);
    Some(local / support.size)
}

pub(super) fn axis_distance(yaw: f32) -> f32 {
    (4. * yaw).sin().atan2((4. * yaw).cos()).abs() * 0.25
}

pub(super) fn measure(scene: &IndoorManifest, objects: &[&IndoorObject]) -> KindStats {
    let mut sizes = BTreeMap::<[i32; 3], usize>::new();
    let mut stats = KindStats {
        objects: objects.len(),
        equal_size_pairs_1cm: 0,
        size_pairs: objects.len() * objects.len().saturating_sub(1),
        axis_aligned_2deg: 0,
        supported: 0,
        support_centered_2pct: 0,
    };
    for o in objects {
        let size = o.size.to_array().map(|v| (v * 100.).round() as i32);
        *sizes.entry(size).or_default() += 1;
        stats.axis_aligned_2deg += usize::from(axis_distance(o.yaw) <= 2_f32.to_radians());
        if let Some(local) = support_coordinates(scene, o) {
            stats.supported += 1;
            stats.support_centered_2pct += usize::from(local.x.abs().max(local.z.abs()) <= 0.02);
        }
    }
    stats.equal_size_pairs_1cm = sizes.values().map(|&n| n * n.saturating_sub(1)).sum();
    stats
}

pub(super) fn record(scene: &IndoorManifest, numeric: &mut NumericCollector) -> io::Result<()> {
    let mut kinds = BTreeMap::<String, Vec<&IndoorObject>>::new();
    for o in scene.objects.iter().filter(|o| !o.neighbor) {
        kinds.entry(format!("{:?}", o.kind)).or_default().push(o);
    }
    for (kind, objects) in kinds {
        let prefix = format!("regularity_{kind}");
        let stats = measure(scene, &objects);
        for (key, value) in [
            ("objects", stats.objects),
            ("size_pairs", stats.size_pairs),
            ("equal_size_pairs_1cm", stats.equal_size_pairs_1cm),
            ("supported", stats.supported),
            ("support_centered_2pct", stats.support_centered_2pct),
        ] {
            numeric.push(&format!("{prefix}_{key}"), value as f64)?;
        }
        numeric.push(
            &format!("{prefix}_axis_aligned_fraction_2deg"),
            stats.axis_aligned_2deg as f64 / stats.objects as f64,
        )?;
        if stats.size_pairs > 0 {
            numeric.push(
                &format!("{prefix}_equal_size_pair_fraction_1cm"),
                stats.equal_size_pairs_1cm as f64 / stats.size_pairs as f64,
            )?;
        }
        if stats.supported > 0 {
            numeric.push(
                &format!("{prefix}_support_centered_fraction_2pct"),
                stats.support_centered_2pct as f64 / stats.supported as f64,
            )?;
        }
        for o in objects {
            for (axis, value) in ["width_m", "height_m", "depth_m"]
                .into_iter()
                .zip(o.size.to_array())
            {
                numeric.push(&format!("{prefix}_{axis}"), value as f64)?;
            }
            if let Some(local) = support_coordinates(scene, o) {
                numeric.push(&format!("{prefix}_support_x_fraction"), local.x as f64)?;
                numeric.push(&format!("{prefix}_support_z_fraction"), local.z as f64)?;
                let yaw = o.yaw - scene.objects[o.support.unwrap()].yaw;
                numeric.push(
                    &format!("{prefix}_support_yaw_radians"),
                    yaw.sin().atan2(yaw.cos()) as f64,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
