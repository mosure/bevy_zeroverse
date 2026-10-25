//! Baseline-program contracts for rigs mixing mounting heights and lens coverage.
//! Raw metric baselines and directed visibility exports are never normalized.
//! This module describes admission, independently of how candidates are proposed.
use super::{
    diversity::{geometry, Track},
    multiview::{MultiViewSettings, OverlapSample},
    IndoorCamera, IndoorManifest,
};
use bevy::prelude::*;
use serde::Serialize;
pub(super) mod footprint;
#[cfg(test)]
mod tests;

/// Expanded, inspectable version-one baseline program. These are proxy admission
/// bounds, not guarantees of rendered pixel coverage. The ordinary multiview
/// fields still bound physical all-pair spacing, group spread and relative motion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MixedRigContract {
    pub version: u32,
    pub baseline: f32,
    /// Cross-height rigs cannot be close in 3D. Bound their horizontal distance;
    /// within each mounting group retain a 3D diameter of twice the ordinary
    /// reference radius (and the original radius around camera zero).
    pub max_cross_mount_horizontal_m: f32,
    /// Nominal bidirectional floor, capped per pair by half its aligned sensor
    /// footprint overlap (minimum 0.005). See `pair_overlap_floor`.
    pub min_directed_overlap: f32,
    /// For equal mounting roles, one view must meet the ordinary min_overlap.
    /// A downward overview / room view has this separate containment bound.
    pub min_cross_mount_containment: f32,
}

impl MixedRigContract {
    pub(super) fn new(policy: &MultiViewSettings) -> Option<Self> {
        let baseline = policy.mixed_rig_baseline()?;
        Some(Self {
            version: 1,
            baseline,
            max_cross_mount_horizontal_m: 0.80 + 9.20 * baseline,
            min_directed_overlap: 0.05 - 0.03 * baseline,
            min_cross_mount_containment: 0.45 - 0.25 * baseline,
        })
    }

    pub(super) fn accepts_visibility(
        self,
        policy: &MultiViewSettings,
        a: &IndoorCamera,
        b: &IndoorCamera,
        samples: &[OverlapSample],
        floor: f32,
    ) -> bool {
        let containment = if a.is_ceiling_mounted() != b.is_ceiling_mounted() {
            self.min_cross_mount_containment
        } else {
            policy.min_overlap
        };
        samples.len() == super::multiview::CHECK_TIMES.len()
            && samples.iter().all(|s| {
                let lo = s.reference_to_view.min(s.view_to_reference);
                let hi = s.reference_to_view.max(s.view_to_reference);
                s.reference_to_view.is_finite()
                    && s.view_to_reference.is_finite()
                    && lo >= floor
                    && hi >= containment
            })
    }

    /// Calibrated, centered sensor-footprint cap, independent of scene/occlusion.
    /// Measured overlaps remain raw. This keeps nested wide/narrow sensors
    /// feasible without permitting a completely unsupported direction.
    pub fn pair_overlap_floor(self, a: &IndoorCamera, b: &IndoorCamera, aspect: f32) -> f32 {
        self.min_directed_overlap
            .min((0.5 * footprint::overlap(a, b, aspect).unwrap_or(1.)).max(0.005))
    }

    /// Same-role groups each have a bounded diameter; otherwise a ceiling reference
    /// could permit room cameras to spread arbitrarily even at baseline zero.
    pub(super) fn accepts_geometry(
        self,
        policy: &MultiViewSettings,
        cameras: &[&IndoorCamera],
        tracks: &[&Track],
        complete: bool,
    ) -> bool {
        for (i, (a, ta)) in cameras.iter().zip(tracks).enumerate() {
            for (b, tb) in cameras.iter().zip(tracks).skip(i + 1) {
                let same_mount = a.is_ceiling_mounted() == b.is_ceiling_mounted();
                for (pa, pb) in ta.0.iter().zip(tb.0) {
                    let distance = pa.distance(pb);
                    if distance + 1e-5 < policy.min_baseline
                        || (i == 0 && distance + 1e-5 < policy.min_reference_baseline)
                        || (same_mount
                            && distance > policy.max_baseline * if i == 0 { 1. } else { 2. } + 1e-5)
                        || (i == 0
                            && !same_mount
                            && (pa.xz().distance(pb.xz())
                                > self.max_cross_mount_horizontal_m + 1e-5
                                || pa.xz().distance(pb.xz()) + 1e-5
                                    < policy.min_reference_baseline))
                    {
                        return false;
                    }
                }
                // Keep synchronized relative-motion diversity, including mixed
                // static/moving roles. Reuse the same public geometry oracle.
                if super::diversity::relative_motion(ta, tb)
                    .is_some_and(|v| v + 1e-5 < 0.15 * policy.trajectory_variation)
                {
                    return false;
                }
            }
        }
        if complete && tracks.len() >= 3 {
            let tracks: Vec<_> = tracks.iter().map(|t| Track(t.0)).collect();
            if geometry(&tracks, None)
                .and_then(|g| g.min_horizontal_spread)
                .is_some_and(|s| s + 1e-5 < policy.min_spread)
            {
                return false;
            }
        }
        true
    }
}

pub(super) fn accepts_geometry(
    policy: &MultiViewSettings,
    contract: Option<MixedRigContract>,
    cameras: &[IndoorCamera],
    tracks: &[Track],
    extra: Option<(&IndoorCamera, &Track)>,
    complete: bool,
) -> bool {
    if let Some(contract) = contract {
        let cameras: Vec<_> = cameras.iter().chain(extra.map(|e| e.0)).collect();
        let tracks: Vec<_> = tracks.iter().chain(extra.map(|e| e.1)).collect();
        return contract.accepts_geometry(policy, &cameras, &tracks, complete);
    }
    geometry(tracks, extra.map(|e| e.1)).is_none_or(|mut g| {
        if !complete {
            // Adding a view can increase or decrease group covariance spread.
            // It cannot reject a prefix; pair spacing and relative motion can.
            g.min_horizontal_spread = None;
        }
        g.accepts(policy)
    })
}

impl IndoorManifest {
    /// Inspect the effective baseline contract without interpreting raw
    /// co-visibility as a normalized/undirected score. None means exact policy.
    pub fn mixed_rig_contract(&self) -> Option<MixedRigContract> {
        (self.cameras.len() > 1
            && self.camera_settings.overlap_mixture.is_none()
            && self
                .cameras
                .iter()
                .any(|c| c.is_ceiling_mounted() || c.fisheye.is_some()))
        .then(|| MixedRigContract::new(self.camera_settings.multiview.as_ref()?))
        .flatten()
    }

    pub fn accepts_camera_geometry(&self) -> bool {
        self.camera_settings
            .multiview
            .as_ref()
            .is_none_or(|policy| {
                accepts_geometry(
                    policy,
                    self.mixed_rig_contract(),
                    &self.cameras,
                    &self.cameras.iter().map(Track::new).collect::<Vec<_>>(),
                    None,
                    true,
                )
            })
    }
}
