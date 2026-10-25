//! Peer-aware placement and bounded framing refinement for the baseline preset.
//! Sharing camera zero does not imply that two peers see each other. Targets are
//! preferred-search constraints, not new rejection criteria for valid rooms.
use super::{
    coverage::{Coverage, VisibilityView},
    CameraSettings, IndoorCamera, IndoorManifest,
};
use serde::Serialize;
mod refine;
#[cfg(test)]
mod tests;
pub(super) use refine::refine;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CameraCoherencePolicy {
    pub version: u32,
    pub baseline: f32,
    pub target_min_directed_overlap: f32,
    pub target_min_containment: f32,
    /// Fixed upper bound on prospective framings, independent of rejection rate.
    pub max_candidates: usize,
    pub max_turn_degrees: f32,
    pub preferred_min_directed_overlap: f32,
    pub preferred_min_containment: f32,
}
impl CameraCoherencePolicy {
    pub(super) fn new(settings: &CameraSettings, count: usize) -> Option<Self> {
        if count < 3 || settings.overlap_mixture.is_some() || settings.handheld.is_some() {
            return None;
        }
        let baseline = settings.multiview.as_ref()?.mixed_rig_baseline()?;
        Some(Self {
            version: 1,
            baseline,
            target_min_directed_overlap: 0.10 - 0.06 * baseline,
            target_min_containment: 0.40 - 0.30 * baseline,
            max_candidates: (12 * count).min(144),
            max_turn_degrees: 60.,
            preferred_min_directed_overlap: 0.05 - 0.04 * baseline,
            preferred_min_containment: 0.10 - 0.08 * baseline,
        })
    }

    /// A wide sensor cannot devote most of its pixels to a nested narrow view.
    /// This target cap uses the calibrated aligned footprint, not fabricated
    /// normalized overlap. Exported overlap always remains the measured fraction.
    pub fn pair_overlap_target(self, a: &IndoorCamera, b: &IndoorCamera, aspect: f32) -> f32 {
        self.target_min_directed_overlap
            .min((0.5 * super::rig::footprint::overlap(a, b, aspect).unwrap_or(1.)).max(0.005))
    }

    pub(super) fn preferred_floor(self, a: &IndoorCamera, b: &IndoorCamera, aspect: f32) -> f32 {
        self.preferred_min_directed_overlap
            .min(self.pair_overlap_target(a, b, aspect))
    }

    pub(super) fn supports(
        self,
        coverage: &Coverage,
        a: &[VisibilityView],
        b: &[VisibilityView],
        floor: f32,
    ) -> bool {
        a.iter().zip(b).all(|(a, b)| {
            let ab = coverage.shared(a, b).0;
            ab >= floor && {
                let ba = coverage.shared(b, a).0;
                ba >= floor && ab.max(ba) >= self.preferred_min_containment
            }
        })
    }
}
impl IndoorManifest {
    pub fn camera_coherence_policy(&self) -> Option<CameraCoherencePolicy> {
        CameraCoherencePolicy::new(&self.camera_settings, self.cameras.len())
    }
}
