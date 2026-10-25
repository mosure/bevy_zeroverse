//! Search effort and recovery receipts. Counts include discarded anchors; final
//! path distributions are exported separately from these proposal diagnostics.
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraSamplingDiagnostics {
    pub reference_proposals: usize,
    pub long_reference_requests: usize,
    pub short_reference_recoveries: usize,
    pub reference_rejections: [usize; 5],
    pub group_restarts: usize,
    /// Bounded mixed-rig search reused another feasible prefix candidate.
    #[serde(default)]
    pub prefix_backtracks: usize,
    /// Connected search may use the stricter reference-star search to complete
    /// a cramped rig. This is recorded; no overlap or spacing bound is relaxed.
    #[serde(default)]
    pub connected_reference_recoveries: usize,
    pub peer_proposals: usize,
    pub peer_rejections: [usize; 6],
    pub endpoint_adjustments: usize,
    /// A bounded peer-aware placement attempt succeeded without fallback.
    #[serde(default)]
    pub peer_search_succeeded: bool,
    #[serde(default)]
    pub peer_search_fallbacks: usize,
    /// Additional discarded work before the ordinary sampler's counters above.
    #[serde(default)]
    pub peer_search_discarded_proposals: usize,
    #[serde(default)]
    pub peer_search_discarded_reference_proposals: usize,
    /// Gaze-only refinement of already-valid rigs; never placement retries.
    #[serde(default)]
    pub peer_framing_candidates: usize,
    #[serde(default)]
    pub peer_framing_updates: usize,
    #[serde(default)]
    pub peer_weak_pairs_before: usize,
    #[serde(default)]
    pub peer_weak_pairs_after: usize,
}
