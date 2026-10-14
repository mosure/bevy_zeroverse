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
    pub peer_proposals: usize,
    pub peer_rejections: [usize; 6],
    pub endpoint_adjustments: usize,
}
