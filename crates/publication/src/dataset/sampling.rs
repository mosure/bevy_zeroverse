use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::sampling::{
    base_factor, window_detail_budget, CaptureSampling, SamplingDecision,
};
use serde_json::Value;

use crate::config::Protocol;

pub(super) fn validate(
    capabilities: &Value,
    protocol: &Protocol,
    metrics_schema: u32,
    generator: u32,
) -> Result<()> {
    let size = [protocol.width, protocol.height];
    let configured = protocol.ssaa_factor;
    ensure!(
        capabilities["ssaa_configured_factor"] == configured,
        "capture SSAA configuration differs from publication protocol"
    );
    let actual =
        match serde_json::from_value::<Option<CaptureSampling>>(capabilities["sampling"].clone())
            .context("invalid capture sampling metadata")?
        {
            None => 1,
            Some(sampling) => {
                sampling.validate().map_err(anyhow::Error::msg)?;
                ensure!(sampling.ssaa_factor > 1, "1× captures omit SSAA metadata");
                sampling.ssaa_factor
            }
        };
    let base = base_factor(size, configured);
    if metrics_schema >= 25 {
        let decision: SamplingDecision =
            serde_json::from_value(capabilities["ssaa_decision"].clone())
                .context("missing or invalid scene-specific SSAA decision")?;
        let limit = decision.max_texture_dimension_2d;
        let budget = window_detail_budget(size, protocol.cameras, limit);
        ensure!(
            actual == decision.expected_factor
                && size[0].max(size[1]) <= limit / actual
                && (actual == base || (configured == 0 && base == 2 && actual == budget)),
            "capture SSAA factor {actual} differs from the scene decision or publication budget"
        );
    } else {
        // Older receipts lack the scene decision and adapter limit. Generator
        // 131 introduced adaptive coverage. Its observed factor must fit the rig.
        let maximum = if generator >= 131 && configured == 0 && base == 2 {
            window_detail_budget(size, protocol.cameras, u32::MAX)
        } else {
            base
        };
        ensure!(
            (base..=maximum).contains(&actual),
            "archived capture SSAA factor {actual} differs from publication policy"
        );
    }
    Ok(())
}
