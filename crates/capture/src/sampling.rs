/// Capture raster policy; width, height, K and flow units always describe the
/// final images. RGB uses a linear-light box filter after tone mapping. Metric
/// annotations and categorical masks retain exact final-grid pixel-center rays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSampling {
    pub schema_version: u32,
    pub ssaa_factor: u32,
}
impl CaptureSampling {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("unknown capture sampling schema".into());
        }
        if !(1..=4).contains(&self.ssaa_factor) {
            return Err(
                "capture sampling metadata must contain the resolved factor (1..=4)".into(),
            );
        }
        Ok(())
    }
}

/// Scene-independent policy for automatic or explicit supersampling.
pub fn base_factor(size: [u32; 2], configured: u32) -> u32 {
    if configured == 0 {
        if u64::from(size[0]) * u64::from(size[1]) < 1280 * 720 {
            2
        } else {
            1
        }
    } else {
        configured
    }
}

/// Highest automatic factor available to a rig with fine projected window
/// coverings, under the renderer's per-view, rig-wide and device limits.
pub fn window_detail_budget(size: [u32; 2], cameras: usize, device_limit: u32) -> u32 {
    let pixels = u64::from(size[0]) * u64::from(size[1]);
    if cameras == 0 || pixels == 0 || pixels > 512 * 512 {
        return 2;
    }
    // At most the usual 768² @ 2× per camera, and eight extra megapixels
    // across the entire rig. Large rigs keep their bounded attachment budget.
    (3..=4)
        .rev()
        .find(|&f| {
            let samples = u64::from(f * f);
            size[0].max(size[1]) <= device_limit / f
                && pixels * samples <= 1536 * 1536
                && (pixels * (samples - 4)).saturating_mul(cameras as u64) <= 8 * 1024 * 1024
        })
        .unwrap_or(2)
}

/// Scene-specific decision independently recomputed by the qualification
/// renderer from the archived cameras, window geometry and adapter limit.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDecision {
    pub expected_factor: u32,
    pub max_texture_dimension_2d: u32,
}
