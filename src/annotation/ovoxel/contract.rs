//! Shared preflight and payload validation for native, Python and CLI capture.
use crate::{
    app::{BevyZeroverseConfig, OvoxelMode},
    human_motion::HumanMotionConfig,
    sample::OvoxelSample,
};

pub fn validate_config(
    mode: OvoxelMode,
    steps: u32,
    motion: Option<&str>,
    resolution: u32,
    capacity: u32,
) -> Result<(), String> {
    if mode == OvoxelMode::Disabled {
        return Ok(());
    }
    if steps != 1 {
        return Err(
            "O-voxel export requires playback_steps=1; disable O-voxel for multi-timestep captures"
                .into(),
        );
    }
    if let Some(json) = motion {
        let policy = HumanMotionConfig::parse(json)?;
        if policy.fraction > 0.0 || !policy.trajectories.is_empty() {
            return Err("O-voxel export requires human motion disabled (no generated or explicit trajectories)".into());
        }
    }
    // Zero retains the existing 'use default resolution' convention.
    if resolution > 1536 || capacity == 0 {
        return Err(
            "O-voxel requires resolution <= 1536 (0 selects default) and positive output capacity"
                .into(),
        );
    }
    Ok(())
}

impl BevyZeroverseConfig {
    pub fn validate_ovoxel(&self) -> Result<(), String> {
        validate_config(
            self.ovoxel_mode,
            self.playback_steps,
            self.human_motion.as_deref(),
            self.ovoxel_resolution,
            self.ovoxel_max_output_voxels,
        )
    }
}

impl OvoxelSample {
    /// Sparse surface cells, aligned fields and a valid semantic palette.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.coords.len();
        if self.resolution == 0
            || self.resolution > 1536
            || self.aabb.iter().flatten().any(|v| !v.is_finite())
            || (0..3).any(|a| self.aabb[1][a] <= self.aabb[0][a])
        {
            return Err("invalid O-voxel bounds or resolution".into());
        }
        if [
            self.dual_vertices.len(),
            self.intersected.len(),
            self.base_color.len(),
            self.semantics.len(),
        ]
        .into_iter()
        .any(|len| len != n)
        {
            return Err("O-voxel fields have mismatched lengths".into());
        }
        if self.semantic_labels.first().map(String::as_str) != Some("unlabeled")
            || self
                .semantics
                .iter()
                .any(|&s| s as usize >= self.semantic_labels.len())
            || self.intersected.iter().any(|&v| v > 7)
            || self.coords.iter().flatten().any(|&v| v >= self.resolution)
            || self.coords.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("O-voxel coordinates must be unique, sorted and in bounds, with valid palette indices and xyz flags".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_contract_is_explicit_and_disabled_mode_is_free() {
        for mode in [OvoxelMode::CpuAsync, OvoxelMode::GpuCompute] {
            assert!(validate_config(mode, 1, None, 128, 1000).is_ok());
            assert!(validate_config(mode, 2, None, 128, 1000)
                .unwrap_err()
                .contains("playback_steps=1"));
            assert!(validate_config(mode, 0, None, 128, 1000).is_err());
            assert!(validate_config(mode, 1, Some("{}"), 128, 1000)
                .unwrap_err()
                .contains("human motion disabled"));
            assert!(validate_config(mode, 1, Some(r#"{"fraction":0}"#), 128, 1000).is_ok());
            let explicit = r#"{"fraction":0,"trajectories":[{"actor_id":0,"prompt":"walk","waypoints":[{"frame":0,"position":[0,0,0]},{"frame":159,"position":[1,0,0]}]}]}"#;
            HumanMotionConfig::parse(explicit).expect("valid explicit trajectory policy");
            assert!(validate_config(mode, 1, Some(explicit), 128, 1000)
                .unwrap_err()
                .contains("human motion disabled"));
            assert!(validate_config(mode, 1, None, 2000, 1000).is_err());
            assert!(validate_config(mode, 1, None, 128, 0).is_err());
        }
        assert!(validate_config(OvoxelMode::Disabled, 3, Some("{}"), 128, 0).is_ok());
    }
    #[test]
    fn malformed_payloads_cannot_silently_truncate() {
        let valid = OvoxelSample {
            coords: vec![[0, 1, 2]],
            dual_vertices: vec![[0, 255, 128]],
            intersected: vec![7],
            base_color: vec![[1, 2, 3, 255]],
            semantics: vec![1],
            semantic_labels: vec!["unlabeled".into(), "wall".into()],
            resolution: 16,
            aabb: [[0.; 3], [1.; 3]],
        };
        valid.validate().unwrap();
        let mut bad = valid.clone();
        bad.semantics.clear();
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.coords[0][0] = 16;
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.semantics[0] = 2;
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.intersected[0] = 8;
        assert!(bad.validate().is_err());
        let mut bad = valid;
        bad.aabb[0][0] = f32::NAN;
        assert!(bad.validate().is_err());
    }
}
