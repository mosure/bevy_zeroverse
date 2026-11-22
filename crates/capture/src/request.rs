//! Room-level capture policy, independent of scene preparation and renderer transport.
use serde::{Deserialize, Serialize};

/// A timestep-major list of logical camera indices. Empty rows request clean RGB.
pub type CameraRows = Vec<Vec<usize>>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureRequest {
    /// Absolute attempt index, including any dataset offset and rejected requests.
    pub sequence: usize,
    /// All timesteps and cameras in this request share this room seed.
    pub seed: u64,
    /// Omission permits all cameras. Eligibility never resamples the camera's optics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgb_artifact_eligible_cameras_by_timestep: Option<CameraRows>,
}

impl CaptureRequest {
    pub fn new(sequence: usize, seed: u64) -> Self {
        Self {
            sequence,
            seed,
            rgb_artifact_eligible_cameras_by_timestep: None,
        }
    }

    pub fn validate(&self, cameras: usize, steps: usize) -> Result<(), String> {
        validate_eligibility(
            self.rgb_artifact_eligible_cameras_by_timestep.as_deref(),
            cameras,
            steps,
        )
    }
}

pub fn validate_eligibility(
    rows: Option<&[Vec<usize>]>,
    cameras: usize,
    steps: usize,
) -> Result<(), String> {
    let Some(rows) = rows else { return Ok(()) };
    if steps == 0 || cameras == 0 || rows.len() != steps {
        return Err(format!("RGB artifact eligibility requires one row per captured timestep ({steps}) and a nonempty rig"));
    }
    for (step, row) in rows.iter().enumerate() {
        for (i, &camera) in row.iter().enumerate() {
            if camera >= cameras {
                return Err(format!("RGB artifact eligibility timestep {step}: camera {camera} outside 0..{cameras}"));
            }
            if row[..i].contains(&camera) {
                return Err(format!(
                    "RGB artifact eligibility timestep {step}: duplicate camera {camera}"
                ));
            }
        }
    }
    Ok(())
}

/// Channels report operations that ran, including the union when archive augmentation is used.
/// All rows are in ascending logical camera order. The request field distinguishes explicit
/// all-camera eligibility from the default.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgbArtifactReceipt {
    pub rgb_artifact_eligible_cameras_by_timestep: Option<CameraRows>,
    pub eligible_cameras_by_timestep: CameraRows,
    pub optics_applied_cameras_by_timestep: CameraRows,
    pub archive_applied_cameras_by_timestep: CameraRows,
    pub applied_cameras_by_timestep: CameraRows,
}

impl RgbArtifactReceipt {
    pub fn for_capture(
        rows: Option<&[Vec<usize>]>,
        cameras: usize,
        steps: usize,
        sampled: impl Fn(usize) -> bool,
    ) -> Self {
        let requested = rows.map(|r| {
            r.iter()
                .map(|row| {
                    let mut row = row.clone();
                    row.sort_unstable();
                    row
                })
                .collect::<CameraRows>()
        });
        let eligible = requested
            .clone()
            .unwrap_or_else(|| vec![(0..cameras).collect(); steps]);
        let profiles: Vec<_> = (0..cameras).map(sampled).collect();
        let applied: CameraRows = eligible
            .iter()
            .map(|row| {
                row.iter()
                    .copied()
                    .filter(|&cam| profiles.get(cam) == Some(&true))
                    .collect()
            })
            .collect();
        Self {
            rgb_artifact_eligible_cameras_by_timestep: requested,
            eligible_cameras_by_timestep: eligible,
            applied_cameras_by_timestep: applied.clone(),
            optics_applied_cameras_by_timestep: applied,
            archive_applied_cameras_by_timestep: vec![vec![]; steps],
        }
    }

    pub fn mark_archive_applied(&mut self) {
        self.archive_applied_cameras_by_timestep
            .clone_from(&self.eligible_cameras_by_timestep);
        self.applied_cameras_by_timestep
            .clone_from(&self.eligible_cameras_by_timestep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validate_rows_and_record_actual_camera_decisions() {
        for s in [
            "[1]",
            "[[0],null]",
            "[[-1],[]]",
            "[[0.5],[]]",
            "[[true],[]]",
        ] {
            assert!(serde_json::from_str::<CameraRows>(s).is_err());
        }
        for rows in [
            vec![],
            vec![vec![]],
            vec![vec![0, 0], vec![]],
            vec![vec![], vec![3]],
        ] {
            assert!(validate_eligibility(Some(&rows), 3, 2).is_err());
        }
        let rows = vec![vec![2, 0], vec![1]];
        validate_eligibility(Some(&rows), 3, 2).unwrap();
        let mut r = RgbArtifactReceipt::for_capture(Some(&rows), 3, 2, |c| c != 0);
        assert_eq!(r.eligible_cameras_by_timestep, [vec![0, 2], vec![1]]);
        assert_eq!(r.applied_cameras_by_timestep, [vec![2], vec![1]]);
        r.mark_archive_applied();
        assert_eq!(
            r.applied_cameras_by_timestep,
            r.eligible_cameras_by_timestep
        );
        assert_eq!(r.optics_applied_cameras_by_timestep, [vec![2], vec![1]]);
        assert!(serde_json::to_value(CaptureRequest::new(1, 8))
            .unwrap()
            .get("rgb_artifact_eligible_cameras_by_timestep")
            .is_none());
    }
}
