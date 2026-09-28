//! One continuous camera-spacing control expands into explicit, archived settings.
//! Custom advanced settings never get overwritten by a hidden preset on regeneration.
use super::{multiview::MultiViewSettings, CameraSettings};
use serde::{Deserialize, Deserializer};

impl MultiViewSettings {
    /// 0 is a close camera rig; 1 permits room-spanning baselines with lower overlap.
    /// Distances are metres. This controls proposals, not measured pixel visibility.
    pub fn from_baseline(baseline: f32) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&baseline) {
            return Err("camera baseline must be finite and in [0,1]".into());
        }
        Ok(Self {
            min_overlap: 0.65 - 0.55 * baseline.powf(0.6),
            min_baseline: 0.04 + 0.46 * baseline,
            min_reference_baseline: 0.06 + 2.30 * baseline,
            max_baseline: 0.35 + 9.65 * baseline,
            min_spread: 0.20 + 0.10 * baseline,
            trajectory_variation: baseline,
        })
    }

    /// Return the slider value only when all advanced controls match its program.
    /// Editing any advanced value leaves the configuration in explicit custom mode.
    pub fn baseline(&self) -> Option<f32> {
        let value = ((self.min_reference_baseline - 0.06) / 2.30).clamp(0.0, 1.0);
        let expected = Self::from_baseline(value).ok()?;
        [
            self.min_overlap - expected.min_overlap,
            self.min_baseline - expected.min_baseline,
            self.min_reference_baseline - expected.min_reference_baseline,
            self.max_baseline - expected.max_baseline,
            self.min_spread - expected.min_spread,
            self.trajectory_variation - expected.trajectory_variation,
        ]
        .iter()
        .all(|d| d.abs() < 1e-5)
        .then_some(value)
    }
}

impl<'de> Deserialize<'de> for CameraSettings {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Expand the shorthand before parsing concrete settings. Serializing always
        // records the exact advanced values, so archives have no preset precedence.
        let mut value = serde_json::Value::deserialize(deserializer)?;
        if let Some(object) = value.as_object_mut() {
            if let Some(baseline) = object.remove("baseline") {
                if object.contains_key("multiview") {
                    return Err(serde::de::Error::custom(
                        "indoor_camera: use baseline or explicit multiview settings, not both",
                    ));
                }
                let baseline: f32 =
                    serde_json::from_value(baseline).map_err(serde::de::Error::custom)?;
                let policy =
                    MultiViewSettings::from_baseline(baseline).map_err(serde::de::Error::custom)?;
                object.insert("multiview".into(), serde_json::to_value(policy).unwrap());
            }
        }
        #[derive(Deserialize)]
        #[serde(default, deny_unknown_fields)]
        struct Concrete {
            primary_room: bool,
            path_length_min: f32,
            path_length_max: f32,
            long_path_fraction: f32,
            multiview: Option<MultiViewSettings>,
        }
        impl Default for Concrete {
            fn default() -> Self {
                let c = CameraSettings::default();
                Self {
                    primary_room: c.primary_room,
                    path_length_min: c.path_length_min,
                    path_length_max: c.path_length_max,
                    long_path_fraction: c.long_path_fraction,
                    multiview: c.multiview,
                }
            }
        }
        let c: Concrete = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            primary_room: c.primary_room,
            path_length_min: c.path_length_min,
            path_length_max: c.path_length_max,
            long_path_fraction: c.long_path_fraction,
            multiview: c.multiview,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation::validate_layout,
    };

    #[test]
    fn narrow_and_wide_programs_control_realized_paths_without_changing_rooms() {
        for seed in 24000..24008 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25)
                    .unwrap();
            let original = scene.clone();
            for baseline in [0.0, 1.0] {
                let policy = MultiViewSettings::from_baseline(baseline).unwrap();
                scene
                    .resample_cameras(
                        4,
                        CameraSettings {
                            multiview: Some(policy.clone()),
                            ..Default::default()
                        },
                        4.0 / 3.0,
                    )
                    .unwrap();
                validate_layout(&scene).unwrap();
                assert_eq!(scene.objects, original.objects);
                assert_eq!(scene.humans, original.humans);
                let geometry = scene.camera_group_geometry().unwrap();
                assert!(geometry.accepts(&policy));
                if baseline == 0.0 {
                    assert!(geometry.max_reference_baseline_m <= 0.35001);
                } else {
                    assert!(geometry.min_reference_baseline_m >= 2.35999);
                }
            }
        }
    }

    #[test]
    fn baseline_expands_roundtrips_and_does_not_overwrite_custom_controls() {
        for value in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let c = CameraSettings::parse(&format!(r#"{{"baseline":{value}}}"#)).unwrap();
            let p = c.multiview.as_ref().unwrap();
            assert!((p.baseline().unwrap() - value).abs() < 1e-5);
            p.validate().unwrap();
            let json = serde_json::to_string(&c).unwrap();
            assert_eq!(CameraSettings::parse(&json).unwrap(), c);
            let mut custom = p.clone();
            custom.min_overlap = 0.123;
            assert!(custom.baseline().is_none());
        }
        for json in [
            r#"{"baseline":-0.1}"#,
            r#"{"baseline":1.1}"#,
            r#"{"baseline":null}"#,
            r#"{"baseline":0.5,"multiview":null}"#,
            r#"{"baseline":0.5,"multiview":{}}"#,
        ] {
            assert!(CameraSettings::parse(json).is_err(), "{json}");
        }
        assert!(MultiViewSettings::from_baseline(f32::NAN).is_err());
        assert!(MultiViewSettings::from_baseline(f32::INFINITY).is_err());
    }
}
