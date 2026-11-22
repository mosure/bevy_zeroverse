//! Interactive launch defaults. A kiosk is a starting configuration, so normal Studio
//! edits, explicit CLI flags and shared URL overrides continue to work after launch.
use super::{view_state::ViewState, BevyZeroverseConfig, PlaybackMode};
use crate::{
    camera::lens::LensSampling,
    scene::procedural_indoor::cameras::{CameraSettings, RollDistribution},
};

const SCENE_SECONDS: u32 = 30;

pub(super) fn default_view(kiosk: bool) -> ViewState {
    ViewState {
        viewport_only: kiosk,
        ..Default::default()
    }
}

fn kiosk_camera() -> CameraSettings {
    CameraSettings {
        path_length_min: 2.,
        long_path_fraction: 1.,
        roll: RollDistribution {
            max_degrees: 10.,
            spread: 1.,
            wide_probability: 0.,
        },
        lenses: Some(LensSampling {
            circular_probability: 0.,
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub(super) fn kiosk_defaults(
    mut config: BevyZeroverseConfig,
    is_default: impl Fn(&str) -> bool,
) -> BevyZeroverseConfig {
    macro_rules! set {
        ($field:ident, $value:expr) => {
            if is_default(stringify!($field)) {
                config.$field = $value;
            }
        };
    }
    config.kiosk = true;
    set!(num_cameras, 1);
    set!(camera_grid, true);
    set!(hdr, true);
    set!(fullscreen, true);
    set!(regenerate_ms, SCENE_SECONDS * 1000);
    set!(playback_mode, PlaybackMode::Smoothstep);
    set!(playback_speed, 1. / SCENE_SECONDS as f32);
    set!(
        viewer_state,
        Some(serde_json::to_string(&default_view(true)).expect("serializable kiosk view"))
    );
    set!(
        indoor_camera,
        Some(serde_json::to_string(&kiosk_camera()).expect("serializable kiosk camera policy"))
    );
    config
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn parse_cli<I, T>(args: I) -> Result<BevyZeroverseConfig, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    use clap::{parser::ValueSource, CommandFactory, FromArgMatches};
    let matches = BevyZeroverseConfig::command().try_get_matches_from(args)?;
    let mut config = BevyZeroverseConfig::from_arg_matches(&matches)?;
    // A windowless indoor run has no editor camera. Give it one capture view by default.
    if config.headless
        && config.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor
        && matches.value_source("num_cameras") == Some(ValueSource::DefaultValue)
    {
        config.num_cameras = 1;
    }
    Ok(if config.kiosk {
        kiosk_defaults(config, |key| {
            matches.value_source(key) != Some(ValueSource::CommandLine)
        })
    } else {
        super::studio_lens_defaults(config)
    })
}

#[cfg(test)]
mod tests;
