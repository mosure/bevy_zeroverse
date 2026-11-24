use super::*;
use crate::app::{query, web_viewer_defaults};
use crate::scene::ZeroverseSceneType;

fn url(pairs: &[(&str, &str)]) -> Result<BevyZeroverseConfig, String> {
    query::parse(
        web_viewer_defaults(),
        pairs.iter().map(|&(k, v)| (k.into(), v.into())),
    )
}

fn policy(config: &BevyZeroverseConfig) -> CameraSettings {
    CameraSettings::parse(config.indoor_camera.as_deref().unwrap()).unwrap()
}

fn view(config: &BevyZeroverseConfig) -> ViewState {
    ViewState::parse(config.viewer_state.as_deref().unwrap()).unwrap()
}

#[test]
fn viewer_launch_defaults_and_kiosk_share_the_same_policy_on_both_platforms() {
    let web = url(&[]).unwrap();
    assert_eq!(web.scene_type, ZeroverseSceneType::ProceduralIndoor);
    assert_eq!(web.preview_aa, PreviewAa::Auto);
    assert!(!web.kiosk && !web.fullscreen && !web.camera_grid && !view(&web).viewport_only);
    assert_eq!(
        BevyZeroverseConfig::default().scene_type,
        ZeroverseSceneType::Object
    );
    let web = url(&[("kiosk", "")]).unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let configs = {
        let cli = parse_cli(["bevy_zeroverse"]).unwrap();
        assert_eq!(cli.scene_type, ZeroverseSceneType::ProceduralIndoor);
        assert_eq!(cli.preview_aa, PreviewAa::Auto);
        assert!(!cli.kiosk);
        assert_eq!(
            parse_cli(["bevy_zeroverse", "--headless"])
                .unwrap()
                .num_cameras,
            1
        );
        [web, parse_cli(["bevy_zeroverse", "--kiosk"]).unwrap()]
    };
    #[cfg(target_arch = "wasm32")]
    let configs = [web];
    for config in configs {
        assert!(
            config.kiosk && config.hdr && config.fullscreen && config.camera_grid && config.editor
        );
        assert!(view(&config).viewport_only);
        assert_eq!(config.preview_aa, PreviewAa::Taa);
        assert_eq!(config.num_cameras, 1);
        assert_eq!(config.regenerate_ms, 30_000);
        assert_eq!(config.playback_mode, PlaybackMode::Smoothstep);
        assert_eq!(config.playback_speed, 1. / 30.);
        let camera = policy(&config);
        assert_eq!(camera.motion_fraction, 1.);
        assert_eq!(camera.path_length_min, 2.);
        assert_eq!(camera.roll.support_degrees(), 10.);
        assert_eq!(camera.roll.wide_probability, 0.);
        assert_eq!(camera.lenses.unwrap().circular_probability, 0.);
    }
}

#[test]
fn kiosk_links_are_sparse_and_preserve_edits_including_revealed_controls() {
    for flag in ["", "true"] {
        let config = url(&[("kiosk", flag)]).unwrap();
        assert_eq!(query::encode(&config, &view(&config)), "kiosk=true");
    }
    assert!(!url(&[("kiosk", "false")]).unwrap().kiosk);
    assert!(url(&[("kiosk", "yes")]).is_err());
    assert!(url(&[("kiosk", ""), ("kiosk", "false")]).is_err());
    let pairs = [
        ("fullscreen", "false"),
        ("preview_aa", "off"),
        ("view.viewport_only", "false"),
        ("camera.path_length_min", "3"),
        ("kiosk", "true"),
    ];
    let config = url(&pairs).unwrap();
    assert_eq!(config.preview_aa, PreviewAa::Off);
    assert_eq!(policy(&config).path_length_min, 3.);
    assert!(!config.fullscreen && !view(&config).viewport_only);
    let encoded = query::encode(&config, &view(&config));
    let decoded = encoded
        .split('&')
        .map(|p| p.split_once('=').unwrap())
        .collect::<Vec<_>>();
    assert_eq!(decoded.len(), pairs.len(), "{encoded}");
    let restored = url(&decoded).unwrap();
    assert_eq!(query::expand(&restored), query::expand(&config));
    assert!(!view(&restored).viewport_only);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn explicit_cli_values_override_kiosk_even_when_equal_to_normal_defaults() {
    let config = parse_cli([
        "bevy_zeroverse",
        "--num-cameras",
        "0",
        "--camera-grid=false",
        "--hdr=false",
        "--preview-aa",
        "auto",
        "--fullscreen=false",
        "--regenerate-ms",
        "0",
        "--scene-type",
        "object",
        "--indoor-camera",
        "{}",
        "--viewer-state",
        "{}",
        "--playback-mode",
        "sin",
        "--kiosk",
    ])
    .unwrap();
    assert_eq!(config.scene_type, ZeroverseSceneType::Object);
    assert_eq!((config.num_cameras, config.regenerate_ms), (0, 0));
    assert!(
        !config.hdr && !config.fullscreen && !config.camera_grid && !view(&config).viewport_only
    );
    assert_eq!(config.playback_mode, PlaybackMode::Sin);
    assert_eq!(config.preview_aa, PreviewAa::Auto);
    assert_eq!(policy(&config), CameraSettings::default());
}

#[test]
fn kiosk_generated_paths_clear_furnishings_and_stay_within_the_roll_limit() {
    use crate::calibration::FisheyeFit;
    use crate::scene::procedural_indoor::{cameras::horizon_roll_radians, layout::IndoorManifest};
    let config = url(&[("kiosk", "true")]).unwrap();
    let mut fisheyes = 0;
    for seed in 0..24 {
        let mut room = IndoorManifest::generate_with_humans(
            seed,
            config.indoor_layout,
            config.indoor_density,
            0,
            config.indoor_human_density,
        )
        .unwrap();
        let size = if seed % 2 == 0 {
            [640, 360]
        } else {
            [384, 384]
        };
        let [width, height] = size.map(|v| v as f32);
        let aspect = width / height;
        room.resample_cameras(config.num_cameras, policy(&config), aspect)
            .unwrap();
        assert_eq!(room.cameras.len(), 1);
        let camera = &room.cameras[0];
        assert!(room.camera_curve_clear(camera));
        assert!(camera.path_length() >= 2. - 1e-4, "seed {seed}");
        if let Some(lens) = camera.fisheye {
            assert_eq!(lens.fit, FisheyeFit::Diagonal);
            let offset = camera.principal_point_offset.unwrap_or_default().to_array();
            let calibration = lens.calibration(size, offset).unwrap();
            for x in [0.5, width - 0.5] {
                for y in [0.5, height - 0.5] {
                    assert!(calibration.ray([x, y]).is_some(), "seed {seed}");
                }
            }
            fisheyes += 1;
        }
        for i in 0..=16 {
            let roll = horizon_roll_radians(camera.transform_at(i as f32 / 16.).rotation).unwrap();
            assert!(roll.to_degrees().abs() <= 10.001, "seed {seed}");
        }
    }
    assert!((1..24).contains(&fisheyes));
}
