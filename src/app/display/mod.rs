//! Display negotiation stays outside the headless renderer. Floating-point
//! dataset cameras can capture HDR independently of the attached monitor.
use bevy::{prelude::*, render::view::Tonemapping};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use super::BevyZeroverseConfig;
use crate::render::RenderMode;

mod fullscreen;
mod screenshot;
mod surface;
mod transfer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum DisplayOutput {
    Pending,
    Sdr,
    ScRgb,
    ExtendedSrgb,
    Pq,
}

#[derive(Clone, Default, Resource)]
pub(crate) struct DisplayStatus(Arc<AtomicU8>);

impl DisplayStatus {
    fn report(&self, value: DisplayOutput) {
        self.0.store(value as u8, Ordering::Relaxed);
    }

    fn sdr_fallback(&self) -> bool {
        self.0.load(Ordering::Relaxed) == DisplayOutput::Sdr as u8
    }

    pub(crate) fn active(&self) -> bool {
        self.0.load(Ordering::Relaxed) > DisplayOutput::Sdr as u8
    }

    #[cfg(feature = "viewer")]
    pub(crate) fn label(&self) -> &'static str {
        match self.0.load(Ordering::Relaxed) {
            x if x == DisplayOutput::Sdr as u8 => "SDR display fallback",
            x if x == DisplayOutput::ScRgb as u8 => "HDR · linear scRGB",
            x if x == DisplayOutput::ExtendedSrgb as u8 => "HDR · extended sRGB",
            x if x == DisplayOutput::Pq as u8 => "HDR · BT.2100 PQ",
            _ => "HDR requested",
        }
    }
}

pub(crate) fn hdr_destination(
    config: &BevyZeroverseConfig,
    display: Option<&DisplayStatus>,
) -> bool {
    config.hdr
        && (config.image_copiers || config.headless || display.is_none_or(DisplayStatus::active))
}

pub(crate) fn camera_curve(
    mode: &RenderMode,
    config: &BevyZeroverseConfig,
    display: Option<&DisplayStatus>,
) -> Tonemapping {
    if *mode != RenderMode::Color {
        return Tonemapping::None;
    }
    config.tonemapping.resolve(hdr_destination(config, display))
}

pub(super) struct DisplayPlugin;
impl Plugin for DisplayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DisplayStatus>();
        fullscreen::install(app);
        #[cfg(feature = "viewer")]
        app.add_systems(
            PostUpdate,
            grid_output
                .after(super::EditorCameraSetup)
                .before(bevy::camera::CameraUpdateSystems),
        );
        screenshot::install(app);
        surface::install(app);
    }
}

/// Capture images have already been tone mapped. Preserve extended values in
/// the grid compositor; the transparent editor overlay keeps its separate LDR target.
#[cfg(feature = "viewer")]
fn grid_output(
    config: Res<BevyZeroverseConfig>,
    status: Res<DisplayStatus>,
    grid: Query<(Entity, Has<bevy::camera::Hdr>), With<super::MaterialGridCameraMarker>>,
    mut commands: Commands,
) {
    let desired = config.camera_grid
        && !config.room_schematic
        && config.render_mode == RenderMode::Color
        && config.hdr
        && status.active();
    for (entity, active) in &grid {
        if desired != active {
            if desired {
                commands
                    .entity(entity)
                    .insert((bevy::camera::Hdr, Tonemapping::None));
            } else {
                commands.entity(entity).remove::<bevy::camera::Hdr>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{query, view_state::ViewState, web_viewer_defaults};
    use crate::render::color::RgbTonemapping;
    use clap::{Parser, ValueEnum};

    #[test]
    fn display_settings_roundtrip_every_bevy_curve_through_cli_and_sparse_url() {
        let defaults = web_viewer_defaults();
        let view = ViewState::default();
        let url = query::encode(&defaults, &view);
        for key in ["tonemapping", "hdr=", "fullscreen"] {
            assert!(!url.contains(key));
        }
        let mut curves = std::collections::HashSet::new();
        for &choice in RgbTonemapping::value_variants() {
            let name = choice.to_possible_value().unwrap();
            let name = name.get_name();
            let cli = BevyZeroverseConfig::try_parse_from([
                "viewer",
                "--tonemapping",
                name,
                "--hdr",
                "--fullscreen",
            ])
            .unwrap();
            let restored = query::parse(
                defaults.clone(),
                [
                    ("tonemapping".into(), name.into()),
                    ("hdr".into(), "true".into()),
                    ("fullscreen".into(), "true".into()),
                ],
            )
            .unwrap();
            assert_eq!(
                (cli.tonemapping, cli.hdr, cli.fullscreen),
                (choice, true, true)
            );
            assert_eq!(
                (restored.tonemapping, restored.hdr, restored.fullscreen),
                (choice, true, true)
            );
            let url = query::encode(&restored, &view);
            assert_eq!(
                url.contains(&format!("tonemapping={name}")),
                choice != RgbTonemapping::Auto
            );
            assert!(url.contains("hdr=true") && url.contains("fullscreen=true"));
            if choice != RgbTonemapping::Auto {
                assert!(curves.insert(choice.resolve(false)));
            }
        }
        assert_eq!(curves.len(), 10);
    }

    #[test]
    fn automatic_rgb_preserves_sdr_and_offscreen_hdr_and_excludes_annotations() {
        let mut cfg = BevyZeroverseConfig {
            editor: true,
            image_copiers: false,
            headless: false,
            ..default()
        };
        let status = DisplayStatus::default();
        assert_eq!(
            camera_curve(&RenderMode::Color, &cfg, Some(&status)),
            Tonemapping::TonyMcMapface
        );
        cfg.hdr = true;
        status.report(DisplayOutput::Sdr);
        assert_eq!(
            camera_curve(&RenderMode::Color, &cfg, Some(&status)),
            Tonemapping::TonyMcMapface
        );
        status.report(DisplayOutput::ScRgb);
        assert_eq!(
            camera_curve(&RenderMode::Color, &cfg, Some(&status)),
            Tonemapping::Linear
        );
        status.report(DisplayOutput::Sdr);
        cfg.image_copiers = true;
        assert_eq!(
            camera_curve(&RenderMode::Color, &cfg, Some(&status)),
            Tonemapping::Linear
        );
        cfg.tonemapping = RgbTonemapping::AgX;
        assert_eq!(
            camera_curve(&RenderMode::Color, &cfg, Some(&status)),
            Tonemapping::AgX
        );
        for mode in RenderMode::value_variants()
            .iter()
            .filter(|m| **m != RenderMode::Color)
        {
            assert_eq!(camera_curve(mode, &cfg, Some(&status)), Tonemapping::None);
        }
    }
}
