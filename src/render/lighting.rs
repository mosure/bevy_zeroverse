//! Lighting for interactive RGB presentation. Calibrated archive and tensor
//! requests always use the generator's raster PBR and shared irradiance bake.
use crate::app::BevyZeroverseConfig;
use bevy::prelude::*;
use bevy_args::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Reflect, ValueEnum, Serialize, Deserialize,
)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int))]
pub enum PreviewLighting {
    #[default]
    Raster,
    Solari,
    /// Adds Bevy’s spatial and temporal reservoir resampling.
    SolariRestir,
}

impl std::fmt::Display for PreviewLighting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Raster => "raster",
            Self::Solari => "solari",
            Self::SolariRestir => "solari-restir",
        })
    }
}

#[cfg(all(feature = "solari", not(target_arch = "wasm32")))]
mod native;
#[cfg(all(feature = "solari", not(target_arch = "wasm32")))]
pub use native::SolariPreviewPlugin;

/// Resolve live UI and render-mode changes before selecting camera attachments.
#[cfg(any(
    feature = "viewer",
    all(feature = "solari", not(target_arch = "wasm32"))
))]
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SelectPreviewLighting;

/// Device capability and the active indoor RGB preview. Absent in ordinary capture workers.
#[derive(Resource, Default, Debug)]
pub struct SolariPreviewStatus {
    pub supported: bool,
    pub active: bool,
    pub temporal_reuse: bool,
    /// Maximum physical pixels in one 48-byte-per-pixel ReSTIR reservoir.
    pub max_reservoir_pixels: u64,
    pub traced_instances: usize,
    pub traced_meshes: usize,
}

pub(crate) fn install_preview(app: &mut App, config: &BevyZeroverseConfig) {
    if config.headless || config.image_copiers {
        if config.preview_lighting != PreviewLighting::Raster {
            warn!("Solari is a Studio preview. Dataset capture uses raster PBR.");
            app.world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .preview_lighting = PreviewLighting::Raster;
        }
        return;
    }
    #[cfg(all(feature = "solari", not(target_arch = "wasm32")))]
    app.add_plugins(SolariPreviewPlugin);
    #[cfg(not(all(feature = "solari", not(target_arch = "wasm32"))))]
    if config.preview_lighting != PreviewLighting::Raster {
        warn!("Solari preview needs a native build with --features solari. Using raster PBR.");
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .preview_lighting = PreviewLighting::Raster;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_args::Parser;

    #[test]
    fn preview_lighting_roundtrips_cli_query_and_older_config() {
        let config = BevyZeroverseConfig::default();
        let mut legacy = serde_json::to_value(&config).unwrap();
        legacy.as_object_mut().unwrap().remove("preview_lighting");
        assert_eq!(
            serde_json::from_value::<BevyZeroverseConfig>(legacy)
                .unwrap()
                .preview_lighting,
            PreviewLighting::Raster
        );
        let cli =
            BevyZeroverseConfig::try_parse_from(["bevy_zeroverse", "--preview-lighting", "solari"])
                .unwrap();
        assert_eq!(cli.preview_lighting, PreviewLighting::Solari);
        let query =
            crate::app::query::parse(config, [("preview_lighting".into(), "solari".into())])
                .unwrap();
        assert_eq!(query.preview_lighting, cli.preview_lighting);
        let temporal = BevyZeroverseConfig::try_parse_from([
            "bevy_zeroverse",
            "--preview-lighting",
            "solari-restir",
        ])
        .unwrap();
        let link = crate::app::query::encode(&temporal, &default());
        assert!(link.contains("preview_lighting=solari-restir"));
        assert!(
            !crate::app::query::encode(&BevyZeroverseConfig::default(), &default())
                .contains("preview_lighting")
        );
    }

    #[test]
    fn production_workers_select_raster_even_in_feature_builds() {
        for (headless, image_copiers) in [(true, false), (false, true), (true, true)] {
            let mut app = App::new();
            let config = BevyZeroverseConfig {
                headless,
                image_copiers,
                preview_lighting: PreviewLighting::Solari,
                ..default()
            };
            app.insert_resource(config.clone());
            install_preview(&mut app, &config);
            assert_eq!(
                app.world()
                    .resource::<BevyZeroverseConfig>()
                    .preview_lighting,
                PreviewLighting::Raster
            );
            assert!(!app.world().contains_resource::<SolariPreviewStatus>());
        }
    }
}
