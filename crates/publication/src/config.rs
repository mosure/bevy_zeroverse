use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub captures: PathBuf,
    pub capture: Protocol,
    pub paper: Paper,
    pub references: Vec<Reference>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    pub seed: u64,
    pub audit_rooms: usize,
    pub rendered_rooms: usize,
    pub cameras: usize,
    pub width: u32,
    pub height: u32,
    pub playback_steps: usize,
    pub density: f32,
    pub human_density: f32,
    pub gi_rays: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Paper {
    pub source: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub id: String,
    pub generator_version: u32,
    pub gallery: PathBuf,
    pub page: PathBuf,
}
impl Config {
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join("publication.toml");
        let config: Self =
            toml::from_str(&std::fs::read_to_string(&path)?).context("publication.toml")?;
        ensure!(
            config.schema_version == 1,
            "unknown publication protocol schema"
        );
        let p = &config.capture;
        ensure!(
            p.audit_rooms >= p.rendered_rooms && p.rendered_rooms >= 4,
            "invalid cohort sizes"
        );
        ensure!(
            p.cameras == 4 && p.playback_steps == 2,
            "current explorer protocol requires four cameras and two endpoints"
        );
        ensure!(
            (64..=4096).contains(&p.width) && (64..=4096).contains(&p.height),
            "invalid image size"
        );
        ensure!(
            p.seed.checked_add(p.audit_rooms as u64).is_some(),
            "seed range overflow"
        );
        ensure!(
            (0.0..=1.0).contains(&p.density) && (0.0..=1.0).contains(&p.human_density),
            "invalid density"
        );
        ensure!((64..=16384).contains(&p.gi_rays), "invalid GI ray count");
        safe_relative(&config.captures)?;
        ensure!(
            config.captures.starts_with("out")
                && !Path::new("out/publication").starts_with(&config.captures),
            "capture cache must be below out/ and cannot replace the publication work directory"
        );
        safe_relative(&config.paper.source)?;
        ensure!(
            config.paper.source == Path::new("tex/bevy_zeroverse.tex"),
            "the publication protocol owns tex/bevy_zeroverse.tex"
        );
        for reference in &config.references {
            safe_relative(&reference.gallery)?;
            safe_relative(&reference.page)?;
        }
        Ok(config)
    }
}

/// Publication inputs and archive names must stay inside the checkout.
pub fn safe_relative(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "unsafe publication path: {}",
        path.display()
    );
    Ok(())
}
