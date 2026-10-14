//! Public artifacts describe one generator identity. Historical evidence stays
//! in Git or local study outputs, rather than becoming a second public gallery.
use crate::io;
use anyhow::{ensure, Result};
use std::{collections::BTreeMap, fs, path::Path};

pub fn retired(name: &str) -> bool {
    if let Some(media) = name.strip_prefix("www/project/static/media/") {
        return ![
            "architecture/",
            "qualification/",
            "cameras/",
            "objects/",
            "materials/",
        ]
        .iter()
        .any(|prefix| media.starts_with(prefix))
            && !matches!(media, "social.jpg" | "whitepaper.webp");
    }
    if let Some(generated) = name.strip_prefix("tex/generated/") {
        return ![
            "architecture_",
            "publication_",
            "camera_count_",
            "object_",
            "material_",
            "pipeline_",
        ]
        .iter()
        .any(|prefix| generated.starts_with(prefix));
    }
    false
}

pub fn prune(stage: &Path) -> Result<()> {
    for directory in ["www/project/static/media", "tex/generated"] {
        for name in io::files(stage, &stage.join(directory))?.into_keys() {
            if retired(&name) {
                fs::remove_file(stage.join(name))?;
            }
        }
    }
    Ok(())
}

pub fn paper_source(name: &str, artifacts: &BTreeMap<String, String>) -> Result<()> {
    ensure!(
        !name.starts_with("tex/generated/") || artifacts.contains_key(name),
        "paper uses an unmanaged or historical figure/result: {name}; rebuild from current captures"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_current_gallery_and_retires_historical_captures() {
        for name in [
            "www/project/static/media/architecture/s0-t0-c0-co_visibility.webp",
            "www/project/static/media/qualification/receipt.json",
            "www/project/static/media/cameras/camera-count-heatmap-b1.svg",
            "www/project/static/media/social.jpg",
            "www/project/static/media/whitepaper.webp",
            "www/project/static/css/architecture.css",
            "tex/generated/publication_evaluation.tex",
            "tex/generated/architecture_co_visibility.jpg",
            "tex/generated/camera_count_statistics.png",
            "tex/generated/pipeline_receipt.json",
            "tex/generated/pipeline_engine.png",
        ] {
            assert!(!retired(name), "retired current output {name}");
        }
        for name in [
            "www/project/static/media/baseline/gallery.json",
            "www/project/static/media/population-v21.json",
            "www/project/static/media/motion_13.mp4",
            "tex/generated/camera_results.tex",
            "tex/generated/gallery_v21.jpg",
        ] {
            assert!(retired(name), "preserved historical output {name}");
        }
    }

    #[test]
    fn paper_dependencies_require_current_artifact_provenance() {
        let artifacts = BTreeMap::from([(
            "tex/generated/architecture_co_visibility.jpg".into(),
            "current hash".into(),
        )]);
        paper_source("tex/generated/architecture_co_visibility.jpg", &artifacts).unwrap();
        paper_source("tex/arxiv.sty", &artifacts).unwrap();
        assert!(paper_source("tex/generated/gallery_v21.jpg", &artifacts).is_err());
        assert!(paper_source("tex/generated/unbound_statistics.tex", &artifacts).is_err());
    }
}
