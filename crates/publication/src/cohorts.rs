//! Local, checksum-bound qualification of distinct capture sampling populations.
//! Strata remain separate from population estimates; no scene is quality-filtered.
use crate::{config::Protocol, dataset::Dataset, io};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
pub mod publication;
mod reconstruction;

fn protocol(directory: &Path) -> Result<Protocol> {
    let audit: Value = io::read(&directory.join("distribution.json"))?;
    let metrics: Value = io::read(&directory.join("metrics.json"))?;
    let selection: Value = io::read(&directory.join("render_selection.json"))?;
    let integer = |v: &Value| v.as_u64().context("missing qualification integer");
    let p = Protocol {
        seed: integer(&audit["first_seed"])?,
        audit_rooms: integer(&audit["seeds"])? as usize,
        rendered_rooms: selection["selected_seeds"]
            .as_array()
            .context("missing selected seeds")?
            .len(),
        cameras: integer(&audit["cameras_per_scene"])? as usize,
        width: integer(&metrics["image_size"][0])? as u32,
        height: integer(&metrics["image_size"][1])? as u32,
        playback_steps: integer(&selection["playback_steps"])? as usize,
        density: metrics["density"].as_f64().context("missing density")? as f32,
        human_density: metrics["human_density"]
            .as_f64()
            .context("missing human density")? as f32,
        gi_rays: integer(&selection["gi_rays"])? as u32,
        ssaa_factor: integer(&selection["ssaa_factor"])? as u32,
    };
    ensure!(
        p.rendered_rooms > 0
            && p.rendered_rooms <= 4096
            && p.audit_rooms >= p.rendered_rooms
            && (1..=16).contains(&p.cameras)
            && (1..=16).contains(&p.playback_steps),
        "invalid cohort size"
    );
    Ok(p)
}

pub fn audit(root: &Path, inputs: &[PathBuf], output: &Path) -> Result<()> {
    let destination = crate::study::local_destination(root, output)?;
    let report = collect(root, inputs)?;
    std::fs::create_dir_all(&destination)?;
    io::write(&destination.join("report.json"), &report)?;
    println!(
        "Qualified {} distinct rooms, {} views across {} cohorts",
        report["rooms"],
        report["views"],
        inputs.len()
    );
    Ok(())
}

pub(super) fn collect(root: &Path, inputs: &[PathBuf]) -> Result<Value> {
    ensure!(!inputs.is_empty(), "at least one cohort is required");
    let mut cohorts = Vec::new();
    let mut seeds = BTreeSet::new();
    let mut identity = None;
    let mut views = 0;
    for input in inputs {
        let directory = root.join(input).canonicalize()?;
        let protocol = protocol(&directory)?;
        let data = Dataset::load_qualification(&directory, &protocol)
            .with_context(|| format!("qualifying {}", input.display()))?;
        ensure!(
            identity.as_ref().is_none_or(|i| i == &data.identity),
            "cohorts contain different generator builds"
        );
        identity = Some(data.identity.clone());
        for scene in &data.scenes {
            ensure!(
                seeds.insert(scene.capture.seed),
                "room seed repeated across cohorts"
            );
            views += scene.capture.views.len();
        }
        let reconstruction = reconstruction::summarize(&data.scenes, &protocol)?;
        cohorts.push(json!({"directory":directory,"protocol":protocol,
            "selection_policy":data.selection["policy"],"selected_seeds":data.selection["selected_seeds"],
            "summary":data.summary,"reconstruction":reconstruction,"sources_sha256":data.input_sha256}));
    }
    let report = json!({"schema_version":1,"identity":identity,"rooms":seeds.len(),"views":views,
        "aggregation":"Each cohort is reported separately. Prespecified strata are not an IID population estimate. Every selected scene and annotation is verified; no quality filtering.",
        "cohorts":cohorts});
    Ok(report)
}
