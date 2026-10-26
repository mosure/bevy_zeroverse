//! Instantaneous rendered-population evidence. Matched before/after studies use
//! the separate local-only study API and cannot enter this publication path.
use crate::{dataset::number, io, pipeline};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::GeneratorIdentity;
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

pub const DIRECTORY: &str = "www/project/static/media/cohorts";

pub fn prepare(root: &Path, inputs: &[PathBuf], output: &Path) -> Result<()> {
    ensure!(!output.exists(), "cohort-study destination exists");
    let mut report = super::collect(root, inputs)?;
    let identity = pipeline::identity(root)?;
    validate(&report, &identity)?;
    // Keep reproducible protocols, selected seeds and all input digests without
    // leaking workstation paths into the distributable receipt.
    for cohort in report["cohorts"].as_array_mut().unwrap() {
        cohort.as_object_mut().unwrap().remove("directory");
    }
    report["scope"] = json!("Current native rendered cohorts. Each sampling population remains separate. Source-image hashes attest the verified local capture run; this compact receipt does not ship every training image. No photographic or downstream-utility claim.");
    fs::create_dir_all(output)?;
    io::write(&output.join("report.json"), &report)?;
    Ok(())
}

fn validate(report: &Value, identity: &GeneratorIdentity) -> Result<()> {
    ensure!(
        report["schema_version"] == 1 && report["identity"] == json!(identity),
        "stale or unsupported room qualification"
    );
    let cohorts = report["cohorts"]
        .as_array()
        .context("missing rendered cohorts")?;
    ensure!(
        !cohorts.is_empty() && cohorts.len() <= 16,
        "invalid cohort count"
    );
    let mut seeds = BTreeSet::new();
    let mut views = 0u64;
    for cohort in cohorts {
        let protocol: crate::config::Protocol = serde_json::from_value(cohort["protocol"].clone())?;
        ensure!(
            (1..=16).contains(&protocol.cameras)
                && (1..=16).contains(&protocol.playback_steps)
                && protocol.rendered_rooms > 0
                && protocol.rendered_rooms <= 4096
                && protocol.audit_rooms >= protocol.rendered_rooms,
            "invalid rendered cohort protocol"
        );
        let selected = cohort["selected_seeds"]
            .as_array()
            .context("missing selected seeds")?;
        ensure!(
            selected.len() == protocol.rendered_rooms,
            "cohort room denominator mismatch"
        );
        for seed in selected {
            ensure!(
                seeds.insert(seed.as_u64().context("invalid seed")?),
                "duplicate seed across rendered cohorts"
            );
        }
        let expected =
            (protocol.rendered_rooms * protocol.cameras * protocol.playback_steps) as u64;
        ensure!(
            cohort["summary"]["co_visibility"]["views"] == expected,
            "cohort view denominator mismatch"
        );
        views += expected;
        let graph = &cohort["reconstruction"];
        let groups = (protocol.rendered_rooms * protocol.playback_steps) as u64;
        ensure!(
            graph["schema_version"] == 1
                && graph["camera_time_groups"] == groups
                && graph["minimum_bidirectional_overlap"] == 0.1
                && graph["primary_connected_groups"]
                    .as_u64()
                    .is_some_and(|v| v <= groups)
                && graph["whole_frame_connected_groups"]
                    .as_u64()
                    .is_some_and(|v| v <= groups),
            "invalid reconstruction denominator"
        );
        let sources = cohort["sources_sha256"]
            .as_object()
            .context("missing capture digests")?;
        ensure!(
            !sources.is_empty()
                && sources.values().all(|v| v
                    .as_str()
                    .is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))),
            "invalid capture digest"
        );
        ensure!(
            cohort["summary"]["co_visibility"]["shared_fraction_valid"]
                .as_f64()
                .is_some_and(|n| n.is_finite() && (0.0..=1.0).contains(&n)),
            "invalid overlap statistic"
        );
    }
    ensure!(
        report["rooms"] == seeds.len() && report["views"] == views,
        "aggregate cohort denominator mismatch"
    );
    Ok(())
}

pub fn build(
    root: &Path,
    stage: &Path,
    input: Option<&Path>,
    identity: &GeneratorIdentity,
) -> Result<()> {
    let output = stage.join(DIRECTORY);
    if output.exists() {
        fs::remove_dir_all(&output)?;
    }
    if let Some(input) = input {
        let report = verify_input(&root.join(input), identity)?;
        fs::create_dir_all(&output)?;
        io::write(&output.join("report.json"), &report)?;
    }
    Ok(())
}

pub(crate) fn verify_input(input: &Path, identity: &GeneratorIdentity) -> Result<Value> {
    let report: Value = io::read(&input.join("report.json"))?;
    validate(&report, identity)?;
    Ok(report)
}

pub fn verify(stage: &Path, enabled: bool, identity: &GeneratorIdentity) -> Result<()> {
    let file = stage.join(DIRECTORY).join("report.json");
    ensure!(
        file.exists() == enabled,
        "room qualification presence differs from recipe"
    );
    if enabled {
        validate(&io::read(&file)?, identity)?;
    }
    Ok(())
}

pub fn page(stage: &Path, gallery_rooms: usize) -> Result<String> {
    let path = stage.join(DIRECTORY).join("report.json");
    if !path.exists() {
        return Ok(String::new());
    }
    let r: Value = io::read(&path)?;
    Ok(format!("<p class=\"note\"><strong>Rendered-cohort qualification: {} rooms · {} calibrated views.</strong> Each sampling population retains its own denominator and selection policy. Counts belong to this receipt and must not be added to the {gallery_rooms}-room gallery or factor sweep: their room populations can overlap. <a href=\"static/media/cohorts/report.json\">Rendered-cohort receipt ↗</a></p>", r["rooms"], r["views"]))
}

pub fn paper(stage: &Path) -> Result<String> {
    let path = stage.join(DIRECTORY).join("report.json");
    if !path.exists() {
        return Ok(String::new());
    }
    let r: Value = io::read(&path)?;
    let mut text = format!("\\subsection{{Rendered-cohort qualification}}\nCurrent-source qualification covers {} distinct rooms and {} camera/time views. All requested captures and aligned annotations are checked before the receipt is emitted. The table keeps each sampling population separate; selected strata are not an IID estimate of ordinary generation. The automatic snapshot qualifies the same captures shown in the interactive gallery. These counts are not additional independent evidence and must not be added to gallery or factor-sweep totals. The compact downloadable receipt retains protocols, selected seeds, checks and all local input hashes; it does not bundle all training images.\n\n\\begin{{table}}[htbp]\\centering\\small\n\\begin{{tabular}}{{lrrrr}}\\toprule\nCohort start & Rooms & Views & Shared (\\%) & Min. classes \\\\\n\\midrule\n", r["rooms"],r["views"]);
    for cohort in r["cohorts"].as_array().unwrap() {
        let p = &cohort["protocol"];
        let s = &cohort["summary"];
        text.push_str(&format!(
            "{} & {} & {} & {:.2} & {} \\\\\n",
            p["seed"],
            p["rendered_rooms"],
            s["co_visibility"]["views"],
            100. * number(&s["co_visibility"]["shared_fraction_valid"]),
            s["minimum_semantic_classes"]
        ));
    }
    text.push_str("\\bottomrule\\end{tabular}\n\\caption{Rendered populations. Shared means valid source pixels visible in any peer, pooled within each cohort; it is not unique surface area. Seeds and selection policies are retained in the receipt. No downstream model inference filters the samples.}\\label{tab:rendered-populations}\\end{table}\n\n");
    text.push_str("\\begin{table}[htbp]\\centering\\small\n\\begin{tabular}{lrrr}\\toprule\nCohort start & Connected / groups & Median baseline (m) & Median angle ($^\\circ$) \\\\\n\\midrule\n");
    for cohort in r["cohorts"].as_array().unwrap() {
        let g = &cohort["reconstruction"];
        let median = |metric: &str| {
            g[metric]["p05_p50_p95"][1]
                .as_f64()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "---".into())
        };
        text.push_str(&format!(
            "{} & {} / {} & {} & {} \\\\\n",
            cohort["protocol"]["seed"],
            g["primary_connected_groups"],
            g["camera_time_groups"],
            median("pair_baseline_metres"),
            median("primary_pair_median_triangulation_degrees")
        ));
    }
    text.push_str("\\bottomrule\\end{tabular}\n\\caption{Measured primary-room reconstruction geometry. An edge requires at least 10\\% shared first-surface pixels in both directions. Groups count camera sets at each time, not independent rooms. Baselines weight each pair equally; angle values are medians of pair medians from 0.5-degree pixel-weighted source-view histograms. Empty pairs remain in the receipt and have no angle. The primary region excludes neighboring rooms and exterior context.}\\label{tab:primary-reconstruction}\\end{table}\n\n");
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshots_reject_stale_sources_duplicate_rooms_and_changed_denominators() {
        let id = GeneratorIdentity {
            schema_version: 1,
            crate_version: "test".into(),
            generator_version: 41,
            source_sha256: "1".repeat(64),
        };
        let cohort = json!({"protocol":{"seed":7,"audit_rooms":1,"rendered_rooms":1,"cameras":2,"width":512,"height":512,"playback_steps":2,"density":0.65,"human_density":0.25,"gi_rays":256},"selected_seeds":[7],"sources_sha256":{"capture.json":"2".repeat(64)},"summary":{"co_visibility":{"views":4,"shared_fraction_valid":0.8}},"reconstruction":{"schema_version":1,"camera_time_groups":2,"minimum_bidirectional_overlap":0.1,"primary_connected_groups":2,"whole_frame_connected_groups":2}});
        let r = json!({"schema_version":1,"identity":id,"rooms":1,"views":4,"cohorts":[cohort.clone()]});
        validate(&r, &id).unwrap();
        let mut stale = r.clone();
        stale["identity"]["source_sha256"] = json!("3".repeat(64));
        assert!(validate(&stale, &id).is_err());
        let mut repeated = r.clone();
        repeated["cohorts"].as_array_mut().unwrap().push(cohort);
        assert!(validate(&repeated, &id).is_err());
        let mut truncated = r;
        truncated["cohorts"][0]["summary"]["co_visibility"]["views"] = json!(3);
        assert!(validate(&truncated, &id).is_err());
    }
}
