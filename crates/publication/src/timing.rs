//! Paper-only, current-source timing receipts. No page edits or GPU dependency.
mod figures;
mod paper;
use crate::{io, pipeline};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::GeneratorIdentity;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const STAGES: [(&str, &str); 13] = [
    ("layout_seconds", "Layout and appearance"),
    ("cameras_seconds", "Camera paths and overlap"),
    ("validation_seconds", "Layout validation"),
    ("motion_planning_seconds", "Motion planning"),
    ("assembly_seconds", "Geometry assemblies"),
    ("material_selection_seconds", "Consumed material selection"),
    ("parallel_assets_seconds", "Materials / transport / meshes"),
    ("transport_binding_seconds", "Transport material binding"),
    ("environment_seconds", "HDR reflection environment"),
    ("gi_request_seconds", "Lighting bake preparation"),
    ("serial_meshes_seconds", "Serial mesh tail (Wasm)"),
    ("group_binding_seconds", "Render group binding"),
    ("other_seconds", "Scheduling and bookkeeping"),
];
fn number(v: &Value) -> Result<f64> {
    let n = v.as_f64().context("missing numeric timing")?;
    ensure!(n.is_finite() && n >= 0., "negative or non-finite timing");
    Ok(n)
}
fn integer(v: &Value) -> Result<usize> {
    v.as_u64().map(|v| v as usize).context("missing integer")
}

pub fn validate_wall(wall: &Value) -> Result<()> {
    let sum = STAGES
        .iter()
        .map(|(k, _)| number(&wall[k]))
        .sum::<Result<f64>>()?;
    let total = number(&wall["total_seconds"])?;
    ensure!(
        total > 0. && (total - sum).abs() < 1e-8_f64.max(total * 1e-8),
        "preparation wall partition does not close: {sum} versus {total}"
    );
    Ok(())
}
fn matches_identity(v: &Value, id: &GeneratorIdentity) -> Result<()> {
    ensure!(
        v["source_sha256"] == id.source_sha256
            && v["crate_version"] == id.crate_version
            && v["generator_version"] == id.generator_version,
        "stale timing source identity; remeasure current generator"
    );
    Ok(())
}
fn validate_run(run: &Value, identity: &GeneratorIdentity) -> Result<()> {
    ensure!(
        matches!(run["kind"].as_str(), Some("preparation" | "capture")),
        "unknown timing workload"
    );
    let summary = &run["summary"];
    matches_identity(&summary["build_provenance"], identity)?;
    let config = &summary["config"];
    ensure!(
        config["indoor_quality"] == "Auto"
            && config["indoor_gpu_materials"] == true
            && config["human_motion"].is_null()
            && config["indoor_human_density"] == 0.25,
        "paper timing protocol requires full native quality, GPU maps and static people"
    );
    let rows = run["records"].as_array().context("timing records")?;
    let prepare = run["kind"] == "preparation";
    let expected = if prepare {
        integer(&summary["rooms"])?
    } else {
        integer(&summary["scenes"])?
            .checked_sub(integer(&summary["warmup_scenes"])?)
            .context("warm-up exceeds population")?
    };
    ensure!(
        rows.len() == expected && expected > 0,
        "timing population incomplete"
    );
    let first = if prepare {
        summary["seed"].as_u64().context("seed")?
    } else {
        summary["config"]["indoor_seed"]
            .as_u64()
            .context("seed")?
            .checked_add(integer(&summary["warmup_scenes"])? as u64)
            .context("seed overflow")?
    };
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row["seed"].as_u64() == first.checked_add(i as u64),
            "nonconsecutive or duplicated timing seed"
        );
        validate_wall(&row["preparation_stages"]["wall"])?;
        for key in [
            "materials_seconds",
            "transport_seconds",
            "meshes_seconds",
            "asset_insertion_seconds",
        ] {
            number(&row["preparation_stages"][key])?;
        }
        if !prepare {
            number(&row["elapsed_seconds"])?;
        }
    }
    if prepare {
        ensure!(
            summary["schema_version"] == 2 && summary["fingerprints_enabled"] == false,
            "not a timed preparation run"
        );
        let seconds = number(&summary["seconds"])?;
        ensure!(
            seconds > 0.
                && (number(&summary["prepared_rooms_per_second"])? - expected as f64 / seconds)
                    .abs()
                    < 1e-8,
            "CPU throughput denominator mismatch"
        );
    } else {
        ensure!(config["num_cameras"] == 3 && config["width"] == 512. && config["height"] == 512.
            && config["playback_steps"] == 1 && config["render_modes"] == json!(["Color","Depth","Position","Normal","Semantic","CoVisibility"])
            && summary["gi_settings"]["enabled"] == true && summary["gi_settings"]["gpu"] == true
            && summary["gi_settings"]["bake"]["rays_per_probe"] == 256
            && summary["cpu_scene_prefetch_depth"] == 3,
            "capture protocol must be three 512-square views, six modes, default lookahead and full GI");
        ensure!(summary["schema_version"] == 4 && summary["saved_lossless_planes"] == false
            && summary["fixed_scene"] == false && summary["gpu_timings_enabled"] == false,
            "capture timing requires regenerating scenes, no file export and no diagnostic GPU timestamp instrumentation");
        let stream = number(&summary["measured_stream_seconds"])?;
        let sum = rows
            .iter()
            .map(|r| number(&r["elapsed_seconds"]))
            .sum::<Result<f64>>()?;
        ensure!(
            stream > 0.
                && stream >= sum
                && (sum - number(&summary["measured_capture_seconds"])?).abs() < 1e-7,
            "capture stream accounting mismatch"
        );
        ensure!(
            (number(&summary["stream_scenes_per_second"])? - expected as f64 / stream).abs() < 1e-8
                && (number(&summary["stream_views_per_second"])? - 3. * expected as f64 / stream)
                    .abs()
                    < 1e-8,
            "capture throughput denominator mismatch"
        );
    }
    Ok(())
}

fn stats(values: impl IntoIterator<Item = f64>) -> Value {
    let mut v: Vec<_> = values.into_iter().collect();
    v.sort_by(f64::total_cmp);
    let sum: f64 = v.iter().sum();
    let quantile = |p: f64| {
        v[((v.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(v.len() - 1)]
    };
    json!({"count":v.len(),"sum":sum,"mean":sum/v.len() as f64,"p50":quantile(0.5),"p95":quantile(0.95),"min":v[0],"max":v[v.len()-1]})
}

fn statistics_agree(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| statistics_agree(v, w)))
        }
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            (a - b).abs() <= 1e-10 * b.abs().max(1.)
        }
        _ => a == b,
    }
}
fn summarize(run: &Value) -> Value {
    let rows = run["records"].as_array().unwrap();
    let stages: BTreeMap<_, _> = STAGES
        .iter()
        .map(|(key, _)| {
            (
                *key,
                stats(
                    rows.iter()
                        .map(|r| r["preparation_stages"]["wall"][key].as_f64().unwrap()),
                ),
            )
        })
        .collect();
    let total = stats(rows.iter().map(|r| {
        r["preparation_stages"]["wall"]["total_seconds"]
            .as_f64()
            .unwrap()
    }));
    json!({"wall":stages,"total":total})
}

pub fn prepare(
    root: &Path,
    captures: &[PathBuf],
    preparation: &[PathBuf],
    environment: &Path,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "timing output exists");
    ensure!(
        !captures.is_empty() && !preparation.is_empty(),
        "both CPU preparation and completed captures are required"
    );
    let identity = pipeline::identity(root)?;
    let mut inputs = BTreeMap::new();
    let mut runs = Vec::new();
    for path in preparation {
        let mut summary: Value = io::read(path)?;
        let mut rows = summary
            .as_object_mut()
            .context("CPU benchmark object")?
            .remove("records")
            .context("CPU records")?;
        for row in rows.as_array_mut().context("CPU records array")? {
            let timings = row
                .as_object_mut()
                .context("CPU record object")?
                .remove("timings")
                .context("CPU timings")?;
            row["preparation_stages"] = timings;
        }
        inputs.insert(path.display().to_string(), io::hash(path)?);
        runs.push(json!({"kind":"preparation","summary":summary,"records":rows}));
    }
    for dir in captures {
        let summary: Value = io::read(&dir.join("summary.json"))?;
        let warmup = integer(&summary["warmup_scenes"])?;
        let rows: Vec<Value> = fs::read_to_string(dir.join("scenes.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            rows.len() == integer(&summary["scenes"])? && warmup < rows.len(),
            "capture rows incomplete"
        );
        for (i, r) in rows.iter().enumerate() {
            ensure!(
                r["index"] == i && r["run_id"] == summary["run_id"],
                "capture identity mismatch"
            );
        }
        let cold = rows[0]["elapsed_seconds"].clone();
        let rows:Vec<_> = rows.into_iter().skip(warmup).map(|r| json!({
            "seed":r["seed"],"elapsed_seconds":r["elapsed_seconds"],"preparation_stages":r["preparation_stages"],
            "update_phase_seconds":r["update_phase_seconds"],"material_bakes":r["material_bakes"],
            "prefetch":r["prefetch"],"humans":r["humans"],"gi":r["gi"],"rss_bytes":r["rss_bytes"]})).collect();
        for name in ["summary.json", "scenes.jsonl"] {
            inputs.insert(
                dir.join(name).display().to_string(),
                io::hash(&dir.join(name))?,
            );
        }
        runs.push(json!({"kind":"capture","summary":summary,"records":rows,"cold_first_scene_seconds":cold}));
    }
    for run in &mut runs {
        validate_run(run, &identity)?;
        run["statistics"] = summarize(run);
    }
    let environment_value: Value = io::read(environment)?;
    for key in ["hardware", "contention", "build"] {
        ensure!(
            environment_value[key]
                .as_str()
                .is_some_and(|v| !v.is_empty()),
            "environment missing {key}"
        );
    }
    inputs.insert(environment.display().to_string(), io::hash(environment)?);
    let receipt = json!({"schema_version":1,"identity":identity,"environment":environment_value,"input_sha256":inputs,"runs":runs});
    fs::create_dir_all(output)?;
    io::write(&output.join("pipeline_receipt.json"), &receipt)?;
    write_paper(output, &receipt, &identity)
}

pub fn write_paper(output: &Path, receipt: &Value, identity: &GeneratorIdentity) -> Result<()> {
    ensure!(receipt["schema_version"] == 1, "timing receipt schema");
    ensure!(
        serde_json::from_value::<GeneratorIdentity>(receipt["identity"].clone())? == *identity,
        "stale paper timing receipt"
    );
    let runs = receipt["runs"].as_array().context("timing runs")?;
    ensure!(!runs.is_empty(), "empty timing receipt");
    for run in runs {
        validate_run(run, identity)?;
        ensure!(
            statistics_agree(&run["statistics"], &summarize(run)),
            "timing statistics mismatch"
        );
    }
    paper::write(output, receipt)?;
    figures::write(output, receipt)
}

/// The full publisher regenerates this paper-only section and rejects stale
/// measurements. Missing receipts are explicit; historical numbers never linger.
pub fn refresh_paper(stage: &Path, identity: &GeneratorIdentity) -> Result<()> {
    let generated = stage.join("tex/generated");
    let path = generated.join("pipeline_receipt.json");
    if path.is_file() {
        write_paper(&generated, &io::read(&path)?, identity)
    } else {
        fs::write(generated.join("pipeline_evaluation.tex"), "\\section{Generation pipeline measurements}\nNo timing receipt is attached to this generator snapshot.\n").map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_overlap_missing_negative_and_nonclosing_stage_windows() {
        let mut wall = json!({"total_seconds":1.});
        for (key, _) in STAGES {
            wall[key] = json!(0.);
        }
        wall["parallel_assets_seconds"] = json!(0.75);
        wall["other_seconds"] = json!(0.25);
        validate_wall(&wall).unwrap();
        wall["assembly_seconds"] = json!(0.4);
        assert!(validate_wall(&wall).is_err());
        wall["assembly_seconds"] = json!(-0.4);
        assert!(validate_wall(&wall).is_err());
        wall.as_object_mut().unwrap().remove("assembly_seconds");
        assert!(validate_wall(&wall).is_err());
    }
    #[test]
    fn quantiles_use_nearest_rank_and_preserve_tail() {
        let s = stats([4., 1., 3., 2.]);
        assert_eq!(s["p50"], 2.);
        assert_eq!(s["p95"], 4.);
        assert_eq!(s["mean"], 2.5);
    }
    fn fixture() -> (GeneratorIdentity, Value) {
        let identity = GeneratorIdentity {
            schema_version: 1,
            crate_version: "1.0.0".into(),
            generator_version: 40,
            source_sha256: "current".into(),
        };
        let mut wall = json!({"total_seconds":1.});
        for (key, _) in STAGES {
            wall[key] = json!(0.);
        }
        wall["other_seconds"] = json!(1.);
        let run = json!({"kind":"capture", "summary":{
            "build_provenance":identity, "schema_version":4, "scenes":2, "warmup_scenes":1,
            "saved_lossless_planes":false,"fixed_scene":false,"gpu_timings_enabled":false,
            "measured_stream_seconds":2.,"measured_capture_seconds":1.5,
            "stream_scenes_per_second":0.5,"stream_views_per_second":1.5,"cpu_scene_prefetch_depth":3,
            "gi_settings":{"enabled":true,"gpu":true,"bake":{"rays_per_probe":256}},
            "config":{"indoor_quality":"Auto","indoor_gpu_materials":true,"human_motion":null,
                "indoor_human_density":0.25,"indoor_seed":399,"num_cameras":3,"width":512.,"height":512.,
                "playback_steps":1,"render_modes":["Color","Depth","Position","Normal","Semantic","CoVisibility"]}
            },"records":[{"seed":400,"elapsed_seconds":1.5,"preparation_stages":{"wall":wall,
                "materials_seconds":0.,"transport_seconds":0.,"meshes_seconds":0.,"asset_insertion_seconds":0.}}]});
        (identity, run)
    }
    #[test]
    fn rejects_incomplete_populations_seed_drift_and_inflated_throughput() {
        let (identity, run) = fixture();
        validate_run(&run, &identity).unwrap();
        let mut bad = run.clone();
        bad["records"] = json!([]);
        assert!(validate_run(&bad, &identity).is_err());
        let mut bad = run.clone();
        bad["records"][0]["seed"] = json!(399);
        assert!(validate_run(&bad, &identity).is_err());
        let mut bad = run.clone();
        bad["summary"]["stream_scenes_per_second"] = json!(1.);
        assert!(validate_run(&bad, &identity).is_err());
        let mut bad = run.clone();
        bad["summary"]["measured_stream_seconds"] = json!(1.);
        assert!(validate_run(&bad, &identity).is_err());
        let mut bad = run;
        bad["summary"]["config"]["indoor_quality"] = json!("Portable");
        assert!(validate_run(&bad, &identity).is_err());
    }
    #[test]
    fn statistics_verify_after_json_roundtrip_but_reject_modified_values() {
        let (_, run) = fixture();
        let summary = summarize(&run);
        let mut decoded: Value =
            serde_json::from_slice(&serde_json::to_vec(&summary).unwrap()).unwrap();
        assert!(statistics_agree(&decoded, &summary));
        decoded["total"]["mean"] = json!(2.);
        assert!(!statistics_agree(&decoded, &summary));
    }
    #[test]
    fn refuses_stale_source_even_at_the_same_package_version() {
        let id = GeneratorIdentity {
            schema_version: 1,
            crate_version: "1.0.0".into(),
            generator_version: 40,
            source_sha256: "current".into(),
        };
        assert!(matches_identity(
            &json!({"crate_version":"1.0.0","generator_version":40,"source_sha256":"old"}),
            &id
        )
        .is_err());
    }
}
