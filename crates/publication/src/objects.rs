//! Current-generator object qualification. Expensive descriptors stay in the
//! local cache; the shipped atlas binds every measurement and rendered example.
mod paper;
use crate::{io, pipeline};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::GeneratorIdentity;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

pub const DIRECTORY: &str = "www/project/static/media/objects";
pub const SAMPLES: usize = 256;
pub const TOOLS: [&str; 2] = ["examples/audit_objects.rs", "examples/review_objects.rs"];

fn tool_sources(root: &Path) -> Result<BTreeMap<String, String>> {
    TOOLS
        .iter()
        .map(|p| Ok(((*p).into(), io::hash(&root.join(p))?)))
        .collect()
}

pub fn refresh_cache(root: &Path, cache: &Path, identity: &GeneratorIdentity) -> Result<()> {
    if let Ok(atlas) = verify(cache, identity) {
        if atlas["tool_sources"] == json!(tool_sources(root)?) {
            return Ok(());
        }
    }
    let work = tempfile::Builder::new()
        .prefix("object-qualification-")
        .tempdir_in(cache.parent().context("no cache parent")?)?
        .keep();
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    ensure!(
        Command::new(&cargo)
            .current_dir(root)
            .args([
                "build",
                "--locked",
                "--no-default-features",
                "--features",
                "multi_threaded",
                "--example",
                "audit_objects",
                "--example",
                "review_objects"
            ])
            .status()?
            .success(),
        "object qualification build failed"
    );
    let output = Command::new(&cargo)
        .current_dir(root)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()?;
    ensure!(output.status.success(), "cargo metadata failed");
    let metadata: Value = serde_json::from_slice(&output.stdout)?;
    let binaries = Path::new(
        metadata["target_directory"]
            .as_str()
            .context("missing target directory")?,
    )
    .join("debug/examples");
    for (binary, args) in [
        (
            "audit_objects",
            vec![
                "--output".into(),
                work.join("audit").into_os_string(),
                "--samples".into(),
                SAMPLES.to_string().into(),
            ],
        ),
        (
            "review_objects",
            vec![
                "--audit".into(),
                work.join("audit/audit.json").into_os_string(),
                "--output".into(),
                work.join("renders").into_os_string(),
            ],
        ),
    ] {
        ensure!(
            Command::new(binaries.join(format!("{binary}{}", std::env::consts::EXE_SUFFIX)))
                .current_dir(root)
                .args(args)
                .status()?
                .success(),
            "{binary} failed; diagnostics in {}",
            work.display()
        );
    }
    prepare(
        root,
        &work.join("audit/audit.json"),
        &work.join("renders"),
        &work.join("completed"),
    )?;
    verify(&work.join("completed"), identity)?;
    if cache.exists() {
        fs::remove_dir_all(cache)?;
    }
    fs::rename(work.join("completed"), cache)?;
    // Retain raw local diagnostic evidence, including the descriptors.
    Ok(())
}

fn summary(mut data: Vec<f64>) -> Value {
    data.sort_by(f64::total_cmp);
    json!({"min":data[0], "median":data[data.len()/2], "p95":data[(data.len()-1)*95/100], "max":data[data.len()-1]})
}

fn validate_samples(samples: &[Value], kind: &str) -> Result<()> {
    ensure!(samples.len() == SAMPLES, "incomplete {kind} samples");
    for (i, s) in samples.iter().enumerate() {
        ensure!(
            s["object"]["kind"] == kind
                && s["object"]["seed"] == i
                && s["program"].is_object()
                && s["mesh"]["triangles"].as_u64().is_some_and(|n| n > 0)
                && s["mesh"]["batches"].as_u64().is_some_and(|n| n > 0)
                && s["mesh"]["max_envelope_overrun_m"]
                    .as_f64()
                    .is_some_and(|n| (0.0..=0.003).contains(&n))
                && s["build_ms"]
                    .as_f64()
                    .is_some_and(|n| n.is_finite() && n >= 0.),
            "invalid/duplicated object sample {kind}/{i}"
        );
    }
    Ok(())
}

fn parameters(samples: &[Value]) -> Value {
    let mut result = BTreeMap::new();
    let keys: std::collections::BTreeSet<_> = samples
        .iter()
        .flat_map(|s| s["program"].as_object().unwrap().keys())
        .collect();
    for key in keys {
        let values: Vec<_> = samples
            .iter()
            .filter_map(|s| s["program"].get(key))
            .filter(|v| !v.is_null())
            .cloned()
            .collect();
        let present = values.len();
        if present == 0 {
            continue;
        }
        if values.iter().all(Value::is_number) {
            let numbers: Vec<_> = values.iter().map(|v| v.as_f64().unwrap()).collect();
            let stats = summary(numbers.clone());
            let lo = stats["min"].as_f64().unwrap();
            let hi = stats["max"].as_f64().unwrap();
            let mut bins = [0usize; 8];
            for x in numbers {
                bins[if hi > lo {
                    (((x - lo) / (hi - lo) * 8.).floor() as usize).min(7)
                } else {
                    0
                }] += 1;
            }
            result.insert(
                key.clone(),
                json!({"type":"numeric", "present":present, "total":samples.len(), "observed":stats, "bins":bins}),
            );
        } else {
            let mut counts = BTreeMap::<String, usize>::new();
            for v in values {
                *counts.entry(v.to_string()).or_default() += 1;
            }
            result.insert(key.clone(), json!({"type":"categorical", "present":present, "total":samples.len(), "counts":counts}));
        }
    }
    json!(result)
}

/// Explicit local entry point; both inputs must come from the current renderer.
/// This does not publish or install anything in the project page.
pub fn prepare(root: &Path, audit: &Path, renders: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "object report output already exists");
    let identity = pipeline::identity(root)?;
    let mut report: Value = io::read(audit)?;
    let rendered: Value = io::read(&renders.join("renders.json"))?;
    for p in [&report["build_provenance"], &rendered["build_provenance"]] {
        ensure!(
            p["source_sha256"] == identity.source_sha256
                && p["generator_version"] == identity.generator_version
                && p["crate_version"] == identity.crate_version,
            "stale object audit or renderer"
        );
    }
    ensure!(
        report["sample_count_per_category"] == SAMPLES
            && rendered["image_size"] == json!([320, 320]),
        "unexpected object protocol"
    );
    validate_inventory(&report)?;
    let categories = report["categories"]
        .as_array_mut()
        .context("missing object categories")?;
    ensure!(
        rendered["categories"]
            .as_array()
            .is_some_and(|c| c.len() == categories.len()),
        "incomplete object inventory"
    );
    fs::create_dir_all(output)?;
    let mut files = BTreeMap::new();
    for category in categories {
        let kind = category["kind"]
            .as_str()
            .context("missing kind")?
            .to_string();
        ensure!(
            kind.chars().all(|c| c.is_ascii_alphanumeric()),
            "unsafe category"
        );
        ensure!(
            category["failures"].as_array().is_some_and(Vec::is_empty),
            "failed object meshes: {kind}"
        );
        let samples = category["samples"].as_array().context("missing samples")?;
        validate_samples(samples, &kind)?;
        let rendered = rendered["categories"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["kind"] == kind)
            .context("missing category renders")?;
        let selected = rendered["selected"]
            .as_array()
            .context("missing selected rows")?;
        ensure!(
            selected.len() == 6 && selected.iter().all(|s| samples.contains(s)),
            "rendered samples do not match audit"
        );
        let selected_seeds: Vec<_> = selected
            .iter()
            .map(|s| s["object"]["seed"].clone())
            .collect();
        let parameters = parameters(samples);
        let mesh: BTreeMap<_, _> = ["triangles", "batches", "max_envelope_overrun_m"]
            .iter()
            .map(|field| {
                (
                    (*field).to_string(),
                    summary(
                        samples
                            .iter()
                            .map(|s| s["mesh"][field].as_f64().unwrap())
                            .collect(),
                    ),
                )
            })
            .collect();
        let cost = summary(
            samples
                .iter()
                .map(|s| s["build_ms"].as_f64().unwrap())
                .collect(),
        );
        category["parameters"] = parameters;
        category["mesh_summary"] = json!(mesh);
        category["build_ms_summary"] = cost;
        category["selected_seeds"] = json!(selected_seeds);
        category
            .as_object_mut()
            .unwrap()
            .remove("depth_descriptors");
        let samples_file = format!("{kind}-samples.json");
        io::write(&output.join(&samples_file), &category["samples"])?;
        files.insert(samples_file.clone(), io::hash(&output.join(&samples_file))?);
        category["samples_file"] = json!(samples_file);
        category.as_object_mut().unwrap().remove("samples");
        for i in 0..6 {
            for mode in ["color", "normal"] {
                let name = format!("{kind}-{i}-{mode}.webp");
                let image = image::open(renders.join(format!("{kind}-{i}-{mode}.png")))?;
                ensure!(
                    image.width() == 320 && image.height() == 320,
                    "invalid studio image"
                );
                image.save(output.join(&name))?;
                files.insert(name.clone(), io::hash(&output.join(name))?);
            }
        }
    }
    report["generator_identity"] = json!(identity);
    report["tool_sources"] = json!(tool_sources(root)?);
    report["inputs"] = json!({"audit_sha256":io::hash(audit)?, "renders_sha256":io::hash(&renders.join("renders.json"))?});
    report["render_protocol"] = json!({"image_size":[320,320],"selected_per_category":6,"lighting":rendered["lighting"],"selection":"deterministic farthest-spread depth-map examples; correlated samples, not isolated parameter interventions","constant_factors":"metric envelope, camera, exposure, finish seed and studio lights within each category"});
    report["files"] = json!(files);
    io::write(&output.join("atlas.json"), &report)?;
    verify(output, &identity)?;
    Ok(())
}

fn validate_inventory(report: &Value) -> Result<()> {
    let total = report["catalog_total_count"]
        .as_u64()
        .context("missing source catalog size")? as usize;
    ensure!(
        report["audit_scope"] == "all_object_kinds"
            && total > 0
            && report["categories"]
                .as_array()
                .is_some_and(|c| c.len() == total),
        "incomplete source catalog inventory"
    );
    Ok(())
}

pub fn verify(directory: &Path, identity: &GeneratorIdentity) -> Result<Value> {
    let atlas: Value = io::read(&directory.join("atlas.json"))?;
    ensure!(
        atlas["schema_version"] == 1
            && atlas["generator_identity"] == json!(identity)
            && atlas["sample_count_per_category"] == SAMPLES
            && atlas["resolution"] == 24,
        "stale/incomplete object atlas"
    );
    let categories = atlas["categories"]
        .as_array()
        .context("missing categories")?;
    validate_inventory(&atlas)?;
    let mut names = std::collections::BTreeSet::new();
    let mut expected = std::collections::BTreeSet::new();
    for category in categories {
        let kind = category["kind"].as_str().context("missing kind")?;
        ensure!(
            names.insert(kind) && kind.chars().all(|c| c.is_ascii_alphanumeric()),
            "duplicate/unsafe category"
        );
        ensure!(
            category["failures"].as_array().is_some_and(Vec::is_empty),
            "failed geometry in atlas"
        );
        let sample_file = format!("{kind}-samples.json");
        ensure!(
            category["samples_file"] == sample_file,
            "unexpected category data path"
        );
        expected.insert(sample_file.clone());
        let sample_data: Value = io::read(&directory.join(sample_file))?;
        let samples = sample_data.as_array().context("missing samples")?;
        validate_samples(samples, kind)?;
        ensure!(
            category["parameters"] == parameters(samples),
            "parameter marginals do not match samples"
        );
        ensure!(
            category["build_ms_summary"]
                == summary(
                    samples
                        .iter()
                        .map(|s| s["build_ms"].as_f64().unwrap())
                        .collect()
                ),
            "build timings do not match samples"
        );
        for field in ["triangles", "batches", "max_envelope_overrun_m"] {
            ensure!(
                category["mesh_summary"][field]
                    == summary(
                        samples
                            .iter()
                            .map(|s| s["mesh"][field].as_f64().unwrap())
                            .collect()
                    ),
                "mesh summary does not match samples"
            );
        }
        ensure!(
            samples.len() == SAMPLES
                && category["nearest_shape_distances"]
                    .as_array()
                    .is_some_and(|a| a.len() == SAMPLES),
            "sample denominator mismatch"
        );
        for (i, s) in samples.iter().enumerate() {
            ensure!(
                s["object"]["kind"] == kind
                    && s["object"]["seed"] == i
                    && s["mesh"]["triangles"].as_u64().is_some_and(|n| n > 0)
                    && s["mesh"]["max_envelope_overrun_m"]
                        .as_f64()
                        .is_some_and(|n| (0.0..=0.003).contains(&n)),
                "invalid audited object"
            );
        }
        let coverage = category["coverage"]
            .as_array()
            .context("missing coverage")?;
        ensure!(coverage.len() == 3, "missing sensitivity analysis");
        for (c, t) in coverage.iter().zip([0.005, 0.015, 0.03]) {
            ensure!(
                c["threshold"]
                    .as_f64()
                    .is_some_and(|v| (v - t).abs() < 1e-7),
                "unexpected threshold"
            );
            let ids = c["representatives"]
                .as_array()
                .context("missing representatives")?;
            ensure!(
                !ids.is_empty()
                    && ids
                        .iter()
                        .all(|v| v.as_u64().is_some_and(|n| n < SAMPLES as u64))
                    && ids.windows(2).all(|w| w[0].as_u64() < w[1].as_u64()),
                "invalid coverage set"
            );
        }
        let chosen = category["selected_seeds"]
            .as_array()
            .context("missing examples")?;
        ensure!(
            chosen.len() == 6
                && chosen
                    .iter()
                    .all(|s| s.as_u64().is_some_and(|n| n < SAMPLES as u64)),
            "invalid example seeds"
        );
        for i in 0..6 {
            for mode in ["color", "normal"] {
                expected.insert(format!("{kind}-{i}-{mode}.webp"));
            }
        }
    }
    let files: BTreeMap<String, String> = serde_json::from_value(atlas["files"].clone())?;
    ensure!(
        files
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            == expected,
        "object image closure differs"
    );
    for (name, hash) in files {
        ensure!(
            io::hash(&directory.join(&name))? == hash,
            "changed object image {name}"
        );
    }
    Ok(atlas)
}

pub fn build(cache: &Path, stage: &Path, identity: &GeneratorIdentity) -> Result<()> {
    verify(cache, identity)?;
    let output = stage.join(DIRECTORY);
    if output.exists() {
        fs::remove_dir_all(&output)?;
    }
    io::copy_tree(cache, &output)?;
    paper::write(stage)
}
pub fn write_paper(stage: &Path) -> Result<()> {
    paper::write(stage)
}
pub fn page() -> &'static str {
    include_str!("../templates/objects.html")
}

/// The page has one geometry browser, including architectural assemblies.
pub fn verify_explorer(html: &str) -> Result<()> {
    ensure!(
        html.matches("id=\"object-atlas\"").count() == 1
            && html.matches("id=\"object-category\"").count() == 1
            && html.matches("id=\"object-mode\"").count() == 1
            && html.matches("id=\"object-study\"").count() == 1
            && html.contains("data-assemblies=\"static/media/doors/atlas.json\"")
            && !html.contains("id=\"doors\"")
            && !html.contains("src=\"static/media/doors/index.html\""),
        "duplicate or disjoint object/door presentation"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn single_geometry_explorer_includes_doors_and_cabinetry() {
        verify_explorer(page()).unwrap();
        assert!(!include_str!("../templates/project.html").contains("@DOOR_STUDY@"));
        for duplicate in [
            page(),
            "<section id=\"doors\"></section>",
            "<iframe src=\"static/media/doors/index.html\"></iframe>",
        ] {
            assert!(verify_explorer(&format!("{}{duplicate}", page())).is_err());
        }
    }
    #[test]
    fn full_catalog_gate_rejects_partial_and_stale_inventory_receipts() {
        validate_inventory(
            &json!({"audit_scope":"all_object_kinds","catalog_total_count":2,"categories":[{},{}]}),
        )
        .unwrap();
        for report in [
            json!({"categories":[{},{}]}),
            json!({"audit_scope":"all_object_kinds","catalog_total_count":3,"categories":[{},{}]}),
            json!({"audit_scope":"public_program_items","catalog_total_count":2,"categories":[{},{}]}),
        ] {
            assert!(validate_inventory(&report).is_err());
        }
    }

    #[test]
    fn missing_duplicate_or_out_of_envelope_samples_fail_closed() {
        let mut samples: Vec<_> = (0..SAMPLES).map(|seed|json!({"object":{"kind":"Chair","seed":seed},"program":{"width":1.},"mesh":{"triangles":12,"batches":1,"max_envelope_overrun_m":0.},"build_ms":0.1})).collect();
        validate_samples(&samples, "Chair").unwrap();
        assert!(validate_samples(&samples[..SAMPLES - 1], "Chair").is_err());
        samples[1]["object"]["seed"] = json!(0);
        assert!(validate_samples(&samples, "Chair").is_err());
        samples[1]["object"]["seed"] = json!(1);
        samples[2]["mesh"]["max_envelope_overrun_m"] = json!(0.0031);
        assert!(validate_samples(&samples, "Chair").is_err());
    }
    #[test]
    fn conditional_and_numeric_parameter_marginals_keep_exact_denominators() {
        let samples = vec![
            json!({"program":{"size":1.,"arms":false}}),
            json!({"program":{"size":2.,"arms":true,"open_angle":42.}}),
            json!({"program":{"size":1.,"arms":false}}),
        ];
        let p = parameters(&samples);
        assert_eq!(p["arms"]["counts"]["false"], 2);
        assert_eq!(p["open_angle"]["type"], "numeric");
        assert_eq!(p["open_angle"]["present"], 1);
        assert_eq!(p["open_angle"]["total"], 3);
        assert_eq!(p["open_angle"]["observed"]["min"], 42.);
        assert_eq!(
            p["size"]["bins"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .sum::<u64>(),
            3
        );
        assert_eq!(p["size"]["observed"]["median"], 1.);
    }
}
