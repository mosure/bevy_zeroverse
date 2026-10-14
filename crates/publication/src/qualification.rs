//! Small, explicit factor sweeps using the real native capture engine. Failed
//! cases stay in the receipt; there is no image-score selection or retry curation.
use crate::io;

pub const PUBLISHED_DIRECTORY: &str = "www/project/static/media/qualification";

/// Cached only for the exact current executable inputs and the canonical recipe.
pub fn verify_current(
    output: &Path,
    identity: &bevy_zeroverse_capture::GeneratorIdentity,
) -> Result<Value> {
    verify(output)?;
    let receipt: Value = io::read(&output.join("receipt.json"))?;
    ensure!(
        receipt["success"] == true
            && receipt["generator_identity"] == json!(identity)
            && receipt["recipe"] == json!(Recipe::default()),
        "stale or incomplete factor qualification"
    );
    Ok(receipt)
}

pub fn refresh_cache(
    root: &Path,
    cache: &Path,
    identity: &bevy_zeroverse_capture::GeneratorIdentity,
) -> Result<()> {
    if verify_current(cache, identity).is_ok() {
        return Ok(());
    }
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    ensure!(
        Command::new(&cargo)
            .current_dir(root)
            .args([
                "build",
                "--locked",
                "--bin",
                "indoor_validate",
                "--no-default-features",
                "--features",
                "multi_threaded"
            ])
            .status()?
            .success(),
        "qualification validator build failed"
    );
    let metadata = Command::new(&cargo)
        .current_dir(root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()?;
    ensure!(
        metadata.status.success(),
        "could not locate Cargo target directory"
    );
    let metadata: Value = serde_json::from_slice(&metadata.stdout)?;
    let validator = Path::new(
        metadata["target_directory"]
            .as_str()
            .context("missing target directory")?,
    )
    .join("debug")
    .join(format!("indoor_validate{}", std::env::consts::EXE_SUFFIX));
    let temporary = tempfile::Builder::new()
        .prefix("qualification-")
        .tempdir_in(
            cache
                .parent()
                .context("qualification cache has no parent")?,
        )?
        .keep();
    let completed = temporary.join("completed");
    run(root, &validator, &completed, None)?;
    verify_current(&completed, identity)?;
    if cache.exists() {
        fs::remove_dir_all(cache)?;
    }
    fs::rename(completed, cache)?;
    fs::remove_dir(temporary)?;
    Ok(())
}
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub name: String,
    #[serde(default = "two", skip_serializing_if = "is_two")]
    pub cameras: usize,
    pub image_size: [u32; 2],
    pub camera: Value,
    pub appearance: Value,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub schema_version: u32,
    pub seed: u64,
    pub rooms_per_case: usize,
    #[serde(default = "two", skip_serializing_if = "is_two")]
    pub playback_steps: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub human_density: f32,
    pub cases: Vec<Case>,
}
fn two() -> usize {
    2
}
fn is_two(value: &usize) -> bool {
    *value == 2
}
fn is_zero(value: &f32) -> bool {
    *value == 0.
}

#[cfg(test)]
mod count_contract_tests {
    use super::*;
    #[test]
    fn changing_camera_diagnostics_keeps_family_geometry_but_object_changes_do_not() {
        let a =
            json!({"objects":[{"size":[1,2,3]}],"camera_diagnostics":{"attempts":1},"cameras":[1]});
        let mut b = a.clone();
        b["camera_diagnostics"] = json!({"attempts":9});
        b["cameras"] = json!([1, 2, 3]);
        assert_eq!(geometry_hash(&a).unwrap(), geometry_hash(&b).unwrap());
        b["objects"][0]["size"] = json!([1, 2, 4]);
        assert_ne!(geometry_hash(&a).unwrap(), geometry_hash(&b).unwrap());
    }
    #[test]
    fn count_controls_preserve_legacy_serialization_and_validate_single_view() {
        let value = json!({"schema_version":1,"seed":0,"rooms_per_case":2,"cases":[{"name":"reference","image_size":[256,160],"camera":{},"appearance":{}}]});
        let mut recipe: Recipe = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&recipe).unwrap(), value);
        validate_recipe(&recipe).unwrap();
        recipe.cases[0].cameras = 1;
        recipe.playback_steps = 3;
        recipe.human_density = 0.25;
        validate_recipe(&recipe).unwrap();
        for n in [0, 17] {
            recipe.cases[0].cameras = n;
            assert!(validate_recipe(&recipe).is_err());
        }
    }
}
impl Default for Recipe {
    fn default() -> Self {
        let camera = json!({"duration_seconds":2.,"path_length_min":0.03,"path_length_max":0.6,"long_path_fraction":0.});
        let base = Case {
            name: "reference".into(),
            cameras: 2,
            image_size: [256, 160],
            camera,
            appearance: json!({}),
        };
        let mut cases = vec![base.clone()];
        for (name, size, camera, appearance) in [
            ("portrait", [160, 256], json!({}), json!({})),
            (
                "wide_baseline",
                [256, 160],
                json!({"baseline":0.9}),
                json!({}),
            ),
            (
                "handheld",
                [256, 160],
                json!({"handheld":{},"path_length_min":0.01}),
                json!({}),
            ),
            (
                "low_overlap",
                [256, 160],
                json!({"overlap_mixture":{"weights":[0,1,0]}}),
                json!({}),
            ),
            (
                "no_proxy_overlap",
                [256, 160],
                json!({"overlap_mixture":{"weights":[0,0,1]}}),
                json!({}),
            ),
            (
                "material_seed",
                [256, 160],
                json!({}),
                json!({"material_seed":55}),
            ),
            (
                "low_detail",
                [256, 160],
                json!({}),
                json!({"material_detail":0.15}),
            ),
            (
                "lighting_seed",
                [256, 160],
                json!({}),
                json!({"lighting_seed":88}),
            ),
            (
                "dim_illumination",
                [256, 160],
                json!({}),
                json!({"illumination_scale":0.1}),
            ),
            (
                "exposure",
                [256, 160],
                json!({}),
                json!({"exposure_ev100_offset":1.5}),
            ),
        ] {
            let mut case = base.clone();
            case.name = name.into();
            case.image_size = size;
            case.camera
                .as_object_mut()
                .unwrap()
                .extend(camera.as_object().unwrap().clone());
            case.appearance = appearance;
            cases.push(case);
        }
        Self {
            schema_version: 1,
            seed: 0,
            rooms_per_case: 2,
            playback_steps: 2,
            human_density: 0.,
            cases,
        }
    }
}

/// Output is immutable: use a new folder to preserve failed qualification runs.
pub fn run(root: &Path, validator: &Path, output: &Path, recipe_path: Option<&Path>) -> Result<()> {
    let root = root.canonicalize()?;
    let validator = validator
        .canonicalize()
        .context("build indoor_validate first")?;
    let recipe: Recipe = recipe_path.map(io::read).transpose()?.unwrap_or_default();
    validate_recipe(&recipe)?;
    ensure!(
        !output.exists(),
        "qualification output already exists; choose a new folder"
    );
    fs::create_dir_all(output)?;
    let output = output.canonicalize()?;
    io::write(&output.join("recipe.json"), &recipe)?;
    let identity = crate::pipeline::identity(&root)?;
    // Package verification and checkout binaries can share a target directory.
    // Reject the wrong executable before paying for any rendered factor cases.
    let validator_sha256 = io::hash(&validator)?;
    let preflight = output.join("preflight");
    let preflight_log = fs::File::create(output.join("preflight.log"))?;
    ensure!(
        Command::new(&validator)
            .current_dir(&root)
            .args([
                "--asset-root",
                root.to_str().context("non-UTF8 root")?,
                "--audit-seeds",
                "1",
                "--renders",
                "0",
                "--cameras",
                "1",
                "--human-density",
                "0",
                "--output",
                preflight.to_str().context("non-UTF8 preflight path")?
            ])
            .stdout(Stdio::from(preflight_log.try_clone()?))
            .stderr(Stdio::from(preflight_log))
            .status()?
            .success(),
        "validator preflight failed"
    );
    let compiled_identity: bevy_zeroverse_capture::GeneratorIdentity =
        io::read(&preflight.join("generator_identity.json"))?;
    ensure!(
        compiled_identity == identity,
        "validator binary is stale; rebuild from this checkout before capturing"
    );
    let mut cases = Vec::new();
    let mut families = BTreeSet::new();
    let mut family_geometry = BTreeMap::<String, String>::new();
    let mut images = BTreeMap::<String, usize>::new();
    let mut compiled_provenance = None;
    for case in &recipe.cases {
        ensure!(
            io::hash(&validator)? == validator_sha256,
            "validator executable changed during qualification"
        );
        let dir = output.join(&case.name);
        fs::create_dir(&dir)?;
        let log = fs::File::create(dir.join("capture.log"))?;
        let args = vec![
            "--asset-root".into(),
            root.to_string_lossy().into_owned(),
            "--seed".into(),
            recipe.seed.to_string(),
            "--audit-seeds".into(),
            recipe.rooms_per_case.to_string(),
            "--renders".into(),
            recipe.rooms_per_case.to_string(),
            "--cameras".into(),
            case.cameras.to_string(),
            "--playback-steps".into(),
            recipe.playback_steps.to_string(),
            "--width".into(),
            case.image_size[0].to_string(),
            "--height".into(),
            case.image_size[1].to_string(),
            "--human-density".into(),
            recipe.human_density.to_string(),
            "--labels".into(),
            "--co-visibility".into(),
            "--no-raw".into(),
            "--indoor-camera".into(),
            serde_json::to_string(&case.camera)?,
            "--indoor-appearance".into(),
            serde_json::to_string(&case.appearance)?,
            "--output".into(),
            dir.to_string_lossy().into_owned(),
        ];
        println!(
            "qualification: {} ({} rooms)",
            case.name, recipe.rooms_per_case
        );
        let status = Command::new(&validator)
            .current_dir(&root)
            .args(&args)
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .status();
        let mut result = json!({"name":case.name,"camera_count":case.cameras,"command_args":args,"success":false});
        let evaluation = (|| -> Result<Value> {
            ensure!(
                status.context("launch validator")?.success(),
                "validator failed; inspect capture.log"
            );
            let complete: Value = io::read(&dir.join("run_complete.json"))?;
            ensure!(
                complete["captured_scenes"] == recipe.rooms_per_case,
                "incomplete capture"
            );
            let mut rooms = Vec::new();
            let mut overlap = Vec::new();
            let mut baseline = Vec::new();
            let mut angles = Vec::new();
            let mut reprojection = Vec::new();
            for i in 0..recipe.rooms_per_case {
                let seed = recipe.seed + i as u64;
                let folder = format!("seed_{seed:06}");
                let room = dir.join(&folder);
                let capture: Value = io::read(&room.join("capture.json"))?;
                let manifest: Value = io::read(&room.join("manifest.json"))?;
                let provenance = &capture["build_provenance"];
                ensure!(
                    provenance["source_sha256"] == identity.source_sha256
                        && capture["seed"] == seed
                        && manifest["seed"] == seed,
                    "stale executable or scene identity"
                );
                ensure!(
                    capture["image_size"] == json!(case.image_size),
                    "wrong capture resolution"
                );
                ensure!(
                    compiled_provenance.as_ref().is_none_or(|p| p == provenance),
                    "mixed capture executable provenance"
                );
                compiled_provenance = Some(provenance.clone());
                let qualification = &capture["camera_qualification"];
                let family = qualification["scene_family"]["id"]
                    .as_str()
                    .context("missing family identity")?;
                families.insert(family.to_owned());
                let geometry_hash = geometry_hash(&manifest)?;
                ensure!(
                    family_geometry
                        .get(family)
                        .is_none_or(|previous| previous == &geometry_hash),
                    "geometry changed within a matched family"
                );
                family_geometry.insert(family.to_owned(), geometry_hash.clone());
                let views = capture["views"]
                    .as_array()
                    .context("missing captured views")?;
                ensure!(
                    views.len() == case.cameras * recipe.playback_steps,
                    "incomplete camera/time capture"
                );
                let mut view_ids = BTreeSet::new();
                for (v, view) in views.iter().enumerate() {
                    let camera = view["camera_index"]
                        .as_u64()
                        .context("missing camera index")? as usize;
                    let step = view["step_index"]
                        .as_u64()
                        .context("missing timestep index")? as usize;
                    ensure!(
                        camera < case.cameras
                            && step < recipe.playback_steps
                            && view_ids.insert((camera, step)),
                        "invalid/repeated camera-time identity"
                    );
                    let calibration: bevy_zeroverse_capture::calibration::CameraCalibration =
                        serde_json::from_value(view["calibration"].clone())?;
                    calibration.validate().map_err(anyhow::Error::msg)?;
                    ensure!(
                        calibration.image_size == case.image_size,
                        "calibration size mismatch"
                    );
                    reprojection.push(
                        view["annotation_alignment"]["reprojection_p99_pixels"]
                            .as_f64()
                            .context("missing reprojection oracle")?,
                    );
                    *images
                        .entry(io::hash(&room.join(format!("view_{v:02}_color.png")))?)
                        .or_default() += 1;
                }
                let pairs = qualification["rendered_overlap"]
                    .as_array()
                    .context("missing measured overlap")?;
                ensure!(
                    pairs.len() == case.cameras * (case.cameras - 1) * recipe.playback_steps,
                    "missing directed pair measurements"
                );
                let mut pair_ids = BTreeSet::new();
                for pair in pairs {
                    let source = pair["source_camera"]
                        .as_u64()
                        .context("missing pair source")? as usize;
                    let target = pair["target_camera"]
                        .as_u64()
                        .context("missing pair target")? as usize;
                    let step = pair["step_index"].as_u64().context("missing pair time")? as usize;
                    ensure!(
                        source < case.cameras
                            && target < case.cameras
                            && source != target
                            && step < recipe.playback_steps
                            && pair_ids.insert((source, target, step)),
                        "invalid/repeated directed pair identity"
                    );
                    if let Some(v) = pair["shared_fraction_valid"].as_f64() {
                        overlap.push(v);
                    }
                    if let Some(v) = pair["baseline_m"].as_f64() {
                        baseline.push(v);
                    }
                    if let Some(v) = pair["mean_triangulation_degrees"].as_f64() {
                        angles.push(v);
                    }
                }
                rooms.push(json!({"seed":seed,"folder":folder,"family":qualification["scene_family"],
                    "geometry_sha256":geometry_hash,"layout":manifest["layout"],"envelope":manifest["envelope"],"domain":manifest["program"]["domain"],
                    "object_count":manifest["objects"].as_array().map(Vec::len),"renderer":capture["renderer"],
                    "camera_qualification":qualification,"views":views}));
            }
            Ok(
                json!({"success":true,"rooms":rooms,"directed_rendered_overlap":distribution(overlap),"baseline_m":distribution(baseline),
                "triangulation_degrees":distribution(angles),"per_view_reprojection_p99_pixels":distribution(reprojection)}),
            )
        })();
        match evaluation {
            Ok(value) => result
                .as_object_mut()
                .unwrap()
                .extend(value.as_object().unwrap().clone()),
            Err(error) => {
                result["error"] = json!(format!("{error:#}"));
                eprintln!("{}: {error:#}", case.name);
            }
        }
        cases.push(result);
    }
    let success = cases.iter().all(|v| v["success"] == true);
    let mut receipt = json!({"schema_version":1,"qualification":"bounded factor sweep; not photographic realism or downstream transfer evidence",
        "success":success,"recipe":recipe,"generator_identity":identity,"build_provenance":compiled_provenance,
        "validator_sha256":validator_sha256,"family_count":families.len(),"distinct_geometry_families":family_geometry.values().collect::<BTreeSet<_>>().len(),"captured_rgb_images":images.values().sum::<usize>(),
        "distinct_rgb_sha256":images.len(),"exact_rgb_duplicate_count":images.values().map(|n|n-1).sum::<usize>(),
        "duplicate_policy":"byte-identical RGB PNGs across all cases, views and times; intentional matched variants are included; no embedding-distance claim",
        "selection":"all prespecified cases and consecutive seeds; failures retained; no learned filtering; people disabled to isolate camera and appearance factors",
        "cases":cases});
    gallery(&output, &receipt)?;
    receipt["artifacts"] = serde_json::to_value(io::files(&output, &output)?)?;
    io::write(&output.join("receipt.json"), &receipt)?;
    verify(&output)?;
    ensure!(
        success,
        "qualification contains failed cases; receipt and gallery retain them"
    );
    Ok(())
}
pub fn verify(output: &Path) -> Result<()> {
    let receipt: Value = io::read(&output.join("receipt.json"))?;
    ensure!(
        receipt["schema_version"] == 1,
        "unsupported qualification receipt"
    );
    let recorded: BTreeMap<String, String> = serde_json::from_value(receipt["artifacts"].clone())?;
    let mut actual = io::files(output, output)?;
    actual.remove("receipt.json");
    ensure!(
        recorded == actual,
        "qualification artifacts changed or are incomplete"
    );
    Ok(())
}
fn validate_recipe(recipe: &Recipe) -> Result<()> {
    ensure!(
        recipe.schema_version == 1
            && (1..=256).contains(&recipe.rooms_per_case)
            && (1..=64).contains(&recipe.playback_steps)
            && (0.0..=1.0).contains(&recipe.human_density)
            && !recipe.cases.is_empty(),
        "invalid qualification recipe size/version"
    );
    ensure!(
        recipe
            .seed
            .checked_add(recipe.rooms_per_case as u64)
            .is_some(),
        "seed range overflow"
    );
    let mut names = BTreeSet::new();
    for case in &recipe.cases {
        ensure!(
            !case.name.is_empty()
                && case
                    .name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
                && names.insert(&case.name),
            "case names must be unique simple identifiers"
        );
        ensure!(
            case.image_size.iter().all(|v| (64..=4096).contains(v))
                && (1..=16).contains(&case.cameras)
                && case.camera.is_object()
                && case.appearance.is_object(),
            "invalid factor case"
        );
    }
    Ok(())
}
fn geometry_hash(manifest: &Value) -> Result<String> {
    let mut geometry = manifest.clone();
    for key in [
        "appearance",
        "cameras",
        "camera_settings",
        "camera_aspect_ratio",
        "camera_diagnostics",
        "lighting",
        "sun_elevation",
        "sun_azimuth",
        "light_kelvin",
        "target_lux",
        "daylight_lux",
    ] {
        geometry
            .as_object_mut()
            .context("manifest must be an object")?
            .remove(key);
    }
    if let Some(program) = geometry["program"].as_object_mut() {
        program.remove("materials");
        if let Some(domain) = program.get_mut("domain").and_then(Value::as_object_mut) {
            for key in ["photometry", "target_lux", "fixture_kelvin"] {
                domain.remove(key);
            }
        }
    }
    Ok(bevy_zeroverse_capture::provenance::sha(
        &serde_json::to_vec(&geometry)?,
    ))
}
fn distribution(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return json!({"count":0});
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    json!({"count":n,"min":values[0],"max":values[n-1],"mean":values.iter().sum::<f64>()/n as f64,
        "p05_p25_p50_p75_p95":([0.05,0.25,0.5,0.75,0.95].map(|q|values[((n-1) as f64*q).round() as usize]))})
}
fn gallery(output: &Path, receipt: &Value) -> Result<()> {
    let mut html=String::from("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Zeroverse camera qualification</title><style>body{font:16px system-ui;background:#101820;color:#e9f0f4;max-width:1200px;margin:auto;padding:2rem}a{color:#8adfd8}h1{font-size:2.3rem}article{margin:2rem 0;padding:1.2rem;background:#1b2934;border-radius:1rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}figure{margin:0}img{width:100%;height:240px;object-fit:contain;background:#091118}figcaption{padding:.5rem;font-size:.9rem}select{padding:.5rem;font:inherit}pre{overflow:auto;max-height:18rem;font-size:.8rem}.error{color:#ffa18f}</style><h1>Multi-view qualification</h1><p>Matched geometry, independently varied camera and appearance factors. All planned cases are retained, including failures. This bounded check measures capture correctness and sampled diversity; it does not establish photographic realism or training transfer.</p><p><a href='receipt.json'>Qualification receipt</a> · <a href='recipe.json'>Reproducible recipe</a></p><label>Annotation <select id=mode><option value=color>RGB</option><option value=co_visibility>Co-visibility</option><option value=depth>Depth</option><option value=normal>Normal</option><option value=semantic>Semantic</option><option value=position>Position</option></select></label>");
    html.push_str("<p>Co-visibility colors identify peer cameras. Each case has its own deterministic legend. Black means no shared camera or invalid background; numeric validity masks distinguish them.</p>");
    for case in receipt["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cameras = case["camera_count"].as_u64().unwrap_or(2) as usize;
        html.push_str(&format!(
            "<article><h2>{}</h2>",
            io::html(&name.replace('_', " "))
        ));
        html.push_str("<p>Co-visibility legend: ");
        for camera in 0..cameras {
            let [r, g, b] = bevy_zeroverse_capture::camera_color(camera, cameras);
            html.push_str(&format!("<span style='border-left:1rem solid rgb({r},{g},{b});padding:.4rem'>camera {camera}</span> "));
        }
        html.push_str("</p>");
        if case["success"] != true {
            html.push_str(&format!(
                "<p class=error>{}</p>",
                io::html(case["error"].as_str().unwrap_or("failed"))
            ));
        }
        if let Some(rooms) = case["rooms"].as_array() {
            for room in rooms {
                html.push_str(&format!(
                    "<h3>Scene seed {}</h3><div class=grid>",
                    room["seed"]
                ));
                for v in 0..room["views"].as_array().unwrap().len() {
                    let prefix =
                        format!("{}/{}/view_{v:02}_", name, room["folder"].as_str().unwrap());
                    let view = &room["views"][v];
                    let time = view["time_seconds"]
                        .as_f64()
                        .map(|t| format!("{t:.2} s"))
                        .unwrap_or_else(|| format!("progress {}", view["time"]));
                    let camera = view["camera_index"].as_u64().unwrap();
                    html.push_str(&format!("<figure><img loading=lazy data-prefix='{prefix}' src='{prefix}color.png' alt='Rendered scene, camera {camera}'><figcaption>Camera {camera} · {}</figcaption></figure>",io::html(&time)));
                }
                html.push_str("</div>");
            }
        }
        let metrics = json!({"overlap":case["directed_rendered_overlap"],"baseline_m":case["baseline_m"],"triangulation_degrees":case["triangulation_degrees"],"reprojection_p99_pixels":case["per_view_reprojection_p99_pixels"]});
        html.push_str(&format!(
            "<details><summary>Measured distributions</summary><pre>{}</pre></details></article>",
            io::html(&serde_json::to_string_pretty(&metrics)?)
        ));
    }
    html.push_str("<script>document.querySelector('#mode').addEventListener('change',e=>{for(const image of document.querySelectorAll('[data-prefix]'))image.src=image.dataset.prefix+e.target.value+'.png'})</script></html>");
    fs::write(output.join("index.html"), html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factor_recipe_is_explicit_and_rejects_paths_and_duplicate_cases() {
        let mut recipe = Recipe::default();
        validate_recipe(&recipe).unwrap();
        recipe.cases.push(recipe.cases[0].clone());
        assert!(validate_recipe(&recipe).is_err());
        recipe.cases.pop();
        recipe.cases[0].name = "../escape".into();
        assert!(validate_recipe(&recipe).is_err());
        assert_eq!(distribution(vec![0., 1., 0.5])["mean"], 0.5);
    }
}
