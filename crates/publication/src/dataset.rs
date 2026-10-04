use crate::{
    config::Protocol,
    io,
    visibility::{self, Stats},
};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::{
    GeneratorIdentity, GENERATOR_VERSION, INDOOR_METRICS_SCHEMA_VERSION, PUBLICATION_MODES,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct View {
    pub camera_index: usize,
    pub step_index: usize,
    pub time: f32,
    pub pose_max_absolute_error: f32,
    pub world_from_view: [[f32; 4]; 4],
    pub fov_y: f32,
    pub fx_pixels: f32,
    pub fy_pixels: f32,
    pub near: f32,
    pub far: f32,
    pub semantic_colors: usize,
    pub semantic_pixel_counts: BTreeMap<String, u64>,
    pub annotation_alignment: Value,
    pub co_visibility: Stats,
    #[serde(flatten)]
    pub other: BTreeMap<String, Value>,
}

// Decode the portions of the architectural program used in figures before
// accessing generic JSON. Missing fields and malformed indices are errors,
// rather than panics or silently incorrect geometry diagrams.
#[derive(Deserialize)]
struct Architecture {
    seed: u64,
    room_size: [f64; 3],
    envelope: Envelope,
    partitions: Option<Vec<Partition>>,
}
#[derive(Deserialize)]
struct Envelope {
    footprint: Vec<[f64; 2]>,
    ceiling_drop: [f64; 2],
    floor_patches: Vec<Patch>,
    pillars: Vec<Pillar>,
    walls: Vec<Wall>,
    mezzanine: Option<Mezzanine>,
}
#[derive(Deserialize)]
struct Patch {
    min: [f64; 2],
    max: [f64; 2],
    height: f64,
}
#[derive(Deserialize)]
struct Pillar {
    center: [f64; 2],
    radius: f64,
}
#[derive(Deserialize)]
struct Wall {
    edge: usize,
    facade: Option<Facade>,
}
#[derive(Deserialize)]
struct Facade {
    openings: Vec<Opening>,
}
#[derive(Deserialize)]
struct Opening {
    min: [f64; 2],
    max: [f64; 2],
}
#[derive(Deserialize)]
struct Mezzanine {
    deck: Patch,
    stair_min: [f64; 2],
    stair_max: [f64; 2],
    stair_axis: usize,
    steps: u64,
    thickness: f64,
}
#[derive(Deserialize)]
struct Partition {
    axis: usize,
    coordinate: f64,
    start: f64,
    end: f64,
    door_center: f64,
    door_width: f64,
    arch_rise: f64,
}
#[derive(Deserialize)]
struct Alignment {
    checked_pixels: u64,
    depth_position_p99_metres: f64,
    reprojection_p99_pixels: f64,
    normal_length_max_error: f64,
    position_quantization_budget_p99_ratio: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Capture {
    pub run_id: String,
    pub seed: u64,
    pub image_size: [u32; 2],
    pub elapsed_seconds: f64,
    pub annotation_precision: String,
    pub capabilities: Value,
    pub co_visibility_metadata: Value,
    pub views: Vec<View>,
    #[serde(flatten)]
    pub other: BTreeMap<String, Value>,
}
pub struct Scene {
    pub folder: PathBuf,
    pub capture: Capture,
    pub manifest: Value,
    pub architecture: Value,
}
pub struct Dataset {
    pub identity: GeneratorIdentity,
    pub metrics: Value,
    pub selection: Value,
    pub rows: Vec<Value>,
    pub scenes: Vec<Scene>,
    pub input_sha256: BTreeMap<String, String>,
    pub summary: Value,
}

pub fn features(row: &Value) -> BTreeMap<String, bool> {
    let e = &row["envelope"];
    let points = e["footprint"].as_array().expect("validated footprint");
    let edges: Vec<[f64; 2]> = (0..points.len())
        .map(|i| {
            let (a, b) = (&points[i], &points[(i + 1) % points.len()]);
            [number(&b[0]) - number(&a[0]), number(&b[1]) - number(&a[1])]
        })
        .collect();
    BTreeMap::from([
        (
            "Oblique walls".into(),
            edges.iter().any(|e| e[0].abs().min(e[1].abs()) > 0.001),
        ),
        ("Chamfered corners".into(), matches!(points.len(), 6 | 10)),
        (
            "Exterior cut-in".into(),
            (0..edges.len()).any(|i| {
                let (a, b) = (edges[i], edges[(i + 1) % edges.len()]);
                a[0] * b[1] - a[1] * b[0] < -0.0001
            }),
        ),
        (
            "Sloping ceiling".into(),
            number(&e["ceiling_drop"][0]).hypot(number(&e["ceiling_drop"][1])) > 0.001,
        ),
        (
            "Raised floor".into(),
            e["floor_patches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| number(&p["height"]) > 0.001),
        ),
        (
            "Sunken floor".into(),
            e["floor_patches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| number(&p["height"]) < -0.001),
        ),
        (
            "Interior pillars".into(),
            !e["pillars"].as_array().unwrap().is_empty(),
        ),
        (
            "Arched portals".into(),
            row["partitions"]
                .as_array()
                .is_some_and(|p| p.iter().any(|p| number(&p["arch_rise"]) > 0.001)),
        ),
        ("Mezzanine".into(), !e["mezzanine"].is_null()),
    ])
}
pub fn number(v: &Value) -> f64 {
    v.as_f64().expect("validated numeric field")
}
pub fn area(points: &Value) -> f64 {
    let points = points.as_array().unwrap();
    (0..points.len())
        .map(|i| {
            let (a, b) = (&points[i], &points[(i + 1) % points.len()]);
            number(&a[0]) * number(&b[1]) - number(&b[0]) * number(&a[1])
        })
        .sum::<f64>()
        .abs()
        / 2.0
}
pub fn same_f32(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) if a.is_u64() && b.is_u64() => a == b,
        (Value::Number(a), Value::Number(b)) if a.is_i64() && b.is_i64() => a == b,
        (Value::Number(a), Value::Number(b)) => {
            (a.as_f64().unwrap() as f32).to_bits() == (b.as_f64().unwrap() as f32).to_bits()
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_f32(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, a)| b.get(k).is_some_and(|b| same_f32(a, b)))
        }
        _ => a == b,
    }
}
fn finite_json(value: &Value) -> Result<()> {
    match value {
        Value::Number(n) => ensure!(n.as_f64().is_some_and(f64::is_finite), "nonfinite metric"),
        Value::Array(a) => {
            for v in a {
                finite_json(v)?;
            }
        }
        Value::Object(o) => {
            for v in o.values() {
                finite_json(v)?;
            }
        }
        _ => (),
    }
    Ok(())
}
fn shape(value: &Value, lengths: &[usize]) -> bool {
    if lengths.is_empty() {
        return value.as_f64().is_some_and(f64::is_finite);
    }
    value
        .as_array()
        .is_some_and(|v| v.len() == lengths[0] && v.iter().all(|v| shape(v, &lengths[1..])))
}
fn validate_row(row: &Value) -> Result<()> {
    let decoded: Architecture =
        serde_json::from_value(row.clone()).context("architectural figure contract")?;
    ensure!(
        decoded.seed == row["seed"].as_u64().unwrap() && decoded.room_size.iter().all(|v| *v > 0.0),
        "invalid architecture size"
    );
    let envelope = &decoded.envelope;
    ensure!(
        envelope.footprint.len() >= 3 && envelope.ceiling_drop.iter().all(|v| v.is_finite()),
        "invalid envelope"
    );
    let patch_valid =
        |p: &Patch| p.min.iter().zip(p.max).all(|(a, b)| *a < b) && p.height.is_finite();
    ensure!(
        envelope.floor_patches.iter().all(patch_valid)
            && envelope
                .pillars
                .iter()
                .all(|p| p.center.iter().all(|v| v.is_finite()) && p.radius > 0.0),
        "invalid floor/pillar geometry"
    );
    for wall in &envelope.walls {
        ensure!(
            wall.edge < envelope.footprint.len(),
            "invalid envelope edge index"
        );
        if let Some(f) = &wall.facade {
            ensure!(
                f.openings
                    .iter()
                    .all(|p| p.min.iter().zip(p.max).all(|(a, b)| *a < b)),
                "invalid aperture"
            );
        }
    }
    if let Some(m) = &envelope.mezzanine {
        ensure!(
            patch_valid(&m.deck)
                && m.stair_axis < 2
                && m.steps > 0
                && m.steps < 128
                && m.thickness > 0.0
                && m.stair_min.iter().zip(m.stair_max).all(|(a, b)| *a < b),
            "invalid mezzanine"
        );
    }
    for p in decoded.partitions.iter().flatten() {
        ensure!(
            p.axis < 2
                && p.start < p.end
                && p.door_width > 0.0
                && [p.coordinate, p.door_center, p.arch_rise]
                    .iter()
                    .all(|v| v.is_finite()),
            "invalid partition"
        );
    }
    let e = &row["envelope"];
    ensure!(
        row["seed"].is_u64() && shape(&row["room_size"], &[3]),
        "invalid architecture identity/size"
    );
    let points = e["footprint"].as_array().context("missing footprint")?;
    ensure!(
        points.len() >= 3
            && points.iter().all(|p| shape(p, &[2]))
            && shape(&e["ceiling_drop"], &[2]),
        "invalid envelope coordinates"
    );
    ensure!(
        e["floor_patches"].is_array() && e["pillars"].is_array() && e["walls"].is_array(),
        "missing envelope programs"
    );
    for patch in e["floor_patches"].as_array().unwrap() {
        ensure!(
            shape(&patch["min"], &[2]) && shape(&patch["max"], &[2]) && patch["height"].is_number(),
            "invalid floor patch"
        );
    }
    for p in e["pillars"].as_array().unwrap() {
        ensure!(
            shape(&p["center"], &[2]) && p["radius"].is_number(),
            "invalid pillar"
        );
    }
    finite_json(row)?;
    Ok(())
}

pub fn distribution(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let percentile = |p: f64| values[((n - 1) as f64 * p).round_ties_even() as usize];
    json!({"count":n,"min":values[0],"max":values[n-1],"mean":values.iter().sum::<f64>()/n as f64,"p05_p50_p95":[percentile(0.05),percentile(0.5),percentile(0.95)]})
}

impl Dataset {
    pub fn load(root: &Path, identity: &GeneratorIdentity, protocol: &Protocol) -> Result<Self> {
        let actual: GeneratorIdentity = io::read(&root.join("generator_identity.json"))?;
        ensure!(
            actual == *identity,
            "captures are stale: refresh with the latest compiled generator"
        );
        let metrics: Value = io::read(&root.join("metrics.json"))?;
        let selection: Value = io::read(&root.join("render_selection.json"))?;
        let completion: Value = io::read(&root.join("run_complete.json"))?;
        let audit: Value = io::read(&root.join("distribution.json"))?;
        ensure!(
            metrics["schema_version"] == INDOOR_METRICS_SCHEMA_VERSION
                && metrics["generator_version"] == GENERATOR_VERSION
                && metrics["scenes"] == protocol.audit_rooms
                && metrics["image_size"] == json!([protocol.width, protocol.height]),
            "incompatible population schema/protocol"
        );
        ensure!(
            same_f32(&metrics["density"], &json!(protocol.density))
                && same_f32(&metrics["human_density"], &json!(protocol.human_density)),
            "population density mismatch"
        );
        ensure!(
            audit["invalid_seeds"]
                .as_array()
                .context("missing layout gates")?
                .is_empty()
                && audit["first_seed"] == protocol.seed
                && audit["seeds"] == protocol.audit_rooms
                && audit["cameras_per_scene"] == protocol.cameras,
            "invalid architecture audit"
        );
        let run_id = selection["run_id"].as_str().context("missing run ID")?;
        ensure!(
            !run_id.is_empty()
                && completion["run_id"] == run_id
                && completion["identity"] == json!(identity)
                && selection["identity"] == json!(identity),
            "incomplete or mixed capture run"
        );
        let seeds: Vec<u64> = (0..protocol.rendered_rooms)
            .map(|i| protocol.seed + i as u64)
            .collect();
        ensure!(
            selection["selected_seeds"] == json!(seeds)
                && completion["selected_seeds"] == json!(seeds)
                && completion["captured_scenes"] == protocol.rendered_rooms,
            "capture selection/completion mismatch"
        );
        ensure!(
            selection["co_visibility"] == true
                && selection["quality"] == "Auto"
                && selection["playback_steps"] == protocol.playback_steps
                && selection["gi_rays"] == protocol.gi_rays
                && selection["diffuse_gi_enabled"] == true,
            "capture settings differ from publication protocol"
        );
        let rows: Vec<Value> = fs::read_to_string(root.join("architecture.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            rows.len() == protocol.audit_rooms,
            "architecture denominator mismatch"
        );
        let mut by_seed = BTreeMap::new();
        for row in &rows {
            validate_row(row)?;
            let seed = row["seed"].as_u64().unwrap();
            ensure!(
                by_seed.insert(seed, row).is_none(),
                "duplicate architecture seed"
            );
        }
        ensure!(
            by_seed
                .keys()
                .copied()
                .eq((0..protocol.audit_rooms).map(|i| protocol.seed + i as u64)),
            "nonconsecutive audit population"
        );
        finite_json(&metrics)?;
        validate_population(&metrics, protocol)?;
        let mut scenes = Vec::new();
        let mut hashes = BTreeMap::new();
        for name in [
            "generator_identity.json",
            "metrics.json",
            "render_selection.json",
            "run_complete.json",
            "distribution.json",
            "architecture.jsonl",
        ] {
            hashes.insert(name.into(), io::hash(&root.join(name))?);
        }
        for seed in seeds {
            let folder = root.join(format!("seed_{seed:06}"));
            let capture: Capture = io::read(&folder.join("capture.json"))?;
            let manifest: Value = io::read(&folder.join("manifest.json"))?;
            let row = by_seed[&seed];
            ensure!(
                capture.seed == seed
                    && manifest["seed"] == seed
                    && capture.run_id == run_id
                    && manifest["generator_version"] == GENERATOR_VERSION
                    && capture.image_size == [protocol.width, protocol.height],
                "mixed room capture/manifest"
            );
            ensure!(
                manifest["world_yaw"].as_f64() == Some(0.0)
                    && same_f32(&manifest["envelope"], &row["envelope"])
                    && same_f32(&manifest["room_size"], &row["room_size"]),
                "capture/audit geometry mismatch"
            );
            ensure!(
                capture.capabilities["shadows"] == true
                    && capture.capabilities["ssao"] == true
                    && capture.capabilities["quality"] == "Auto"
                    && capture.annotation_precision == "float32_geometry",
                "capture capability/precision mismatch"
            );
            ensure!(
                capture.views.len() == protocol.cameras * protocol.playback_steps
                    && manifest["cameras"]
                        .as_array()
                        .context("missing camera paths")?
                        .len()
                        == protocol.cameras,
                "capture view denominator mismatch"
            );
            visibility::validate_metadata(&capture.co_visibility_metadata, protocol.cameras)?;
            let mut identities = BTreeSet::new();
            for (index, view) in capture.views.iter().enumerate() {
                ensure!(
                    view.camera_index < protocol.cameras
                        && view.step_index < protocol.playback_steps
                        && identities.insert((view.step_index, view.camera_index)),
                    "duplicate/out-of-range camera/time"
                );
                let time = view.step_index as f32 / (protocol.playback_steps - 1) as f32;
                ensure!(
                    view.time == time
                        && view.near > 0.0
                        && view.far > view.near
                        && view.fx_pixels > 0.0
                        && view.fy_pixels > 0.0
                        && view.fov_y > 0.0
                        && view.fov_y < std::f32::consts::PI,
                    "invalid camera intrinsics/time"
                );
                finite_json(&json!(view))?;
                ensure!(
                    (0.0..=0.0001).contains(&view.pose_max_absolute_error),
                    "rendered camera differs from its planned pose"
                );
                let focal = protocol.height as f64 / (2.0 * (view.fov_y as f64 / 2.0).tan());
                ensure!(
                    (view.fx_pixels as f64 - focal).abs() < focal * 0.0001
                        && (view.fy_pixels as f64 - focal).abs() < focal * 0.0001,
                    "intrinsics/FOV mismatch"
                );
                let matrix = &view.world_from_view;
                ensure!(
                    (matrix[3][3] - 1.0).abs() < 1e-5 && (0..3).all(|i| matrix[i][3].abs() < 1e-5),
                    "non-affine camera transform"
                );
                for i in 0..3 {
                    for j in 0..3 {
                        let dot = (0..3).map(|k| matrix[i][k] * matrix[j][k]).sum::<f32>();
                        ensure!(
                            (dot - if i == j { 1.0 } else { 0.0 }).abs() < 0.0001,
                            "camera rotation is not orthonormal"
                        );
                    }
                }
                let align: Alignment = serde_json::from_value(view.annotation_alignment.clone())
                    .context("missing/invalid annotation alignment")?;
                ensure!(
                    align.checked_pixels > 0
                        && (0.0..=0.001).contains(&align.depth_position_p99_metres)
                        && (0.0..=0.05).contains(&align.reprojection_p99_pixels)
                        && (0.0..=0.0001).contains(&align.normal_length_max_error)
                        && (0.0..=1.0).contains(&align.position_quantization_budget_p99_ratio),
                    "annotation alignment gate failed"
                );
                let mut preview = None;
                for mode in PUBLICATION_MODES {
                    let file = folder.join(format!("view_{index:02}_{mode}.png"));
                    let image = image::open(&file)
                        .with_context(|| file.display().to_string())?
                        .into_rgb8();
                    ensure!(
                        image.dimensions() == (protocol.width, protocol.height),
                        "annotation image size mismatch"
                    );
                    if mode == "co_visibility" {
                        preview = Some(image);
                    }
                }
                visibility::decode(
                    &fs::read(folder.join(format!("view_{index:02}_co_visibility_mask.png")))?,
                    &fs::read(folder.join(format!("view_{index:02}_co_visibility_valid.png")))?,
                    preview.as_ref(),
                    capture.image_size,
                    view.camera_index,
                    protocol.cameras,
                    &view.co_visibility,
                )?;
            }
            hashes.extend(io::files(root, &folder)?);
            scenes.push(Scene {
                folder,
                capture,
                manifest,
                architecture: row.clone(),
            });
        }
        let summary = summarize(&rows, &scenes, &metrics)?;
        Ok(Self {
            identity: actual,
            metrics,
            selection,
            rows,
            scenes,
            input_sha256: hashes,
            summary,
        })
    }

    /// Recover measured programs/reports from an already verified publication.
    /// This does not invent captures or reconstruct raw training annotations.
    pub fn published(
        root: &Path,
        identity: GeneratorIdentity,
        capture_sources: BTreeMap<String, String>,
    ) -> Result<Self> {
        use std::io::Read;
        let media = root.join("www/project/static/media/architecture");
        let gallery: Value = io::read(&media.join("gallery.json"))?;
        let metrics: Value = io::read(&media.join("population.json"))?;
        let mut archive =
            zip::ZipArchive::new(fs::File::open(media.join("calibration-and-programs.zip"))?)?;
        let mut programs = String::new();
        archive
            .by_name("architecture.jsonl")?
            .read_to_string(&mut programs)?;
        let rows: Vec<Value> = programs
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        let mut scenes = Vec::new();
        for scene in gallery["scenes"].as_array().context("missing scenes")? {
            let metadata: Value =
                io::read(&media.join(scene["metadata"].as_str().context("missing metadata")?))?;
            scenes.push(Scene {
                folder: PathBuf::new(),
                capture: serde_json::from_value(metadata["capture"].clone())?,
                manifest: metadata["manifest"].clone(),
                architecture: metadata["architecture"].clone(),
            });
        }
        let summary = summarize(&rows, &scenes, &metrics)?;
        Ok(Self {
            identity,
            metrics,
            selection: json!({"capture_engine":gallery["capture_engine"],"run_id":scenes[0].capture.run_id}),
            rows,
            scenes,
            input_sha256: capture_sources,
            summary,
        })
    }
}

fn validate_population(metrics: &Value, protocol: &Protocol) -> Result<()> {
    fn total<'a>(mut values: impl Iterator<Item = &'a Value>) -> Result<u64> {
        values.try_fold(0u64, |sum, value| {
            sum.checked_add(value.as_u64().context("histogram count must be uint64")?)
                .context("histogram count overflow")
        })
    }
    for (key, value) in metrics["numeric"]
        .as_object()
        .context("missing histograms")?
    {
        let count = value["count"]
            .as_u64()
            .context("missing histogram denominator")?;
        let bins = value["bin_counts"].as_array().context("missing bins")?;
        let edges = value["bin_edges"].as_array().context("missing bin edges")?;
        ensure!(
            edges.len() == bins.len() + 1
                && edges.iter().all(|e| e.as_f64().is_some_and(f64::is_finite))
                && edges.windows(2).all(|e| number(&e[0]) < number(&e[1]))
                && total(bins.iter())? == count,
            "histogram denominator/edges mismatch: {key}"
        );
    }
    for value in metrics["object_counts_per_scene"]
        .as_object()
        .context("missing instance distributions")?
        .values()
    {
        ensure!(
            total(
                value
                    .as_object()
                    .context("invalid count histogram")?
                    .values()
            )? == protocol.audit_rooms as u64,
            "instance histogram omits rooms"
        );
    }
    let grid = metrics["heatmap_grid_size"]
        .as_u64()
        .context("missing heatmap grid size")? as usize;
    ensure!((1..=256).contains(&grid), "invalid heatmap grid size");
    for value in metrics["placement_heatmaps"]
        .as_object()
        .context("missing heatmaps")?
        .values()
    {
        ensure!(
            value
                .as_array()
                .is_some_and(|v| v.len() == grid * grid && v.iter().all(Value::is_u64)),
            "invalid heatmap"
        );
    }
    Ok(())
}
fn summarize(rows: &[Value], scenes: &[Scene], metrics: &Value) -> Result<Value> {
    let mut counts = BTreeMap::<String, u64>::new();
    let mut signatures = BTreeSet::new();
    for row in rows {
        for (key, enabled) in features(row) {
            *counts.entry(key).or_default() += u64::from(enabled);
        }
        let e = &row["envelope"];
        let mut program = json!({"footprint":e["footprint"],"ceiling_drop":e["ceiling_drop"],"floor_patches":e["floor_patches"],"pillars":e["pillars"],"mezzanine":e["mezzanine"]});
        fn quantize(value: &mut Value) {
            match value {
                Value::Number(n) if n.is_f64() => {
                    *value = json!((n.as_f64().unwrap() * 100.0).round_ties_even() / 100.0)
                }
                Value::Array(a) => {
                    for v in a {
                        quantize(v)
                    }
                }
                Value::Object(o) => {
                    for v in o.values_mut() {
                        quantize(v)
                    }
                }
                _ => (),
            }
        }
        quantize(&mut program);
        signatures.insert(serde_json::to_string(&program)?);
    }
    let mut main = Vec::new();
    let mut primary = Vec::new();
    let mut invisible_main = Vec::new();
    let mut invisible_primary = Vec::new();
    for scene in scenes {
        let people = scene.manifest["humans"]
            .as_array()
            .context("missing human manifest")?;
        let people: Vec<_> = people.iter().filter(|p| p["neighbor"] == false).collect();
        let zones = scene.manifest["program"]["zones"].as_array();
        let largest = zones.and_then(|z| {
            z.iter()
                .enumerate()
                .max_by(|(ai, a), (bi, b)| {
                    let area = |v: &Value| {
                        (number(&v["max"][0]) - number(&v["min"][0]))
                            * (number(&v["max"][1]) - number(&v["min"][1]))
                    };
                    area(a).total_cmp(&area(b)).then(ai.cmp(bi))
                })
                .map(|(_, z)| z)
        });
        let has_primary = people.iter().any(|p| {
            largest.is_none_or(|z| {
                [0usize, 1].iter().all(|&i| {
                    let q = number(&p["position"][i * 2]);
                    number(&z["min"][i]) <= q && q <= number(&z["max"][i])
                })
            })
        });
        let visible = scene
            .capture
            .views
            .iter()
            .any(|v| v.semantic_pixel_counts.get("person").copied().unwrap_or(0) > 0);
        if !people.is_empty() {
            main.push(scene.capture.seed);
            if !visible {
                invisible_main.push(scene.capture.seed);
            }
        }
        if has_primary {
            primary.push(scene.capture.seed);
            if !visible {
                invisible_primary.push(scene.capture.seed);
            }
        }
    }
    let views: Vec<_> = scenes.iter().flat_map(|s| s.capture.views.iter()).collect();
    let annotation: BTreeMap<_, _> = [
        "depth_position_p99_metres",
        "reprojection_p99_pixels",
        "normal_length_max_error",
        "position_quantization_budget_p99_ratio",
    ]
    .into_iter()
    .map(|key| {
        (
            key,
            distribution(
                views
                    .iter()
                    .map(|v| number(&v.annotation_alignment[key]))
                    .collect(),
            ),
        )
    })
    .collect();
    let valid: u64 = views.iter().map(|v| v.co_visibility.valid_pixels).sum();
    let shared: u64 = views.iter().map(|v| v.co_visibility.shared_pixels).sum();
    let count = scenes[0].capture.co_visibility_metadata["camera_count"]
        .as_u64()
        .unwrap() as usize;
    let cardinality: Vec<u64> = (0..count)
        .map(|i| {
            views
                .iter()
                .map(|v| v.co_visibility.cardinality_pixels[i])
                .sum()
        })
        .collect();
    Ok(
        json!({"generator_version":GENERATOR_VERSION,"audit_scenes":rows.len(),"feature_room_counts":counts,
            "quantized_structural_signatures":signatures.len(),"quantization_m":0.01,
            "annotation_metrics":annotation,"capture_wall_seconds":distribution(scenes.iter().map(|s|s.capture.elapsed_seconds).collect()),
            "minimum_semantic_classes":views.iter().map(|v|v.semantic_colors).min(),
            "semantic_coverage":{"rooms_with_main_envelope_humans":main.len(),"rooms_with_camera_primary_zone_humans":primary.len(),"main_envelope_human_rooms_without_person_pixels":invisible_main,"camera_primary_zone_human_rooms_without_person_pixels":invisible_primary,"views_with_fewer_than_four_semantic_classes":views.iter().filter(|v|v.semantic_colors<4).count()},
            "co_visibility":{"views":views.len(),"valid_pixels":valid,"shared_pixels":shared,"cardinality_pixels":cardinality,"shared_fraction_valid":shared as f64/valid.max(1) as f64},
            "architecture_numeric":metrics["numeric"].as_object().unwrap().iter().filter(|(k,_)|k.starts_with("envelope_")||k.starts_with("mezzanine_")||k.starts_with("exterior_")).collect::<BTreeMap<_,_>>()
        }),
    )
}
