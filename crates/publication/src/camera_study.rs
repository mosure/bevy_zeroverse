//! A matched camera-count study, separate from rendered population qualification.
//! Raw pose traces stay in the local audit; this self-contained report retains
//! every requested cell, its failures, denominators and measured distributions.
mod examples;
mod figures;
pub mod model;
use crate::io;
use anyhow::{ensure, Context, Result};
use model::{measure, Group, TIMES};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader},
    path::Path,
};

pub const DIRECTORY: &str = "www/project/static/media/cameras";
const BINS: usize = 24;

pub fn distribution(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    json!({"count":values.len(),"min":at(0.),"p05":at(0.05),"median":at(0.5),"p95":at(0.95),"max":at(1.),"mean":values.iter().sum::<f64>()/values.len() as f64})
}

fn validate_group(group: &Group) -> Result<()> {
    ensure!(
        (0.0..=1.0).contains(&group.baseline) && (1..=12).contains(&group.camera_count),
        "invalid camera cell"
    );
    if group.error.is_some() {
        ensure!(
            group.cameras.is_empty(),
            "rejected cell exposes stale camera poses"
        );
        return Ok(());
    }
    ensure!(
        group.cameras.len() == group.camera_count,
        "incomplete accepted camera set"
    );
    for path in &group.cameras {
        ensure!(
            path.positions.len() == TIMES && path.forwards.len() == TIMES,
            "incomplete synchronized trajectory"
        );
        ensure!(
            path.positions
                .iter()
                .chain(&path.forwards)
                .flatten()
                .all(|p| p.is_finite())
                && path.target.iter().all(|p| p.is_finite()),
            "non-finite camera pose"
        );
        ensure!(
            path.forwards
                .iter()
                .all(|f| (model::length(*f) - 1.).abs() < 1e-5),
            "non-unit optical axis"
        );
        ensure!(
            path.length_m.is_finite()
                && path.length_m >= 0.
                && path.fov_degrees.is_finite()
                && (0.1..179.).contains(&path.fov_degrees),
            "invalid camera path or lens"
        );
    }
    let edges = group
        .reference_overlap
        .as_array()
        .context("missing placement overlap")?;
    ensure!(
        edges.len() == group.camera_count - 1,
        "missing reference-edge qualification"
    );
    Ok(())
}

/// Produce a portable report from a completed CPU audit and optional native
/// qualification. Inputs are checksum-verified before any statistics are used.
pub fn prepare(input: &Path, output: &Path, rendered: Option<&Path>) -> Result<()> {
    ensure!(
        !output.exists(),
        "camera study output exists; use a new folder"
    );
    let receipt: Value = io::read(&input.join("receipt.json"))?;
    ensure!(
        receipt["schema_version"] == 1 && receipt["kind"] == "cpu_camera_sampler_audit",
        "unsupported camera audit"
    );
    for (name, expected) in receipt["artifacts"]
        .as_object()
        .context("audit hashes missing")?
    {
        crate::config::safe_relative(Path::new(name))?;
        ensure!(
            expected.as_str() == Some(io::hash(&input.join(name))?.as_str()),
            "audit artifact changed: {name}"
        );
    }
    let recipe: Value = io::read(&input.join("recipe.json"))?;
    ensure!(recipe == receipt["recipe"], "camera recipe mismatch");
    let rooms: Vec<Value> = io::read(&input.join("rooms.json"))?;
    ensure!(
        rooms.len()
            == receipt["completed_rooms"]
                .as_u64()
                .context("missing room count")? as usize,
        "room denominator mismatch"
    );
    let by_seed: BTreeMap<_, _> = rooms
        .iter()
        .map(|room| (room["seed"].as_u64().unwrap(), room))
        .collect();
    ensure!(by_seed.len() == rooms.len(), "duplicate room family");
    let mut groups = Vec::new();
    let mut keys = BTreeSet::new();
    for line in BufReader::new(fs::File::open(input.join("groups.jsonl"))?).lines() {
        let group: Group = serde_json::from_str(&line?)?;
        validate_group(&group)?;
        ensure!(
            keys.insert((group.seed, group.baseline.to_bits(), group.camera_count)),
            "duplicate camera cell"
        );
        ensure!(
            by_seed.contains_key(&group.seed),
            "camera cell lacks its room family"
        );
        groups.push(group);
    }
    ensure!(
        groups.len() == receipt["completed_groups"].as_u64().unwrap() as usize,
        "group denominator mismatch"
    );
    let baselines = recipe["baselines"]
        .as_array()
        .context("baselines missing")?;
    let counts = recipe["camera_counts"]
        .as_array()
        .context("counts missing")?;
    ensure!(
        groups.len() == rooms.len() * baselines.len() * counts.len(),
        "missing requested camera cells"
    );
    for &seed in by_seed.keys() {
        let singletons: Vec<_> = groups
            .iter()
            .filter(|g| g.seed == seed && g.camera_count == 1 && g.error.is_none())
            .collect();
        if let Some(first) = singletons.first() {
            ensure!(
                singletons
                    .iter()
                    .all(|g| serde_json::to_value(&g.cameras).ok()
                        == serde_json::to_value(&first.cameras).ok()),
                "baseline changed a single-camera trajectory"
            );
        }
    }
    let mut cells = Vec::new();
    let mut records = Vec::new();
    let example_seed = recipe["seed"]
        .as_u64()
        .context("first audit seed missing")?;
    let mut examples = Vec::new();
    for baseline in baselines {
        let b = baseline.as_f64().context("invalid baseline")?;
        for count in counts {
            let n = count.as_u64().context("invalid count")? as usize;
            let selected: Vec<_> = groups
                .iter()
                .filter(|g| g.baseline == b && g.camera_count == n)
                .collect();
            ensure!(selected.len() == rooms.len(), "missing camera-count cell");
            let mut pools = BTreeMap::<&str, Vec<f64>>::new();
            let mut path_bins = vec![0u64; BINS * BINS];
            let mut start_bins = path_bins.clone();
            let mut target_bins = path_bins.clone();
            let (mut path_outside, mut target_outside) = (0, 0);
            let (mut similar_pose, mut exact_pose, mut similar_shape, mut pairs, mut moving_pairs) =
                (0, 0, 0, 0, 0);
            let mut accepted = 0;
            let mut wide_groups = 0;
            let mut failures = Vec::new();
            for group in &selected {
                if let Some(error) = &group.error {
                    failures.push(json!({"seed":group.seed,"error":error}));
                    records.push(json!({"seed":group.seed,"baseline":b,"camera_count":n,"accepted":false,"error":error,"policy":group.policy,"diagnostics":group.metadata}));
                    continue;
                }
                accepted += 1;
                let m = measure(&group.cameras);
                similar_pose += m.similar_pose_pairs;
                exact_pose += m.exact_pose_duplicate_pairs;
                similar_shape += m.similar_moving_shape_pairs;
                pairs += m.pair_count;
                moving_pairs += m.moving_pair_count;
                wide_groups +=
                    usize::from(m.max_start_optical_angle_degrees.is_some_and(|a| a > 90.));
                let mut push =
                    |name, values: Vec<f64>| pools.entry(name).or_default().extend(values);
                push("nearest_neighbor_m", m.start_nearest_neighbor_m.clone());
                push("pair_baseline_m", m.pair_baseline_m.clone());
                push(
                    "pair_optical_angle_degrees",
                    m.pair_optical_angle_degrees.clone(),
                );
                push("path_length_m", m.path_lengths_m.clone());
                push(
                    "vertical_fov_degrees",
                    group.cameras.iter().map(|c| c.fov_degrees).collect(),
                );
                push(
                    "initial_height_room_datum_m",
                    group.cameras.iter().map(|c| c.positions[0][1]).collect(),
                );
                push("relative_motion", m.pair_relative_motion.clone());
                for (key, value) in [
                    (
                        "group_max_start_optical_angle_degrees",
                        m.max_start_optical_angle_degrees,
                    ),
                    (
                        "group_max_optical_angle_degrees",
                        m.max_optical_angle_degrees,
                    ),
                    ("start_azimuth_span_degrees", m.start_azimuth_span_degrees),
                    ("target_diameter_m", m.target_diameter_m),
                    ("common_axis_residual_m", m.common_axis_residual_m),
                    (
                        "horizontal_spread",
                        group.geometry["min_horizontal_spread"].as_f64(),
                    ),
                ] {
                    push(key, value.into_iter().collect());
                }
                let mut proxy = Vec::new();
                for edge in group.reference_overlap.as_array().unwrap() {
                    for sample in edge["samples"].as_array().context("proxy times missing")? {
                        let a = sample["reference_to_view"]
                            .as_f64()
                            .context("missing proxy overlap")?;
                        let c = sample["view_to_reference"]
                            .as_f64()
                            .context("missing reverse proxy overlap")?;
                        ensure!(
                            (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&c),
                            "invalid proxy overlap"
                        );
                        proxy.push(a.min(c));
                    }
                }
                push("reference_proxy_overlap", proxy.clone());
                let bounds = &by_seed[&group.seed]["primary_bounds"];
                let lo = [
                    bounds[0][0].as_f64().unwrap(),
                    bounds[0][1].as_f64().unwrap(),
                ];
                let hi = [
                    bounds[1][0].as_f64().unwrap(),
                    bounds[1][1].as_f64().unwrap(),
                ];
                let bin = |p: model::Point| -> Option<usize> {
                    let x = (p[0] - lo[0]) / (hi[0] - lo[0]);
                    let z = (p[2] - lo[1]) / (hi[1] - lo[1]);
                    ((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&z)).then(|| {
                        (z * BINS as f64).floor().min((BINS - 1) as f64) as usize * BINS
                            + (x * BINS as f64).floor().min((BINS - 1) as f64) as usize
                    })
                };
                for camera in &group.cameras {
                    if let Some(i) = bin(camera.positions[0]) {
                        start_bins[i] += 1;
                    }
                    for &p in &camera.positions {
                        if let Some(i) = bin(p) {
                            path_bins[i] += 1;
                        } else {
                            path_outside += 1;
                        }
                    }
                    if let Some(i) = bin(camera.target) {
                        target_bins[i] += 1;
                    } else {
                        target_outside += 1;
                    }
                }
                let mut compact = json!(m);
                for field in [
                    "pair_baseline_m",
                    "pair_optical_angle_degrees",
                    "pair_relative_motion",
                ] {
                    let values = compact[field]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect();
                    compact[field] = distribution(values);
                }
                records.push(json!({"seed":group.seed,"baseline":b,"camera_count":n,"accepted":true,"metrics":compact,"geometry":group.geometry,"policy":group.policy,"diagnostics":group.metadata,"reference_proxy_overlap":distribution(proxy)}));
                if group.seed == example_seed {
                    examples.push(json!({"baseline":b,"camera_count":n,"cameras":group.cameras}));
                }
            }
            ensure!(
                path_outside == 0
                    && path_bins.iter().sum::<u64>() == (accepted * n * TIMES) as u64
                    && start_bins.iter().sum::<u64>() == (accepted * n) as u64,
                "camera trajectory escapes primary-room bounds"
            );
            let statistics: BTreeMap<_, _> = pools
                .into_iter()
                .map(|(name, values)| (name, distribution(values)))
                .collect();
            cells.push(json!({"baseline":b,"camera_count":n,"requested_rooms":recipe["rooms"],"accepted_rooms":accepted,"failures":failures,
                "statistics":statistics,"group_max_start_angle_above_90_count":wide_groups,"pairs":pairs,"moving_pairs":moving_pairs,"similar_pose_pairs":similar_pose,"exact_pose_duplicate_pairs":exact_pose,"similar_moving_shape_pairs":similar_shape,
                "heatmaps":{"bins":BINS,"path":path_bins,"start":start_bins,"target":target_bins,"path_outside_primary_bounds":path_outside,"target_outside_primary_bounds":target_outside}}));
        }
    }
    let native = if let Some(path) = rendered {
        crate::qualification::verify(path)?;
        let mut native: Value = io::read(&path.join("receipt.json"))?;
        ensure!(
            native["generator_identity"] == receipt["generator_identity"],
            "rendered study uses a different generator"
        );
        let native_recipe = native["recipe"].clone();
        for case in native["cases"]
            .as_array_mut()
            .context("native cases missing")?
        {
            if case["success"] != true {
                continue;
            }
            let case_name = case["name"]
                .as_str()
                .context("native case name missing")?
                .to_owned();
            // Read the actual control, without inferring a baseline from names.
            let native_case = native_recipe["cases"]
                .as_array()
                .context("native case recipe missing")?
                .iter()
                .find(|c| c["name"] == case_name)
                .context("native case recipe missing")?;
            let b = native_case["camera"]["baseline"]
                .as_f64()
                .context("native baseline missing")?;
            ensure!(
                receipt["recipe"]["baselines"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v.as_f64() == Some(b)),
                "native baseline not in CPU recipe"
            );
            let n = case["camera_count"].as_u64().unwrap() as usize;
            let mut any_peer = Vec::new();
            let mut components = Vec::new();
            let mut reference_min = Vec::new();
            for room in case["rooms"]
                .as_array_mut()
                .context("native room missing")?
            {
                let folder = room["folder"]
                    .as_str()
                    .context("native room folder missing")?;
                let capture: Value =
                    io::read(&path.join(&case_name).join(folder).join("capture.json"))?;
                ensure!(
                    capture["capabilities"]["shadows"] == true
                        && capture["capabilities"]["ssao"] == true
                        && !capture["diffuse_gi"].is_null(),
                    "rendered camera study reduced lighting quality"
                );
                room["capabilities"] = capture["capabilities"].clone();
                room["diffuse_gi"] = capture["diffuse_gi"].clone();
                let seed = room["seed"].as_u64().context("native room seed missing")?;
                let planned = groups
                    .iter()
                    .find(|g| {
                        g.seed == seed
                            && g.camera_count == n
                            && g.baseline == b
                            && g.error.is_none()
                    })
                    .context("native camera set not in matched CPU audit")?;
                for view in room["views"].as_array().unwrap() {
                    let camera = view["camera_index"].as_u64().unwrap() as usize;
                    let progress = view["trajectory_progress"]
                        .as_f64()
                        .context("missing trajectory progress")?;
                    let index = (progress * 32.).round() as usize;
                    ensure!(
                        index < TIMES && (progress - index as f64 / 32.).abs() < 1e-6,
                        "native timestep outside the synchronized audit grid"
                    );
                    let expected = &planned.cameras[camera];
                    for axis in 0..3 {
                        ensure!(
                            (view["world_from_view"][3][axis].as_f64().unwrap()
                                - expected.positions[index][axis])
                                .abs()
                                < 1e-4,
                            "native camera origin differs from CPU audit"
                        );
                        ensure!(
                            (-view["world_from_view"][2][axis].as_f64().unwrap()
                                - expected.forwards[index][axis])
                                .abs()
                                < 1e-5,
                            "native optical axis differs from CPU audit"
                        );
                    }
                    if let Some(v) = view["co_visibility"]["shared_fraction_valid"].as_f64() {
                        any_peer.push(v);
                    }
                }
                let pairs = room["camera_qualification"]["rendered_overlap"]
                    .as_array()
                    .unwrap();
                let steps: BTreeSet<_> = pairs
                    .iter()
                    .filter_map(|p| p["step_index"].as_u64())
                    .collect();
                for step in steps {
                    let mut overlap = vec![vec![0.; n]; n];
                    for p in pairs.iter().filter(|p| p["step_index"] == step) {
                        let a = p["source_camera"].as_u64().unwrap() as usize;
                        let b = p["target_camera"].as_u64().unwrap() as usize;
                        overlap[a][b] = p["shared_fraction_valid"].as_f64().unwrap_or(0.);
                    }
                    reference_min.push(
                        (1..n)
                            .map(|i| overlap[0][i].min(overlap[i][0]))
                            .fold(1., f64::min),
                    );
                    components.push(component_count(&overlap, 0.05) as f64);
                }
            }
            case["any_peer_shared_fraction_valid"] = distribution(any_peer);
            case["rendered_connected_components_at_005"] = distribution(components);
            case["minimum_reference_rendered_overlap"] = distribution(reference_min);
        }
        Some(native)
    } else {
        None
    };
    let rendered_examples = rendered
        .zip(native.as_ref())
        .map(|(root, native)| examples::collect(root, native))
        .transpose()?
        .unwrap_or(Value::Null);
    let report = json!({"schema_version":1,"kind":"matched_camera_count_study","generator_identity":receipt["generator_identity"],"audit_receipt":receipt,"recipe":recipe,
        "analysis_metric_source_sha256":bevy_zeroverse_capture::provenance::sha(include_bytes!("camera_study/model.rs")),
        "measurement_contract":{"trajectory_samples":TIMES,"placement_proxy":"13x9 rays, five synchronized times, reference edges only; not rendered pixel overlap","heatmap":"primary room bounding rectangle normalized independently per room, +Z down; equal camera/path-sample weighting","similarity":"pose similarity and relative displacement similarity are separate; no perceptual duplicate or training-utility claim","camera_counts":"geometry and people are matched; each camera count is resampled, not guaranteed to be a nested prefix","single_camera":"baseline has no pairwise effect; pair and motion-pair metrics are absent"},
        "cells":cells,"groups":records,"schematic_example":{"selection":"first prespecified audit seed, irrespective of appearance","room":by_seed.get(&example_seed),"groups":examples},"rendered_qualification":native,"rendered_examples":rendered_examples});
    fs::create_dir_all(output)?;
    io::write(&output.join("study.json"), &report)?;
    let output_receipt = json!({"schema_version":1,"kind":"matched_camera_count_study","generator_identity":report["generator_identity"],"report_sha256":io::hash(&output.join("study.json"))?});
    io::write(&output.join("receipt.json"), &output_receipt)?;
    verify(output, None)?;
    Ok(())
}

fn component_count(overlap: &[Vec<f64>], threshold: f64) -> usize {
    let mut seen = vec![false; overlap.len()];
    let mut components = 0;
    for i in 0..overlap.len() {
        if seen[i] {
            continue;
        }
        components += 1;
        let mut frontier = vec![i];
        seen[i] = true;
        while let Some(a) = frontier.pop() {
            for (b, visit) in seen.iter_mut().enumerate() {
                if !*visit && overlap[a][b].min(overlap[b][a]) >= threshold {
                    *visit = true;
                    frontier.push(b);
                }
            }
        }
    }
    components
}

pub fn verify(
    input: &Path,
    identity: Option<&bevy_zeroverse_capture::GeneratorIdentity>,
) -> Result<Value> {
    let receipt: Value = io::read(&input.join("receipt.json"))?;
    ensure!(
        receipt["schema_version"] == 1
            && receipt["kind"] == "matched_camera_count_study"
            && receipt["report_sha256"].as_str()
                == Some(io::hash(&input.join("study.json"))?.as_str()),
        "camera study receipt/hash mismatch"
    );
    let report: Value = io::read(&input.join("study.json"))?;
    ensure!(
        receipt["generator_identity"] == report["generator_identity"],
        "camera study identity mismatch"
    );
    if let Some(identity) = identity {
        ensure!(
            report["generator_identity"] == json!(identity),
            "stale camera-count study; rerun the matched audit"
        );
    }
    examples::verify(
        &report["rendered_examples"],
        &report["rendered_qualification"],
    )?;
    let cells = report["cells"].as_array().context("camera cells missing")?;
    let groups = report["groups"]
        .as_array()
        .context("camera groups missing")?;
    ensure!(
        cells.len()
            == report["recipe"]["baselines"].as_array().unwrap().len()
                * report["recipe"]["camera_counts"].as_array().unwrap().len(),
        "missing camera-count cells"
    );
    for cell in cells {
        let members: Vec<_> = groups
            .iter()
            .filter(|g| {
                g["baseline"] == cell["baseline"] && g["camera_count"] == cell["camera_count"]
            })
            .collect();
        let accepted = members.iter().filter(|g| g["accepted"] == true).count();
        ensure!(
            accepted == cell["accepted_rooms"].as_u64().unwrap() as usize,
            "camera study acceptance denominator mismatch"
        );
        ensure!(
            members.len() == report["audit_receipt"]["completed_rooms"].as_u64().unwrap() as usize,
            "camera study omits a requested family"
        );
        ensure!(
            cell["exact_pose_duplicate_pairs"] == 0,
            "duplicate camera poses in accepted set"
        );
    }
    Ok(report)
}

pub fn build(root: &Path, stage: &Path, input: Option<&Path>) -> Result<()> {
    let Some(input) = input else {
        let media = stage.join(DIRECTORY);
        if media.exists() {
            fs::remove_dir_all(media)?;
        }
        fs::write(
            stage.join("tex/generated/camera_count_evaluation.tex"),
            "% No camera-count study configured.\n",
        )?;
        return Ok(());
    };
    let report = verify(&root.join(input), Some(&crate::pipeline::identity(root)?))?;
    let dest = stage.join(DIRECTORY);
    fs::create_dir_all(&dest)?;
    // Preserve checksum-bound floating-point text byte for byte. Parsing and
    // reserializing an input report can change the last displayed decimal.
    fs::copy(root.join(input).join("study.json"), dest.join("study.json"))?;
    fs::copy(
        root.join(input).join("receipt.json"),
        dest.join("receipt.json"),
    )?;
    figures::build(&report, stage, &dest)?;
    examples::build(&report, stage, &dest)?;
    fs::write(
        stage.join("tex/generated/camera_count_evaluation.tex"),
        paper(&report),
    )?;
    Ok(())
}

pub fn abstract_summary(report: &Value) -> Result<String> {
    let counts = report["recipe"]["camera_counts"]
        .as_array()
        .context("camera counts missing")?
        .iter()
        .map(|v| v.as_u64().context("invalid camera count"))
        .collect::<Result<Vec<_>>>()?;
    let first = counts.iter().min().context("empty camera counts")?;
    let last = counts.iter().max().unwrap();
    let baselines = report["recipe"]["baselines"]
        .as_array()
        .context("baselines missing")?
        .len();
    let mut summary = format!(
        "A matched camera study covers {} rooms, {first}--{last} cameras and {baselines} baseline values",
        report["recipe"]["rooms"]
    );
    if let Some(families) = report["rendered_qualification"]["family_count"].as_u64() {
        summary.push_str(&format!(
            ", with native shared-surface measurements from {families} prespecified room families"
        ));
    }
    summary.push('.');
    Ok(summary)
}

pub fn page(report: &Value) -> Result<String> {
    let cells = report["cells"]
        .as_array()
        .context("camera study cells missing")?;
    let wide = |n| {
        cells
            .iter()
            .find(|c| c["baseline"] == 1.0 && c["camera_count"] == n)
            .context("canonical camera-count study must cover baseline 1 and counts 4/12")
    };
    let four = wide(4)?;
    let twelve = wide(12)?;
    let statistic = |cell: &Value, name: &str, precision: usize| -> String {
        cell["statistics"][name]["median"]
            .as_f64()
            .map(|v| format!("{v:.precision$}"))
            .unwrap_or_else(|| "unmeasured".into())
    };
    let native = &report["rendered_qualification"];
    let mut native_html = if let Some(cases) = native["cases"].as_array() {
        let configurations = cases.len();
        let images = &native["captured_rgb_images"];
        let pairs = cases
            .iter()
            .filter_map(|c| c["directed_rendered_overlap"]["count"].as_u64())
            .sum::<u64>();
        format!(
            r#"<h3>Rendered shared surfaces</h3><p>{} prespecified room families, {configurations} configurations, {images} camera/time views and {pairs} directed pair observations. Native shadows, material synthesis and indirect lighting were retained at 320×200. Counts 2, 4, 8 and 12 use the same rooms at both path endpoints. This is a small rendered check alongside the larger placement audit.</p><figure><a href="static/media/cameras/camera-count-rendered.svg"><img src="static/media/cameras/camera-count-rendered.svg" width="1280" height="465" loading="lazy" alt="Native rendered any-peer visibility, directed pair overlap and triangulation angle by camera count and baseline"></a><figcaption>Any-peer visibility is averaged over source views; pair overlap and angles are averaged over directed camera pairs. Co-visibility uses valid first-surface pixels, including solid annotation glass. These are different denominators and do not count unique 3D points.</figcaption></figure>"#,
            native["family_count"]
        )
    } else {
        String::new()
    };
    if let Some(rooms) = report["rendered_examples"]["rooms"].as_array() {
        native_html.push_str("<h3>See the baseline in the same room</h3><p>Each column resamples a complete four-camera group over identical geometry, people, materials and lighting. All four views are shown at t=0. Camera indices identify group members, rather than fixed physical viewpoints across baselines.</p>");
        for room in rooms {
            let seed = &room["seed"];
            native_html.push_str(&format!(r#"<figure><a href="static/media/cameras/camera-baseline-s{seed}.svg"><img src="static/media/cameras/camera-baseline-s{seed}.svg" width="1288" height="1256" loading="lazy" alt="All four RGB views of room {seed} at narrow, default and wide camera baselines"></a><figcaption>Seed {seed}, the same scene at baselines 0, 0.5 and 1. Native 320×200 images, full extent, with no exposure adjustment or appearance filtering. These families are part of the rendered camera-count study, not additional population samples.</figcaption></figure>"#));
        }
    }
    crate::media::fill(
        include_str!("../templates/camera-study.html"),
        &BTreeMap::from([
            ("ROOMS", report["recipe"]["rooms"].to_string()),
            (
                "REJECTED",
                report["audit_receipt"]["rejected_groups"].to_string(),
            ),
            (
                "REQUESTED",
                report["audit_receipt"]["requested_groups"].to_string(),
            ),
            (
                "ANGLE4",
                statistic(four, "group_max_start_optical_angle_degrees", 1),
            ),
            (
                "ANGLE12",
                statistic(twelve, "group_max_start_optical_angle_degrees", 1),
            ),
            ("NN4", statistic(four, "nearest_neighbor_m", 2)),
            ("NN12", statistic(twelve, "nearest_neighbor_m", 2)),
            ("TARGET4", statistic(four, "target_diameter_m", 2)),
            ("TARGET12", statistic(twelve, "target_diameter_m", 2)),
            (
                "EXACT",
                cells
                    .iter()
                    .filter_map(|c| c["exact_pose_duplicate_pairs"].as_u64())
                    .sum::<u64>()
                    .to_string(),
            ),
            ("NATIVE", native_html),
        ]),
    )
}

fn paper(report: &Value) -> String {
    let mut text = format!(
        r"% Generated by the Rust camera-count study.
\section{{Camera-count and baseline audit}}
The current sampler was evaluated on {} consecutive matched rooms (seeds {}--{}), 1--12 cameras and baseline values 0, 0.5 and 1 at aspect ratio {}. Geometry, people and appearance were unchanged between requests; complete camera sets were resampled, so different counts are not necessarily nested prefixes. {} of {} requested sets were rejected by the bounded sampler; all failures remain in the downloadable report. Camera-center and optical-axis measurements use 33 synchronized path samples. Reference-edge overlap uses the production 13$\times$9 first-surface proxy at five times, rather than rendered pixels.

At one camera the baseline imposes no pair constraint. The serialized controls and per-count mixed-rig/coherence diagnostics accompany every requested set. Nominal advanced controls describe proposals; admission also accounts for calibrated lens footprints and room clearance. Group covariance spread constrains overall anisotropy, not every nearest neighbor. Visible-region target variation samples reference-visible surface patches and ranks feasible candidates for count-aware nearest-peer spacing, optical-axis separation and additional primary-room proxy surface cells. The soft distance scale is room diagonal divided by the square root of camera count. Temporal paths can remain related deformations of an existing route; varied gaze is not independent route topology. There is no explicit 90-degree optical-angle cap.

\begin{{table}}[htbp]\centering\small
\begin{{tabular}}{{rrrrrr}}\toprule
$b$ & Cameras & Accepted & NN (m) & Max angle ($^\circ$) & Targets (m) \\
\midrule
",
        report["recipe"]["rooms"],
        report["recipe"]["seed"],
        report["recipe"]["seed"].as_u64().unwrap() + report["recipe"]["rooms"].as_u64().unwrap()
            - 1,
        format_args!("{:.2}", report["recipe"]["aspect"].as_f64().unwrap()),
        report["audit_receipt"]["rejected_groups"],
        report["audit_receipt"]["requested_groups"]
    );
    for cell in report["cells"].as_array().unwrap() {
        let n = cell["camera_count"].as_u64().unwrap();
        if ![2, 4, 8, 12].contains(&n) {
            continue;
        }
        let stats = &cell["statistics"];
        let scalar = |key: &str, precision: usize| {
            stats[key]["median"]
                .as_f64()
                .map(|v| format!("{v:.precision$}"))
                .unwrap_or_else(|| "--".into())
        };
        text.push_str(&format!(
            "{} & {} & {}/{} & {} & {} & {} \\\\\n",
            cell["baseline"],
            n,
            cell["accepted_rooms"],
            cell["requested_rooms"],
            scalar("nearest_neighbor_m", 2),
            scalar("group_max_start_optical_angle_degrees", 1),
            scalar("target_diameter_m", 2)
        ));
    }
    text.push_str(r"\bottomrule\end{tabular}
\caption{Current matched placement audit. NN is the pooled camera-level initial nearest-neighbor median; angle and target diameter are room-level medians. Failed sets have no placement measurements. These denominators differ from rendered pixel overlap.}\label{tab:camera-counts}
\end{table}

Similar absolute trajectories have RMS center separation below 0.15 m and RMS optical-axis difference below $5^\circ$. Shape similarity instead removes each start position and normalizes the RMS displacement residual by the larger RMS travel, with a diagnostic threshold of 0.25; static pairs are excluded. Close stereo views can intentionally satisfy the pose threshold. A common-axis least-squares residual measures convergence of initial optical-axis lines; parallel or ill-conditioned lines have no reported focus. Neither statistic proves perceptual repetition or reconstruction utility.

\begin{figure}[p]\centering\includegraphics[width=\linewidth]{generated/camera_count_statistics.png}
\caption{Camera-count distributions at three baseline settings. Room-level viewing-angle/target statistics and camera-level spacing use their own denominators. Repeated path shapes are measured separately from repeated absolute poses.}\label{fig:camera-count-statistics}\end{figure}
\begin{figure}[p]\centering\includegraphics[width=\linewidth]{generated/camera_count_heatmap.png}
\caption{Baseline 1 camera-path heatmaps for 1--12 views: 33 synchronized samples per camera, primary-room bounds normalized per room and +Z down. A common probability color scale is used. These show positions, not reconstructed-surface coverage.}\label{fig:camera-count-heatmap}\end{figure}
\begin{figure}[p]\centering\includegraphics[width=\linewidth]{generated/camera_count_schematics.png}
\caption{Prespecified first room, fixed geometry and people, default/wide baselines and increasing camera counts. Full paths, initial optical directions and look targets expose the shared-content proposal. Rejected requests are retained.}\label{fig:camera-count-schematics}\end{figure}

\clearpage

");
    if let Some(cases) = report["rendered_qualification"]["cases"].as_array() {
        let native = &report["rendered_qualification"];
        text.push_str(&format!(r"\subsection{{Native rendered overlap across camera counts}}
The rendered check uses {} prespecified matched room families, {} configurations and {} camera/time views: counts 2, 4, 8 and 12 at all three baselines and both trajectory endpoints, with 320$\times$200 images, static human density 0.25 and native shadows, material synthesis and diffuse GI retained. The executable and every channel are hash-bound. Rendered centers and optical axes match the CPU audit within $10^{{-4}}$ m and $10^{{-5}}$ per direction component. Table~\ref{{tab:camera-count-rendered}} separates mean directed pair overlap from mean source-view visibility in any peer, and reports the smallest reference-edge overlap over the sampled rooms/times. These do not measure unique scene-surface coverage.

\begin{{table}}[htbp]\centering\small
\begin{{tabular}}{{rrrrrr}}\toprule
$b$ & Cameras & Pair (\%) & Any peer (\%) & Min ref. (\%) & Components \\
\midrule
",native["family_count"],cases.len(),native["captured_rgb_images"]));
        for case in cases {
            let control = native["recipe"]["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == case["name"])
                .unwrap();
            text.push_str(&format!(
                "{} & {} & {:.1} & {:.1} & {:.1} & {} \\\\\n",
                control["camera"]["baseline"],
                case["camera_count"],
                case["directed_rendered_overlap"]["mean"].as_f64().unwrap() * 100.,
                case["any_peer_shared_fraction_valid"]["mean"]
                    .as_f64()
                    .unwrap()
                    * 100.,
                case["minimum_reference_rendered_overlap"]["min"]
                    .as_f64()
                    .unwrap()
                    * 100.,
                case["rendered_connected_components_at_005"]["max"]
                    .as_f64()
                    .unwrap() as usize
            ));
        }
        text.push_str(r"\bottomrule\end{tabular}
\caption{Native first-surface co-visibility; glass follows the solid annotation convention. Components is the largest number of connected components when an undirected edge requires at least 5\% shared pixels in both directions. Room families are the independent units; camera pairs and endpoints are correlated.}\label{tab:camera-count-rendered}\end{table}
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{generated/camera_count_rendered.png}
\caption{Measured native visibility and triangulation at increasing camera counts. Any-peer visibility is averaged over source views, other means over directed pairs. More peer views increase any-peer visibility without proving broader unique surface coverage.}\label{fig:camera-count-rendered}\end{figure}

");
    }
    if let Some(rooms) = report["rendered_examples"]["rooms"].as_array() {
        text.push_str("\\subsection{Visual effect of camera baseline}\nEach column resamples a complete four-camera group over identical geometry, people, materials and lighting. Camera indices identify group members, rather than fixed physical viewpoints across columns. These are the first two prespecified native families, with every view retained at t=0 and no per-image exposure correction. They contribute no extra rooms to the population denominator.\n\n");
        for room in rooms {
            let seed = &room["seed"];
            text.push_str(&format!("\\begin{{figure}}[p]\\centering\\includegraphics[width=\\linewidth,height=.82\\textheight,keepaspectratio]{{generated/camera_count_baseline_s{seed}.png}}\n\\caption{{Fixed room {seed}: narrow, default and wide baseline settings (0, 0.5 and 1) from left to right. Rows retain all four native RGB views at 320$\\times$200. Perspective and shared content change with the camera policy; scene content is invariant.}}\\label{{fig:camera-baseline-{seed}}}\\end{{figure}}\n\n"));
        }
    }
    text.push_str(
        r"\clearpage
",
    );
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_and_rejected_sets_cannot_become_accepted_pose_measurements() {
        let mut group = Group {
            seed: 0,
            baseline: 1.,
            camera_count: 2,
            error: Some("infeasible".into()),
            cameras: vec![],
            policy: Value::Null,
            geometry: Value::Null,
            reference_overlap: Value::Null,
            metadata: Default::default(),
        };
        validate_group(&group).unwrap();
        group.error = None;
        assert!(validate_group(&group).is_err());
        assert_eq!(distribution(vec![]), Value::Null);
    }
    #[test]
    fn rendered_connectivity_requires_both_directions_but_not_every_pair() {
        let mut overlap = vec![vec![0.; 3]; 3];
        overlap[0][1] = 0.2;
        overlap[1][0] = 0.01;
        assert_eq!(component_count(&overlap, 0.05), 3);
        overlap[1][0] = 0.1;
        overlap[1][2] = 0.3;
        overlap[2][1] = 0.3;
        assert_eq!(component_count(&overlap, 0.05), 1);
        assert_eq!(component_count(&[], 0.05), 0);
    }
}
