//! Matched, complete capture comparisons. Historical identities are retained;
//! no room is selected or discarded by its appearance or embedding score.
mod embeddings;
mod page;

use crate::{
    config::Protocol,
    dataset::{self, Dataset},
    graphics, io,
};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::{provenance::sha, PUBLICATION_MODES};
use image::imageops::FilterType;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::BufRead,
    path::Path,
};

/// Studies are experiments, never release-page inputs or deployable outputs.
/// Resolve existing ancestors to enforce both the ignored-cache boundary and
/// the reserved publication-cache boundary through directory symlinks.
pub(crate) fn local_destination(root: &Path, output: &Path) -> Result<std::path::PathBuf> {
    crate::config::safe_relative(output)?;
    ensure!(
        output.starts_with("out")
            && output != Path::new("out")
            && !output.starts_with("out/publication"),
        "matched studies must remain below ignored out/, outside the publication cache"
    );
    let root = root.canonicalize()?;
    let cache = root.join("out");
    fs::create_dir_all(&cache)?;
    let cache = cache.canonicalize()?;
    ensure!(
        cache == root.join("out"),
        "study cache must not redirect into a publication directory"
    );
    let destination = root.join(output);
    let mut ancestor = destination.as_path();
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .context("study has no existing ancestor")?;
    }
    let ancestor = ancestor.canonicalize()?;
    ensure!(
        ancestor.starts_with(&cache),
        "study destination escapes local cache"
    );
    let publication = cache.join("publication");
    ensure!(
        !ancestor.starts_with(&publication)
            && (!publication.exists() || !ancestor.starts_with(publication.canonicalize()?)),
        "study destination redirects into the publication cache"
    );
    Ok(destination)
}

#[allow(clippy::too_many_arguments)]
pub fn compare(
    root: &Path,
    before: &Path,
    after: &Path,
    output: &Path,
    recipe: Option<&Path>,
    embedding_run: Option<&Path>,
    before_population: Option<&Path>,
    after_population: Option<&Path>,
) -> Result<()> {
    compare_with_matching(
        root,
        before,
        after,
        output,
        recipe,
        embedding_run,
        before_population,
        after_population,
        false,
    )
}

/// Seed-matched generator experiments may deliberately change architecture.
/// Geometry matching remains the default contract for appearance-only studies.
#[allow(clippy::too_many_arguments)]
pub fn compare_with_matching(
    root: &Path,
    before: &Path,
    after: &Path,
    output: &Path,
    recipe: Option<&Path>,
    embedding_run: Option<&Path>,
    before_population: Option<&Path>,
    after_population: Option<&Path>,
    allow_architecture_changes: bool,
) -> Result<()> {
    let destination = local_destination(root, output)?;
    let protocol: Protocol = if let Some(path) = recipe {
        toml::from_str(&fs::read_to_string(root.join(path))?)?
    } else {
        Protocol {
            seed: 0,
            audit_rooms: 64,
            rendered_rooms: 64,
            cameras: 3,
            width: 512,
            height: 512,
            playback_steps: 1,
            density: 0.65,
            human_density: 0.25,
            gi_rays: 1024,
        }
    };
    ensure!(
        (2..=1024).contains(&protocol.rendered_rooms)
            && protocol.audit_rooms >= protocol.rendered_rooms
            && (1..=16).contains(&protocol.cameras)
            && (1..=16).contains(&protocol.playback_steps),
        "invalid study protocol"
    );
    let before_root = root.join(before).canonicalize()?;
    let after_root = root.join(after).canonicalize()?;
    let before = Dataset::load_study(&before_root, &protocol).context("baseline qualification")?;
    let after = Dataset::load_study(&after_root, &protocol).context("expanded qualification")?;
    ensure!(
        before.identity.source_sha256 != after.identity.source_sha256,
        "comparison has identical generator source"
    );
    for field in [
        "camera_settings",
        "appearance",
        "density",
        "human_density",
        "image_size",
    ] {
        ensure!(
            dataset::same_f32(&before.metrics[field], &after.metrics[field]),
            "unmatched capture control: {field}"
        );
    }
    let mut camera_displacement = Vec::new();
    let mut unchanged_cameras = 0;
    for (b, a) in before.scenes.iter().zip(&after.scenes) {
        ensure!(
            b.capture.seed == a.capture.seed
                && (allow_architecture_changes
                    || dataset::same_f32(&b.architecture, &a.architecture)),
            "unmatched room geometry"
        );
        for field in [
            "layout",
            "palette",
            "floor_style",
            "architecture_style",
            "lighting",
            "sun_elevation",
            "sun_azimuth",
            "light_kelvin",
            "target_lux",
            "daylight_lux",
        ] {
            ensure!(
                dataset::same_f32(&b.manifest[field], &a.manifest[field]),
                "unmatched room prior: {field}"
            );
        }
        ensure!(
            b.capture.capabilities == a.capture.capabilities,
            "unmatched renderer capabilities"
        );
        for (b, a) in b.capture.views.iter().zip(&a.capture.views) {
            let distance = (0..3)
                .map(|i| (b.world_from_view[3][i] as f64 - a.world_from_view[3][i] as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            camera_displacement.push(distance);
            unchanged_cameras += usize::from(
                dataset::same_f32(&json!(b.world_from_view), &json!(a.world_from_view))
                    && b.fov_y.to_bits() == a.fov_y.to_bits(),
            );
        }
    }
    ensure!(
        destination.file_name().is_some(),
        "report destination must be a named directory"
    );
    if destination.exists() {
        ensure!(
            destination.is_dir() && destination.join("receipt.json").is_file(),
            "refusing to replace a directory not owned by this report generator"
        );
    }
    let parent = destination
        .parent()
        .context("report destination has no parent")?;
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".zeroverse-study-")
        .tempdir_in(parent)?;
    let output = stage.path();
    let mut samples = Vec::new();
    for (cohort, data) in [("before", &before), ("after", &after)] {
        for scene in &data.scenes {
            for (i, view) in scene.capture.views.iter().enumerate() {
                let path = scene.folder.join(format!("view_{i:02}_color.png"));
                samples.push(
                    json!({"path":path,"sha256":io::hash(&path)?,"cohort":cohort,
                    "seed":scene.capture.seed,"camera":view.camera_index,"time":view.time}),
                );
            }
        }
    }
    let index = json!({"schema_version":1,"samples":samples,"cohorts":{
        "before":{"identity":before.identity,"run_id":before.selection["run_id"]},
        "after":{"identity":after.identity,"run_id":after.selection["run_id"]}}});
    io::write(&output.join("embedding-index.json"), &index)?;
    let embedding_metrics = embedding_run
        .map(|p| embeddings::analyze(&root.join(p), &index, &output.join("embedding-index.json")))
        .transpose()?;
    let b_stats = census(&before)?;
    let a_stats = census(&after)?;
    let new_items: Vec<_> = a_stats["item_occurrences"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| {
            !b_stats["item_occurrences"]
                .as_object()
                .unwrap()
                .contains_key(*k)
        })
        .cloned()
        .collect();
    let mut geometry = BTreeMap::new();
    for (name, data, directory) in [
        ("before", &before, &before_root),
        ("after", &after, &after_root),
    ] {
        let mut meshes = 0_u64;
        let mut object_bounds = 0_u64;
        let mut human_bounds = 0_u64;
        let mut max_overrun = 0.0_f64;
        for scene in &data.scenes {
            let stats: Value =
                io::read(&directory.join(format!("geometry_{}.json", scene.capture.seed)))?;
            ensure!(
                stats["degenerate_triangles"] == 0
                    && stats["object_envelopes_checked"]
                        == scene.manifest["objects"].as_array().unwrap().len()
                    && stats["human_envelopes_checked"]
                        == scene.manifest["humans"].as_array().unwrap().len()
                    && stats["object_envelope_max_overrun_metres"]
                        .as_f64()
                        .is_some_and(|v| (0.0..=0.003).contains(&v)),
                "geometry qualification denominator or tolerance mismatch"
            );
            meshes += stats["triangles"]
                .as_u64()
                .context("missing triangle count")?;
            object_bounds += stats["object_envelopes_checked"].as_u64().unwrap();
            human_bounds += stats["human_envelopes_checked"].as_u64().unwrap();
            max_overrun = max_overrun.max(
                stats["object_envelope_max_overrun_metres"]
                    .as_f64()
                    .unwrap(),
            );
        }
        geometry.insert(name, json!({"audited_rooms":data.scenes.len(),"triangles":meshes,
            "object_envelopes_checked":object_bounds,"human_envelopes_checked":human_bounds,"max_object_overrun_metres":max_overrun}));
    }
    let populations = match (before_population, after_population) {
        (Some(b), Some(a)) => Some(populations(&root.join(b), &root.join(a), &before, &after)?),
        (None, None) => None,
        _ => anyhow::bail!("provide both population cohorts"),
    };
    for (name, directory) in [("before", before_population), ("after", after_population)] {
        if let Some(directory) = directory {
            for (source, suffix) in [
                ("metrics.json", "metrics.json"),
                ("placement_heatmaps.svg", "placements.svg"),
            ] {
                fs::copy(
                    root.join(directory).join(source),
                    output.join(format!("population-{name}-{suffix}")),
                )?;
            }
        }
    }
    let report = json!({"schema_version":1,"visibility":"local_only","protocol":protocol,"identities":{"before":before.identity,"after":after.identity},
        "capture_provenance":{"before":{"engine":before.selection["capture_engine"],"build":before.scenes[0].capture.other.get("build_provenance"),
            "renderer":before.scenes[0].capture.other.get("renderer"),"capabilities":before.scenes[0].capture.capabilities},
            "after":{"engine":after.selection["capture_engine"],"build":after.scenes[0].capture.other.get("build_provenance"),
            "renderer":after.scenes[0].capture.other.get("renderer"),"capabilities":after.scenes[0].capture.capabilities}},
        "before":b_stats,"after":a_stats,"new_item_kinds":new_items,"geometry":geometry,
        "matching":{"architecture":if allow_architecture_changes {"seed_matched_changed_geometry"} else {"identical_geometry"}},
        "camera_control":{"policy":"Matched seeds, lighting priors and camera settings. Architecture matching is declared separately. Collision-safe replanning can change poses; views are not pixel-matched.",
            "unchanged_views":unchanged_cameras,"position_displacement_metres":dataset::distribution(camera_displacement)},
        "population":populations,"embedding":embedding_metrics,
        "limits":[format!("{} rooms per cohort cannot establish 10M-sample learning utility or total perceptual support.",protocol.rendered_rooms),
            "Effective storage constructions and service combinations are local program factors, not a multiplicative estimate of total room diversity.",
            "Camera viewpoint sampling and collision-safe replanning are confounds for image embeddings; all consecutive rooms and views remain included.",
            "Source revisions can vary construction, finishes and camera poses. Image changes are not attributable to item count alone.",
            "Embedding distances do not establish photographic realism. Timings are diagnostic, not an isolated throughput benchmark."]});
    io::write(&output.join("report.json"), &report)?;
    // Self-contained gallery: all modes are lossless WebP, with measured
    // co-visibility membership and a single deterministic legend per camera count.
    let mut gallery = Vec::new();
    let mut review_tiles = Vec::new();
    for (b, a) in before.scenes.iter().zip(&after.scenes) {
        let mut paired = BTreeMap::new();
        for (name, scene) in [("before", b), ("after", a)] {
            let relative = format!("media/{name}/seed_{:06}", scene.capture.seed);
            let directory = output.join(&relative);
            fs::create_dir_all(&directory)?;
            for i in 0..scene.capture.views.len() {
                for mode in PUBLICATION_MODES {
                    let image = image::open(scene.folder.join(format!("view_{i:02}_{mode}.png")))?
                        .into_rgb8();
                    graphics::save_webp(
                        image,
                        &directory.join(format!("view_{i:02}_{mode}.webp")),
                    )?;
                }
                for suffix in ["co_visibility_mask", "co_visibility_valid"] {
                    let file = format!("view_{i:02}_{suffix}.png");
                    fs::copy(scene.folder.join(&file), directory.join(file))?;
                }
            }
            let views: Vec<_> = scene.capture.views.iter().collect();
            let plan = graphics::plan(&scene.architecture, Some(&scene.manifest), &views);
            graphics::save(&plan, &directory.join("plan.png"))?;
            let thumbnail = image::open(scene.folder.join("view_00_color.png"))?
                .resize_exact(256, 256, FilterType::Triangle)
                .into_rgb8();
            graphics::save_webp(thumbnail.clone(), &directory.join("thumbnail.webp"))?;
            if gallery.len() < 8 {
                review_tiles.push(thumbnail);
            }
            paired.insert(name, json!({"media":relative,"layout":scene.manifest["layout"],"objects":scene.manifest["objects"],
                "people":scene.manifest["humans"].as_array().unwrap().len(),"views":scene.capture.views}));
        }
        gallery.push(json!({"seed":b.capture.seed,"cohorts":paired}));
    }
    graphics::combined(&review_tiles, 4).save(output.join("paired-review.png"))?;
    io::write(
        &output.join("gallery.json"),
        &json!({"visibility":"local_only","protocol":protocol,"rooms":gallery}),
    )?;
    fs::write(output.join("index.html"), page::render(&report, &gallery)?)?;
    let mut inputs = BTreeMap::new();
    for (name, data) in [("before", &before), ("after", &after)] {
        for (path, hash) in &data.input_sha256 {
            inputs.insert(format!("{name}/{path}"), hash.clone());
        }
        let directory = if name == "before" {
            &before_root
        } else {
            &after_root
        };
        for scene in &data.scenes {
            let file = format!("geometry_{}.json", scene.capture.seed);
            inputs.insert(format!("{name}/{file}"), io::hash(&directory.join(file))?);
        }
    }
    for (name, directory) in [
        ("before-population", before_population),
        ("after-population", after_population),
    ] {
        if let Some(directory) = directory {
            for file in [
                "metrics.json",
                "distribution.json",
                "generator_identity.json",
                "objects.csv",
                "placement_heatmaps.svg",
            ] {
                inputs.insert(
                    format!("{name}/{file}"),
                    io::hash(&root.join(directory).join(file))?,
                );
            }
        }
    }
    if let Some(path) = embedding_run {
        for file in ["embeddings.json", "embeddings.f32"] {
            inputs.insert(
                format!("embedding/{file}"),
                io::hash(&root.join(path).join(file))?,
            );
        }
    }
    let mut artifacts = io::files(output, output)?;
    artifacts.remove("receipt.json");
    io::write(
        &output.join("receipt.json"),
        &json!({"schema_version":1,"visibility":"local_only","protocol":protocol,
        "identities":report["identities"],"inputs":inputs,"artifacts":artifacts,
        "report_sha256":sha(&fs::read(output.join("report.json"))?)}),
    )?;
    let backup = tempfile::Builder::new()
        .prefix(".zeroverse-study-previous-")
        .tempdir_in(parent)?;
    let previous = backup.path().join("report");
    if destination.exists() {
        fs::rename(&destination, &previous)?;
    }
    if let Err(error) = fs::rename(stage.path(), &destination) {
        if previous.exists() {
            fs::rename(&previous, &destination)
                .context("restore previous study after failed installation")?;
        }
        return Err(error.into());
    }
    println!(
        "Compared {} consecutive rooms per cohort, {} views each; report {}",
        protocol.rendered_rooms,
        protocol.rendered_rooms * protocol.cameras * protocol.playback_steps,
        destination.join("index.html").display()
    );
    Ok(())
}

pub(super) fn effective(counts: impl Iterator<Item = u64>) -> Value {
    let counts: Vec<_> = counts.filter(|&n| n > 0).collect();
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return json!({"observed":0,"effective":0.,"count":0});
    }
    let entropy = counts
        .iter()
        .map(|&n| {
            let p = n as f64 / total as f64;
            -p * p.ln()
        })
        .sum::<f64>();
    json!({"observed":counts.len(),"effective":entropy.exp(),"count":total})
}

/// Offline byte verification of the complete gallery and its measurement receipt.
pub fn verify(output: &Path) -> Result<()> {
    let receipt: Value = io::read(&output.join("receipt.json"))?;
    ensure!(receipt["schema_version"] == 1, "unsupported study receipt");
    ensure!(
        receipt["visibility"] == "local_only",
        "study is not marked local only"
    );
    let mut files = io::files(output, output)?;
    files.remove("receipt.json");
    ensure!(
        json!(files) == receipt["artifacts"]
            && receipt["report_sha256"] == io::hash(&output.join("report.json"))?,
        "study artifacts changed or are missing"
    );
    let report: Value = io::read(&output.join("report.json"))?;
    ensure!(
        report["visibility"] == "local_only",
        "study report is not local only"
    );
    ensure!(
        receipt["identities"] == report["identities"] && receipt["protocol"] == report["protocol"],
        "mixed study identity or protocol"
    );
    println!("Verified {} study artifacts", files.len());
    Ok(())
}

fn census(data: &Dataset) -> Result<Value> {
    let mut occurrences = BTreeMap::<String, u64>::new();
    let mut rooms = BTreeMap::<String, u64>::new();
    let mut compositions = BTreeMap::<String, u64>::new();
    let mut service = BTreeMap::<u32, u64>::new();
    let mut per_room = Vec::new();
    let mut plants_per_room = Vec::new();
    let mut cables_per_room = Vec::new();
    let mut fixtures_per_room = Vec::new();
    let mut times = Vec::new();
    let mut visible_people = 0;
    let mut people_rooms = 0;
    let mut shared = Vec::new();
    let mut low_semantic = 0;
    let mut align_depth = 0.0_f64;
    let mut align_pixels = 0.0_f64;
    let mut hashes = BTreeMap::<String, BTreeSet<u64>>::new();
    for scene in &data.scenes {
        let mut kinds = BTreeMap::<String, u64>::new();
        for object in scene.manifest["objects"]
            .as_array()
            .context("missing objects")?
            .iter()
            .filter(|o| o["neighbor"] == false)
        {
            let kind = object["kind"].as_str().context("missing kind")?.to_owned();
            *kinds.entry(kind.clone()).or_default() += 1;
            *occurrences.entry(kind).or_default() += 1;
        }
        for kind in kinds.keys() {
            *rooms.entry(kind.clone()).or_default() += 1;
        }
        per_room.push(kinds.len() as f64);
        plants_per_room.push(kinds.get("Plant").copied().unwrap_or(0) as f64);
        cables_per_room.push(kinds.get("Cable").copied().unwrap_or(0) as f64);
        fixtures_per_room.push(
            ["DeskLamp", "WallSconce", "CeilingPendant"]
                .iter()
                .map(|k| kinds.get(*k).copied().unwrap_or(0) as f64)
                .sum(),
        );
        let signature = serde_json::to_string(
            &kinds
                .iter()
                .map(|(k, v)| (k, (*v).min(3)))
                .collect::<BTreeMap<_, _>>(),
        )?;
        *compositions.entry(signature).or_default() += 1;
        let mask = [
            "SideTable",
            "MobileCart",
            "DrawerUnit",
            "Bench",
            "Ottoman",
            "RoomDivider",
        ]
        .iter()
        .enumerate()
        .fold(0, |mask, (i, k)| {
            mask | if kinds.contains_key(*k) { 1 << i } else { 0 }
        });
        *service.entry(mask).or_default() += 1;
        let people = !scene.manifest["humans"].as_array().unwrap().is_empty();
        people_rooms += usize::from(people);
        visible_people += usize::from(
            people
                && scene
                    .capture
                    .views
                    .iter()
                    .any(|v| v.semantic_pixel_counts.get("person").copied().unwrap_or(0) > 0),
        );
        times.push(scene.capture.elapsed_seconds);
        for (index, view) in scene.capture.views.iter().enumerate() {
            low_semantic += usize::from(view.semantic_colors < 4);
            shared.push(view.co_visibility.shared_fraction_valid);
            align_depth = align_depth.max(
                view.annotation_alignment["depth_position_max_metres"]
                    .as_f64()
                    .context("missing depth alignment")?,
            );
            align_pixels = align_pixels.max(
                view.annotation_alignment["reprojection_max_pixels"]
                    .as_f64()
                    .context("missing reprojection alignment")?,
            );
            let hash = io::hash(&scene.folder.join(format!("view_{index:02}_color.png")))?;
            hashes.entry(hash).or_default().insert(scene.capture.seed);
        }
    }
    let exact_duplicate_room_pairs: usize =
        hashes.values().map(|s| s.len() * (s.len() - 1) / 2).sum();
    let kinds_entropy = effective(occurrences.values().copied());
    let front = data.metrics["categories"]["storage_front_handle"]
        .as_object()
        .map(|c| c.values().filter_map(Value::as_u64).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![occurrences.get("Cabinet").copied().unwrap_or(0)]);
    let programs: BTreeMap<_, _> = [
        "plant_growth_habit",
        "plant_pot_sides",
        "screen_content",
        "screen_dormant",
        "whiteboard_content",
        "art_subject",
        "art_frame",
        "accepted_workspace_arrangement",
        "table_support",
        "chair_arm_support",
    ]
    .into_iter()
    .map(|key| {
        let counts = data.metrics["categories"][key]
            .as_object()
            .map(|c| c.values().filter_map(Value::as_u64).collect::<Vec<_>>())
            .unwrap_or_default();
        (key, effective(counts.into_iter()))
    })
    .collect();
    Ok(
        json!({"rendered_rooms":data.scenes.len(),"views":shared.len(),"item_occurrences":occurrences,"item_room_counts":rooms,
        "catalog_item_kinds":data.metrics["object_counts_per_scene"].as_object().context("missing catalog coverage")?.keys()
            .filter(|name| name.starts_with("main/") && name.as_str() != "main/Person").count(),
        "kind_coverage":kinds_entropy,"kinds_per_room":dataset::distribution(per_room),
        "plants_per_primary_room":dataset::distribution(plants_per_room),"cables_per_primary_room":dataset::distribution(cables_per_room),
        "task_fixtures_per_primary_room":dataset::distribution(fixtures_per_room),"content_programs":programs,
        "coarse_composition":effective(compositions.values().copied()),"service_constellations":effective(service.values().copied()),
        "service_presence_counts":service,"storage_front_handle":effective(front.into_iter()),"capture_seconds":dataset::distribution(times),
        "co_visibility":dataset::distribution(shared),"rooms_with_people":people_rooms,"rooms_showing_people":visible_people,
        "views_with_fewer_than_four_semantic_classes":low_semantic,"cross_room_exact_image_duplicate_pairs":exact_duplicate_room_pairs,
        "annotation_max_depth_position_metres":align_depth,"annotation_max_reprojection_pixels":align_pixels}),
    )
}

fn populations(b: &Path, a: &Path, before: &Dataset, after: &Dataset) -> Result<Value> {
    let mut output = BTreeMap::new();
    let mut denominator = None;
    for (name, path, data) in [("before", b, before), ("after", a, after)] {
        let identity: Value = io::read(&path.join("generator_identity.json"))?;
        ensure!(
            identity == json!(data.identity),
            "population compiled identity differs from rendered cohort"
        );
        let metrics: Value = io::read(&path.join("metrics.json"))?;
        let audit: Value = io::read(&path.join("distribution.json"))?;
        let count = metrics["scenes"]
            .as_u64()
            .context("missing population denominator")?;
        ensure!(
            audit["invalid_seeds"]
                .as_array()
                .context("missing population gates")?
                .is_empty()
                && denominator.is_none_or(|d| d == count)
                && count >= data.scenes.len() as u64
                && audit["first_seed"] == json!(data.scenes[0].capture.seed)
                && audit["seeds"] == count
                && audit["cameras_per_scene"]
                    == data.scenes[0].capture.co_visibility_metadata["camera_count"]
                && audit["generator_version"] == json!(data.identity.generator_version)
                && [
                    "camera_settings",
                    "density",
                    "human_density",
                    "image_size",
                    "appearance"
                ]
                .iter()
                .all(|field| dataset::same_f32(&metrics[*field], &data.metrics[*field])),
            "unmatched or invalid population"
        );
        denominator = Some(count);
        let construction = if let Some(counts) =
            metrics["categories"]["storage_front_handle"].as_object()
        {
            effective(counts.values().filter_map(Value::as_u64))
        } else {
            json!({"observed":1,"effective":1.0,"definition":"All baseline cabinet fronts use full-height hinged doors with vertical bar handles."})
        };
        let furnishings =
            population_furnishings(path, audit["first_seed"].as_u64().unwrap(), count)?;
        output.insert(name,json!({"rooms":count,"invalid_layouts":0,"storage_front_handle":construction,"furnishings":furnishings,
            "categories":metrics["categories"],"object_counts":metrics["object_counts_per_scene"],
            "heatmap_grid_size":metrics["heatmap_grid_size"],"placement_heatmaps":metrics["placement_heatmaps"]}));
    }
    Ok(json!(output))
}

/// The generator's unquoted numeric/enum CSV contract; include zero-service
/// rooms in the denominator and measure the joint accepted constellation.
fn population_furnishings(path: &Path, first: u64, count: u64) -> Result<Value> {
    ensure!(
        count <= 100_000 && first.checked_add(count).is_some(),
        "unbounded population audit"
    );
    let mut room_kinds = vec![BTreeSet::new(); count as usize];
    let file = fs::File::open(path.join("objects.csv"))?;
    let mut rows = std::io::BufReader::new(file).lines();
    ensure!(
        rows.next()
            .transpose()?
            .context("empty object CSV")?
            .starts_with("seed,layout,id,kind,neighbor,support,"),
        "unsupported object CSV"
    );
    for line in rows {
        let line = line?;
        let fields: Vec<_> = line.split(',').collect();
        ensure!(
            !line.contains('"') && fields.len() == 16,
            "malformed object CSV"
        );
        let seed = fields[0].parse::<u64>()?;
        ensure!(
            seed >= first && seed < first + count && matches!(fields[4], "true" | "false"),
            "object outside audited population"
        );
        if fields[4] == "false" {
            room_kinds[(seed - first) as usize].insert(fields[3].to_owned());
        }
    }
    let mut room_counts = BTreeMap::<String, u64>::new();
    let mut constellations = BTreeMap::<u32, u64>::new();
    let mut counts = Vec::new();
    for kinds in room_kinds {
        counts.push(kinds.len() as f64);
        for kind in &kinds {
            *room_counts.entry(kind.clone()).or_default() += 1;
        }
        let mask = [
            "SideTable",
            "MobileCart",
            "DrawerUnit",
            "Bench",
            "Ottoman",
            "RoomDivider",
        ]
        .into_iter()
        .enumerate()
        .fold(0, |mask, (i, kind)| {
            mask | if kinds.contains(kind) { 1 << i } else { 0 }
        });
        *constellations.entry(mask).or_default() += 1;
    }
    Ok(
        json!({"item_room_counts":room_counts,"kinds_per_room":dataset::distribution(counts),
        "service_constellations":effective(constellations.values().copied()),"service_presence_counts":constellations}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn studies_cannot_target_release_directories_or_escape_local_cache() {
        let root = tempfile::tempdir().unwrap();
        for path in [
            "www/project/study",
            "tex/generated/study",
            "out/publication/study",
            "out",
            "out/../www/study",
        ] {
            assert!(
                local_destination(root.path(), Path::new(path)).is_err(),
                "{path}"
            );
        }
        assert!(local_destination(root.path(), Path::new("out/local-study/report")).is_ok());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.path().join("www"), root.path().join("out/link"))
                .unwrap();
            fs::create_dir_all(root.path().join("www")).unwrap();
            assert!(local_destination(root.path(), Path::new("out/link/report")).is_err());
        }
    }
    #[test]
    fn effective_support_does_not_multiply_correlated_marginals() {
        assert_eq!(effective([10, 0, 0].into_iter())["effective"], 1.);
        assert_eq!(effective([0, 0].into_iter())["effective"], 0.);
        assert!(
            (effective([3, 3, 3, 3].into_iter())["effective"]
                .as_f64()
                .unwrap()
                - 4.)
                .abs()
                < 1e-12
        );
        assert!(
            effective([1000, 1, 1, 1].into_iter())["effective"]
                .as_f64()
                .unwrap()
                < 1.1
        );
    }

    #[test]
    fn population_constellations_include_empty_rooms_and_exclude_neighbors() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("objects.csv"),
            "seed,layout,id,kind,neighbor,support,x_m,y_m,z_m,width_m,height_m,depth_m,yaw_radians,variant,u,v\n0,Lounge,0,SideTable,false,,0,0,0,1,1,1,0,0,0.5,0.5\n0,Lounge,1,SideTable,false,,0,0,0,1,1,1,0,0,0.5,0.5\n1,Lounge,0,SideTable,true,,0,0,0,1,1,1,0,0,0.5,0.5\n").unwrap();
        let report = population_furnishings(dir.path(), 0, 2).unwrap();
        assert_eq!(report["service_constellations"]["effective"], 2.);
        assert_eq!(report["service_presence_counts"], json!({"0":1,"1":1}));
        assert_eq!(report["item_room_counts"]["SideTable"], 1);
        assert_eq!(report["kinds_per_room"]["mean"], 0.5);
        assert!(population_furnishings(dir.path(), 1, 2).is_err());
    }
}
