use crate::{
    camera_study,
    config::{Config, Protocol},
    dataset::{Capture, Dataset},
    doors, io, materials, media, objects, paper, qualification, snapshot, visibility,
};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::{
    provenance::{publisher_inputs, source_digest, source_inputs},
    GeneratorIdentity, CAPTURE_SCHEMA_VERSION, GENERATOR_VERSION,
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

const ATTESTATION: &str = "www/project/publication.json";
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attestation {
    pub schema_version: u32,
    pub generator_identity: GeneratorIdentity,
    pub protocol: Protocol,
    pub run_id: String,
    pub capture_engine: String,
    pub renderer_inputs: BTreeMap<String, String>,
    pub publication_inputs: BTreeMap<String, String>,
    pub capture_sources: BTreeMap<String, String>,
    pub artifacts: BTreeMap<String, String>,
    pub reference_studies: Vec<Value>,
}

pub fn identity(root: &Path) -> Result<GeneratorIdentity> {
    let manifest: toml::Value = toml::from_str(&fs::read_to_string(root.join("Cargo.toml"))?)?;
    let version = manifest["package"]["version"]
        .as_str()
        .context("missing generator crate version")?;
    Ok(GeneratorIdentity {
        schema_version: CAPTURE_SCHEMA_VERSION,
        crate_version: version.into(),
        generator_version: GENERATOR_VERSION,
        source_sha256: source_digest(&source_inputs(root)?),
    })
}
fn publication_inputs(root: &Path, config: &Config) -> Result<BTreeMap<String, String>> {
    let pub_dir = root.join("crates/publication");
    let mut inputs = BTreeMap::new();
    for (key, value) in publisher_inputs(&pub_dir)? {
        inputs.insert(format!("crates/publication/{key}"), value);
    }
    for name in ["publication.toml", "tex/arxiv.sty", "tex/references.bib"] {
        inputs.insert(name.into(), io::hash(&root.join(name))?);
    }
    for name in objects::TOOLS
        .into_iter()
        .chain(materials::TOOLS)
        .chain([doors::TOOL])
    {
        inputs.insert(name.into(), io::hash(&root.join(name))?);
    }
    for dir in ["www/project/static", "tex/generated"] {
        for (name, value) in io::files(root, &root.join(dir))? {
            if !is_generated(&name) && !snapshot::retired(&name) {
                inputs.insert(name, value);
            }
        }
    }
    if let Some(input) = &config.camera_study {
        for (name, hash) in io::files(root, &root.join(input))? {
            inputs.insert(name, hash);
        }
    }
    if let Some(input) = &config.room_qualification {
        inputs.extend(io::files(root, &root.join(input))?);
    }
    for input in &config.capacity_receipts {
        inputs.insert(
            input.to_string_lossy().into_owned(),
            io::hash(&root.join(input))?,
        );
    }
    let lens_input = root.join(crate::fisheye::EVIDENCE);
    if lens_input.join("receipt.json").exists() {
        inputs.extend(io::files(root, &lens_input)?);
    }
    Ok(inputs)
}
fn is_generated(name: &str) -> bool {
    name == "www/project/index.html"
        || name == "www/project/reference.html"
        || name.starts_with("www/project/fisheye/")
        || name.starts_with("www/project/static/media/architecture/")
        || name.starts_with("www/project/static/media/qualification/")
        || name.starts_with("www/project/static/media/cohorts/")
        || name.starts_with("www/project/static/media/capacity/")
        || name.starts_with("www/project/static/media/cameras/")
        || name.starts_with("www/project/static/media/objects/")
        || name.starts_with("www/project/static/media/doors/")
        || name.starts_with("www/project/static/media/materials/")
        || name.starts_with("www/project/static/papers/")
        || matches!(
            name,
            "www/project/static/media/social.jpg"
                | "www/project/static/media/whitepaper.webp"
                | "tex/bevy_zeroverse.tex"
        )
        || name.starts_with("tex/generated/architecture_")
        || name.starts_with("tex/generated/publication_")
        || name.starts_with("tex/generated/camera_count_")
        || name.starts_with("tex/generated/object_")
        || name.starts_with("tex/generated/door_")
        || name.starts_with("tex/generated/material_")
        || name.starts_with("tex/generated/pipeline_")
        || name.starts_with("tex/generated/capacity_")
}
fn artifacts(root: &Path) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for dir in ["www/project/static", "www/project/fisheye", "tex/generated"] {
        out.extend(
            io::files(root, &root.join(dir))?
                .into_iter()
                .filter(|(name, _)| is_generated(name)),
        );
    }
    for path in [
        "www/project/index.html",
        "www/project/reference.html",
        "tex/bevy_zeroverse.tex",
    ] {
        out.insert(path.into(), io::hash(&root.join(path))?);
    }
    Ok(out)
}
pub fn verify(root: &Path, release_version: Option<&str>) -> Result<Attestation> {
    verify_for_rebuild(root, release_version, false)
}
fn verify_for_rebuild(
    root: &Path,
    release_version: Option<&str>,
    allow_page_changes: bool,
) -> Result<Attestation> {
    ensure!(
        source_digest(&publisher_inputs(&root.join("crates/publication"))?)
            == env!("PUBLICATION_SOURCE_SHA256"),
        "publisher binary is stale: rebuild from this checkout"
    );
    let config = Config::load(root)?;
    let attestation: Attestation = io::read(&root.join(ATTESTATION))
        .context("publication missing: run cargo run -p bevy_zeroverse_publication -- refresh")?;
    ensure!(
        attestation.schema_version == 1,
        "unsupported publication attestation"
    );
    let current = identity(root)?;
    let renderer_inputs = source_inputs(root)?;
    let changed_inputs = attestation
        .renderer_inputs
        .keys()
        .chain(renderer_inputs.keys())
        .filter(|name| attestation.renderer_inputs.get(*name) != renderer_inputs.get(*name))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    ensure!(
        attestation.protocol == config.capture,
        "capture recipe changed"
    );
    if let Some(version) = release_version {
        ensure!(
            version.trim_start_matches('v') == current.crate_version,
            "release tag does not match generator version"
        );
    }
    let inputs = publication_inputs(root, &config)?;
    if let Some(retained) = &config.retained_evidence {
        retained.verify(&current, &attestation, &inputs)?;
    } else {
        ensure!(
            attestation.generator_identity == current,
            "stale generator: publication refresh required; expected source {}, found {}; changed renderer inputs [{}]",
            attestation.generator_identity.source_sha256,
            current.source_sha256,
            changed_inputs
        );
        ensure!(
            attestation.renderer_inputs == renderer_inputs,
            "renderer source set/hash changed"
        );
    }
    if !allow_page_changes || config.retained_evidence.is_none() {
        ensure!(
            attestation.publication_inputs == inputs,
            "publication inputs changed: rebuild the retained page or refresh current captures"
        );
    }
    let mut actual_artifacts = artifacts(root)?;
    // First installation of a separately qualified lens chapter. Existing
    // architecture/paper hashes remain exact; the lens bundle is independently
    // checked below and becomes fully bound by the new top-level attestation.
    if allow_page_changes
        && !attestation
            .artifacts
            .keys()
            .any(|name| name.starts_with("www/project/fisheye/"))
    {
        actual_artifacts.retain(|name, _| !name.starts_with("www/project/fisheye/"));
    }
    ensure!(
        attestation.artifacts == actual_artifacts,
        "publication artifacts missing/changed; rebuild rather than edit generated files"
    );
    ensure!(
        attestation.reference_studies.is_empty(),
        "the current snapshot cannot include historical reference studies"
    );
    verify_bundle(root, &config, &attestation, allow_page_changes)?;
    if !allow_page_changes {
        crate::objects::verify_explorer(&fs::read_to_string(root.join("www/project/index.html"))?)?;
    }
    Ok(attestation)
}

/// The exported bundle is self-contained: CI never needs out/, GPU captures,
/// neural models, Python, plotting packages or the original machine's paths.
fn verify_bundle(
    root: &Path,
    config: &Config,
    attestation: &Attestation,
    rebuilding: bool,
) -> Result<()> {
    crate::fisheye::verify_installed(root, rebuilding)?;
    doors::verify(
        &root.join(doors::DIRECTORY),
        &attestation.generator_identity,
    )?;
    materials::verify(
        &root.join(materials::DIRECTORY),
        &attestation.generator_identity,
    )?;
    objects::verify(
        &root.join(objects::DIRECTORY),
        &attestation.generator_identity,
    )?;
    if config.camera_study.is_some() {
        camera_study::verify(
            &root.join(camera_study::DIRECTORY),
            Some(&attestation.generator_identity),
        )?;
    }
    qualification::verify_current(
        &root.join(qualification::PUBLISHED_DIRECTORY),
        &attestation.generator_identity,
    )?;
    crate::cohorts::publication::verify(
        root,
        config.room_qualification.is_some(),
        &attestation.generator_identity,
    )?;
    crate::capacity::publication::verify(
        root,
        config.capacity_receipts.len(),
        &attestation.generator_identity,
    )?;
    let media = root.join("www/project/static/media/architecture");
    let gallery: Value = io::read(&media.join("gallery.json"))?;
    ensure!(
        gallery["generator_identity"] == json!(attestation.generator_identity)
            && gallery["audit_rooms"] == config.capture.audit_rooms
            && gallery["rendered_rooms"] == config.capture.rendered_rooms
            && gallery["rendered_views"]
                == config.capture.rendered_rooms
                    * config.capture.cameras
                    * config.capture.playback_steps
            && gallery["modes"] == json!(bevy_zeroverse_capture::PUBLICATION_MODES),
        "page/capture protocol mismatch"
    );
    let mut archive = zip::ZipArchive::new(fs::File::open(media.join("visibility-masks.zip"))?)?;
    let mut pooled = [0u64; 2];
    let scenes = gallery["scenes"]
        .as_array()
        .context("missing gallery scenes")?;
    ensure!(
        scenes.len() == config.capture.rendered_rooms,
        "gallery room denominator mismatch"
    );
    for (offset, scene) in scenes.iter().enumerate() {
        let seed = config.capture.seed + offset as u64;
        ensure!(scene["seed"] == seed, "nonconsecutive room gallery");
        let metadata_name = scene["metadata"].as_str().context("missing calibration")?;
        crate::config::safe_relative(Path::new(metadata_name))?;
        let metadata: Value = io::read(&media.join(metadata_name))?;
        ensure!(
            metadata["generator_identity"] == json!(attestation.generator_identity),
            "mixed calibration identity"
        );
        let capture: Capture = serde_json::from_value(metadata["capture"].clone())?;
        ensure!(
            capture.run_id == attestation.run_id
                && capture.seed == seed
                && capture.views.len() == config.capture.cameras * config.capture.playback_steps,
            "mixed/incomplete calibration run"
        );
        visibility::validate_metadata(&capture.co_visibility_metadata, config.capture.cameras)?;
        let frames = scene["frames"].as_array().context("missing time frames")?;
        ensure!(
            frames.len() == config.capture.playback_steps,
            "missing time frame"
        );
        for (step, frame) in frames.iter().enumerate() {
            let views = frame["views"].as_array().context("missing camera views")?;
            ensure!(
                views.len() == config.capture.cameras
                    && number_eq(&frame["time"], step as f64 / (frames.len() - 1) as f64),
                "incomplete camera/time set"
            );
            for (camera, view) in views.iter().enumerate() {
                let index = capture
                    .views
                    .iter()
                    .position(|v| v.camera_index == camera && v.step_index == step)
                    .context("missing calibrated view")?;
                ensure!(
                    crate::dataset::same_f32(&view["camera"], &json!(capture.views[index]))
                        && crate::dataset::same_f32(
                            &view["visibility"],
                            &json!(capture.views[index].co_visibility)
                        ),
                    "UI calibration/membership differs from exported source"
                );
                let mut preview = None;
                for mode in bevy_zeroverse_capture::PUBLICATION_MODES {
                    let name = view["images"][mode]
                        .as_str()
                        .context("missing annotation mode")?;
                    crate::config::safe_relative(Path::new(name))?;
                    let image = image::open(media.join(name))?.into_rgb8();
                    ensure!(
                        image.dimensions() == (config.capture.width, config.capture.height),
                        "preview image size mismatch"
                    );
                    if mode == "co_visibility" {
                        preview = Some(image);
                    }
                }
                let mut mask = Vec::new();
                let mut valid = Vec::new();
                archive
                    .by_name(&format!(
                        "seed_{seed:06}/view_{index:02}_co_visibility_mask.png"
                    ))?
                    .read_to_end(&mut mask)?;
                archive
                    .by_name(&format!(
                        "seed_{seed:06}/view_{index:02}_co_visibility_valid.png"
                    ))?
                    .read_to_end(&mut valid)?;
                let plane = visibility::decode(
                    &mask,
                    &valid,
                    preview.as_ref(),
                    capture.image_size,
                    camera,
                    config.capture.cameras,
                    &capture.views[index].co_visibility,
                )?;
                pooled[0] += plane.stats.valid_pixels;
                pooled[1] += plane.stats.shared_pixels;
                let peers = view["images"]["peers"]
                    .as_array()
                    .context("missing peer overlays")?;
                ensure!(
                    peers.len() == config.capture.cameras,
                    "missing peer overlay"
                );
                for peer in peers {
                    let name = peer.as_str().context("invalid peer image")?;
                    crate::config::safe_relative(Path::new(name))?;
                    ensure!(media.join(name).is_file(), "missing peer image");
                }
            }
        }
    }
    ensure!(
        gallery["co_visibility"]["statistics"]["valid_pixels"] == pooled[0]
            && gallery["co_visibility"]["statistics"]["shared_pixels"] == pooled[1],
        "co-visibility pooled denominator mismatch"
    );
    let pdf = root.join("www/project/static/papers/bevy_zeroverse.pdf");
    let provenance: Value = io::read(&root.join("www/project/static/papers/provenance.json"))?;
    ensure!(
        fs::read(&pdf)?.starts_with(b"%PDF-")
            && provenance["pdf_sha256"] == io::hash(&pdf)?
            && provenance["architecture_gallery_sha256"] == io::hash(&media.join("gallery.json"))?
            && provenance["generator_identity"] == json!(attestation.generator_identity),
        "paper/page identity or PDF mismatch"
    );
    let archive_path = root.join("www/project/static/papers/whitepaper-source.zip");
    ensure!(
        provenance["source_archive_sha256"] == io::hash(&archive_path)?,
        "paper source archive mismatch"
    );
    let mut sources = zip::ZipArchive::new(fs::File::open(archive_path)?)?;
    for (name, digest) in provenance["sources"]
        .as_object()
        .context("missing paper dependency hashes")?
    {
        crate::config::safe_relative(Path::new(name))?;
        snapshot::paper_source(name, &attestation.artifacts)?;
        ensure!(
            *digest == io::hash(&root.join(name))?,
            "paper source changed: {name}"
        );
        let mut bytes = Vec::new();
        sources
            .by_name(
                name.strip_prefix("tex/")
                    .context("invalid paper source root")?,
            )?
            .read_to_end(&mut bytes)?;
        ensure!(
            *digest == bevy_zeroverse_capture::provenance::sha(&bytes),
            "paper archive dependency mismatch: {name}"
        );
    }
    check_html(root)?;
    Ok(())
}
fn number_eq(value: &Value, expected: f64) -> bool {
    value.as_f64().is_some_and(|v| (v - expected).abs() < 1e-6)
}
fn check_html(root: &Path) -> Result<()> {
    let page = root.join("www/project/index.html");
    let html = fs::read_to_string(&page)?;
    check_gallery(&html)?;
    ensure!(
        !html.contains("reference.html") && !html.contains("class=\"camera-study-plan\""),
        "historical studies or implementation plans cannot be presented as the current snapshot"
    );
    for directory in ["www/project/static/media", "tex/generated"] {
        for name in io::files(root, &root.join(directory))?.into_keys() {
            ensure!(
                !snapshot::retired(&name),
                "historical snapshot artifact: {name}"
            );
        }
    }
    let expression = regex::Regex::new(r#"(?:src|href)="([^"]+)""#)?;
    for captures in expression.captures_iter(&html) {
        let link = &captures[1];
        if link.starts_with(['#', '/']) || link.contains("://") || link.starts_with("mailto:") {
            continue;
        }
        let path = link.split(['#', '?']).next().unwrap();
        ensure!(
            page.parent().unwrap().join(path).exists(),
            "broken project-page link: {link}"
        );
    }
    Ok(())
}

fn check_gallery(html: &str) -> Result<()> {
    // A static matched panel is useful without JavaScript, but repeats the
    // interactive gallery if it is also presented to scripting browsers.
    let interactive = regex::Regex::new(r"(?s)<noscript>.*?</noscript>")?.replace_all(html, "");
    ensure!(
        interactive.matches("id=\"architecture-explorer\"").count() == 1
            && interactive
                .matches("id=\"architecture-visibility\"")
                .count()
                == 1
            && interactive
                .matches("data-architecture-mode=\"co_visibility\"")
                .count()
                == 1
            && !interactive.contains("id=\"comparison\"")
            && !interactive.contains("static/js/index.js")
            && !regex::Regex::new(r#"<img\b[^>]*src="[^"]*architecture_co_visibility\.jpg""#)?
                .is_match(&interactive),
        "duplicate or disjoint co-visibility presentation"
    );
    Ok(())
}

pub fn refresh(root: &Path, recapture: bool) -> Result<()> {
    let root = root.canonicalize()?;
    let config = Config::load(&root)?;
    crate::retained::require_refresh_allowed(&config)?;
    let work = root.join("out/publication");
    fs::create_dir_all(&work)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(work.join("writer.lock"))?;
    lock.try_lock_exclusive()
        .context("another publication build is running")?;
    ensure!(
        Command::new("latexmk").arg("-v").output().is_ok()
            && Command::new("pdftoppm").arg("-v").output().is_ok(),
        "latexmk/pdflatex and Poppler are required; no captures were started"
    );
    let publisher_digest = source_digest(&publisher_inputs(&root.join("crates/publication"))?);
    ensure!(
        publisher_digest == env!("PUBLICATION_SOURCE_SHA256"),
        "publisher binary is stale: use cargo run or rebuild this tool"
    );
    let current = identity(&root)?;
    // These independently measured inputs are not produced by the gallery
    // capture. Reject stale receipts before spending GPU time on any atlases.
    if let Some(input) = &config.camera_study {
        camera_study::verify(&root.join(input), Some(&current))
            .context("refresh the configured camera-count study before publication refresh")?;
    }
    if let Some(input) = &config.room_qualification {
        crate::cohorts::publication::verify_input(&root.join(input), &current)
            .context("refresh the configured rendered-cohort receipt before publication refresh")?;
    }
    for input in &config.capacity_receipts {
        crate::capacity::publication::read_current(&root.join(input), &current)?;
    }
    let timing = root.join("tex/generated/pipeline_receipt.json");
    if timing.exists() {
        crate::timing::validate_receipt(&io::read(&timing)?, &current)
            .context("remeasure the paper timing study before publication refresh")?;
    }
    let before = publication_inputs(&root, &config)?;
    let capture_root = root.join(&config.captures);
    let dataset = if !recapture {
        Dataset::load(&capture_root, &current, &config.capture).ok()
    } else {
        None
    };
    let dataset = match dataset {
        Some(dataset) => {
            println!("Reuse completed captures with identical compiled generator and protocol");
            dataset
        }
        None => {
            println!(
                "Capture current generator: {} audited / {} rendered rooms",
                config.capture.audit_rooms, config.capture.rendered_rooms
            );
            let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
            let temporary = tempfile::Builder::new()
                .prefix("capture-")
                .tempdir_in(&work)?;
            // Retain diagnostics if post-render contract checks fail.
            let temporary = temporary.keep();
            let p = &config.capture;
            ensure!(
                Command::new(cargo)
                    .args([
                        "run",
                        "--locked",
                        "-p",
                        "bevy_zeroverse",
                        "--bin",
                        "indoor_validate",
                        "--no-default-features",
                        "--features",
                        "multi_threaded",
                        "--"
                    ])
                    .args(["--labels", "--co-visibility", "--no-raw"])
                    .args([
                        "--seed",
                        &p.seed.to_string(),
                        "--audit-seeds",
                        &p.audit_rooms.to_string(),
                        "--renders",
                        &p.rendered_rooms.to_string(),
                        "--cameras",
                        &p.cameras.to_string(),
                        "--width",
                        &p.width.to_string(),
                        "--height",
                        &p.height.to_string(),
                        "--playback-steps",
                        &p.playback_steps.to_string(),
                        "--density",
                        &p.density.to_string(),
                        "--human-density",
                        &p.human_density.to_string(),
                        "--gi-rays",
                        &p.gi_rays.to_string(),
                        "--ssaa-factor",
                        &p.ssaa_factor.to_string()
                    ])
                    .arg("--output")
                    .arg(&temporary)
                    .arg("--asset-root")
                    .arg(&root)
                    .current_dir(&root)
                    .status()?
                    .success(),
                "capture run failed; published output unchanged"
            );
            Dataset::load(&temporary, &current, p)?;
            fs::create_dir_all(capture_root.parent().unwrap())?;
            // Cache replacement is not publication: preserve the last complete
            // cache until the new run has passed every contract check.
            if capture_root.exists() {
                fs::remove_dir_all(&capture_root)?;
            }
            fs::rename(temporary, &capture_root)?;
            Dataset::load(&capture_root, &current, p)?
        }
    };
    let qualification_cache = work.join("qualification");
    qualification::refresh_cache(&root, &qualification_cache, &current)?;
    let door_cache = work.join("doors");
    doors::refresh_cache(&root, &door_cache, &current)?;
    let object_cache = work.join("objects");
    objects::refresh_cache(&root, &object_cache, &current)?;
    let material_cache = work.join("materials");
    materials::refresh_cache(&root, &material_cache, &current)?;
    let staging = tempfile::Builder::new()
        .prefix("stage-")
        .tempdir_in(&work)?;
    let stage = staging.path();
    io::copy_tree(&root.join("www/project"), &stage.join("www/project"))?;
    io::copy_tree(&root.join("tex"), &stage.join("tex"))?;
    snapshot::prune(stage)?;
    let qualification_output = stage.join(qualification::PUBLISHED_DIRECTORY);
    if qualification_output.exists() {
        fs::remove_dir_all(&qualification_output)?;
    }
    io::copy_tree(&qualification_cache, &qualification_output)?;
    // The viewer is built separately, but the project link must resolve in the
    // same staged site root when validation checks relative targets.
    fs::copy(root.join("www/index.html"), stage.join("www/index.html"))?;
    println!("Build lossless media, plans, population metrics and unified HTML");
    let gallery = media::build(&dataset, stage, &config.capture)?;
    camera_study::build(&root, stage, config.camera_study.as_deref())?;
    crate::cohorts::publication::build(
        &root,
        stage,
        config.room_qualification.as_deref(),
        &current,
    )?;
    objects::build(&object_cache, stage, &current)?;
    doors::build(&door_cache, stage, &current)?;
    materials::build(&material_cache, stage, &current)?;
    crate::capacity::publication::build(&root, stage, &config.capacity_receipts, &current)?;
    crate::fisheye::build(&root, stage)?;
    media::page(&gallery, stage, &config.capture)?;
    paper::write_sources(&dataset, &gallery, stage, &config)?;
    println!("Compile paper and package its exact dependency closure");
    paper::compile(stage, &config, &gallery)?;
    crate::fisheye::bind_paper(stage)?;
    ensure!(
        identity(&root)? == current && publication_inputs(&root, &config)? == before,
        "sources changed during build; no artifacts installed"
    );
    let attestation = Attestation {
        schema_version: 1,
        generator_identity: current.clone(),
        protocol: config.capture.clone(),
        run_id: dataset.selection["run_id"].as_str().unwrap().into(),
        capture_engine: dataset.selection["capture_engine"].as_str().unwrap().into(),
        renderer_inputs: source_inputs(&root)?,
        publication_inputs: before,
        capture_sources: dataset.input_sha256,
        artifacts: artifacts(stage)?,
        reference_studies: Vec::new(),
    };
    verify_bundle(stage, &config, &attestation, false)?;
    io::write(&stage.join(ATTESTATION), &attestation)?;
    println!("Install fully validated bundle; attestation is the last commit marker");
    install(&root, stage, &attestation.artifacts)?;
    verify(&root, None)?;
    println!(
        "Publication verified: crate {}, generator {}, {} rooms / {} views",
        current.crate_version,
        current.generator_version,
        config.capture.rendered_rooms,
        config.capture.rendered_rooms * config.capture.cameras * config.capture.playback_steps
    );
    Ok(())
}

/// Rebuild the page and paper at release/deployment time from the self-contained
/// attested capture bundle. Source upgrades require refresh or an explicit
/// release/source-bound retained-evidence policy. Retained captures stay intact;
/// release servers never relabel an old generator's measurements.
pub fn rebuild(root: &Path, release_version: Option<&str>) -> Result<()> {
    let root = root.canonicalize()?;
    let config = Config::load(&root)?;
    let work = root.join("out/publication");
    fs::create_dir_all(&work)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(work.join("writer.lock"))?;
    lock.try_lock_exclusive()
        .context("another publication build is running")?;
    let mut attestation =
        verify_for_rebuild(&root, release_version, config.retained_evidence.is_some())?;
    if config.retained_evidence.is_some() {
        return rebuild_retained_page(&root, &config, attestation, release_version);
    }
    let before = publication_inputs(&root, &config)?;
    let staging = tempfile::Builder::new()
        .prefix("release-")
        .tempdir_in(&work)?;
    let stage = staging.path();
    io::copy_tree(&root.join("www/project"), &stage.join("www/project"))?;
    io::copy_tree(&root.join("tex"), &stage.join("tex"))?;
    snapshot::prune(stage)?;
    fs::copy(root.join("www/index.html"), stage.join("www/index.html"))?;
    let gallery: Value =
        io::read(&stage.join("www/project/static/media/architecture/gallery.json"))?;
    let dataset = Dataset::published(
        &root,
        attestation.generator_identity.clone(),
        attestation.capture_sources.clone(),
    )?;
    camera_study::build(&root, stage, config.camera_study.as_deref())?;
    crate::cohorts::publication::build(
        &root,
        stage,
        config.room_qualification.as_deref(),
        &attestation.generator_identity,
    )?;
    objects::write_paper(stage)?;
    doors::write_paper(stage)?;
    materials::write_paper(stage)?;
    crate::capacity::publication::build(
        &root,
        stage,
        &config.capacity_receipts,
        &attestation.generator_identity,
    )?;
    crate::fisheye::build(&root, stage)?;
    media::page(&gallery, stage, &config.capture)?;
    paper::write_sources(&dataset, &gallery, stage, &config)?;
    paper::compile(stage, &config, &gallery)?;
    crate::fisheye::bind_paper(stage)?;
    ensure!(
        identity(&root)? == attestation.generator_identity
            && publication_inputs(&root, &config)? == before,
        "source changed during release rebuild"
    );
    attestation.artifacts = artifacts(stage)?;
    verify_bundle(stage, &config, &attestation, false)?;
    io::write(&stage.join(ATTESTATION), &attestation)?;
    install(&root, stage, &attestation.artifacts)?;
    verify(&root, release_version)?;
    println!("Rebuilt and verified current page/paper from the attested capture bundle");
    Ok(())
}

fn rebuild_retained_page(
    root: &Path,
    config: &Config,
    mut attestation: Attestation,
    release_version: Option<&str>,
) -> Result<()> {
    let current = identity(root)?;
    let inputs = publication_inputs(root, config)?;
    let staging = tempfile::Builder::new()
        .prefix("retained-page-")
        .tempdir_in(root.join("out/publication"))?;
    let stage = staging.path();
    io::copy_tree(&root.join("www/project"), &stage.join("www/project"))?;
    io::copy_tree(&root.join("tex"), &stage.join("tex"))?;
    let gallery = io::read(&stage.join("www/project/static/media/architecture/gallery.json"))?;
    crate::fisheye::build(root, stage)?;
    media::page(&gallery, stage, &config.capture)?;
    attestation.artifacts = artifacts(stage)?;
    attestation.publication_inputs = inputs.clone();
    config
        .retained_evidence
        .as_ref()
        .context("missing retained evidence policy")?
        .verify(&current, &attestation, &inputs)?;
    ensure!(
        identity(root)? == current && publication_inputs(root, config)? == inputs,
        "source changed during retained page rebuild"
    );
    verify_bundle(stage, config, &attestation, false)?;
    crate::objects::verify_explorer(&fs::read_to_string(stage.join("www/project/index.html"))?)?;
    io::write(&stage.join(ATTESTATION), &attestation)?;
    install(root, stage, &attestation.artifacts)?;
    verify(root, release_version)?;
    println!("Rebuilt page presentation for release {}; retained all captured evidence and paper from crate {} (source {}), without GPU work or relabeling",
        current.crate_version, attestation.generator_identity.crate_version, attestation.generator_identity.source_sha256);
    Ok(())
}

/// The sanctioned registry entry point: no package upload happens before the
/// page/paper has passed the explicit source and evidence policy.
/// Direct Cargo uploads remain outside this tool's control.
pub fn publish(root: &Path, package: &str, dry_run: bool) -> Result<()> {
    if Config::load(root)?.retained_evidence.is_some() {
        // A failed retained-evidence check must never start a GPU refresh.
        rebuild(root, None)?;
    } else if verify(root, None).is_err() {
        refresh(root, false)?;
    } else {
        rebuild(root, None)?;
    }
    let attestation = verify(root, None)?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command
        .args(["publish", "--locked", "--package", package])
        .current_dir(root);
    if dry_run {
        command.args(["--dry-run", "--allow-dirty"]);
    }
    println!("Publication verified: captured crate {}, release crate {}; registry package {package}, dry_run={dry_run}",attestation.generator_identity.crate_version, identity(root)?.crate_version);
    ensure!(
        command.status()?.success(),
        "Cargo publication failed; validated documentation remains available"
    );
    Ok(())
}

/// All scientific work happens in staging. Rename each managed file with a
/// rollback backup; write the attestation last. A process crash cannot pass the
/// release hash gate with a partially installed bundle.
pub fn install(root: &Path, stage: &Path, artifacts: &BTreeMap<String, String>) -> Result<()> {
    let transaction = tempfile::Builder::new()
        .prefix("install-")
        .tempdir_in(root.join("out/publication"))?;
    let mut installed = Vec::<(PathBuf, Option<PathBuf>)>::new();
    let mut names: Vec<_> = artifacts.keys().cloned().map(|name| (name, true)).collect();
    for directory in ["www/project/static", "www/project/fisheye", "tex/generated"] {
        for name in io::files(root, &root.join(directory))?.into_keys() {
            if (is_generated(&name) || snapshot::retired(&name)) && !artifacts.contains_key(&name) {
                names.push((name, false));
            }
        }
    }
    names.push((ATTESTATION.into(), true));
    let result = (|| -> Result<()> {
        for (index, (name, replace)) in names.iter().enumerate() {
            crate::config::safe_relative(Path::new(name))?;
            let destination = root.join(name);
            fs::create_dir_all(destination.parent().unwrap())?;
            let backup = if destination.exists() {
                let backup = transaction.path().join(format!("{index}.old"));
                fs::rename(&destination, &backup)?;
                Some(backup)
            } else {
                None
            };
            installed.push((destination.clone(), backup));
            if *replace {
                fs::rename(stage.join(name), destination)?;
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        for (destination, backup) in installed.into_iter().rev() {
            if destination.exists() {
                fs::remove_file(&destination)?;
            }
            if let Some(backup) = backup {
                fs::rename(backup, destination)?;
            }
        }
        return Err(error.context("publication installation rolled back"));
    }
    Ok(())
}

#[cfg(test)]
mod gallery_tests {
    use super::check_gallery;

    #[test]
    fn rejects_duplicate_presentations_but_preserves_the_no_script_figure() {
        let html = r#"<div id="architecture-explorer"><button data-architecture-mode="co_visibility"></button><div id="architecture-visibility"></div></div><noscript><img src="static/media/architecture/architecture_co_visibility.jpg"></noscript>"#;
        check_gallery(html).unwrap();
        for duplicate in [
            r#"<img src="static/media/architecture/architecture_co_visibility.jpg">"#,
            r#"<button data-architecture-mode="co_visibility"></button>"#,
            r#"<div id="architecture-visibility"></div>"#,
            r#"<div id="architecture-explorer"></div>"#,
            r#"<div id="comparison"></div>"#,
        ] {
            assert!(check_gallery(&format!("{html}{duplicate}")).is_err());
        }
    }
}
