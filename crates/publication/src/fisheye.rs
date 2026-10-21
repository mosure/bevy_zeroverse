//! Independently attested lens qualification and paper addendum. This does not
//! relabel the retained architectural cohort as a new renderer measurement.
use crate::{config::safe_relative, io};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};

pub mod performance;

pub const DIRECTORY: &str = "www/project/fisheye";
pub const EVIDENCE: &str = "docs/evidence/fisheye";

fn valid_lighting_report(value: &Value) -> bool {
    let below = |value: &Value, limit| value.as_f64().is_some_and(|v| v >= 0. && v < limit);
    value["cases"].as_array().is_some_and(|cases| {
        let raw_passes = cases.iter().all(|case| {
            below(&case["linear_rgb_abs_error_mean"], 0.025)
                && below(&case["linear_rgb_abs_error_p95"], 0.08)
                && case["boundaries"].as_array().is_some_and(|boundaries| {
                    boundaries.iter().all(|boundary| {
                        below(&boundary["mean"], 0.025) && below(&boundary["p95"], 0.08)
                    })
                })
        });
        cases.len() == 5
                // Preserve an observed raw-face failure in the receipt and paper;
                // never silently promote final-composite continuity to raw parity.
                && value["passed"].as_bool() == Some(raw_passes)
                && ["ambient", "directional", "point", "spot", "combined"]
                    .iter()
                    .zip(cases)
                    .all(|(name, case)| {
                        let composite = &case["composite_continuity"];
                        case["case"] == *name
                            && below(&case["linear_rgb_abs_error_mean"], f64::INFINITY)
                            && below(&case["linear_rgb_abs_error_p95"], f64::INFINITY)
                            && below(&case["linear_rgb_abs_error_max"], f64::INFINITY)
                            && composite["production_wgsl"] == true
                            && composite["world_ray_approaches"].as_u64().unwrap_or(0) > 1000
                            && below(&composite["linear_rgb_abs_error_mean"], 0.0001)
                            && below(&composite["linear_rgb_abs_error_max"], 0.002)
                            && case["boundaries"].as_array().is_some_and(|boundaries| {
                                boundaries.len() == 8
                                    && boundaries.iter().all(|boundary| {
                                        below(&boundary["mean"], f64::INFINITY)
                                            && below(&boundary["p95"], f64::INFINITY)
                                    })
                            })
                    })
    })
}

pub fn prepare(root: &Path, captures: &Path, analytic: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "lens evidence destination already exists");
    let protocol: Value = io::read(&captures.join("protocol.json"))?;
    let rooms: Vec<Value> = io::read(&captures.join("timings.json"))?;
    let oracle: Value = io::read(analytic)?;
    let sampling: Value = io::read(&captures.join("sampling-audit.json"))?;
    let current = crate::pipeline::identity(root)?;
    let performance: Option<Value> = if captures.join("performance.json").exists() {
        let value: Value = io::read(&captures.join("performance.json"))?;
        ensure!(
            value["generator_identity"] == serde_json::to_value(&current)?,
            "stale lens performance report"
        );
        Some(value)
    } else {
        None
    };
    let lighting: Option<Value> = if captures.join("lighting.json").exists() {
        let value: Value = io::read(&captures.join("lighting.json"))?;
        ensure!(
            value["provenance"]["source_sha256"] == current.source_sha256,
            "stale lighting qualification"
        );
        ensure!(
            valid_lighting_report(&value),
            "invalid lighting evidence or failed final-composite continuity"
        );
        Some(value)
    } else {
        None
    };
    ensure!(
        protocol["provenance"]["source_sha256"] == current.source_sha256
            && oracle["provenance"]["source_sha256"] == current.source_sha256
            && sampling["provenance"]["source_sha256"] == current.source_sha256,
        "stale lens capture or oracle binary"
    );
    ensure!(
        sampling["assignments"] == 12_288
            && sampling["seeds"] == 1024
            && sampling["cameras_per_seed"] == 12
            && sampling["replay_exact"] == true
            && sampling["calibrations_valid"] == true
            && sampling["prior"]["probability"] == 0.25
            && (0.23..0.27).contains(&(sampling["angular"].as_f64().unwrap_or(0.) / 12_288.))
            && (0.17..0.23).contains(
                &(sampling["circular"].as_f64().unwrap_or(0.)
                    / sampling["angular"].as_f64().unwrap_or(0.)),
            ),
        "lens sampling audit did not pass"
    );
    ensure!(
        oracle["rgb_foreground_pixels"].as_u64().unwrap_or(0) > 10
            && oracle["rgb_core_mismatches"] == 0
            && oracle["rgb_core_pixels"].as_u64().unwrap_or(0) > 5000
            && oracle["co_visibility_mismatches"] == 0
            && oracle["co_visibility_relations"].as_u64().unwrap_or(0) > 5000
            && oracle["flow_correspondences"].as_u64().unwrap_or(0) > 5000
            && oracle["flow_max_pixels"].as_f64().unwrap_or(f64::INFINITY) < 0.01
            && oracle["position_max_metres"]
                .as_f64()
                .unwrap_or(f64::INFINITY)
                < 0.001,
        "analytic lens qualification did not pass"
    );
    let config = &protocol["config"];
    let cameras = config["num_cameras"].as_u64().context("camera count")? as usize;
    let steps = config["playback_steps"].as_u64().context("step count")? as usize;
    let dimensions = [
        config["width"].as_f64().context("width")? as u32,
        config["height"].as_f64().context("height")? as u32,
    ];
    let start = protocol["seeds"][0].as_u64().context("seed range")?;
    let end = protocol["seeds"][1].as_u64().context("seed range")?;
    ensure!(
        rooms.len() == (end - start) as usize
            && rooms.len() == 6
            && start == 400
            && end == 406
            && dimensions == [512, 512]
            && cameras == 3
            && steps == 2,
        "incomplete lens population"
    );
    ensure!(
        config["indoor_gi_rays"] == 256
            && config["indoor_human_density"] == 0.25
            && (config["indoor_density"].as_f64().unwrap_or(0.) - 0.65).abs() < 1e-6
            && config["indoor_gpu_materials"] == true
            && config["ssaa_factor"] == 0
            && protocol["adapter"]
                .as_str()
                .is_some_and(|a| a.contains("RTX PRO 6000 Blackwell") && a.contains("Vulkan")),
        "lens study workload/hardware differs from the declared paper protocol"
    );
    let mut times = Vec::new();
    let mut max_error = 0_f64;
    let mut diameters = Vec::new();
    let mut files = BTreeMap::new();
    for (index, room) in rooms.iter().enumerate() {
        let seed = start + index as u64;
        ensure!(room["seed"] == seed, "nonconsecutive lens seeds");
        let measurements = room["measurements"]
            .as_array()
            .context("view measurements")?;
        ensure!(measurements.len() == cameras * steps, "missing lens views");
        times.push(room["seconds"].as_f64().context("completed capture time")?);
        for (i, m) in measurements.iter().enumerate() {
            let calibration: bevy_zeroverse_capture::calibration::CameraCalibration =
                serde_json::from_value(m["calibration"].clone())?;
            calibration.validate().map_err(anyhow::Error::msg)?;
            ensure!(
                calibration.image_size == dimensions && calibration.fisheye.is_some(),
                "lens or size mismatch"
            );
            let error = m["self_reprojection_max_pixels"]
                .as_f64()
                .context("reprojection")?;
            ensure!(
                error < 0.01
                    && m["normal_length_max_error"]
                        .as_f64()
                        .unwrap_or(f64::INFINITY)
                        < 1e-5,
                "annotation qualification failed"
            );
            max_error = max_error.max(error);
            if i < cameras {
                diameters.push(f64::from(
                    calibration.fisheye.unwrap().max_angle.to_degrees() * 2.,
                ));
            }
            for suffix in [
                "",
                "_depth",
                "_normal",
                "_semantic",
                "_position",
                "_co_visibility",
                "_flow",
            ] {
                let name = format!("room_{seed}_view_{i}{suffix}.png");
                ensure!(
                    image::image_dimensions(captures.join(&name))?
                        == (dimensions[0], dimensions[1]),
                    "capture image size mismatch"
                );
                files.insert(name.clone(), io::hash(&captures.join(name))?);
            }
            let name = format!("room_{seed}_view_{i}.json");
            files.insert(name.clone(), io::hash(&captures.join(name))?);
        }
        let name = format!("room_{seed}_manifest.json");
        files.insert(name.clone(), io::hash(&captures.join(name))?);
    }
    for name in ["protocol.json", "timings.json", "sampling-audit.json"] {
        files.insert(name.into(), io::hash(&captures.join(name))?);
    }
    for name in ["performance.json", "lighting.json"] {
        if captures.join(name).exists() {
            files.insert(name.into(), io::hash(&captures.join(name))?);
        }
    }
    fs::create_dir_all(output)?;
    for name in files.keys() {
        fs::copy(captures.join(name), output.join(name))?;
    }
    fs::copy(analytic, output.join("analytic.json"))?;
    files.insert(
        "analytic.json".into(),
        io::hash(&output.join("analytic.json"))?,
    );
    let seconds: f64 = times[1..].iter().sum();
    let receipt = json!({"schema_version":1,"kind":"fisheye_qualification","generator_identity":current,
        "seed_start":start,"rooms":rooms.len(),"cameras":cameras,"steps":steps,"image_size":dimensions,
        "captured_images":rooms.len()*cameras*steps,"self_reprojection_max_pixels":max_error,
        "diameter_degrees":crate::camera_study::distribution(diameters),"analytic":oracle,"sampling":sampling,
        "timing":{"concurrent_training":protocol["concurrent_training"],"first_room_seconds":times[0],"steady_seconds":seconds,
            "rooms_per_second":(rooms.len()-1) as f64/seconds,"images_per_second":((rooms.len()-1)*cameras*steps) as f64/seconds,
            "scope":"completed generation and capture; application startup and PNG/JSON writing excluded; first room reported separately"},
        "performance":performance,"lighting":lighting,"artifacts":files});
    io::write(&output.join("receipt.json"), &receipt)?;
    verify(output)?;
    Ok(())
}

pub fn verify(path: &Path) -> Result<Value> {
    let receipt: Value = io::read(&path.join("receipt.json"))?;
    ensure!(
        receipt["schema_version"] == 1 && receipt["kind"] == "fisheye_qualification",
        "unknown lens study"
    );
    for (name, hash) in receipt["artifacts"]
        .as_object()
        .context("missing lens hashes")?
    {
        safe_relative(Path::new(name))?;
        ensure!(
            hash.as_str() == Some(io::hash(&path.join(name))?.as_str()),
            "changed lens evidence: {name}"
        );
    }
    Ok(receipt)
}

/// Rebuild optional feature media from its own checked receipt. The architecture
/// paper remains immutable; the new lens methods/results are a linked addendum.
pub fn build(root: &Path, stage: &Path) -> Result<()> {
    let input = root.join(EVIDENCE);
    if !input.join("receipt.json").exists() {
        ensure!(
            !root.join(DIRECTORY).join("receipt.json").exists(),
            "published lens gallery needs current-source qualification inputs"
        );
        return Ok(());
    }
    let receipt = verify(&input)?;
    ensure!(
        receipt["generator_identity"] == serde_json::to_value(crate::pipeline::identity(root)?)?,
        "lens examples need current-source qualification"
    );
    let output = stage.join(DIRECTORY);
    if output.exists() {
        fs::remove_dir_all(&output)?;
    }
    fs::create_dir_all(&output)?;
    io::copy_tree(&input, &output)?;
    let rooms: Value = io::read(&input.join("timings.json"))?;
    let data = json!({"receipt":receipt,"rooms":rooms});
    let html = include_str!("../templates/fisheye.html").replace(
        "@DATA@",
        &serde_json::to_string(&data)?.replace('<', "\\u003c"),
    );
    fs::write(output.join("index.html"), html)?;
    let section = paper_section(&receipt);
    fs::write(output.join("paper-section.tex"), &section)?;
    let paper=format!("\\documentclass{{article}}\n\\pdfinfoomitdate=1\n\\pdftrailerid{{}}\n\\usepackage[margin=22mm]{{geometry}}\n\\usepackage{{graphicx,amsmath,booktabs,hyperref}}\n\\title{{bevy\\_zeroverse: calibrated fisheye cameras}}\n\\author{{Mitchell Mosure}}\n\\date{{October 2026}}\n\\begin{{document}}\n\\maketitle\n{section}\n\\end{{document}}\n");
    fs::write(output.join("fisheye_cameras.tex"), paper)?;
    compile(&output, "fisheye_cameras")?;
    bind_paper(stage)?;
    Ok(())
}

fn compile(output: &Path, name: &str) -> Result<()> {
    let status = std::process::Command::new("latexmk")
        .args([
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-no-shell-escape",
        ])
        .arg(format!("{name}.tex"))
        .current_dir(output)
        .env("SOURCE_DATE_EPOCH", "946684800")
        .env("FORCE_SOURCE_DATE", "1")
        .env("TZ", "UTC")
        .output()
        .context("latexmk for lens paper")?;
    ensure!(
        status.status.success(),
        "lens paper compile: {}",
        String::from_utf8_lossy(&status.stdout)
    );
    let log = fs::read_to_string(output.join(format!("{name}.log")))?;
    ensure!(
        !log.contains("Overfull ") && !log.contains("undefined references"),
        "lens paper layout or reference error"
    );
    for ext in ["aux", "log", "fls", "fdb_latexmk", "out"] {
        let _ = fs::remove_file(output.join(format!("{name}.{ext}")));
    }
    Ok(())
}

/// Append the qualified lens chapter to the main paper without rewriting the
/// provenance of its architectural measurements. Called again after main-paper
/// compilation in a full refresh, so the combined PDF cannot retain stale pages.
pub fn bind_paper(stage: &Path) -> Result<()> {
    let output = stage.join(DIRECTORY);
    if !output.join("receipt.json").exists() {
        return Ok(());
    }
    fs::write(
        output.join("whitepaper.tex"),
        r"\documentclass{article}
\pdfinfoomitdate=1
\pdftrailerid{}
\usepackage{pdfpages}
\begin{document}
\includepdf[pages=-]{../static/papers/bevy_zeroverse.pdf}
\includepdf[pages=-]{fisheye_cameras.pdf}
\end{document}
",
    )?;
    compile(&output, "whitepaper")?;
    let mut sources = BTreeMap::new();
    for name in [
        "fisheye_cameras.tex",
        "paper-section.tex",
        "whitepaper.tex",
        "room_401_view_0.png",
        "room_402_view_1.png",
        "room_402_view_1_co_visibility.png",
        "receipt.json",
        "protocol.json",
    ] {
        sources.insert(name.to_owned(), fs::read(output.join(name))?);
    }
    sources.insert("BUILD.txt".into(), format!(
        "Build the lens chapter with latexmk -pdf -no-shell-escape fisheye_cameras.tex.\nThe main page separately provides the architectural paper source ZIP.\nTo reproduce the combined PDF, keep this directory at project/fisheye and the architectural PDF at project/static/papers/bevy_zeroverse.pdf, then run latexmk -pdf -no-shell-escape whitepaper.tex.\nArchitecture PDF SHA-256: {}\nArchitecture source ZIP SHA-256: {}\n",
        io::hash(&stage.join("www/project/static/papers/bevy_zeroverse.pdf"))?,
        io::hash(&stage.join("www/project/static/papers/whitepaper-source.zip"))?,
    ).into_bytes());
    io::archive(&output.join("lens-source.zip"), &sources)?;
    let mut hashes = io::files(&output, &output)?;
    hashes.remove("publication.json");
    io::write(&output.join("publication.json"), &hashes)?;
    Ok(())
}

pub fn page(stage: &Path) -> Result<String> {
    let path = stage.join(DIRECTORY);
    if !path.join("receipt.json").exists() {
        return Ok(String::new());
    }
    let r = verify(&path)?;
    Ok(format!(
        r#"<section class="container section" id="fisheye"><div class="section-heading"><div><div class="eyebrow">CALIBRATED ANGULAR CAMERAS</div><h2>Wide views.<br>Shared geometry.</h2></div><p>{} full-resolution camera/time captures from {} consecutive furnished rooms. Explore RGB, range depth, normals, semantics, positions, optical flow and per-camera co-visibility.</p></div><a href="fisheye/"><img src="fisheye/room_402_view_1.png" width="512" height="512" style="width:32%;border-radius:12px" alt="Circular fisheye room capture"><img src="fisheye/room_401_view_0.png" width="512" height="512" style="width:32%;border-radius:12px" alt="Diagonal fisheye room capture"><img src="fisheye/room_402_view_1_co_visibility.png" width="512" height="512" style="width:32%;border-radius:12px" alt="Aligned additive co-visibility for the circular camera"></a><p><a class="button primary" href="fisheye/">Explore calibrated fisheye captures ↗</a> <a class="button" href="fisheye/fisheye_cameras.pdf">Lens paper addendum ↗</a> <a class="button" href="fisheye/receipt.json">Qualification receipt ↗</a></p></section>"#,
        r["captured_images"], r["rooms"]
    ))
}

fn paper_section(r: &Value) -> String {
    let mut section = format!(
        r"\section{{Projection and sampling}}
The optional central angular camera uses $r(\theta)=\theta(1+k_1\theta^2+k_2\theta^4+k_3\theta^6+k_4\theta^8)$, with $\theta=\operatorname{{atan2}}(\sqrt{{x^2+y^2}},-z)$. Full pixel-space $K$, image dimensions, half-pixel centers, model version, coefficients and calibrated angular domain are exported per view. Monotonicity is certified over the entire domain using interval derivative bounds. The coefficient order matches OpenCV fisheye (Kannala--Brandt), verified against its forward-hemisphere equations; OpenCV integer-centered pixels require subtracting 0.5 from the exported principal point and converting the view basis to right/down/forward. The inverse supports rays behind the optical plane, beyond the domain of OpenCV routines based on the front-hemisphere arctangent. This is a central camera model, not rolling shutter or a non-central lens.

Lens assignment is seeded by scene and camera index before placement retries. Configuration controls its Bernoulli probability, continuous diameter FOV, polynomial variation, anisotropy, principal-point displacement and circular, full-frame/diagonal or explicitly elliptical fit. Circular fits enforce equal focal axes and a contained disk; elliptical fits contain the sampled ellipse; diagonal fits cover every sensor corner, including principal-point shifts. Defaults remain pinhole; enabling the lens program defaults to probability 0.25, a 140--210 degree diameter range and 0.2 circular probability. In 1,024 consecutive seeds with 12 slots each, {angular} of {assignments} assignments are angular and {circular} of those are circular. Calibration validation and exact replay pass for every assignment. The admitted pose distribution is affected by obstacle and shared-surface constraints; lens assignment is preserved. Camera motion and baseline controls remain independent.

\section{{Rendering and annotation contract}}
Up to five co-located perspective faces retain the PBR renderer, shadows, lighting and exposure. Guard bands and a GPU angular resolve produce the final image; faces are implementation details and consume no dataset camera indices. SSAA averages RGB in linear light. Final-pixel-center rays independently intersect authoritative annotation triangles in a shared BVH. Fisheye depth is positive ray range; pinhole depth remains positive axial depth. Invalid lens pixels and surface misses are distinct through the calibrated domain mask. Normals are expressed in the logical camera frame; semantic IDs are never interpolated. Glass follows the selected geometric surface policy, not a refracted or reflected RGB light path.

Forward flow traces the preceding capture's surface and projects its corresponding point at the next controlled timestamp. Co-visibility projects each source surface through the target model and tests symmetric tangent-plane agreement at the nearest target pixel. Finite sampling can leave small holes at thin silhouettes; this is the existing dataset visibility policy. Mixed lenses and circular validity use the same projection contract. Only logical cameras contribute bits to the additive membership mask. Burn and Python tensor archives use a versioned mixed-lens contract with per-view model identifiers and five angular parameters; pinhole-only archives preserve version 1. Direct GPU transports retain the same per-view metadata and on-device color conversion.

\section{{Bounded qualification}}
The analytic fixture uses a moving occluder, two angular cameras and one pinhole camera over three synchronized captures, with unrelated render updates between captures. Maximum position error is {:.4} mm; maximum forward-flow error is {:.6} pixels across {} correspondences. There are {} co-visibility comparisons away from silhouette discontinuities and {} mismatches. These tests exercise occlusion, rearward rays, optical-center displacement and anisotropy.

The furnished-room cohort contains {} consecutive rooms, three cameras and two timestamps at $512\times512$, with automatic $2\times$ SSAA and full scene quality. All {} captures pass self-reprojection (maximum {:.6} pixels) and unit-normal checks. The realized diameter range is {:.1}--{:.1} degrees. Figures show actual, uncorrected renderer output; dim and bright scenes remain in the gallery. Geometry and lens settings, calibration, software provenance and artifact checksums accompany every record.

\begin{{figure}}[ht]\centering
\includegraphics[width=.32\linewidth]{{room_401_view_0.png}}
\includegraphics[width=.32\linewidth]{{room_402_view_1.png}}
\includegraphics[width=.32\linewidth]{{room_402_view_1_co_visibility.png}}
\caption{{Diagonal and circular fisheye RGB; additive co-visibility from the same circular view (camera 1). Red: camera 0; blue: camera 2; magenta: both; black: no shared peer or invalid lens domain. The full gallery includes all rooms and aligned annotations.}}\end{{figure}}

\section{{Throughput and limits}}
On an NVIDIA RTX PRO 6000 Blackwell ({workload}), the completed-generation rate excluding the first room was {:.3} rooms/s or {:.3} camera/time images/s. The first room took {:.3} seconds and is reported separately. Application startup and PNG/JSON report writing are excluded; rendering, scene work and capture readback are included. These are bounded observations of the recorded workload, not a memory-stability guarantee. Five perspective faces cost more than a single pinhole image; shared scene assets/BVH, indexed-vertex transforms, compact BVH partitioning, nearest-first traversal, stable geometry reuse, and the capture activity gate bound redundant work. The disabled path creates no angular textures or BVH.

Measurements use the optimized development profile (optimization level 3), the default 256 diffuse-GI rays, GPU material synthesis, furnishing density 0.65 and human density 0.25. Desktop applications share the GPU; training is separately identified above. This is native Vulkan qualification. The WebAssembly build passes; browser/Metal runtime and cross-vendor performance remain unqualified for this source.

Screen-space effects remain face-local; the guarded color resolve is not a path tracer. RGB uses perspective-face clipping while exact geometric targets use the calibrated ray range. Geometry within the configured operating range is the qualification domain; grazing near/far boundary behavior, arbitrary optical systems, hardware-vendor parity and downstream learning gains require further qualification. The main architectural cohort retains its original capture provenance; this addendum is independently source-bound.
",
        r["analytic"]["position_max_metres"].as_f64().unwrap() * 1000.,
        r["analytic"]["flow_max_pixels"].as_f64().unwrap(),
        r["analytic"]["flow_correspondences"],
        r["analytic"]["co_visibility_relations"],
        r["analytic"]["co_visibility_mismatches"],
        r["rooms"],
        r["captured_images"],
        r["self_reprojection_max_pixels"].as_f64().unwrap(),
        r["diameter_degrees"]["min"].as_f64().unwrap(),
        r["diameter_degrees"]["max"].as_f64().unwrap(),
        r["timing"]["rooms_per_second"].as_f64().unwrap(),
        r["timing"]["images_per_second"].as_f64().unwrap(),
        r["timing"]["first_room_seconds"].as_f64().unwrap(),
        angular = r["sampling"]["angular"],
        assignments = r["sampling"]["assignments"],
        circular = r["sampling"]["circular"],
        workload = if r["timing"]["concurrent_training"] == true {
            "concurrent training workload"
        } else {
            "training paused during capture"
        }
    );
    if let Some(rows) = r["performance"]["rows"].as_array() {
        use std::fmt::Write;
        section.push_str(r"
\subsection{Matched camera-model workload}
The same six rooms, poses, timestamps, materials, GI, shadows and automatic 2x SSAA are used for both projections. The report rejects differences in the geometry manifest and exported poses. Projection coverage differs; a rectangular pinhole image cannot span the complete fisheye domain. Rates include generation, rendering and CPU readback, with the first room excluded and recorded separately. These are not direct Burn/WGPU or CUDA transport benchmarks.
\begin{center}\begin{tabular}{lrr}\toprule
Workload & Pinhole rooms/s & Fisheye rooms/s \\
\midrule
");
        for row in rows {
            writeln!(
                section,
                r"{} & {:.3} & {:.3} \\",
                if row["config"]["playback_steps"] == 2 {
                    "Seven modes, two timestamps"
                } else {
                    "RGB, one timestamp"
                },
                row["pinhole"]["rooms_per_second"].as_f64().unwrap(),
                row["fisheye"]["rooms_per_second"].as_f64().unwrap()
            )
            .unwrap();
        }
        section.push_str("\\bottomrule\\end{tabular}\\end{center}\n");
    }
    if let Some(cases) = r["lighting"]["cases"].as_array() {
        use std::fmt::Write;
        section.push_str(r"
\subsection{Photometric face-boundary qualification}
Identical world rays are compared across all eight adjacent face pairs at fixed exposure, with a continuous environment map. Ambient, shadowed directional, point and spot cases vary the direct illumination while retaining the emissive object and bloom; the combined case also includes SSAO. Glass transmission and metallic reflection are present throughout. Per-boundary and aggregate thresholds are checked. These measurements test raster consistency, not physical accuracy against a path tracer.
\begin{center}\begin{tabular}{lrrr}\toprule
Case & Mean error & 95th percentile & Maximum \\
\midrule
");
        for case in cases {
            writeln!(
                section,
                r"{} & {:.5} & {:.5} & {:.5} \\",
                case["case"].as_str().unwrap(),
                case["linear_rgb_abs_error_mean"].as_f64().unwrap(),
                case["linear_rgb_abs_error_p95"].as_f64().unwrap(),
                case["linear_rgb_abs_error_max"].as_f64().unwrap()
            )
            .unwrap();
        }
        section.push_str("\\bottomrule\\end{tabular}\\end{center}\nErrors are maximum-channel absolute differences in tone-mapped linear RGB before the composite. Maximum outliers are retained; screen-space and finite shadow-map approximations are not claimed to be exactly invariant across faces. The final composite blends only the guarded overlap. A separate GPU probe executes the production compositor on these rendered textures, approaching each boundary from either side by $10^{-6}$ radians; every illumination case must have maximum discontinuity below 0.002 and mean below 0.0001. Geometric and categorical targets are not blended.\n");
        if r["lighting"]["passed"] == false {
            section.push_str("\\paragraph{Remaining photometric limitation.} The raw, pre-composite boundary budget did not pass for every pair. This result is retained as a failed diagnostic, even though all final-composite continuity checks passed. View-local postprocessing can produce different radiance estimates for the same ray; blending removes a hard boundary, but does not establish global photometric invariance or path-traced accuracy.\n");
        }
    }
    section
}

/// The supplemental bundle has its own manifest so extending the paper never
/// changes the scientific identity of the retained main architectural cohort.
pub fn verify_published(root: &Path) -> Result<()> {
    verify_installed(root, false)
}

/// Before a transactional rebuild, the installed chapter may describe the
/// preceding source. Its bytes must still match its receipt and the caller's
/// top-level attestation. build() then requires current-source inputs; final
/// verification never permits the previous identity.
pub(crate) fn verify_installed(root: &Path, rebuilding: bool) -> Result<()> {
    let output = root.join(DIRECTORY);
    if !output.join("receipt.json").exists() {
        ensure!(
            rebuilding || !root.join(EVIDENCE).join("receipt.json").exists(),
            "lens evidence needs a generated gallery"
        );
        return Ok(());
    }
    let receipt = verify(&output)?;
    // Staged bundles do not contain generator source; their inputs are checked
    // by build() before the atomic page installation.
    if !rebuilding && root.join("Cargo.toml").exists() {
        ensure!(
            receipt["generator_identity"]
                == serde_json::to_value(crate::pipeline::identity(root)?)?,
            "published lens cohort is stale"
        );
        let source_receipt: Value = io::read(&root.join(EVIDENCE).join("receipt.json"))?;
        ensure!(
            receipt == source_receipt,
            "lens gallery differs from its qualification receipt"
        );
    }
    let mut actual = io::files(&output, &output)?;
    actual.remove("publication.json");
    let expected: BTreeMap<String, String> = io::read(&output.join("publication.json"))?;
    ensure!(actual == expected, "lens gallery/paper artifact changed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_accepts_intact_previous_chapter_but_never_tampering_or_missing_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("Cargo.toml"), "[package]\nversion=\"2.0.0\"\n").unwrap();
        let gallery = root.join(DIRECTORY);
        fs::create_dir_all(&gallery).unwrap();
        fs::write(gallery.join("image.png"), b"previous capture").unwrap();
        let receipt = json!({"schema_version":1,"kind":"fisheye_qualification",
            "generator_identity":{"crate_version":"1.0.0"},
            "artifacts":{"image.png":io::hash(&gallery.join("image.png")).unwrap()}});
        io::write(&gallery.join("receipt.json"), &receipt).unwrap();
        io::write(
            &gallery.join("publication.json"),
            &io::files(&gallery, &gallery).unwrap(),
        )
        .unwrap();
        verify_installed(root, true).unwrap();
        assert!(verify_published(root)
            .unwrap_err()
            .to_string()
            .contains("stale"));
        assert!(build(root, root)
            .unwrap_err()
            .to_string()
            .contains("qualification inputs"));
        fs::write(gallery.join("image.png"), b"tampered capture").unwrap();
        assert!(verify_installed(root, true).is_err());
        assert!(verify_published(root).is_err());
    }

    #[test]
    fn initial_evidence_must_be_installed_before_strict_verification() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join(EVIDENCE);
        fs::create_dir_all(&input).unwrap();
        fs::write(input.join("receipt.json"), b"pending input").unwrap();
        verify_installed(dir.path(), true).unwrap();
        assert!(verify_published(dir.path()).is_err());
    }
    #[test]
    fn lighting_receipt_requires_each_boundary_and_final_gpu_composite() {
        let cases: Vec<_> = ["ambient", "directional", "point", "spot", "combined"]
            .iter()
            .map(|name| {
                json!({"case":name,"linear_rgb_abs_error_mean":0.01,"linear_rgb_abs_error_p95":0.02,"linear_rgb_abs_error_max":0.03,
                    "boundaries":vec![json!({"mean":0.01,"p95":0.02});8],
                    "composite_continuity":{"production_wgsl":true,"world_ray_approaches":1400,
                        "linear_rgb_abs_error_mean":0.00001,"linear_rgb_abs_error_max":0.0001}})
            })
            .collect();
        let receipt = json!({"passed":true,"cases":cases});
        assert!(valid_lighting_report(&receipt));
        let mut bad = receipt.clone();
        bad["cases"][2]["boundaries"][3]["mean"] = json!(0.1);
        assert!(!valid_lighting_report(&bad));
        bad["passed"] = json!(false);
        assert!(valid_lighting_report(&bad));
        let mut bad = receipt.clone();
        bad["cases"][4]["composite_continuity"]["linear_rgb_abs_error_max"] = json!(0.03);
        assert!(!valid_lighting_report(&bad));
        let mut bad = receipt;
        bad["cases"][0]["composite_continuity"] = Value::Null;
        assert!(!valid_lighting_report(&bad));
    }

    #[test]
    fn lens_receipt_rejects_changed_pixels_and_path_escape() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("image.png"), b"pixels").unwrap();
        let mut receipt = json!({"schema_version":1,"kind":"fisheye_qualification","artifacts":{"image.png":io::hash(&dir.path().join("image.png")).unwrap()}});
        io::write(&dir.path().join("receipt.json"), &receipt).unwrap();
        verify(dir.path()).unwrap();
        fs::write(dir.path().join("image.png"), b"changed").unwrap();
        assert!(verify(dir.path()).is_err());
        receipt["artifacts"] = json!({"../image.png":"0"});
        io::write(&dir.path().join("receipt.json"), &receipt).unwrap();
        assert!(verify(dir.path()).is_err());
    }
}
