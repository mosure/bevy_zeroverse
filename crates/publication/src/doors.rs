//! Architectural assemblies are a separate inventory from placeable furniture.
//! Capture, category grid, semantics and paper figure share a source-bound receipt.
use crate::{io, pipeline};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::GeneratorIdentity;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, process::Command};
pub const DIRECTORY: &str = "www/project/static/media/doors";
pub const TOOL: &str = "examples/review_doors.rs";
const CATEGORIES: [&str; 10] = [
    "HingedDoor",
    "PairedDoor",
    "SlidingDoor",
    "TelescopingDoor",
    "RevolvingDoor",
    "Cabinet",
    "KitchenCounter",
    "KitchenCupboard",
    "Pantry",
    "KitchenSink",
];
const MODES: [&str; 3] = ["color", "normal", "semantic"];

fn validate(audit: &Value, renders: &Value, identity: &GeneratorIdentity) -> Result<()> {
    for report in [audit, renders] {
        ensure!(
            report["build_provenance"]["source_sha256"] == identity.source_sha256,
            "stale door/cabinet evidence"
        );
        let categories = report["categories"]
            .as_array()
            .context("missing door categories")?;
        ensure!(
            categories.len() == CATEGORIES.len(),
            "incomplete door category inventory"
        );
        for (row, name) in categories.iter().zip(CATEGORIES) {
            ensure!(row["kind"] == name, "reordered/duplicate door inventory");
        }
    }
    ensure!(
        audit["samples_per_category"] == 64,
        "door protocol requires 64 consecutive seeds per category"
    );
    for (i, name) in CATEGORIES.iter().enumerate() {
        let rows = audit["categories"][i]["samples"]
            .as_array()
            .context("missing construction samples")?;
        ensure!(rows.len() == 64, "incomplete door samples");
        for (seed, row) in rows.iter().enumerate() {
            ensure!(
                row["seed"] == seed
                    && row["mesh"]["triangles"].as_u64().is_some_and(|n| n > 0)
                    && row["mesh"]["max_envelope_overrun_m"]
                        .as_f64()
                        .is_some_and(|n| (0.0..=0.003).contains(&n))
                    && row["build_ms"]
                        .as_f64()
                        .is_some_and(|n| n.is_finite() && n >= 0.),
                "invalid {name}/{seed}"
            );
        }
        let selected = renders["categories"][i]["selected"]
            .as_array()
            .context("missing rendered selections")?;
        ensure!(selected.len() == 6, "missing state examples");
        for row in selected {
            let seed = row["seed"].as_u64().context("missing seed")? as usize;
            ensure!(
                seed < 64 && row == &rows[seed],
                "rendered parameters differ from audit"
            );
        }
        if i < 4 {
            let states: Vec<_> = selected[..3]
                .iter()
                .map(|row| row["program"]["program"]["opening"].as_f64())
                .collect();
            ensure!(
                states[0] == Some(0.)
                    && states[1].is_some_and(|v| v > 0.1 && v < 0.7)
                    && states[2].is_some_and(|v| v > 0.85 && v <= 1.),
                "missing shut/ajar/open examples"
            );
        }
    }
    ensure!(
        renders["renderer"].as_str().is_some_and(|s| !s.is_empty()),
        "missing actual renderer provenance"
    );
    for row in renders["categories"].as_array().unwrap() {
        let counts = row["semantic_pixel_counts"]
            .as_array()
            .context("missing semantic checks")?;
        ensure!(counts.len() == 6, "missing semantic views");
        for count in counts {
            let owner = if row["kind"].as_str().is_some_and(|s| s.ends_with("Door")) {
                "door"
            } else {
                "cabinet"
            };
            ensure!(
                count
                    .as_object()
                    .is_some_and(|p| p.values().all(|v| v.as_u64().is_some())
                        && p.values().filter_map(Value::as_u64).sum::<u64>() == 320 * 320),
                "semantic denominator mismatch"
            );
            ensure!(
                count[owner].as_u64().is_some_and(|n| n > 20),
                "missing owner semantic pixels"
            );
        }
    }
    ensure!(
        renders["image_size"] == json!([320, 320]),
        "unexpected door image dimensions"
    );
    Ok(())
}
pub fn prepare(root: &Path, input: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "door output already exists");
    let parent = output.parent().context("no door output parent")?;
    fs::create_dir_all(parent)?;
    let transaction = tempfile::Builder::new()
        .prefix("doors-stage-")
        .tempdir_in(parent)?;
    let staged = transaction.path().join("atlas");
    prepare_at(root, input, &staged)?;
    fs::rename(staged, output)?;
    Ok(())
}
fn prepare_at(root: &Path, input: &Path, output: &Path) -> Result<()> {
    let identity = pipeline::identity(root)?;
    let audit: Value = io::read(&input.join("audit.json"))?;
    let renders: Value = io::read(&input.join("renders.json"))?;
    validate(&audit, &renders, &identity)?;
    ensure!(!output.exists(), "door output already exists");
    fs::create_dir_all(output)?;
    let mut files = BTreeMap::new();
    for name in CATEGORIES {
        for i in 0..6 {
            for mode in MODES {
                let name = format!("{name}-{i}-{mode}.png");
                let image = image::open(input.join(&name))?;
                ensure!(
                    image.width() == 320 && image.height() == 320,
                    "invalid door image {name}"
                );
                fs::copy(input.join(&name), output.join(&name))?;
                files.insert(name.clone(), io::hash(&output.join(name))?);
            }
        }
    }
    for (name, value) in [("audit.json", &audit), ("renders.json", &renders)] {
        io::write(&output.join(name), value)?;
        files.insert(name.into(), io::hash(&output.join(name))?);
    }
    for (stem, start) in [("doors", 0), ("cabinetry", 5)] {
        let mut grid = image::RgbImage::new(960, 1600);
        for row in 0..5 {
            for col in 0..3 {
                let img = image::open(
                    output.join(format!("{}-{col}-color.png", CATEGORIES[start + row])),
                )?
                .to_rgb8();
                image::imageops::replace(&mut grid, &img, col * 320, row as i64 * 320);
            }
        }
        let name = format!("{stem}_grid.png");
        grid.save(output.join(&name))?;
        files.insert(name.clone(), io::hash(&output.join(name))?);
    }
    fs::write(
        output.join("index.html"),
        include_str!("../templates/doors.html"),
    )?;
    files.insert("index.html".into(), io::hash(&output.join("index.html"))?);
    let paper = paper_text()
        .replace("generated/door_doors_grid.png", "doors_grid.png")
        .replace("generated/door_cabinetry_grid.png", "cabinetry_grid.png");
    fs::write(output.join("paper.tex"), format!("\\documentclass{{article}}\n\\usepackage[margin=22mm]{{geometry}}\n\\usepackage{{graphicx}}\n\\begin{{document}}\n\\section{{Doors and cabinetry}}\n{paper}\n\\end{{document}}\n"))?;
    files.insert("paper.tex".into(), io::hash(&output.join("paper.tex"))?);
    io::write(
        &output.join("atlas.json"),
        &json!({"schema_version":1,"generator_identity":identity,"tool_sha256":io::hash(&root.join(TOOL))?,"semantic_palette":renders["build_provenance"]["semantic_palette"],"samples_per_category":64,"categories":renders["categories"],"files":files,
        "protocol":"64 consecutive construction seeds per category; six illustrative state/material examples, all three aligned channels; construction timings exclude materials/uploads/rendering. Cabinet dimensions include moving-front travel; door studio envelopes fixed. Counts are not independent combinations or training-utility estimates."}),
    )?;
    verify(output, &identity)?;
    Ok(())
}
pub fn verify(dir: &Path, identity: &GeneratorIdentity) -> Result<Value> {
    let atlas: Value = io::read(&dir.join("atlas.json"))?;
    ensure!(
        atlas["generator_identity"] == json!(identity),
        "door atlas generator mismatch"
    );
    let audit = io::read(&dir.join("audit.json"))?;
    let renders = io::read(&dir.join("renders.json"))?;
    validate(&audit, &renders, identity)?;
    ensure!(
        atlas["categories"] == renders["categories"],
        "door atlas examples mismatch"
    );
    let files: BTreeMap<String, String> = serde_json::from_value(atlas["files"].clone())?;
    let mut expected = std::collections::BTreeSet::<String>::new();
    for name in CATEGORIES {
        for i in 0..6 {
            for mode in MODES {
                expected.insert(format!("{name}-{i}-{mode}.png"));
            }
        }
    }
    expected.extend(
        [
            "audit.json",
            "renders.json",
            "doors_grid.png",
            "cabinetry_grid.png",
            "index.html",
            "paper.tex",
        ]
        .map(str::to_owned),
    );
    ensure!(
        files
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            == expected,
        "door asset closure incomplete"
    );
    for (name, hash) in files {
        ensure!(
            !name.contains('/') && !name.contains('\\') && name != "..",
            "invalid door asset path"
        );
        ensure!(io::hash(&dir.join(name))? == hash, "changed door asset");
    }
    Ok(atlas)
}
pub fn refresh_cache(root: &Path, cache: &Path, identity: &GeneratorIdentity) -> Result<()> {
    if let Ok(atlas) = verify(cache, identity) {
        if atlas["tool_sha256"] == io::hash(&root.join(TOOL))? {
            return Ok(());
        }
    }
    let work = tempfile::Builder::new()
        .prefix("doors-")
        .tempdir_in(cache.parent().context("no cache parent")?)?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    ensure!(
        Command::new(cargo)
            .current_dir(root)
            .args([
                "run",
                "--locked",
                "--no-default-features",
                "--features",
                "multi_threaded",
                "--example",
                "review_doors",
                "--",
                "--output"
            ])
            .arg(work.path().join("capture"))
            .status()?
            .success(),
        "door qualification failed"
    );
    let prepared = work.path().join("prepared");
    prepare(root, &work.path().join("capture"), &prepared)?;
    if cache.exists() {
        fs::remove_dir_all(cache)?;
    }
    fs::rename(prepared, cache)?;
    Ok(())
}
pub fn build(cache: &Path, stage: &Path, identity: &GeneratorIdentity) -> Result<()> {
    verify(cache, identity)?;
    let output = stage.join(DIRECTORY);
    if output.exists() {
        fs::remove_dir_all(&output)?;
    }
    io::copy_tree(cache, &output)?;
    write_paper(stage)
}
pub fn write_paper(stage: &Path) -> Result<()> {
    let atlas: Value = io::read(&stage.join(DIRECTORY).join("atlas.json"))?;
    let categories = atlas["categories"]
        .as_array()
        .context("missing door atlas")?;
    ensure!(categories.len() == 10, "door paper inventory incomplete");
    let generated = stage.join("tex/generated");
    fs::create_dir_all(&generated)?;
    for stem in ["doors", "cabinetry"] {
        fs::copy(
            stage.join(DIRECTORY).join(format!("{stem}_grid.png")),
            generated.join(format!("door_{stem}_grid.png")),
        )?;
    }
    fs::write(generated.join("door_evaluation.tex"), paper_text())?;
    Ok(())
}
fn paper_text() -> &'static str {
    r#"\subsection{Door and cabinetry construction}
Doors form a separate architectural inventory from placeable furniture. Hinged, paired, surface-sliding, telescoping and revolving mechanisms sample continuous states, handedness, dimensions, panel construction and hardware finishes. Revolving entrances require a wider aperture and are reserved as static obstacles; the generator does not simulate automatic door operation. Arch frames have no door-leaf annotation. Cabinet fronts mix hinged, sliding and upper lift-up mechanisms; lower and upper cupboard states share a recorded construction program. Opening travel is admitted before placement. RGB, depth, normals and semantic meshes use the same geometry. Door leaves, frames and hardware form one bounding-box instance, with distinct semantic classes; cabinet glazing retains cabinet ownership and cabinet pulls have their own hardware label.
The category audit checks 64 consecutive samples in each of ten categories (640 assemblies). Six illustrative samples per category are captured in aligned PBR, normal and semantic channels. They illustrate state/material coverage rather than occurrence rates. Cabinet envelopes include opening travel; consequently these are not fixed-envelope perceptual diversity estimates. Triangle checks cover finite attributes, winding, labels and bounds; they do not certify architectural codes or physical dynamics.
\begin{figure}[p]\centering\includegraphics[width=.87\linewidth,height=.78\textheight,keepaspectratio]{generated/door_doors_grid.png}\caption{Architectural door construction. Rows: hinged, paired, sliding, telescoping and revolving. The first four rows show sampled shut, ajar and open states; revolving rows show static rotor phases. Panel and finish parameters also vary.}\end{figure}
\begin{figure}[p]\centering\includegraphics[width=.87\linewidth,height=.78\textheight,keepaspectratio]{generated/door_cabinetry_grid.png}\caption{Cabinetry construction. Rows: office storage, counter base, base-and-overhead cupboard, pantry and sink base. Columns are prespecified seeds 0, 3 and 7, without visual filtering. All six examples and semantic/normal channels are available in the project-page category viewer.}\end{figure}
"#
}
pub fn page() -> &'static str {
    r#"<section id="doors" class="container section"><div class="section-heading"><div><div class="eyebrow">ARCHITECTURAL ASSEMBLIES</div><h2>Doors, hardware<br>and cabinetry.</h2></div><p>Explore construction states and aligned semantic labels. Doors belong to openings; cabinet fronts retain the identity of their supporting furniture.</p></div><iframe title="Door and cabinet construction atlas" src="static/media/doors/index.html" loading="lazy" style="width:100%;height:920px;border:0;border-radius:18px"></iframe><p><a href="static/media/doors/index.html">Open the full category viewer</a> · <a href="static/media/doors/atlas.json">Download source-bound measurements</a></p></section>"#
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_or_missing_door_evidence_fails_closed() {
        let id = GeneratorIdentity {
            schema_version: 1,
            crate_version: "0".into(),
            generator_version: 45,
            source_sha256: "current".into(),
        };
        assert!(validate(&json!({}), &json!({}), &id).is_err());
    }
    #[test]
    fn complete_protocol_rejects_missing_states_mesh_and_semantic_evidence() {
        let id = GeneratorIdentity {
            schema_version: 1,
            crate_version: "test".into(),
            generator_version: 45,
            source_sha256: "current".into(),
        };
        let audit_categories: Vec<_> = CATEGORIES.iter().map(|name| {
            let samples: Vec<_> = (0..64).map(|seed| json!({"seed":seed,"mesh":{"triangles":100,"max_envelope_overrun_m":0.},"build_ms":1.,"program":{"program":{"opening":([0.,0.4,0.95][seed%3])}}})).collect();
            json!({"kind":name,"samples":samples})
        }).collect();
        let render_categories: Vec<_> = audit_categories.iter().map(|row| {
            let owner = if row["kind"].as_str().unwrap().ends_with("Door") { "door" } else { "cabinet" };
            json!({"kind":row["kind"],"selected":&row["samples"].as_array().unwrap()[..6],"semantic_pixel_counts":vec![json!({owner: 400, "floor":102000});6]})
        }).collect();
        let audit = json!({"build_provenance":{"source_sha256":"current"},"samples_per_category":64,"categories":audit_categories});
        let renders = json!({"build_provenance":{"source_sha256":"current"},"renderer":"fixture","image_size":[320,320],"categories":render_categories});
        validate(&audit, &renders, &id).unwrap();
        let mut broken = renders.clone();
        broken["categories"][0]["selected"][0] = audit["categories"][0]["samples"][2].clone();
        assert!(validate(&audit, &broken, &id).is_err());
        let mut broken = audit.clone();
        broken["categories"][9]["samples"][63]["mesh"]["max_envelope_overrun_m"] = json!(0.1);
        assert!(validate(&broken, &renders, &id).is_err());
        let mut broken = renders.clone();
        broken["categories"][0]["semantic_pixel_counts"][0] = json!({"floor":102400});
        assert!(validate(&audit, &broken, &id).is_err());
    }
}
