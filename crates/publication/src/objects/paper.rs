use super::*;
use crate::graphics;
use base64::{engine::general_purpose::STANDARD, Engine};

fn tex(s: &str) -> String {
    s.replace('_', "\\_ ")
}

pub(super) fn write(stage: &Path) -> Result<()> {
    let directory = stage.join(DIRECTORY);
    let atlas: Value = io::read(&directory.join("atlas.json"))?;
    let categories = atlas["categories"]
        .as_array()
        .context("missing categories")?;
    let mut text = format!(
        r"% Generated from the current checksum-bound object atlas.
\section{{Object geometry diversity}}
\label{{sec:object-diversity}}
\subsection{{Fixed-envelope protocol}}
We evaluate all 64 furnishing/prop categories on {SAMPLES} consecutive construction seeds each: {} triangle assemblies. Each category holds its metric placement envelope, orientation and finish seed fixed. This isolates internal construction from changes in object size, pose or color; it deliberately does not measure the full joint scene distribution. The cable fixture varies a controlled centerline; actual endpoint connectivity and collision are separate room-level checks. Humans and architecture are evaluated elsewhere.

For each mesh, six orthographic first-surface depth maps sample pixel centers at $24\times24$ resolution per direction. Depth is normalized by the fixed envelope extent. Pair distance is mean absolute depth difference over the union of hit pixels; missing-versus-present surface contributes one. Greedy coverage retains a sample only when its distance to every earlier representative exceeds the threshold. We report thresholds 0.005, 0.015 and 0.030, and retain seed order. These are finite-resolution, order-dependent coverage counts, not numbers of independent geometries, perceptual embedding volume, or a proof of a multiplicative increase in dataset utility. Thin features and content on nearly planar surfaces can be under-resolved; saturated categories need more samples or a finer descriptor.

All audited assemblies passed finite position/normal/UV, unit-normal, winding, nondegenerate-triangle, index, semantic-label and reserved-envelope checks (maximum allowed overrun 3 mm). This does not prove watertightness, lack of every self-intersection, or physical plausibility. Separate regression tests cover support, cable attachment, overlapping planar faces and deterministic replay. Mesh build time excludes placement, materials, uploads and rendering and is not completed-room throughput.

Six examples per category are selected by deterministic farthest-spread sampling from the same depth descriptors, beginning with seed zero. The native renderer reuses one device, with fixed cameras, exposure, lights and finish seed within each category. The project-page atlas provides RGB and normal views, every example's program inputs, all sampled parameter marginals and the complete machine-readable measurements. These selected examples are correlated draws, not one-parameter interventions; inactive conditional controls are not evidence of realized shape diversity.

\subsection{{Category measurements}}
\small
\begin{{longtable}}{{lrrr r r}}
\caption{{Current construction coverage at three distances; triangles and CPU build milliseconds are per-object medians. All categories use {SAMPLES} seeds.}}\\
\toprule Category & $C_{{.005}}$ & $C_{{.015}}$ & $C_{{.030}}$ & Triangles & ms \\\midrule
\endfirsthead
\toprule Category & $C_{{.005}}$ & $C_{{.015}}$ & $C_{{.030}}$ & Triangles & ms \\\midrule
\endhead
",
        categories.len() * SAMPLES
    );
    for c in categories {
        let counts: Vec<_> = c["coverage"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["representatives"].as_array().unwrap().len())
            .collect();
        text.push_str(&format!(
            "{} & {} & {} & {} & {:.0} & {:.3} \\\\\n",
            tex(c["kind"].as_str().unwrap()),
            counts[0],
            counts[1],
            counts[2],
            c["mesh_summary"]["triangles"]["median"].as_f64().unwrap(),
            c["build_ms_summary"]["median"].as_f64().unwrap()
        ));
    }
    text.push_str("\\bottomrule\n\\end{longtable}\n\\normalsize\n");
    let groups = [
        ["Chair", "Table", "Sofa", "Plant"],
        ["Keyboard", "Mouse", "FloorLamp", "CoatRack"],
        ["StorageBox", "Printer", "Bag", "Headphones"],
        ["WaterBottle", "Microphone", "WallSconce", "FloorOutlet"],
    ];
    for (index, kinds) in groups.iter().enumerate() {
        let mut svg = graphics::begin(1440, 1100);
        for (row, kind) in kinds.iter().enumerate() {
            let c = categories.iter().find(|c| c["kind"] == *kind).unwrap();
            let y = row as u32 * 270;
            graphics::text(&mut svg, 8., (y + 24) as f64, 21, kind);
            for i in 0..6 {
                let file = directory.join(format!("{kind}-{i}-color.webp"));
                let data = STANDARD.encode(fs::read(file)?);
                let x = i * 240;
                svg.push_str(&format!("<image x=\"{x}\" y=\"{}\" width=\"230\" height=\"218\" href=\"data:image/webp;base64,{data}\"/>",y+32));
                graphics::text(
                    &mut svg,
                    (x + 8) as f64,
                    (y + 266) as f64,
                    17,
                    &format!("seed {}", c["selected_seeds"][i]),
                );
            }
        }
        svg.push_str("</g></svg>");
        graphics::save(
            &svg,
            &stage.join(format!("tex/generated/object_plate_{index}.png")),
        )?;
        text.push_str(&format!(r"\begin{{figure}}[p]
\centering\includegraphics[width=\linewidth]{{generated/object_plate_{index}.png}}
\caption{{Matched studio construction examples: {}. All columns within each row share dimensions, viewpoint, lights and finish seed. Seed labels link to the full sampled parameter grid in the downloadable atlas. Example selection emphasizes shape spread, not frequency or image quality; the table includes every sampled seed.}}
\end{{figure}}
", kinds.join(", ")));
    }
    // Couple images to their actual inputs, not to an independently sampled grid.
    let lamp = categories
        .iter()
        .find(|c| c["kind"] == "FloorLamp")
        .unwrap();
    let samples: Vec<Value> = io::read(&directory.join("FloorLamp-samples.json"))?;
    let mut svg = graphics::begin(1440, 280);
    for i in 0..6 {
        let data = STANDARD.encode(fs::read(
            directory.join(format!("FloorLamp-{i}-color.webp")),
        )?);
        let x = i * 240;
        svg.push_str(&format!("<image x=\"{x}\" y=\"0\" width=\"230\" height=\"240\" href=\"data:image/webp;base64,{data}\"/>"));
        graphics::text(
            &mut svg,
            (x + 12) as f64,
            267.,
            19,
            &format!("seed {}", lamp["selected_seeds"][i]),
        );
    }
    svg.push_str("</g></svg>");
    graphics::save(&svg, &stage.join("tex/generated/object_parameter_grid.png"))?;
    text.push_str(
        r"\begin{figure}[p]
\centering\includegraphics[width=\linewidth]{generated/object_parameter_grid.png}
\smallskip
\small\begin{tabular}{lrrrrrr}\toprule
Construction seed",
    );
    for seed in lamp["selected_seeds"].as_array().unwrap() {
        text.push_str(&format!(" & {seed}"));
    }
    text.push_str(" \\\\\n\\midrule\n");
    for (key, label) in [
        ("shade_height", "Shade height (m)"),
        ("shade_radius_fraction", "Radius / envelope width"),
        ("shade_taper", "Upper / lower radius"),
        ("stem_offset", "Stem offset / width"),
        ("shade_sides", "Shade sides"),
        ("support", "Support code"),
    ] {
        text.push_str(label);
        for seed in lamp["selected_seeds"].as_array().unwrap() {
            let sample = samples
                .iter()
                .find(|s| s["object"]["seed"] == *seed)
                .unwrap();
            let v = sample["program"][key].as_f64().unwrap();
            text.push_str(&if matches!(key, "shade_sides" | "support") {
                format!(" & {v:.0}")
            } else {
                format!(" & {v:.3}")
            });
        }
        text.push_str(" \\\\\n");
    }
    text.push_str(r"\bottomrule\end{tabular}\normalsize
\caption{A construction parameter grid with values taken from the pictured meshes. Each lamp occupies the same $0.55\times1.65\times0.55$ m reserved envelope. Support code 0 is a disk and stem, 1 a wooden tripod, and 2 four metal legs. A 32-sided shade approximates a circular section; smaller counts produce polygonal shades. Stem offset and shade radius are jointly bounded by the reserved envelope. The bulb and light emitter use the same computed position. This grid shows correlated sampled programs, not isolated causal effects.}
\end{figure}
\clearpage
");
    text.push_str(r"\clearpage
\subsection{Sampled controls and their interpretation}
The following inventory lists the sampled construction inputs recorded by each builder. Some controls are conditional on a construction choice (for example, a phone's rear lenses affect only face-down configurations); a range does not imply independent influence. Observed ranges, eight-bin numeric marginals, categorical frequencies and all seed-level values accompany the project-page grids. A small coverage count for a flat notebook, notice board or outlet is retained explicitly rather than counting different textures as different geometry.
\small
\begin{longtable}{p{.20\linewidth}p{.74\linewidth}}
\toprule Category & Recorded controls \\\midrule
\endfirsthead
\toprule Category & Recorded controls \\\midrule
\endhead

");
    for c in categories {
        let keys: Vec<_> = c["parameters"]
            .as_object()
            .unwrap()
            .keys()
            .map(|s| tex(s))
            .collect();
        text.push_str(&format!(
            "{} & {} \\\\\n",
            c["kind"].as_str().unwrap(),
            keys.join(", ")
        ));
    }
    text.push_str("\\bottomrule\n\\end{longtable}\n\\normalsize\n");
    fs::write(stage.join("tex/generated/object_evaluation.tex"), text)?;
    Ok(())
}
