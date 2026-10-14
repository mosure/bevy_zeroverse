//! Figures and paper fragment derived from the same verified gallery receipt.
use crate::{graphics, io};
use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};

pub(super) fn write(directory: &Path, report: &Value) -> Result<()> {
    let scenes = report["scenes"].as_array().context("missing scenes")?;
    let mut groups = BTreeMap::<&str, Vec<&Value>>::new();
    for s in scenes {
        groups
            .entry(s["profile"].as_str().context("missing profile")?)
            .or_default()
            .push(s);
    }
    let rows = groups.len().div_ceil(3) as u32;
    let mut plate = graphics::begin(1020, rows * 338 + 70);
    graphics::text(&mut plate, 20., 30., 22, "Functional indoor programs");
    graphics::text(&mut plate,20.,54.,12,"First consecutive room per profile, camera 0. Illustration only; every room is retained in the gallery.");
    let mut chart = graphics::begin(1020, groups.len() as u32 * 60 + 116);
    graphics::text(
        &mut chart,
        20.,
        30.,
        22,
        "Realized content and capture latency",
    );
    graphics::text(&mut chart,20.,54.,12,"Aligned RGB and geometry annotations. Shared GPU; elapsed capture includes validation and image export.");
    let mut tex=String::from("% Generated from verified current-source captures. Local review, not a release attestation.\n\\subsection{Functional indoor programs}\nThe generator composes transit, service, dining, teaching and exhibition programs with required objects, reserved circulation, support levels and explicit assembly membership.\n\\begin{center}\\begin{tabular}{lrrrr}\nProgram & Rooms & Objects/room & People/room & Capture P50 (s) \\\\\n\\hline\n");
    for (i, (profile, group)) in groups.iter().enumerate() {
        let s = group[0];
        let file = directory
            .join(s["path"].as_str().context("missing capture path")?)
            .join("view_00_color.png");
        let image = STANDARD.encode(fs::read(file)?);
        let x = 20 + (i % 3) * 338;
        let y = 72 + (i / 3) * 338;
        plate.push_str(&format!("<image x=\"{x}\" y=\"{y}\" width=\"316\" height=\"292\" preserveAspectRatio=\"xMidYMid meet\" href=\"data:image/png;base64,{image}\"/>"));
        graphics::text(
            &mut plate,
            x as f64,
            y as f64 + 314.,
            16,
            &format!("{profile} · seed {}", s["seed"]),
        );
        let count = group.len() as f64;
        let objects = group
            .iter()
            .map(|s| {
                s["objects"]
                    .as_object()
                    .unwrap()
                    .values()
                    .filter_map(Value::as_f64)
                    .sum::<f64>()
            })
            .sum::<f64>()
            / count;
        let people = group
            .iter()
            .filter_map(|s| s["people"].as_f64())
            .sum::<f64>()
            / count;
        let mut times: Vec<_> = group.iter().filter_map(|s| s["seconds"].as_f64()).collect();
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        let y = 100 + i * 60;
        graphics::text(&mut chart, 20., y as f64, 16, profile);
        chart.push_str(&format!(
            "<rect x=\"210\" y=\"{}\" width=\"{}\" height=\"22\" fill=\"#147d70\"/>",
            y - 18,
            objects * 5.
        ));
        graphics::text(
            &mut chart,
            220. + objects * 5.,
            y as f64,
            14,
            &format!("{objects:.1} objects · {people:.1} people · {median:.2} s"),
        );
        tex.push_str(&format!(
            "{} & {} & {objects:.1} & {people:.1} & {median:.2} \\\\\n",
            profile,
            group.len()
        ));
    }
    tex.push_str("\\end{tabular}\\end{center}\nCapture time includes annotation validation and image export on a GPU shared with concurrent workloads; it is not isolated JIT tensor throughput. Required/accepted capacity, content counts, calibration and directed rendered overlap remain in the accompanying machine-readable receipt. Statistical coverage does not establish photographic equivalence or downstream learning gains.\n\\begin{figure}[t]\\centering\\includegraphics[width=\\linewidth]{program_plate.png}\\caption{Current functional programs: first consecutive room per family, with no image-quality selection.}\\end{figure}\n");
    plate.push_str("</g></svg>");
    chart.push_str("</g></svg>");
    graphics::save(&plate, &directory.join("program_plate.svg"))?;
    graphics::save(&chart, &directory.join("program_metrics.svg"))?;
    fs::write(directory.join("programs.tex"), tex)?;
    io::write(
        &directory.join("figure_sources.json"),
        &serde_json::json!({"identity":report["identity"],"selection":"First consecutive room per profile, camera zero; illustrative only","rooms":report["rooms"],"views":report["views"]}),
    )?;
    Ok(())
}
