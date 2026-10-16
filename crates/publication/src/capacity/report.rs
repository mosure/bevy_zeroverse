//! Current-state evidence report. All figures and paper rows come from the same
//! validated immutable receipt; no hand-entered diversity numbers or old deltas.
use super::*;
use crate::{graphics, io::html};
use std::fmt::Write;

fn tex(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\\' => "\\textbackslash{}".into(),
            '_' => "\\_".into(),
            '%' => "\\%".into(),
            '&' => "\\&".into(),
            '#' => "\\#".into(),
            '{' => "\\{".into(),
            '}' => "\\}".into(),
            '$' => "\\$".into(),
            '^' => "\\textasciicircum{}".into(),
            '~' => "\\textasciitilde{}".into(),
            c => c.to_string(),
        })
        .collect()
}
fn number(v: Option<f64>) -> String {
    v.map_or_else(|| "unqualified".into(), |v| format!("{v:.2}"))
}

pub fn write(receipt_path: &Path, output: &Path) -> Result<()> {
    let receipt: CapacityReceipt = serde_json::from_slice(&fs::read(receipt_path)?)?;
    validate_receipt(&receipt.plan, &receipt).map_err(anyhow::Error::msg)?;
    ensure!(!output.exists(), "capacity report already exists");
    fs::create_dir_all(output)?;
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    let e = &receipt.estimate;
    let p = &receipt.plan.population;
    let c = &p.capture;
    let mut rows = String::new();
    let mut paper_rows = String::new();
    let mut svg = graphics::begin(1000, 570);
    graphics::text(
        &mut svg,
        32.,
        35.,
        22,
        "Observed joint diversity and conditional coverage",
    );
    graphics::text(
        &mut svg,
        32.,
        62.,
        13,
        "Exact observed-cohort rarefaction; this curve does not extrapolate unseen rooms.",
    );
    svg.push_str("<path d=\"M65,410 L490,410 M65,410 L65,95\" fill=\"none\" stroke=\"#678078\"/>");
    let maximum = e
        .definitions
        .iter()
        .map(|d| d.rooms)
        .max()
        .unwrap_or(1)
        .max(1);
    for i in 0..=4 {
        let fraction = f64::from(i) / 4.;
        let label = format!("{:.0}", fraction * f64::from(maximum));
        graphics::text(&mut svg, 58. + 425. * fraction, 430., 11, &label);
        graphics::text(&mut svg, 18., 414. - 305. * fraction, 11, &label);
    }
    graphics::text(&mut svg, 155., 455., 12, "Draws from the observed cohort");
    graphics::text(&mut svg, 65., 85., 12, "Distinct observed classes");
    let colors = ["#147d70", "#626aca", "#c56a33", "#aa4984"];
    for (i, d) in e.definitions.iter().enumerate() {
        let color = colors[i % colors.len()];
        let points = d
            .rarefaction
            .iter()
            .filter_map(|v| {
                v.distinct.point.map(|u| {
                    format!(
                        "{:.2},{:.2}",
                        65. + 425. * v.draws as f64 / f64::from(maximum),
                        410. - 305. * u / f64::from(maximum)
                    )
                })
            })
            .collect::<Vec<_>>()
            .join(" ");
        if i < 4 {
            write!(svg,"<polyline points=\"{points}\" stroke=\"{color}\" fill=\"none\" stroke-width=\"2.5\"/>")?;
            graphics::text(
                &mut svg,
                65.,
                490. + i as f64 * 17.,
                12,
                &format!(
                    "{}: {} classes / {} rooms",
                    d.definition.id, d.observed_classes, d.rooms
                ),
            );
        }
        write!(rows,"<tr><td>{}</td><td>{:?}</td><td>{} / {}</td><td>{}</td><td>{} to {}</td><td>{}</td></tr>",html(&d.definition.id),d.definition.descriptor,d.observed_classes,d.rooms,d.matching_disjoint_pairs,number(d.requested_occupancy.distinct.lower),number(d.requested_occupancy.distinct.upper),d.max_qualified_forecast_draws)?;
        writeln!(
            paper_rows,
            r"{} & {} & {} & {} & {} \\",
            tex(&d.definition.id),
            d.rooms,
            d.observed_classes,
            d.disjoint_pairs,
            number(d.collision_diversity.lower)
        )?;
    }
    if e.definitions.len() > 4 {
        graphics::text(
            &mut svg,
            530.,
            490.,
            12,
            "First four definitions plotted; all definitions appear in the table.",
        );
    }
    let mut profiles: Vec<_> = e
        .diagnostics
        .iter()
        .filter(|(k, _)| k.starts_with("profile/"))
        .collect();
    profiles.sort_by_key(|a| std::cmp::Reverse(a.1.nonzero_rooms));
    let max_profile = profiles
        .iter()
        .map(|(_, d)| d.nonzero_rooms)
        .max()
        .unwrap_or(1)
        .max(1);
    if profiles.is_empty() {
        rendered_coverage(&mut svg, e)?;
    } else {
        graphics::text(&mut svg, 530., 82., 12, "Accepted rooms per program");
    }
    for (i, (name, d)) in profiles.iter().take(20).enumerate() {
        let y = 102. + i as f64 * 18.;
        graphics::text(
            &mut svg,
            530.,
            y,
            11,
            name.strip_prefix("profile/").unwrap_or(name),
        );
        let w = 180. * f64::from(d.nonzero_rooms) / f64::from(max_profile);
        write!(
            svg,
            "<rect x=\"725\" y=\"{}\" width=\"{}\" height=\"10\" fill=\"#147d70\"/>",
            y - 9.,
            w
        )?;
        graphics::text(&mut svg, 920., y, 11, &d.nonzero_rooms.to_string());
    }
    svg.push_str("</g></svg>");
    graphics::save(&svg, &output.join("capacity.svg"))?;
    let mut factors = String::new();
    for f in &receipt.plan.factors {
        write!(
            factors,
            "<tr><td>{}</td><td>{:?}</td><td>{}</td></tr>",
            html(&f.id),
            f.state,
            html(&f.evidence_scope)
        )?;
    }
    let mut coverage = String::new();
    for (name, d) in &e.diagnostics {
        write!(
            coverage,
            "<tr><td>{}</td><td>{}/{}</td><td>{:.3}</td><td>{:.3}–{:.3}</td></tr>",
            html(name),
            d.nonzero_rooms,
            d.rooms,
            d.sum / f64::from(d.rooms.max(1)),
            d.min,
            d.max
        )?;
    }
    let limitations = e
        .limitations
        .iter()
        .map(|s| format!("<li>{}</li>", html(s)))
        .collect::<String>();
    let document = format!(
        r#"<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zeroverse · capacity evidence</title>
<style>body{{font:16px/1.55 system-ui;background:#f7f8f3;color:#20382e;max-width:1150px;margin:3rem auto;padding:0 1.4rem}}h1{{font-size:2.5rem;letter-spacing:-.04em}}p{{max-width:90ch}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{padding:.6rem;border-bottom:1px solid #d8e2d9;text-align:left}}th{{background:#edf1e9}}img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}.card{{background:white;border:1px solid #d8e2d9;border-radius:16px;padding:1.5rem;margin:1.3rem 0}}.scroll{{overflow:auto}}summary{{cursor:pointer;font-weight:650}}a{{color:#147d70}}</style>
<p>BEVY ZEROVERSE / LOCAL QUALIFICATION</p><h1>What this population actually produces</h1>
<p>{accepted} accepted / {requested} requested rooms · {failed} retained failures · {meshes} actual mesh checks · {rendered} completed rendered rooms. Sample unit: {unit:?}. {views} views × {width}×{height}, {times} captured times per room.</p>
<div class="card"><img src="capacity.svg" alt="Observed-cohort rarefaction and accepted room program coverage"><p>Seed-family groups: {families}. Views, annotations and repeated static frames are not independent rooms. Constructed surface area is not pixel coverage.</p></div>
<div class="card scroll"><h2>Observed classes and conservative bounds</h2><p>Requested horizon: {horizon} accepted draws. The interval is a distribution-free bound, not a fitted forecast. Zero extrapolation horizon means that no long-run novelty model is qualified. Bounds use disjoint room pairs, adjusted across prespecified definitions.</p><table><tr><th>Definition</th><th>Representation</th><th>Classes / rooms</th><th>Pair matches</th><th>Expected distinct bound</th><th>Forecast horizon</th></tr>{rows}</table></div>
<div class="card"><h2>Captured factor contract</h2><table><tr><th>Factor</th><th>State</th><th>Evidence scope</th></tr>{factors}</table></div>
<div class="card"><details><summary>Accepted objects, people, materials and functional coverage</summary><div class="scroll"><table><tr><th>Diagnostic</th><th>Nonzero / rooms</th><th>Mean including zero</th><th>Range</th></tr>{coverage}</table></div></details></div>
<div class="card"><h2>Qualification limits</h2><ul>{limitations}</ul><p>Photographic accuracy, browser performance and downstream learning utility require separate qualified experiments. This report does not certify them.</p><p>Generator {generator}, source <code>{source}</code><br>Configuration <code>{config}</code><br>Capture <code>{engine}</code><br>Records <code>{records}</code></p><p><a href="receipt.json">Machine-readable receipt</a> · <a href="capacity.png">Figure PNG</a> · <a href="capacity.tex">Paper table</a></p></div></html>"#,
        accepted = e.accepted_rooms,
        requested = e.requested_rooms,
        failed = e.failed_rooms,
        meshes = e.mesh_checked_rooms,
        rendered = e.rendered_rooms,
        unit = receipt.plan.request.unit,
        views = c.cameras,
        width = c.image_size[0],
        height = c.image_size[1],
        times = c.progress.len(),
        families = e.unique_families,
        horizon = receipt.plan.request.requested_draws,
        generator = p.generator.generator_version,
        source = p.generator.source_sha256,
        config = p.config_sha256,
        engine = html(&p.capture_engine),
        records = receipt.records_sha256
    );
    fs::write(output.join("index.html"), document)?;
    let paper = format!(
        r"% Generated from an immutable capacity receipt; no extrapolated population point.
\begin{{table}}[htbp]\centering\small
\begin{{tabular}}{{lrrrr}}\toprule
Definition & Rooms & Classes & Pairs & $D_2$ lower\\\midrule
{paper_rows}\bottomrule\end{{tabular}}
\caption{{Current configuration qualification: {} accepted of {} requested rooms, {} actual mesh checks and {} rendered rooms. Bounds condition on successful independent draws. Fixed/consecutive runs have no population confidence claim. Source \texttt{{{}}}; configuration \texttt{{{}}}. Counts are joint bins at the declared resolution, not perceptual support or learning capacity. No unseen-tail forecast is qualified.}}\label{{tab:capacity-observed}}
\end{{table}}
",
        e.accepted_rooms,
        e.requested_rooms,
        e.mesh_checked_rooms,
        e.rendered_rooms,
        &p.generator.source_sha256[..12],
        &p.config_sha256[..12]
    );
    fs::write(output.join("capacity.tex"), paper)?;
    Ok(())
}

fn rendered_coverage(svg: &mut String, estimate: &CapacityEstimate) -> Result<()> {
    graphics::text(svg, 530., 82., 12, "Rendered coverage");
    if let (Some(primary), Some(valid)) = (
        estimate.diagnostics.get("rendered/primary_pixels"),
        estimate.diagnostics.get("rendered/valid_pixels"),
    ) {
        graphics::text(
            svg,
            530.,
            110.,
            12,
            &format!(
                "Primary region: {:.1}% of valid pixels",
                100. * primary.sum / valid.sum.max(1.)
            ),
        );
    }
    let mut rows: Vec<_> = estimate
        .diagnostics
        .iter()
        .filter_map(|(name, value)| {
            name.strip_prefix("rendered/view_")?
                .strip_suffix("/shared_fraction")?
                .parse::<usize>()
                .ok()
                .map(|index| (index, value))
        })
        .collect();
    rows.sort_by_key(|row| row.0);
    graphics::text(
        svg,
        530.,
        139.,
        12,
        "Shared visibility: mean and room range",
    );
    for (i, (index, value)) in rows.iter().take(12).enumerate() {
        let y = 166. + i as f64 * 22.;
        let mean = value.sum / f64::from(value.rooms.max(1));
        graphics::text(svg, 530., y + 4., 11, &format!("View row {index}"));
        write!(svg,"<path d=\"M680,{y} H900\" stroke=\"#d8e2d9\"/><path d=\"M{},{y} H{}\" stroke=\"#678078\" stroke-width=\"3\"/><circle cx=\"{}\" cy=\"{y}\" r=\"4\" fill=\"#147d70\"/>",680.+220.*value.min,680.+220.*value.max,680.+220.*mean)?;
        graphics::text(svg, 914., y + 4., 11, &format!("{:.1}%", mean * 100.));
    }
    let y = 187. + rows.len().min(12) as f64 * 22.;
    graphics::text(svg, 680., y, 11, "0%");
    graphics::text(svg, 875., y, 11, "100%");
    graphics::text(
        svg,
        530.,
        y + 27.,
        11,
        "Rows identify captured camera/time observations.",
    );
    if rows.is_empty() {
        graphics::text(
            svg,
            530.,
            y + 48.,
            11,
            "Shared visibility was not measured in this cohort.",
        );
    } else if rows.len() > 12 {
        graphics::text(
            svg,
            530.,
            y + 48.,
            11,
            "First 12 rows shown; all rows are in the receipt.",
        );
    }
    Ok(())
}
