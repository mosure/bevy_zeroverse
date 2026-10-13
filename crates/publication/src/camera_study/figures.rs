//! Native SVG/raster figures share the portable camera-study measurements.
use super::{BINS, DIRECTORY};
use crate::{graphics, io};
use anyhow::Result;
use serde_json::Value;
use std::{fs, path::Path};

const COLORS: [&str; 3] = ["#147d70", "#626aca", "#c56a33"];
fn number(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.)
}
fn end(mut svg: String) -> String {
    svg.push_str("</g></svg>");
    svg
}
fn path(points: &[[f64; 2]]) -> String {
    points
        .iter()
        .enumerate()
        .map(|(i, p)| format!("{} {:.2},{:.2}", if i == 0 { "M" } else { "L" }, p[0], p[1]))
        .collect::<Vec<_>>()
        .join(" ")
}
fn save(
    svg: &str,
    destination: &Path,
    stage: &Path,
    name: &str,
    tex_name: Option<&str>,
) -> Result<()> {
    graphics::save(svg, &destination.join(format!("{name}.svg")))?;
    if let Some(tex_name) = tex_name {
        fs::copy(
            destination.join(format!("{name}.png")),
            stage.join(format!("tex/generated/{tex_name}.png")),
        )?;
    }
    Ok(())
}

fn statistics(report: &Value) -> String {
    let mut svg = graphics::begin(1280, 870);
    graphics::text(
        &mut svg,
        28.,
        32.,
        22,
        "Camera count and baseline shape the view distribution",
    );
    graphics::text(&mut svg,28.,61.,15,"Matched rooms, 33 synchronized path samples; accepted sets only. Rejected requests remain in the report.");
    for (i, label) in [
        "Baseline 0 · narrow",
        "Baseline 0.5 · default",
        "Baseline 1 · wide",
    ]
    .iter()
    .enumerate()
    {
        svg.push_str(&format!(
            "<path d=\"M{},85 h32\" stroke=\"{}\" stroke-width=\"4\"/>",
            34 + i * 405,
            COLORS[i]
        ));
        graphics::text(&mut svg, (76 + i * 405) as f64, 90., 16, label);
    }
    let panels = [
        (
            "nearest_neighbor_m",
            "Initial nearest-neighbor distance",
            "metres",
            0.,
        ),
        (
            "group_max_start_optical_angle_degrees",
            "Largest optical-axis angle in a room",
            "degrees",
            180.,
        ),
        (
            "target_diameter_m",
            "Spread of initial look targets",
            "metres",
            0.,
        ),
        ("path_length_m", "Camera travel length", "metres", 0.),
        (
            "similar_pose",
            "Similar absolute pose sequences",
            "fraction of pairs",
            1.,
        ),
        (
            "similar_shape",
            "Similar motion after removing origins",
            "fraction of moving pairs",
            1.,
        ),
    ];
    let cells = report["cells"].as_array().unwrap();
    for (panel, (key, title, unit, limit)) in panels.iter().enumerate() {
        let x = 28. + (panel % 3) as f64 * 420.;
        let y = 136. + (panel / 3) as f64 * 351.;
        graphics::text(&mut svg, x, y, 17, title);
        graphics::text(&mut svg, x, y + 23., 13, unit);
        let ymax = if *limit > 0. {
            *limit
        } else {
            cells
                .iter()
                .filter_map(|c| c["statistics"][key]["p95"].as_f64())
                .fold(0.1, f64::max)
                * 1.10
        };
        let map = |n: f64, v: f64| [x + 45. + (n - 1.) / 11. * 333., y + 269. - v / ymax * 211.];
        for tick in 0..=4 {
            let v = ymax * tick as f64 / 4.;
            let p = map(1., v);
            svg.push_str(&format!(
                "<path d=\"M{},{} h333\" stroke=\"#d9e1d4\"/>",
                p[0], p[1]
            ));
            graphics::text(&mut svg, x, p[1] + 5., 12, &format!("{v:.2}"));
        }
        if *key == "group_max_start_optical_angle_degrees" {
            let p = map(1., 90.);
            svg.push_str(&format!(
                "<path d=\"M{},{} h333\" stroke=\"#a16d54\" stroke-dasharray=\"5 5\"/>",
                p[0], p[1]
            ));
        }
        for n in [1, 2, 4, 6, 8, 10, 12] {
            let p = map(n as f64, 0.);
            graphics::text(&mut svg, p[0] - 5., p[1] + 21., 12, &n.to_string());
        }
        graphics::text(&mut svg, x + 155., y + 314., 14, "cameras / room");
        for (bi, b) in [0., 0.5, 1.].into_iter().enumerate() {
            let selected: Vec<_> = cells
                .iter()
                .filter(|c| number(&c["baseline"]) == b)
                .collect();
            let mut values = Vec::new();
            let mut low = Vec::new();
            let mut high = Vec::new();
            for cell in selected {
                let n = number(&cell["camera_count"]);
                let value = if *key == "similar_pose" || *key == "similar_shape" {
                    let (num, den) = if *key == "similar_pose" {
                        ("similar_pose_pairs", "pairs")
                    } else {
                        ("similar_moving_shape_pairs", "moving_pairs")
                    };
                    (number(&cell[den]) > 0.).then(|| number(&cell[num]) / number(&cell[den]))
                } else {
                    cell["statistics"][key]["median"].as_f64()
                };
                if let Some(value) = value {
                    values.push(map(n, value));
                }
                if let Some(l) = cell["statistics"][key]["p05"].as_f64() {
                    low.push(map(n, l));
                    high.push(map(n, number(&cell["statistics"][key]["p95"])));
                }
            }
            if !low.is_empty() {
                high.reverse();
                low.extend(high);
                svg.push_str(&format!(
                    "<path d=\"{} Z\" fill=\"{}\" opacity=\"0.10\"/>",
                    path(&low),
                    COLORS[bi]
                ));
            }
            svg.push_str(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2.4\"/>",
                path(&values),
                COLORS[bi]
            ));
            for p in values {
                svg.push_str(&format!(
                    "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"3.5\" fill=\"{}\"/>",
                    p[0], p[1], COLORS[bi]
                ));
            }
        }
    }
    graphics::text(&mut svg,28.,840.,13,"Bands: 5th–95th percentiles. Pose similarity: <0.15 m RMS + <5° RMS. Shape similarity: relative RMS <0.25.");
    end(svg)
}

fn heatmap(report: &Value, baseline: f64) -> String {
    let mut svg = graphics::begin(1280, 1060);
    graphics::text(
        &mut svg,
        28.,
        34.,
        23,
        &format!("Camera-path placement · baseline {baseline} · all camera counts"),
    );
    graphics::text(&mut svg,28.,63.,15,"Primary-room bounds normalized per room. +Z down. 33 synchronized samples per path; same probability scale.");
    let cells = report["cells"].as_array().unwrap();
    // One common probability scale across counts AND baseline settings.
    let vmax = cells
        .iter()
        .flat_map(|cell| {
            let total = number(&cell["accepted_rooms"]) * number(&cell["camera_count"]) * 33.;
            cell["heatmaps"]["path"]
                .as_array()
                .unwrap()
                .iter()
                .map(move |v| number(v) / total.max(1.))
        })
        .fold(0.0001, f64::max);
    for cell in cells.iter().filter(|c| number(&c["baseline"]) == baseline) {
        let n = cell["camera_count"].as_u64().unwrap() as usize;
        let x = 28. + ((n - 1) % 4) as f64 * 315.;
        let y = 115. + ((n - 1) / 4) as f64 * 297.;
        graphics::text(
            &mut svg,
            x,
            y,
            18,
            &format!("{n} camera{}", if n == 1 { "" } else { "s" }),
        );
        let total = number(&cell["accepted_rooms"]) * n as f64 * 33.;
        graphics::text(
            &mut svg,
            x,
            y + 23.,
            13,
            &format!(
                "{}/{} rooms · {} samples",
                cell["accepted_rooms"], cell["requested_rooms"], total as u64
            ),
        );
        let size = 211.;
        let pixel = size / BINS as f64;
        for (i, value) in cell["heatmaps"]["path"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let density = number(value) / total.max(1.);
            let q = (density / vmax).sqrt();
            let rgb = [242., 245., 235.]
                .into_iter()
                .zip([20., 125., 112.])
                .map(|(a, b)| (a + (b - a) * q).round() as u8)
                .collect::<Vec<_>>();
            svg.push_str(&format!("<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.3}\" height=\"{:.3}\" fill=\"rgb({},{},{})\"/>",x+35.+(i%BINS) as f64*pixel,y+40.+(i/BINS) as f64*pixel,pixel+0.1,pixel+0.1,rgb[0],rgb[1],rgb[2]));
        }
        svg.push_str(&format!("<rect x=\"{}\" y=\"{}\" width=\"{size}\" height=\"{size}\" fill=\"none\" stroke=\"#9cab99\"/>",x+35.,y+40.));
        graphics::text(&mut svg, x + 29., y + 267., 12, "−0.5");
        graphics::text(&mut svg, x + 135., y + 267., 12, "0");
        graphics::text(&mut svg, x + 232., y + 267., 12, "+0.5 X");
        graphics::text(&mut svg, x + 2., y + 54., 12, "−.5");
        graphics::text(&mut svg, x + 2., y + 242., 12, "+.5");
    }
    for i in 0..220 {
        let q = i as f64 / 219.;
        let rgb = [242., 245., 235.]
            .into_iter()
            .zip([20., 125., 112.])
            .map(|(a, b)| (a + (b - a) * q).round() as u8)
            .collect::<Vec<_>>();
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"1003\" width=\"1.2\" height=\"15\" fill=\"rgb({},{},{})\"/>",
            28 + i,
            rgb[0],
            rgb[1],
            rgb[2]
        ));
    }
    graphics::text(
        &mut svg,
        28.,
        1042.,
        13,
        &format!(
            "0% to {:.2}% per bin (square-root color scale)",
            vmax * 100.
        ),
    );
    graphics::text(
        &mut svg,
        405.,
        1015.,
        15,
        "Repeated path samples are correlated; rooms are independent sampling units.",
    );
    graphics::text(
        &mut svg,
        405.,
        1040.,
        14,
        "Heatmap density measures camera positions, not visible or reconstructed surfaces.",
    );
    end(svg)
}

fn schematics(report: &Value) -> String {
    let mut svg = graphics::begin(1280, 780);
    let example = &report["schematic_example"];
    let room = &example["room"];
    let manifest = &room["manifest"];
    let bounds = &room["primary_bounds"];
    let lo = [number(&bounds[0][0]), number(&bounds[0][1])];
    let hi = [number(&bounds[1][0]), number(&bounds[1][1])];
    graphics::text(
        &mut svg,
        25.,
        32.,
        22,
        &format!(
            "Fixed geometry · prespecified seed {} · paths, gaze and camera count",
            room["seed"]
        ),
    );
    graphics::text(&mut svg,25.,61.,14,"Primary-room close-up. Dots: initial centers. Arrows: optical axes. Thin rays: look targets. Curves: full camera paths.");
    for (row, b) in [0.5, 1.].into_iter().enumerate() {
        for (col, n) in [1, 2, 4, 8, 12].into_iter().enumerate() {
            let x = 18. + col as f64 * 253.;
            let y = 109. + row as f64 * 324.;
            graphics::text(
                &mut svg,
                x,
                y,
                16,
                &format!("{n} camera{} · baseline {b}", if n == 1 { "" } else { "s" }),
            );
            let scale = (215. / (hi[0] - lo[0])).min(242. / (hi[1] - lo[1]));
            let map = |px: f64, pz: f64| {
                [
                    x + 119. + (px - (lo[0] + hi[0]) * 0.5) * scale,
                    y + 157. + (pz - (lo[1] + hi[1]) * 0.5) * scale,
                ]
            };
            let clip = format!("clip-{row}-{col}");
            svg.push_str(&format!("<defs><clipPath id=\"{clip}\"><rect x=\"{x}\" y=\"{}\" width=\"238\" height=\"281\"/></clipPath></defs><g clip-path=\"url(#{clip})\">",y+15.));
            let pts = manifest["envelope"]["footprint"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let p = map(number(&p[0]), number(&p[1]));
                    format!("{:.2},{:.2}", p[0], p[1])
                })
                .collect::<Vec<_>>()
                .join(" ");
            svg.push_str(&format!(
                "<polygon points=\"{pts}\" fill=\"#eef2e9\" stroke=\"#9baa95\"/>"
            ));
            for o in manifest["objects"].as_array().unwrap() {
                if o["neighbor"] != false || o["solid"] != true {
                    continue;
                }
                let w = number(&o["size"][0]) * 0.5;
                let d = number(&o["size"][2]) * 0.5;
                let (s, c) = number(&o["yaw"]).sin_cos();
                let pts = [[-w, -d], [w, -d], [w, d], [-w, d]]
                    .into_iter()
                    .map(|p| {
                        let p = map(
                            c * p[0] + s * p[1] + number(&o["position"][0]),
                            -s * p[0] + c * p[1] + number(&o["position"][2]),
                        );
                        format!("{:.2},{:.2}", p[0], p[1])
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                svg.push_str(&format!("<polygon points=\"{pts}\" fill=\"#d9e0d2\" stroke=\"#b7c2ae\" stroke-width=\"0.7\"/>"));
            }
            let top = map(lo[0], lo[1]);
            svg.push_str(&format!("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#20382e\" stroke-width=\"1.4\" stroke-dasharray=\"4 3\"/>",top[0],top[1],(hi[0]-lo[0])*scale,(hi[1]-lo[1])*scale));
            let group = example["groups"]
                .as_array()
                .unwrap()
                .iter()
                .find(|g| number(&g["baseline"]) == b && g["camera_count"] == n);
            if let Some(group) = group {
                for (i, camera) in group["cameras"].as_array().unwrap().iter().enumerate() {
                    let poses = camera["positions"].as_array().unwrap();
                    let start = map(number(&poses[0][0]), number(&poses[0][2]));
                    let target = map(number(&camera["target"][0]), number(&camera["target"][2]));
                    let [r, g, b] = bevy_zeroverse_capture::camera_color(i, n as usize);
                    let color = format!("rgb({r},{g},{b})");
                    let points: Vec<_> = poses
                        .iter()
                        .map(|p| map(number(&p[0]), number(&p[2])))
                        .collect();
                    svg.push_str(&format!("<path d=\"{}\" stroke=\"{color}\" stroke-width=\"{}\" fill=\"none\" opacity=\"0.85\"/>",path(&points),if i==0 {2.0}else{1.4}));
                    svg.push_str(&format!("<path d=\"M{},{} L{},{}\" stroke=\"{color}\" stroke-width=\"0.8\" opacity=\"0.3\"/><circle cx=\"{}\" cy=\"{}\" r=\"2\" fill=\"#ab4984\"/>",start[0],start[1],target[0],target[1],target[0],target[1]));
                    let f = [
                        number(&camera["forwards"][0][0]),
                        number(&camera["forwards"][0][2]),
                    ];
                    let norm = f[0].hypot(f[1]).max(1e-9);
                    let v = [f[0] / norm, f[1] / norm];
                    let tip = [start[0] + v[0] * 15., start[1] + v[1] * 15.];
                    svg.push_str(&format!("<path d=\"M{},{} L{},{} M{},{} L{},{} L{},{}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"1.8\"/>",start[0],start[1],tip[0],tip[1],tip[0]-v[0]*5.-v[1]*3.,tip[1]-v[1]*5.+v[0]*3.,tip[0],tip[1],tip[0]-v[0]*5.+v[1]*3.,tip[1]-v[1]*5.-v[0]*3.));
                    svg.push_str(&format!("<circle cx=\"{}\" cy=\"{}\" r=\"3.8\" fill=\"{color}\" stroke=\"#f7f8f3\" stroke-width=\"1\"/>",start[0],start[1]));
                    graphics::text(&mut svg, start[0] + 5., start[1] - 5., 10, &i.to_string());
                }
            } else {
                graphics::text(&mut svg, x + 30., y + 150., 16, "REQUEST REJECTED");
            }
            svg.push_str("</g>");
        }
    }
    graphics::text(&mut svg,25.,763.,14,"Dashed rectangle: primary-room bounds. Furnishings are solid footprint proxies; diagram omits roof and vertical clearance.");
    end(svg)
}

fn rendered(report: &Value) -> Option<String> {
    let receipt = &report["rendered_qualification"];
    let cases = receipt["cases"].as_array()?;
    let recipe = receipt["recipe"]["cases"].as_array()?;
    let mut svg = graphics::begin(1280, 465);
    graphics::text(
        &mut svg,
        28.,
        33.,
        22,
        "Native rendered co-visibility · distinct from placement estimates",
    );
    graphics::text(&mut svg,28.,62.,15,"Shadows/materials/GI retained. Any-peer means use source views; pair metrics use directed pairs and times.");
    let panels = [
        (
            "any_peer_shared_fraction_valid",
            "Visible in at least one peer",
            "fraction of valid source pixels",
            1.,
        ),
        (
            "directed_rendered_overlap",
            "Directed pair overlap",
            "fraction of valid source pixels",
            1.,
        ),
        (
            "triangulation_degrees",
            "Shared-surface triangulation angle",
            "degrees",
            0.,
        ),
    ];
    for (panel, (key, title, unit, limit)) in panels.into_iter().enumerate() {
        let x = 30. + panel as f64 * 416.;
        let y = 112.;
        let ymax = if limit > 0. {
            limit
        } else {
            cases
                .iter()
                .filter_map(|c| c[key]["p05_p25_p50_p75_p95"][4].as_f64())
                .fold(1., f64::max)
                * 1.1
        };
        graphics::text(&mut svg, x, y, 18, title);
        graphics::text(&mut svg, x, y + 23., 14, unit);
        let map = |n: f64, v: f64| [x + 53. + (n - 2.) / 10. * 330., y + 260. - v / ymax * 203.];
        for tick in 0..=4 {
            let v = ymax * tick as f64 / 4.;
            let p = map(2., v);
            svg.push_str(&format!(
                "<path d=\"M{},{} h330\" stroke=\"#d9e1d4\"/>",
                p[0], p[1]
            ));
            graphics::text(&mut svg, x, p[1] + 5., 13, &format!("{v:.2}"));
        }
        for n in [2, 4, 8, 12] {
            let p = map(n as f64, 0.);
            graphics::text(&mut svg, p[0] - 5., p[1] + 20., 14, &n.to_string());
        }
        for (i, b) in [0., 0.5, 1.].into_iter().enumerate() {
            let mut selected: Vec<_> = cases
                .iter()
                .filter(|c| {
                    c["success"] == true
                        && recipe
                            .iter()
                            .find(|r| r["name"] == c["name"])
                            .is_some_and(|r| number(&r["camera"]["baseline"]) == b)
                })
                .collect();
            selected
                .sort_by(|a, b| number(&a["camera_count"]).total_cmp(&number(&b["camera_count"])));
            let points: Vec<_> = selected
                .iter()
                .filter_map(|c| {
                    c[key]["mean"]
                        .as_f64()
                        .map(|v| map(number(&c["camera_count"]), v))
                })
                .collect();
            svg.push_str(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2.5\"/>",
                path(&points),
                COLORS[i]
            ));
            for point in points {
                svg.push_str(&format!(
                    "<circle cx=\"{}\" cy=\"{}\" r=\"4\" fill=\"{}\"/>",
                    point[0], point[1], COLORS[i]
                ));
            }
            graphics::text(
                &mut svg,
                x + 15. + i as f64 * 125.,
                436.,
                14,
                &format!("Baseline {b}"),
            );
            svg.push_str(&format!(
                "<path d=\"M{},449 h88\" stroke=\"{}\" stroke-width=\"3\"/>",
                x + 15. + i as f64 * 125.,
                COLORS[i]
            ));
        }
    }
    Some(end(svg))
}

pub(super) fn build(report: &Value, stage: &Path, dest: &Path) -> Result<()> {
    save(
        &statistics(report),
        dest,
        stage,
        "camera-count-statistics",
        Some("camera_count_statistics"),
    )?;
    for baseline in [0., 0.5, 1.] {
        save(
            &heatmap(report, baseline),
            dest,
            stage,
            &format!("camera-count-heatmap-b{baseline}"),
            (baseline == 1.).then_some("camera_count_heatmap"),
        )?;
    }
    save(
        &schematics(report),
        dest,
        stage,
        "camera-count-schematics",
        Some("camera_count_schematics"),
    )?;
    if let Some(svg) = rendered(report) {
        save(
            &svg,
            dest,
            stage,
            "camera-count-rendered",
            Some("camera_count_rendered"),
        )?;
    }
    io::write(
        &dest.join("figure_contract.json"),
        &serde_json::json!({"schema_version":1,"source_report":"study.json","directory":DIRECTORY,"schematic_seed":report["schematic_example"]["room"]["seed"],"probability_color_scale":"same across baseline and camera count; square root","path_samples":33}),
    )?;
    Ok(())
}
