use crate::{
    dataset::{area, number, View},
    io::html,
};
use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, RgbImage};
use serde_json::Value;
use std::{fs, path::Path};

const FONT: &[u8] = include_bytes!("../fonts/DejaVuSans.ttf");
const COLORS: [&str; 4] = ["#147d70", "#c56a33", "#626aca", "#aa4984"];

pub fn begin(width: u32, height: u32) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f7f8f3"/><g font-family="DejaVu Sans" fill="#20382e">"##
    )
}
pub fn text(svg: &mut String, x: f64, y: f64, size: u32, value: &str) {
    svg.push_str(&format!(
        "<text x=\"{x:.3}\" y=\"{y:.3}\" font-size=\"{size}\">{}</text>",
        html(value)
    ));
}
fn line(svg: &mut String, a: [f64; 2], b: [f64; 2], color: &str, width: f64) {
    svg.push_str(&format!("<path d=\"M{:.3},{:.3} L{:.3},{:.3}\" stroke=\"{color}\" stroke-width=\"{width}\" fill=\"none\"/>",a[0],a[1],b[0],b[1]));
}
pub fn raster(svg: &str) -> Result<RgbImage> {
    let mut options = resvg::usvg::Options::default();
    options.fontdb_mut().load_font_data(FONT.to_vec());
    let tree = resvg::usvg::Tree::from_str(svg, &options).context("parse publication SVG")?;
    let size = tree.size().to_int_size();
    let mut pixels = resvg::tiny_skia::Pixmap::new(size.width(), size.height())
        .context("allocate publication figure")?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixels.as_mut(),
    );
    let rgb = pixels
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| p[..3].iter().copied())
        .collect();
    RgbImage::from_raw(size.width(), size.height(), rgb).context("raster figure dimensions")
}
pub fn save(svg: &str, path: &Path) -> Result<()> {
    let image = raster(svg)?;
    if path.extension().is_some_and(|x| x == "svg") {
        fs::write(path, format!("{svg}\n"))?;
        image.save(path.with_extension("png"))?;
    } else {
        image.save(path)?;
    }
    Ok(())
}

/// Uses the serialized metric envelope; this is a diagram, never a scene renderer.
pub fn plan(row: &Value, manifest: Option<&Value>, views: &[&View]) -> String {
    let mut svg = begin(1000, 510);
    let e = &row["envelope"];
    let size = &row["room_size"];
    let w = number(&size[0]);
    let d = number(&size[2]);
    let h = number(&size[1]);
    text(
        &mut svg,
        24.0,
        29.0,
        18,
        &format!(
            "Seed {} · {:.1} m² · plan and metric envelope section",
            row["seed"],
            area(&e["footprint"])
        ),
    );
    let scale = (420.0 / (w + 1.0)).min(375.0 / (d + 1.0));
    let map = |p: [f64; 2]| [250.0 + p[0] * scale, 260.0 + p[1] * scale];
    let p2 = |p: &Value| [number(&p[0]), number(&p[1])];
    let points = e["footprint"].as_array().unwrap();
    let poly = |svg: &mut String, pts: Vec<[f64; 2]>, fill: &str, stroke: &str| {
        let values = pts
            .iter()
            .map(|p| format!("{:.3},{:.3}", p[0], p[1]))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!("<polygon points=\"{values}\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"/>"));
    };
    poly(
        &mut svg,
        points.iter().map(|p| map(p2(p))).collect(),
        "#eff3e9",
        "#20382e",
    );
    if let Some(m) = manifest {
        for o in m["objects"].as_array().unwrap() {
            if o["neighbor"] != false || o["solid"] != true {
                continue;
            }
            let a = number(&o["yaw"]);
            let (s, c) = a.sin_cos();
            let ow = number(&o["size"][0]) / 2.0;
            let od = number(&o["size"][2]) / 2.0;
            // Match Bevy's Y-axis rotation: x'=cx+sz, z'=-sx+cz.
            let pts = [[-ow, -od], [ow, -od], [ow, od], [-ow, od]]
                .into_iter()
                .map(|p| {
                    map([
                        c * p[0] + s * p[1] + number(&o["position"][0]),
                        -s * p[0] + c * p[1] + number(&o["position"][2]),
                    ])
                })
                .collect();
            poly(&mut svg, pts, "#dce3d6", "#b7c2b0");
        }
    }
    let rectangle = |svg: &mut String, min: [f64; 2], max: [f64; 2], color: &str| {
        poly(
            svg,
            vec![
                map(min),
                map([max[0], min[1]]),
                map(max),
                map([min[0], max[1]]),
            ],
            color,
            "#899781",
        )
    };
    for f in e["floor_patches"].as_array().unwrap() {
        rectangle(
            &mut svg,
            p2(&f["min"]),
            p2(&f["max"]),
            if number(&f["height"]) > 0.0 {
                "#e3ae8c"
            } else {
                "#a4cee0"
            },
        );
        let a = map([
            (number(&f["min"][0]) + number(&f["max"][0])) / 2.0,
            (number(&f["min"][1]) + number(&f["max"][1])) / 2.0,
        ]);
        text(
            &mut svg,
            a[0] - 23.0,
            a[1],
            12,
            &format!("{:+.2} m", number(&f["height"])),
        );
    }
    for pillar in e["pillars"].as_array().unwrap() {
        let a = map(p2(&pillar["center"]));
        let r = number(&pillar["radius"]) * scale;
        svg.push_str(&format!(
            r##"<circle cx="{}" cy="{}" r="{r}" fill="#45594a"/>"##,
            a[0], a[1]
        ));
    }
    if let Some(partitions) = row["partitions"].as_array() {
        for p in partitions {
            let axis = p["axis"].as_u64().unwrap() as usize;
            for (lo, hi) in [
                (
                    number(&p["start"]),
                    number(&p["door_center"]) - number(&p["door_width"]) / 2.0,
                ),
                (
                    number(&p["door_center"]) + number(&p["door_width"]) / 2.0,
                    number(&p["end"]),
                ),
            ] {
                let mut a = [0.0; 2];
                let mut b = [0.0; 2];
                a[axis] = number(&p["coordinate"]);
                b[axis] = a[axis];
                a[1 - axis] = lo;
                b[1 - axis] = hi;
                line(&mut svg, map(a), map(b), "#65745f", 2.0);
            }
            if number(&p["arch_rise"]) > 0.001 {
                let mut p0 = [0.0; 2];
                p0[axis] = number(&p["coordinate"]);
                p0[1 - axis] = number(&p["door_center"]);
                let a = map(p0);
                poly(
                    &mut svg,
                    vec![
                        [a[0], a[1] - 5.0],
                        [a[0] - 5.0, a[1] + 4.0],
                        [a[0] + 5.0, a[1] + 4.0],
                    ],
                    "#bc6e35",
                    "#bc6e35",
                );
            }
        }
    }
    for wall in e["walls"].as_array().unwrap() {
        if wall["facade"].is_null() {
            continue;
        }
        let i = wall["edge"].as_u64().unwrap() as usize;
        let (a, b) = (p2(&points[i]), p2(&points[(i + 1) % points.len()]));
        let length = (b[0] - a[0]).hypot(b[1] - a[1]);
        for o in wall["facade"]["openings"].as_array().unwrap() {
            let at = |v: &Value| {
                let t = number(v) / length + 0.5;
                map([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
            };
            line(&mut svg, at(&o["min"][0]), at(&o["max"][0]), "#2797a5", 4.0);
        }
    }
    let m = &e["mezzanine"];
    if !m.is_null() {
        rectangle(
            &mut svg,
            p2(&m["deck"]["min"]),
            p2(&m["deck"]["max"]),
            "#c8aed8",
        );
        rectangle(
            &mut svg,
            p2(&m["stair_min"]),
            p2(&m["stair_max"]),
            "#e4d7ec",
        );
        let axis = m["stair_axis"].as_u64().unwrap() as usize;
        let steps = m["steps"].as_u64().unwrap();
        for i in 0..=steps {
            let (mut a, mut b) = (p2(&m["stair_min"]), p2(&m["stair_max"]));
            a[axis] += (b[axis] - a[axis]) * i as f64 / steps as f64;
            b[axis] = a[axis];
            line(&mut svg, map(a), map(b), "#79608f", 0.8);
        }
    }
    for v in views {
        let p = [
            v.world_from_view[3][0] as f64,
            v.world_from_view[3][2] as f64,
        ];
        let a = map(p);
        let b = map([
            p[0] - v.world_from_view[2][0] as f64,
            p[1] - v.world_from_view[2][2] as f64,
        ]);
        let color = COLORS[v.camera_index];
        line(&mut svg, a, b, color, 2.0);
        let angle = (b[1] - a[1]).atan2(b[0] - a[0]);
        for delta in [-0.45, 0.45] {
            line(
                &mut svg,
                b,
                [
                    b[0] - 7.0 * (angle + delta).cos(),
                    b[1] - 7.0 * (angle + delta).sin(),
                ],
                color,
                2.0,
            );
        }
        svg.push_str(&format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"4\" fill=\"{color}\"/>",
            a[0], a[1]
        ));
        text(
            &mut svg,
            a[0] + 6.0,
            a[1] - 7.0,
            13,
            &format!("C{}", v.camera_index),
        );
    }
    text(
        &mut svg,
        40.0,
        490.0,
        13,
        &format!("X/Z metres · {:.1} × {:.1} m bounding envelope", w, d),
    );
    // An exact section of the polygon can contain disconnected intervals.
    let mut axis =
        usize::from(number(&e["ceiling_drop"][1]).abs() > number(&e["ceiling_drop"][0]).abs());
    let mut coordinate = 0.0;
    if let Some(f) = e["floor_patches"].as_array().unwrap().first() {
        coordinate = (number(&f["min"][1 - axis]) + number(&f["max"][1 - axis])) / 2.0;
    }
    if !m.is_null() {
        axis = m["stair_axis"].as_u64().unwrap() as usize;
        coordinate = (number(&m["stair_min"][1 - axis]) + number(&m["stair_max"][1 - axis])) / 2.0;
    }
    let point = |x: f64| {
        if axis == 0 {
            [x, coordinate]
        } else {
            [coordinate, x]
        }
    };
    let mut hits = Vec::new();
    for i in 0..points.len() {
        let (a, b) = (p2(&points[i]), p2(&points[(i + 1) % points.len()]));
        if a[1 - axis].min(b[1 - axis]) <= coordinate && coordinate < a[1 - axis].max(b[1 - axis]) {
            let t = (coordinate - a[1 - axis]) / (b[1 - axis] - a[1 - axis]);
            hits.push(a[axis] + t * (b[axis] - a[axis]));
        }
    }
    hits.sort_by(f64::total_cmp);
    let section = |x: f64, y: f64| [750.0 + x * scale, 420.0 - y * scale];
    let roof = |p: [f64; 2]| {
        h - (0..2)
            .map(|i| {
                let drop = number(&e["ceiling_drop"][i]);
                let extent = if i == 0 { w } else { d };
                let uv = (p[i] / extent + 0.5).clamp(0.0, 1.0);
                drop.abs() * if drop > 0.0 { uv } else { 1.0 - uv }
            })
            .sum::<f64>()
    };
    for interval in hits.as_chunks::<2>().0 {
        let (lo, hi) = (interval[0], interval[1]);
        let (a, b) = (map(point(lo)), map(point(hi)));
        svg.push_str(&format!(r##"<path d="M{:.3},{:.3} L{:.3},{:.3}" stroke="#909a8b" stroke-width="1" stroke-dasharray="5 4" fill="none"/>"##,a[0],a[1],b[0],b[1]));
        line(
            &mut svg,
            section(lo, roof(point(lo))),
            section(hi, roof(point(hi))),
            "#20382e",
            2.3,
        );
        for x in [lo, hi] {
            line(
                &mut svg,
                section(x, 0.0),
                section(x, roof(point(x))),
                "#20382e",
                1.5,
            );
        }
        let mut xs = vec![lo, hi];
        for f in e["floor_patches"].as_array().unwrap() {
            for k in ["min", "max"] {
                xs.push(number(&f[k][axis]).clamp(lo, hi));
            }
        }
        xs.sort_by(f64::total_cmp);
        xs.dedup();
        for pair in xs.windows(2) {
            let p = point((pair[0] + pair[1]) / 2.0);
            let height = e["floor_patches"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| {
                    (0..2).all(|i| number(&f["min"][i]) <= p[i] && p[i] <= number(&f["max"][i]))
                })
                .map_or(0.0, |f| number(&f["height"]));
            line(
                &mut svg,
                section(pair[0], height),
                section(pair[1], height),
                if height > 0.0 {
                    "#bf6738"
                } else if height < 0.0 {
                    "#458eae"
                } else {
                    "#65745f"
                },
                3.0,
            );
            for x in pair {
                line(
                    &mut svg,
                    section(*x, 0.0),
                    section(*x, height),
                    "#8d9a85",
                    1.0,
                );
            }
        }
    }
    if !m.is_null() {
        let deck = &m["deck"];
        let height = number(&deck["height"]);
        line(
            &mut svg,
            section(number(&deck["min"][axis]), height),
            section(number(&deck["max"][axis]), height),
            "#9f80b4",
            number(&m["thickness"]) * scale,
        );
        let steps = m["steps"].as_u64().unwrap();
        let min = number(&m["stair_min"][axis]);
        let max = number(&m["stair_max"][axis]);
        for i in 0..steps {
            let a = min + (max - min) * i as f64 / steps as f64;
            let b = min + (max - min) * (i + 1) as f64 / steps as f64;
            let y = height * (1.0 - i as f64 / steps as f64);
            line(&mut svg, section(a, y), section(b, y), "#79608f", 1.5);
            line(
                &mut svg,
                section(b, y),
                section(b, y - height / steps as f64),
                "#79608f",
                1.5,
            );
        }
    }
    let pitch = (number(&e["ceiling_drop"][0]) / w)
        .hypot(number(&e["ceiling_drop"][1]) / d)
        .atan()
        .to_degrees();
    text(
        &mut svg,
        530.0,
        61.0,
        15,
        &format!("Envelope section · roof pitch {pitch:.1}°"),
    );
    text(
        &mut svg,
        530.0,
        490.0,
        13,
        "Base floor 0 m · furnishings omitted in section",
    );
    svg.push_str("</g></svg>");
    svg
}

pub fn figure_rows(
    rows: &[(String, Vec<(std::path::PathBuf, String)>)],
    destination: &Path,
    width: u32,
    aspect: f64,
) -> Result<()> {
    let pad = 18u32;
    let title = 44;
    let label = 32;
    let height = (width as f64 * aspect).round() as u32;
    let columns = rows
        .iter()
        .map(|(_, r)| r.len())
        .max()
        .context("empty contact figure")? as u32;
    let mut svg = begin(
        columns * (width + pad) + pad,
        rows.len() as u32 * (height + title + label + pad) + pad,
    );
    for (r, (name, images)) in rows.iter().enumerate() {
        let y = pad + r as u32 * (height + title + label + pad);
        text(&mut svg, pad as f64, (y + 25) as f64, 20, name);
        for (c, (path, caption)) in images.iter().enumerate() {
            let x = pad + c as u32 * (width + pad);
            let im = image::open(path)?;
            let thumbnail = im.thumbnail(width, height);
            let mut bytes = std::io::Cursor::new(Vec::new());
            thumbnail.write_to(&mut bytes, image::ImageFormat::Png)?;
            svg.push_str(&format!("<image x=\"{x}\" y=\"{}\" width=\"{width}\" height=\"{height}\" preserveAspectRatio=\"xMinYMin meet\" xlink:href=\"data:image/png;base64,{}\"/>",y+title,STANDARD.encode(bytes.into_inner())));
            text(
                &mut svg,
                x as f64,
                (y + title + height + 24) as f64,
                15,
                caption,
            );
        }
    }
    svg.push_str("</g></svg>");
    save(&svg, destination)
}

pub fn histogram(
    title: &str,
    unit: &str,
    edges: &[f64],
    counts: &[u64],
    width: u32,
    height: u32,
) -> String {
    let mut svg = begin(width, height);
    let total: u64 = counts.iter().sum();
    text(&mut svg, 24.0, 29.0, 16, &format!("{title} · n={total}"));
    let left = 50.0;
    let top = 55.0;
    let plotw = width as f64 - 75.0;
    let ploth = height as f64 - 110.0;
    let maximum = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
    let span = (edges.last().copied().unwrap_or(1.0) - edges[0]).max(1e-9);
    for i in 0..=4 {
        let y = top + ploth * i as f64 / 4.0;
        line(&mut svg, [left, y], [left + plotw, y], "#dfe5d9", 0.8);
        text(
            &mut svg,
            7.0,
            y + 4.0,
            11,
            &format!("{:.0}", maximum * (1.0 - i as f64 / 4.0)),
        );
    }
    for (i, &n) in counts.iter().enumerate() {
        let x = left + (edges[i] - edges[0]) / span * plotw;
        let w = (edges[i + 1] - edges[i]) / span * plotw * 0.9;
        let h = n as f64 / maximum * ploth;
        svg.push_str(&format!(
            r##"<rect x="{x}" y="{}" width="{w}" height="{h}" fill="#287d69"/>"##,
            top + ploth - h
        ));
    }
    for i in 0..=4 {
        let x = left + plotw * i as f64 / 4.0;
        text(
            &mut svg,
            x - 14.0,
            top + ploth + 20.0,
            11,
            &format!("{:.2}", edges[0] + span * i as f64 / 4.0),
        );
    }
    text(&mut svg, 50.0, height as f64 - 10.0, 12, unit);
    svg.push_str("</g></svg>");
    svg
}
pub fn combined(images: &[RgbImage], columns: u32) -> RgbImage {
    let width = images[0].width();
    let height = images[0].height();
    let rows = (images.len() as u32).div_ceil(columns);
    let mut output =
        RgbImage::from_pixel(width * columns, height * rows, image::Rgb([247, 248, 243]));
    for (i, im) in images.iter().enumerate() {
        image::imageops::replace(
            &mut output,
            im,
            (i as u32 % columns * width) as i64,
            (i as u32 / columns * height) as i64,
        );
    }
    output
}
pub fn save_webp(image: RgbImage, path: &Path) -> Result<()> {
    DynamicImage::ImageRgb8(image).save(path)?;
    Ok(())
}
