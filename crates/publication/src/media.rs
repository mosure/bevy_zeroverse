use crate::{
    config::Protocol,
    dataset::{area, features, number, Dataset},
    graphics, io, visibility,
};
use anyhow::{ensure, Result};
use image::{Rgb, RgbImage};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const MODES: [&str; 6] = bevy_zeroverse_capture::PUBLICATION_MODES;

pub fn build(dataset: &Dataset, stage: &Path, protocol: &Protocol) -> Result<Value> {
    let media = stage.join("www/project/static/media/architecture");
    if media.exists() {
        fs::remove_dir_all(&media)?;
    }
    fs::create_dir_all(&media)?;
    let figures = stage.join("tex/generated");
    fs::create_dir_all(&figures)?;
    for entry in fs::read_dir(&figures)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with("architecture_")
        {
            fs::remove_file(entry.path())?;
        }
    }
    let mut records = Vec::new();
    let mut outputs = BTreeMap::new();
    let mut masks_archive = BTreeMap::new();
    let mut calibration_archive = BTreeMap::new();
    let mut metadata = None;
    for scene in &dataset.scenes {
        let capture = &scene.capture;
        let seed = capture.seed;
        if let Some(ref first) = metadata {
            ensure!(
                *first == capture.co_visibility_metadata,
                "co-visibility convention changed within cohort"
            );
        } else {
            metadata = Some(capture.co_visibility_metadata.clone());
        }
        let mut frames = Vec::new();
        let mut room_masks = BTreeMap::new();
        for step in 0..protocol.playback_steps {
            let mut views: Vec<_> = capture
                .views
                .iter()
                .enumerate()
                .filter(|(_, v)| v.step_index == step)
                .collect();
            views.sort_by_key(|(_, v)| v.camera_index);
            let plan = format!("s{seed}-t{step}-plan.svg");
            graphics::save(
                &graphics::plan(
                    &scene.architecture,
                    Some(&scene.manifest),
                    &views.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                ),
                &media.join(&plan),
            )?;
            let mut view_records = Vec::new();
            for (index, view) in views {
                let camera = view.camera_index;
                let mut images = BTreeMap::<String, Value>::new();
                for mode in MODES {
                    let source = scene.folder.join(format!("view_{index:02}_{mode}.png"));
                    let destination = format!("s{seed}-t{step}-c{camera}-{mode}.webp");
                    let rgb = image::open(source)?.into_rgb8();
                    graphics::save_webp(rgb.clone(), &media.join(&destination))?;
                    ensure!(
                        image::open(media.join(&destination))?.into_rgb8() == rgb,
                        "lossless display roundtrip failed"
                    );
                    images.insert(mode.into(), json!(destination));
                }
                let mask = fs::read(
                    scene
                        .folder
                        .join(format!("view_{index:02}_co_visibility_mask.png")),
                )?;
                let valid = fs::read(
                    scene
                        .folder
                        .join(format!("view_{index:02}_co_visibility_valid.png")),
                )?;
                let plane = visibility::decode(
                    &mask,
                    &valid,
                    None,
                    capture.image_size,
                    camera,
                    protocol.cameras,
                    &view.co_visibility,
                )?;
                let color = image::open(scene.folder.join(format!("view_{index:02}_color.png")))?
                    .into_rgb8();
                let mut peers = Vec::new();
                for peer in 0..protocol.cameras {
                    let mut overlay = RgbImage::new(protocol.width, protocol.height);
                    for ((out, rgb), &mask) in
                        overlay.pixels_mut().zip(color.pixels()).zip(&plane.masks)
                    {
                        let shared = mask & (1 << peer) != 0;
                        *out = Rgb(std::array::from_fn(|i| {
                            let c = rgb[i] as f64 / 255.0;
                            let v = if shared {
                                c * 0.65 + [0.12, 0.92, 0.72][i] * 0.35
                            } else {
                                c * 0.20
                            };
                            (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8
                        }));
                    }
                    let destination = format!("s{seed}-t{step}-c{camera}-peer{peer}.webp");
                    graphics::save_webp(overlay, &media.join(&destination))?;
                    peers.push(destination);
                }
                images.insert("peers".into(), json!(peers));
                view_records.push(json!({"camera":view,"images":images,"visibility":plane.stats}));
                for (suffix, bytes) in [("mask", mask), ("valid", valid)] {
                    let name = format!("view_{index:02}_co_visibility_{suffix}.png");
                    room_masks.insert(name.clone(), bytes.clone());
                    masks_archive.insert(format!("seed_{seed:06}/{name}"), bytes);
                }
            }
            frames.push(json!({"time":step as f32/(protocol.playback_steps-1) as f32,"views":view_records,"plan":plan}));
        }
        let filename = format!("s{seed}-capture.json");
        let prefix = format!("seed_{seed:06}/");
        let source_hashes: BTreeMap<_, _> = dataset
            .input_sha256
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .collect();
        io::write(
            &media.join(&filename),
            &json!({"manifest":scene.manifest,"capture":capture,"architecture":scene.architecture,"source_sha256":source_hashes,"generator_identity":dataset.identity}),
        )?;
        let bytes = fs::read(media.join(&filename))?;
        calibration_archive.insert(filename.clone(), bytes.clone());
        masks_archive.insert(format!("seed_{seed:06}/capture.json"), bytes.clone());
        room_masks.insert("capture.json".into(), bytes);
        let masks = format!("s{seed}-visibility.zip");
        io::archive(&media.join(&masks), &room_masks)?;
        let row = &scene.architecture;
        let pitch = (number(&row["envelope"]["ceiling_drop"][0]) / number(&row["room_size"][0]))
            .hypot(number(&row["envelope"]["ceiling_drop"][1]) / number(&row["room_size"][2]))
            .atan()
            .to_degrees();
        records.push(json!({"seed":seed,"activity":scene.manifest["layout"],"features":features(row).into_iter().filter_map(|(k,v)|v.then_some(k)).collect::<Vec<_>>(),"area_m2":area(&row["envelope"]["footprint"]),"roof_pitch_degrees":pitch,"metadata":filename,"masks":masks,"frames":frames}));
    }
    io::archive(&media.join("visibility-masks.zip"), &masks_archive)?;
    let programs = dataset
        .rows
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n")
        + "\n";
    calibration_archive.insert("architecture.jsonl".into(), programs.into_bytes());
    calibration_archive.insert(
        "generator_identity.json".into(),
        serde_json::to_vec_pretty(&dataset.identity)?,
    );
    calibration_archive.insert("README.txt".into(),format!("{} complete consecutively rendered rooms, {} cameras, {} endpoints. Six matched display modes. Previews are not float32 training labels. world_from_view stores columns; forward is -Z. Exact uint16 membership and uint8 validity in visibility-masks.zip. All {} architecture programs included. No temporal flow/continuous video captured in this cohort.\n",protocol.rendered_rooms,protocol.cameras,protocol.playback_steps,protocol.audit_rooms).into_bytes());
    io::archive(
        &media.join("calibration-and-programs.zip"),
        &calibration_archive,
    )?;
    io::write(&media.join("population.json"), &dataset.metrics)?;
    let mut audit = dataset.summary.clone();
    audit["input_sha256"] = json!(dataset.input_sha256);
    audit["generator_identity"] = json!(dataset.identity);
    io::write(&media.join("audit.json"), &audit)?;
    let featured = choose_featured(&records);
    figures_build(dataset, &records, &featured, &media, &figures, protocol)?;
    for (name, digest) in io::files(&media, &media)? {
        outputs.insert(name, digest);
    }
    let mut co_visibility = metadata.unwrap();
    co_visibility["archive"] = json!("visibility-masks.zip");
    co_visibility["statistics"] = dataset.summary["co_visibility"].clone();
    let data = json!({"schema_version":1,"generator_version":dataset.identity.generator_version,"generator_identity":dataset.identity,
        "capture_engine":dataset.selection["capture_engine"],"audit_rooms":protocol.audit_rooms,"rendered_rooms":protocol.rendered_rooms,
        "rendered_views":protocol.rendered_rooms*protocol.cameras*protocol.playback_steps,"capture_size":[protocol.width,protocol.height],"modes":MODES,
        "quantized_structural_signatures":dataset.summary["quantized_structural_signatures"],"frames_per_camera":protocol.playback_steps,"selection":"All consecutive rooms retained; shortcuts chosen by structural feature coverage with stable seed ties.",
        "featured":featured,"feature_room_counts":dataset.summary["feature_room_counts"],"population_plot_totals":plot_totals(&dataset.metrics),"scenes":records,
        "display":{"color":"Original renderer-tone-mapped sRGB, lossless WebP. No exposure correction or crop.","depth":"Grayscale linear depth / 15 m, clamped to [0,1]; 8-bit display preview.","normal":"View-space (n+1)/2; 8-bit display preview.","position":"World position normalized by annotation AABB, clamped for display.","semantic":"Original semantic palette; lossless WebP.","co_visibility":"Same-time first-surface camera membership; additive camera colors; source excluded. Exact masks distinguish valid unshared surfaces from background.","plans":"Manifest metric envelope and floor levels; furniture footprint proxies. Camera arrows indicate direction, not frusta."},
        "co_visibility":co_visibility,"annotation_metrics":dataset.summary["annotation_metrics"],"semantic_coverage":dataset.summary["semantic_coverage"],
        "minimum_semantic_classes":dataset.summary["minimum_semantic_classes"],"capture_wall_seconds":dataset.summary["capture_wall_seconds"],
        "fresh_render":false,"source_sha256":dataset.input_sha256,"output_sha256":outputs});
    io::write(&media.join("gallery.json"), &data)?;
    Ok(data)
}

fn choose_featured(records: &[Value]) -> Vec<Value> {
    let mut used = BTreeSet::new();
    let mut featured = Vec::new();
    for (feature, title) in [
        ("Mezzanine", "Mezzanine & stairs"),
        ("Sunken floor", "Sunken floor"),
        ("Raised floor", "Raised floor"),
        ("Exterior cut-in", "Exterior cut-in"),
    ] {
        let chosen = records
            .iter()
            .find(|r| {
                !used.contains(&r["seed"].as_u64().unwrap())
                    && r["features"].as_array().unwrap().contains(&json!(feature))
            })
            .or_else(|| {
                records
                    .iter()
                    .find(|r| !used.contains(&r["seed"].as_u64().unwrap()))
            })
            .unwrap();
        let title = if chosen["features"]
            .as_array()
            .unwrap()
            .contains(&json!(feature))
        {
            title
        } else {
            "Room program"
        };
        let seed = chosen["seed"].as_u64().unwrap();
        used.insert(seed);
        featured.push(json!({"seed":seed,"title":title}));
    }
    featured
}
fn plot_totals(metrics: &Value) -> Value {
    let mut result = BTreeMap::new();
    for key in [
        "envelope_footprint_area_m2",
        "vertical_fov_degrees",
        "sun_illuminance_lux",
        "camera_reference_baseline_m",
    ] {
        result.insert(key.to_string(), metrics["numeric"][key]["count"].clone());
    }
    for key in ["main/Chair", "main/Person"] {
        result.insert(
            key.into(),
            json!(metrics["object_counts_per_scene"][key]
                .as_object()
                .unwrap()
                .values()
                .map(|n| n.as_u64().unwrap())
                .sum::<u64>()),
        );
    }
    for key in ["main/Chair", "main/Person", "camera_path"] {
        result.insert(
            format!("{key}/placement"),
            json!(metrics["placement_heatmaps"][key]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap())
                .sum::<u64>()),
        );
    }
    json!(result)
}
fn histogram(metrics: &Value, key: &str, title: &str, unit: &str) -> String {
    let d = &metrics["numeric"][key];
    let edges: Vec<_> = d["bin_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(number)
        .collect();
    let counts: Vec<_> = d["bin_counts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    graphics::histogram(title, unit, &edges, &counts, 430, 320)
}
fn distributions(dataset: &Dataset, media: &Path, figures: &Path) -> Result<()> {
    let metrics = &dataset.metrics;
    let mut panels = Vec::new();
    for (key, title, unit) in [
        ("envelope_footprint_area_m2", "Footprint area", "m²"),
        (
            "main/Chair",
            "Chairs per interior",
            "instances, zero included",
        ),
        (
            "main/Person",
            "People per interior",
            "instances, zero included",
        ),
        ("vertical_fov_degrees", "Vertical field of view", "degrees"),
        ("sun_illuminance_lux", "Solar illumination", "lux"),
        (
            "camera_reference_baseline_m",
            "Reference camera distance",
            "metres",
        ),
    ] {
        let svg = if key.starts_with("main/") {
            let d = metrics["object_counts_per_scene"][key].as_object().unwrap();
            let mut counts = BTreeMap::new();
            for (n, c) in d {
                counts.insert(n.parse::<usize>()?, c.as_u64().unwrap());
            }
            let maximum = *counts.keys().max().unwrap();
            let values: Vec<_> = (0..=maximum)
                .map(|i| counts.get(&i).copied().unwrap_or(0))
                .collect();
            let edges: Vec<_> = (0..=maximum + 1).map(|i| i as f64 - 0.5).collect();
            graphics::histogram(title, unit, &edges, &values, 430, 320)
        } else {
            histogram(metrics, key, title, unit)
        };
        let name = if key == "main/Chair" {
            "chair"
        } else if key == "main/Person" {
            "person"
        } else {
            key
        };
        graphics::save(&svg, &media.join(format!("distribution-{name}.svg")))?;
        panels.push(graphics::raster(&svg)?);
    }
    graphics::combined(&panels, 3).save(figures.join("architecture_population.png"))?;
    let mut panels = Vec::new();
    // Feature frequencies have one shared room denominator.
    let counts = dataset.summary["feature_room_counts"].as_object().unwrap();
    let mut svg = graphics::begin(430, 320);
    graphics::text(
        &mut svg,
        20.0,
        26.0,
        16,
        &format!("Features · {} rooms", dataset.rows.len()),
    );
    for (i, (key, count)) in counts.iter().enumerate() {
        let y = 44.0 + i as f64 * 27.0;
        graphics::text(&mut svg, 16.0, y + 13.0, 12, key);
        let w = 190.0 * number(count) / dataset.rows.len() as f64;
        svg.push_str(&format!(
            r##"<rect x="188" y="{y}" width="{w}" height="17" fill="#287d69"/>"##
        ));
        graphics::text(
            &mut svg,
            383.0,
            y + 13.0,
            11,
            &format!("{:.0}", number(count)),
        );
    }
    svg.push_str("</g></svg>");
    panels.push(graphics::raster(&svg)?);
    for (key, title, unit) in [
        (
            "envelope_footprint_fraction",
            "Footprint / bounds",
            "fraction",
        ),
        ("envelope_ceiling_slope_degrees", "Roof pitch", "degrees"),
        ("envelope_footprint_area_m2", "Footprint area", "m²"),
        ("envelope_floor_min_m", "Lowest floor", "metres"),
        (
            "envelope_mezzanine_area_m2",
            "Mezzanine area",
            "m², zero included",
        ),
    ] {
        panels.push(graphics::raster(&histogram(metrics, key, title, unit))?);
    }
    let im = graphics::combined(&panels, 3);
    im.save(media.join("distributions.png"))?;
    im.save(figures.join("architecture_features.png"))?;
    let mut svg = graphics::begin(1200, 380);
    let grid = metrics["heatmap_grid_size"].as_u64().unwrap() as usize;
    for (panel, (key, label)) in [
        ("main/Chair", "Chair centers"),
        ("main/Person", "Person centers"),
        ("camera_path", "Camera path samples"),
    ]
    .iter()
    .enumerate()
    {
        let counts = metrics["placement_heatmaps"][*key].as_array().unwrap();
        let total: u64 = counts.iter().map(|n| n.as_u64().unwrap()).sum();
        let max = counts
            .iter()
            .map(|n| n.as_u64().unwrap())
            .max()
            .unwrap()
            .max(1) as f64;
        let x = panel as f64 * 400.0;
        graphics::text(
            &mut svg,
            x + 20.0,
            30.0,
            17,
            &format!("{label} · n={total}"),
        );
        for (i, n) in counts.iter().enumerate() {
            let intensity = number(n) / max;
            let color = format!(
                "#{:02x}{:02x}{:02x}",
                (242.0 - intensity * 213.0) as u8,
                (246.0 - intensity * 112.0) as u8,
                (237.0 - intensity * 124.0) as u8
            );
            let left = x + 48.0 + (i % grid) as f64 * 12.0;
            let top = 48.0 + (i / grid) as f64 * 12.0;
            svg.push_str(&format!(
                "<rect x=\"{left}\" y=\"{top}\" width=\"12\" height=\"12\" fill=\"{color}\"/>"
            ));
        }
        graphics::text(
            &mut svg,
            x + 45.0,
            362.0,
            12,
            &format!(
                "Normalized X/Z · +Z down · max {:.2}%",
                100.0 * max / total.max(1) as f64
            ),
        );
    }
    svg.push_str("</g></svg>");
    graphics::save(&svg, &media.join("placement.svg"))?;
    graphics::raster(&svg)?.save(figures.join("architecture_placement.png"))?;
    Ok(())
}

fn figures_build(
    dataset: &Dataset,
    records: &[Value],
    featured: &[Value],
    media: &Path,
    figures: &Path,
    protocol: &Protocol,
) -> Result<()> {
    let by_seed: BTreeMap<_, _> = records
        .iter()
        .map(|r| (r["seed"].as_u64().unwrap(), r))
        .collect();
    let picture = |seed: u64, camera: usize, mode: &str| {
        media.join(
            by_seed[&seed]["frames"][0]["views"][camera]["images"][mode]
                .as_str()
                .unwrap(),
        )
    };
    let aspect = protocol.height as f64 / protocol.width as f64;
    let first = featured[0]["seed"].as_u64().unwrap();
    let mut geometry = Vec::new();
    let mut examples = Vec::new();
    let mut annotations = Vec::new();
    for feature in featured {
        let seed = feature["seed"].as_u64().unwrap();
        let title = feature["title"].as_str().unwrap();
        let camera = 3usize;
        geometry.push((
            format!("{title} · seed {seed}"),
            vec![
                (
                    picture(seed, camera, "color"),
                    format!("Native RGB · C{camera} · t=0"),
                ),
                (
                    media.join(format!("s{seed}-t0-plan.png")),
                    "Same-room metric plan and section".into(),
                ),
            ],
        ));
        examples.push((
            format!("{title} · seed {seed}"),
            vec![
                (picture(seed, 0, "color"), "Camera 0 · t=0".into()),
                (picture(seed, 3, "color"), "Camera 3 · t=0".into()),
            ],
        ));
        annotations.push((
            format!("Seed {seed} · {title}"),
            ["color", "depth", "normal", "semantic"]
                .into_iter()
                .map(|m| (picture(seed, 0, m), format!("{m} · C0 · t=0")))
                .collect(),
        ));
    }
    for (rows, name, width) in [
        (&geometry[..], "architecture_geometry.jpg", 640),
        (&geometry[..2], "architecture_levels.jpg", 640),
        (&geometry[2..], "architecture_envelopes.jpg", 640),
        (&examples[..], "architecture_examples.jpg", 640),
        (&annotations[..3], "architecture_annotations.jpg", 480),
    ] {
        graphics::figure_rows(rows, &figures.join(name), width, aspect)?;
        fs::copy(figures.join(name), media.join(name))?;
    }
    let covis: Vec<_> = [
        ("color", "matched RGB"),
        ("co_visibility", "additive co-visibility"),
    ]
    .into_iter()
    .map(|(mode, label)| {
        (
            format!("Seed {first} · {label}"),
            (0..protocol.cameras)
                .map(|c| (picture(first, c, mode), format!("Camera {c} · t=0")))
                .collect(),
        )
    })
    .collect();
    graphics::figure_rows(
        &covis,
        &figures.join("architecture_co_visibility.jpg"),
        480,
        aspect,
    )?;
    fs::copy(
        figures.join("architecture_co_visibility.jpg"),
        media.join("architecture_co_visibility.jpg"),
    )?;
    let teaser = vec![(
        "Current native captures · three generated rooms".into(),
        featured[..3]
            .iter()
            .map(|f| {
                let seed = f["seed"].as_u64().unwrap();
                (
                    picture(seed, 3, "color"),
                    format!("{} · seed {seed}", f["title"].as_str().unwrap()),
                )
            })
            .collect(),
    )];
    graphics::figure_rows(
        &teaser,
        &figures.join("architecture_teaser.jpg"),
        640,
        aspect,
    )?;
    let mut hero = RgbImage::new(protocol.width * 2, protocol.height * 2);
    for (i, (seed, c)) in [
        (first, 0),
        (first, 3),
        (featured[1]["seed"].as_u64().unwrap(), 0),
        (featured[2]["seed"].as_u64().unwrap(), 1),
    ]
    .into_iter()
    .enumerate()
    {
        image::imageops::replace(
            &mut hero,
            &image::open(picture(seed, c, "color"))?.into_rgb8(),
            (i % 2) as i64 * protocol.width as i64,
            (i / 2) as i64 * protocol.height as i64,
        );
    }
    graphics::save_webp(hero.clone(), &media.join("hero.webp"))?;
    hero.save(media.parent().unwrap().join("social.jpg"))?;
    let contact: Vec<_> = records
        .chunks(4)
        .map(|rows| {
            (
                "All consecutive rooms · camera 0 · t=0".into(),
                rows.iter()
                    .map(|r| {
                        (
                            picture(r["seed"].as_u64().unwrap(), 0, "color"),
                            format!("Seed {} · {}", r["seed"], r["activity"].as_str().unwrap()),
                        )
                    })
                    .collect(),
            )
        })
        .collect();
    graphics::figure_rows(
        &contact,
        &figures.join("architecture_consecutive.jpg"),
        384,
        aspect,
    )?;
    fs::copy(
        figures.join("architecture_consecutive.jpg"),
        media.join("consecutive_rooms.jpg"),
    )?;
    // Feature-coverage examples are independent of appearance, with stable ties.
    let mut plans = Vec::new();
    let mut used = BTreeSet::new();
    let mut seen = BTreeMap::<String, usize>::new();
    for _ in 0..dataset.rows.len().min(12) {
        let score = |r: &Value| {
            features(r)
                .into_iter()
                .filter(|(_, v)| *v)
                .map(|(k, _)| 1.0 / (1 + seen.get(&k).copied().unwrap_or(0)) as f64)
                .sum::<f64>()
        };
        let selected = dataset
            .rows
            .iter()
            .filter(|r| !used.contains(&r["seed"].as_u64().unwrap()))
            .max_by(|a, b| {
                score(a)
                    .total_cmp(&score(b))
                    .then_with(|| b["seed"].as_u64().cmp(&a["seed"].as_u64()))
            })
            .unwrap();
        used.insert(selected["seed"].as_u64().unwrap());
        for (key, enabled) in features(selected) {
            if enabled {
                *seen.entry(key).or_default() += 1;
            }
        }
        let im = graphics::raster(&graphics::plan(selected, None, &[]))?;
        plans.push(image::imageops::resize(
            &im,
            600,
            306,
            image::imageops::FilterType::Lanczos3,
        ));
    }
    let im = graphics::combined(&plans, 3);
    im.save(media.join("footprints.png"))?;
    im.save(figures.join("architecture_footprints.png"))?;
    distributions(dataset, media, figures)?;
    Ok(())
}

pub fn fill(template: &str, values: &BTreeMap<&str, String>) -> Result<String> {
    let mut output = template.to_owned();
    for (key, value) in values {
        output = output.replace(&format!("@{key}@"), value);
    }
    ensure!(
        !regex::Regex::new("@[A-Z_]+@")?.is_match(&output),
        "unexpanded publication template token"
    );
    Ok(output)
}
pub fn page(gallery: &Value, stage: &Path, protocol: &Protocol) -> Result<()> {
    let scenes = gallery["scenes"].as_array().unwrap();
    let featured = gallery["featured"].as_array().unwrap();
    let seed = featured[0]["seed"].as_u64().unwrap();
    let first = scenes.iter().find(|r| r["seed"] == seed).unwrap();
    let base = "static/media/architecture/";
    let buttons=featured.iter().map(|f|format!("<button data-architecture-seed=\"{}\" aria-pressed=\"{}\" disabled>{}<span>Seed {} · four views</span></button>",f["seed"],f["seed"]==seed,io::html(f["title"].as_str().unwrap()),f["seed"])).collect::<String>();
    let modes = MODES
        .iter()
        .zip([
            "RGB",
            "Depth",
            "Normals",
            "Semantic",
            "Position",
            "Co-visibility",
        ])
        .map(|(m, l)| {
            format!(
                "<button data-architecture-mode=\"{m}\" aria-pressed=\"{}\" disabled>{l}</button>",
                *m == "color"
            )
        })
        .collect::<String>();
    let views=first["frames"][0]["views"].as_array().unwrap().iter().enumerate().map(|(i,v)|{
        let file=v["images"]["color"].as_str().unwrap();let fov=number(&v["camera"]["fov_y"]).to_degrees();
        format!("<figure data-architecture-camera=\"{i}\"><a href=\"{base}{file}\" aria-label=\"Open seed {seed} camera {i} RGB at full resolution\"><img class=\"architecture-rgb\" src=\"{base}{file}\" width=\"{}\" height=\"{}\" loading=\"lazy\" alt=\"Seed {seed}, camera {i}, t=0, RGB\"><img class=\"architecture-annotation\" src=\"{base}{file}\" width=\"{}\" height=\"{}\" loading=\"lazy\" alt=\"Seed {seed}, camera {i}, t=0, RGB comparison\"></a><figcaption><strong>Camera {i}</strong><span class=\"architecture-fov\">{fov:.1}° FOV</span><span class=\"architecture-shared\" hidden></span></figcaption></figure>",protocol.width,protocol.height,protocol.width,protocol.height)
    }).collect::<String>();
    let cohort=scenes.iter().map(|s|{let file=s["frames"][0]["views"][0]["images"]["color"].as_str().unwrap();let seed=&s["seed"];format!("<figure><a href=\"{base}{file}\"><img src=\"{base}{file}\" width=\"{}\" height=\"{}\" loading=\"lazy\" alt=\"Consecutive seed {seed}, camera 0, t=0\"></a><figcaption><strong>{seed} · {}</strong><br>{}</figcaption></figure>",protocol.width,protocol.height,io::html(s["activity"].as_str().unwrap()),io::html(&s["features"].as_array().unwrap().iter().map(|f|f.as_str().unwrap()).collect::<Vec<_>>().join(", ")))}).collect::<String>();
    let counts=gallery["feature_room_counts"].as_object().unwrap().iter().map(|(k,n)|format!("<div class=\"architecture-feature-stat\"><strong>{:.1}%</strong><span>{} · {}/{}</span></div>",100.0*number(n)/protocol.audit_rooms as f64,io::html(k),n,protocol.audit_rooms)).collect::<String>();
    let plots=[("envelope_footprint_area_m2","Actual footprint areas"),("chair","Chairs including zero"),("person","People including zero"),("vertical_fov_degrees","Vertical camera field of view"),("sun_illuminance_lux","Solar illuminance"),("camera_reference_baseline_m","Reference camera distances")].into_iter().map(|(name,title)|format!("<img src=\"{base}distribution-{name}.svg\" loading=\"lazy\" width=\"430\" height=\"320\" alt=\"{title}\"> ")).collect::<String>();
    let legend=gallery["co_visibility"]["legend"].as_array().unwrap().iter().map(|e|format!("<span class=\"legend-swatch\"><i style=\"background-color:rgb({},{},{})\"></i>Camera {}</span>",e["rgb8"][0],e["rgb8"][1],e["rgb8"][2],e["camera_index"])).collect::<String>();
    let a = &gallery["annotation_metrics"];
    let coverage = &gallery["semantic_coverage"];
    let primary = coverage["rooms_with_camera_primary_zone_humans"]
        .as_u64()
        .unwrap();
    let invisible = coverage["camera_primary_zone_human_rooms_without_person_pixels"]
        .as_array()
        .unwrap()
        .len() as u64;
    let annotation=format!("All {} views passed annotation alignment: maximum per-view depth/position p99 error <strong>{:.3} µm</strong>, reprojection p99 <strong>{:.5} pixels</strong>. Every view contains at least {} semantic classes. {}/{} rooms with people in the camera’s primary functional zone show person pixels. {}/{} rooms containing people anywhere in the main envelope never show them.",gallery["rendered_views"],number(&a["depth_position_p99_metres"]["max"])*1e6,number(&a["reprojection_p99_pixels"]["max"]),gallery["minimum_semantic_classes"],primary-invisible,primary,coverage["main_envelope_human_rooms_without_person_pixels"].as_array().unwrap().len(),coverage["rooms_with_main_envelope_humans"]);
    let visibility=format!("Production co-visibility covers all {} camera/time views. <strong>{:.1}%</strong> of valid source-pixel observations are visible in at least one peer. This pools the exact masks over all {} rooms and both endpoints; it does not count unique 3D points.",gallery["rendered_views"],100.0*number(&gallery["co_visibility"]["statistics"]["shared_fraction_valid"]),protocol.rendered_rooms);
    let values = BTreeMap::from([
        ("FEATURED", buttons),
        ("MODES", modes),
        ("VIEWS", views),
        ("COHORT", cohort),
        ("FEATURE_COUNTS", counts),
        ("DISTRIBUTIONS", plots),
        ("ACTIVITY", io::html(first["activity"].as_str().unwrap())),
        ("VISIBILITY_LEGEND", legend),
        ("ANNOTATION_STATISTICS", annotation),
        ("VISIBILITY_STATISTICS", visibility),
        ("AUDIT_ROOMS", protocol.audit_rooms.to_string()),
        ("RENDER_ROOMS", protocol.rendered_rooms.to_string()),
        ("VIEW_COUNT", gallery["rendered_views"].to_string()),
        ("FIRST_SEED", seed.to_string()),
        (
            "SIGNATURES",
            gallery["quantized_structural_signatures"].to_string(),
        ),
        ("WIDTH", protocol.width.to_string()),
        ("HEIGHT", protocol.height.to_string()),
        ("PLAN_COUNT", protocol.audit_rooms.min(12).to_string()),
        (
            "AUDIT_SEED_RANGE",
            format!(
                "{}–{}",
                protocol.seed,
                protocol.seed + protocol.audit_rooms as u64 - 1
            ),
        ),
        (
            "SEED_RANGE",
            format!(
                "{}–{}",
                protocol.seed,
                protocol.seed + protocol.rendered_rooms as u64 - 1
            ),
        ),
        (
            "DIMENSIONS",
            format!(
                "{:.1} m² footprint · {:.1}° roof pitch",
                number(&first["area_m2"]),
                number(&first["roof_pitch_degrees"])
            ),
        ),
        (
            "FEATURE_LIST",
            first["features"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| format!("<li>{}</li>", io::html(f.as_str().unwrap())))
                .collect(),
        ),
    ]);
    let body = fill(include_str!("../templates/gallery.html"), &values)?;
    let page = fill(
        include_str!("../templates/project.html"),
        &BTreeMap::from([
            ("GALLERY", body),
            ("FIRST_SEED",seed.to_string()),
            ("HERO_CAPTION",format!("Top: seed {}, cameras 0 and 3. Bottom: seed {} camera 0, seed {} camera 1. All at t = 0.",seed,featured[1]["seed"],featured[2]["seed"])),
            ("RENDER_ROOMS", protocol.rendered_rooms.to_string()),
            ("AUDIT_ROOMS", protocol.audit_rooms.to_string()),
            ("MODE_COUNT", MODES.len().to_string()),
            (
                "CRATE_VERSION",
                gallery["generator_identity"]["crate_version"]
                    .as_str()
                    .unwrap()
                    .to_string(),
            ),
        ]),
    )?;
    fs::write(stage.join("www/project/index.html"), page)?;
    Ok(())
}
