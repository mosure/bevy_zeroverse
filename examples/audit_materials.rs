//! CPU qualification of recorded recipes and actual production map diversity.
//! `cargo run --example audit_materials -- out/material-audit.json`
use bevy_zeroverse::scene::procedural_indoor::materials::{program, Surface};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

fn quantiles(mut xs: Vec<f64>) -> Value {
    xs.sort_by(f64::total_cmp);
    let at = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
    json!({"n":xs.len(),"min":at(0.),"p05":at(0.05),"median":at(0.5),"p95":at(0.95),"max":at(1.)})
}
fn moments(xs: impl Iterator<Item = f64>) -> [f64; 2] {
    let values: Vec<_> = xs.collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    [
        mean,
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt(),
    ]
}
fn participation(rows: &[Vec<f64>]) -> Value {
    let n = rows.len();
    let dim = rows[0].len();
    let mean: Vec<_> = (0..dim)
        .map(|k| rows.iter().map(|r| r[k]).sum::<f64>() / n as f64)
        .collect();
    let centered: Vec<Vec<_>> = rows
        .iter()
        .map(|r| r.iter().zip(&mean).map(|(v, m)| v - m).collect())
        .collect();
    let mut trace = 0.;
    let mut squared = 0.;
    for (i, a) in centered.iter().enumerate() {
        for (j, b) in centered.iter().enumerate() {
            let cov = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>() / (n - 1) as f64;
            if i == j {
                trace += cov;
            }
            squared += cov * cov;
        }
    }
    json!({"samples":n,"descriptor_dimensions":dim,"variance_trace":trace,"effective_covariance_rank":if squared>0. {trace*trace/squared} else {0.}})
}

fn map_record(
    seed: u64,
    surface: String,
    floor: u32,
    group: usize,
    r: program::MaterialRecipe,
    linear: &[f64; 256],
) -> anyhow::Result<(String, String, Vec<f64>, Value)> {
    let maps = r.maps(floor);
    let size = maps[0].width() as usize;
    let bytes = maps.each_ref().map(|im| im.data.as_ref().unwrap());
    let mut hash = Sha256::new();
    for (i, b) in bytes.iter().enumerate() {
        if r.surface != Surface::Chrome || i != 1 {
            hash.update(b);
        }
    }
    let fingerprint = format!("{:x}", hash.finalize());
    let color = &bytes[0][..size * size * 4];
    let normals = &bytes[1][..size * size * 4];
    let data = &bytes[2][..size * size * 4];
    let albedo = moments(
        color
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| (p[0] as f64 + p[1] as f64 + p[2] as f64) / 765.),
    );
    let roughness = moments(data.as_chunks::<4>().0.iter().map(|p| p[1] as f64 / 255.));
    let slope = (normals
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            let n = p.map(|c| c as f64 / 127.5 - 1.);
            (n[0].powi(2) + n[1].powi(2)) / n[2].max(0.01).powi(2)
        })
        .sum::<f64>()
        / (size * size) as f64)
        .sqrt();
    let metal = if r.surface == Surface::Chrome { 255 } else { 0 };
    anyhow::ensure!(
        data.as_chunks::<4>().0.iter().all(|p| p[2] == metal),
        "incorrect substrate metalness"
    );
    let base = r.texture_base_color(floor);
    let base = bevy::prelude::Color::srgb(base[0], base[1], base[2]).to_linear();
    let base = [base.red as f64, base.green as f64, base.blue as f64];
    let mut descriptor = Vec::new();
    let block = size / 8;
    for y in 0..8 {
        for x in 0..8 {
            let mut rgb = [0.; 3];
            let mut rough = 0.;
            let mut relief = 0.;
            for dy in 0..block {
                for dx in 0..block {
                    let i = ((y * block + dy) * size + x * block + dx) * 4;
                    for c in 0..3 {
                        rgb[c] += linear[color[i + c] as usize] * base[c];
                    }
                    if r.surface != Surface::Chrome {
                        let n = [normals[i], normals[i + 1], normals[i + 2]]
                            .map(|b| b as f64 / 127.5 - 1.);
                        relief += (n[0] * n[0] + n[1] * n[1]) / n[2].max(0.01).powi(2);
                    }
                    rough += data[i + 1] as f64 / 255.;
                }
            }
            let pixels = (block * block) as f64;
            descriptor.extend([
                rgb[0] / pixels,
                rgb[1] / pixels,
                rgb[2] / pixels,
                rough / pixels,
                (relief / pixels).sqrt(),
            ]);
        }
    }
    let record = json!({"seed":seed,"surface":surface,"structure_group":group,"floor_style":floor,"width":size,"normal_map_bound":r.surface!=Surface::Chrome,"sha256":fingerprint,
            "albedo_mean_sd":albedo,"roughness_mean_sd":roughness,"normal_slope_rms":slope,"texture_base_color":r.texture_base_color(floor),"recipe":r});
    Ok((surface, fingerprint, descriptor, record))
}

fn standardized(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len() as f64;
    let means: Vec<_> = (0..rows[0].len())
        .map(|k| rows.iter().map(|r| r[k]).sum::<f64>() / n)
        .collect();
    let scales: Vec<_> = means
        .iter()
        .enumerate()
        .map(|(k, m)| {
            (rows.iter().map(|r| (r[k] - m).powi(2)).sum::<f64>() / n)
                .sqrt()
                .max(1e-6)
        })
        .collect();
    rows.iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .map(|(k, v)| (v - means[k]) / scales[k])
                .collect()
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("out/material-audit.json".into()),
    );
    let focus = std::env::args().any(|a| a == "--focus-finishes");
    let map_seeds: u64 = std::env::args()
        .find_map(|a| a.strip_prefix("--map-seeds=").map(str::to_owned))
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(32);
    anyhow::ensure!((2..=512).contains(&map_seeds), "map-seeds must be 2..512");
    let mut parameters = BTreeMap::<String, BTreeMap<String, Vec<f64>>>::new();
    let mut topologies = BTreeMap::<String, usize>::new();
    let mut hues = BTreeMap::<String, [usize; 12]>::new();
    let mut identities = BTreeSet::new();
    let mut recipe_count = 0;
    for seed in 0..1024 {
        let mut cohort: Vec<_> = program::sample(seed)
            .into_iter()
            .filter(|r| r.surface != Surface::Floor)
            .map(|r| (format!("{:?}", r.surface), r))
            .collect();
        for floor in 0..3 {
            cohort.push((
                format!("Floor/{floor}"),
                program::sample_with_floor(seed, floor)
                    .unwrap()
                    .remove(Surface::Floor as usize),
            ));
        }
        for (surface, r) in cohort {
            r.validate().map_err(anyhow::Error::msg)?;
            recipe_count += 1;
            let mut fields = vec![
                ("roughness", r.roughness),
                ("repeat_m", r.period_m),
                ("relief_m", r.relief_m),
            ];
            if let Some(t) = &r
                .textile
                .as_ref()
                .filter(|_| surface != "Floor/0" && surface != "Floor/2")
            {
                t.validate().map_err(anyhow::Error::msg)?;
                let key = format!(
                    "{}:r{}-f{}-a{}-b{}-h{}",
                    surface, t.repeat, t.float_length, t.advance, t.bundle, t.herringbone
                );
                *topologies.entry(key).or_default() += 1;
                fields.extend([
                    ("yarn_spacing_m", r.period_m / t.yarns[1] as f32),
                    ("yarn_width", t.width[0]),
                    ("slub", t.slub),
                    ("fuzz", t.fuzz),
                    ("lustre", t.lustre),
                ]);
            }
            if let Some(l) = &r.leather {
                l.validate().map_err(anyhow::Error::msg)?;
                fields.extend([
                    ("grain_spacing_m", r.period_m / l.cells[1] as f32),
                    ("grain_stretch", l.stretch),
                    ("nap", l.nap),
                    ("polish", l.polish),
                    ("coat", l.coat),
                ]);
            }
            if let Some(m) = &r
                .mineral
                .as_ref()
                .filter(|_| surface != "Floor/0" && surface != "Floor/1")
            {
                let tile_repeat = if surface == "Floor/2" {
                    r.floor_repetitions(2)[1]
                } else {
                    1.
                };
                fields.extend([
                    ("aggregate_exposure", m.aggregate_exposure),
                    ("porosity", m.porosity),
                    ("marble_mix", m.marble_mix),
                    ("polish", m.polish),
                    (
                        "aggregate_spacing_m",
                        r.period_m / tile_repeat / m.aggregate_cells[1] as f32,
                    ),
                ]);
                if m.marble_mix > 0. {
                    fields.push(("vein_width", m.vein_width));
                }
                if let Some(c) = &m.casting {
                    fields.extend([
                        ("formwork", c.formwork),
                        ("board_width_m", c.board_width_m),
                        ("form_relief_m", c.form_relief_m),
                        ("cure_variation", c.cure_variation),
                        ("trowel", c.trowel),
                        ("bughole_density", c.bughole_density),
                        ("bughole_radius_m", c.bughole_radius_m),
                        ("bughole_depth_m", c.bughole_depth_m),
                        ("sand_exposure", c.sand_exposure),
                    ]);
                }
            }
            if let Some(c) = &r.coating {
                fields.push(("gloss", c.gloss));
                if c.glaze.is_none() {
                    fields.extend([
                        ("texture_mix", c.texture_mix),
                        ("knockdown", c.knockdown),
                        ("roller", c.roller),
                        ("trowel", c.trowel),
                        ("pinholes", c.pinholes),
                    ]);
                    if c.application.is_none() {
                        fields.push(("stipple_spacing_m", r.period_m / c.cells[1] as f32));
                    }
                }
                fields.extend([
                    ("coat", c.clearcoat),
                    ("coat_roughness", c.coat_roughness),
                    ("crackle", c.crackle),
                ]);
                if let Some(g) = &c.glaze {
                    fields.extend([
                        ("body_grain_m", g.body_grain_m),
                        ("cloud_scale_m", g.cloud_scale_m),
                        ("reactive_mix", g.reactive_mix),
                        ("thickness_variation", g.thickness_variation),
                        ("throwing", g.throwing),
                        ("turning_pitch_m", g.turning_pitch_m),
                        ("speckle_density", g.speckle_density),
                        ("speckle_radius_m", g.speckle_radius_m),
                        ("crack_spacing_m", g.crack_spacing_m),
                    ]);
                }
                if let Some(a) = &c.application {
                    fields.extend([
                        ("spray_spacing_m", a.spray_spacing_m),
                        ("roller_spacing_m", a.roller_spacing_m),
                        ("roller_stretch", a.roller_stretch),
                        ("orange_peel", a.orange_peel),
                        ("orange_peel_m", a.orange_peel_m),
                        ("brush", a.brush),
                        ("trowel_scale_m", a.trowel_scale_m),
                        ("repair_mix", a.repair_mix),
                    ]);
                }
            }
            if let Some(w) = &r
                .wood
                .as_ref()
                .filter(|_| surface != "Floor/1" && surface != "Floor/2")
            {
                fields.extend([
                    ("stain_strength", w.stain_strength),
                    ("bleach", w.bleach),
                    ("pore_fill", w.pore_fill),
                    ("clearcoat", w.clearcoat),
                    ("coat_roughness", w.coat_roughness),
                ]);
            }
            if let Some(l) = &r.leaf {
                fields.extend([
                    ("vein_pairs", l.vein_pairs as f32),
                    ("vein_rake", l.vein_rake),
                    ("variegation", l.variegation),
                    ("wax", l.wax),
                    ("transmission", l.transmission),
                ]);
            }
            let c = bevy::color::Hsla::from(bevy::prelude::Color::srgb(
                r.color[0], r.color[1], r.color[2],
            ));
            fields.extend([("saturation", c.saturation), ("lightness", c.lightness)]);
            hues.entry(surface.clone()).or_default()[(c.hue / 30.).floor() as usize % 12] += 1;
            for (name, v) in fields {
                parameters
                    .entry(surface.clone())
                    .or_default()
                    .entry(name.into())
                    .or_default()
                    .push(v as f64);
            }
            identities.insert(format!("{:x}", Sha256::digest(serde_json::to_vec(&r)?)));
        }
    }
    let linear: [f64; 256] = std::array::from_fn(|i| {
        let c = i as f64 / 255.;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let mut records = Vec::new();
    let mut signatures = BTreeSet::new();
    let mut descriptors = BTreeMap::<String, Vec<Vec<f64>>>::new();
    let mut jobs_ready = Vec::new();
    for seed in 0..map_seeds {
        let recipes = program::sample(seed);
        let mut jobs = Vec::new();
        for surface in [
            Surface::Wood,
            Surface::WoodEdge,
            Surface::Fabric,
            Surface::FabricAlt,
            Surface::Leather,
            Surface::Paint,
            Surface::Accent,
            Surface::Ceiling,
            Surface::Concrete,
            Surface::Ceramic,
            Surface::Terracotta,
            Surface::Soil,
            Surface::Leaf,
            Surface::LeafLight,
            Surface::LeafVariegated,
            Surface::Bark,
            Surface::Metal,
            Surface::Chrome,
            Surface::Plastic,
            Surface::Rubber,
            Surface::Paper,
        ] {
            if focus
                && !matches!(
                    surface,
                    Surface::Paint
                        | Surface::Accent
                        | Surface::Ceiling
                        | Surface::Concrete
                        | Surface::Ceramic
                        | Surface::Terracotta
                )
            {
                continue;
            }
            for group in 0
                ..bevy_zeroverse::scene::procedural_indoor::materials::variants::structure_count(
                    surface,
                )
            {
                jobs.push((
                    format!("{surface:?}"),
                    0,
                    group,
                    recipes[surface as usize].variant(group),
                ));
            }
        }
        for floor in 0..3 {
            if focus && floor != 2 {
                continue;
            }
            jobs.push((
                format!("Floor/{floor}"),
                floor,
                0,
                program::sample_with_floor(seed, floor)
                    .unwrap()
                    .remove(Surface::Floor as usize),
            ));
        }
        for (surface, floor, group, r) in jobs {
            jobs_ready.push((seed, surface, floor, group, r));
        }
    }
    let pool = bevy::tasks::TaskPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(4)))
        .build();
    let ready = pool.scope(|scope| {
        for (seed, surface, floor, group, r) in &jobs_ready {
            let linear = &linear;
            scope.spawn(async move {
                map_record(*seed, surface.clone(), *floor, *group, r.clone(), linear)
            });
        }
    });
    for record in ready {
        let (surface, hash, descriptor, record) = record?;
        signatures.insert(hash);
        descriptors.entry(surface).or_default().push(descriptor);
        records.push(record);
    }
    let nearest: BTreeMap<_, _> = descriptors
        .iter()
        .map(|(surface, rows)| {
            let values = rows
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    rows.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, b)| {
                            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
                                / a.len() as f64
                        })
                        .min_by(f64::total_cmp)
                        .unwrap()
                })
                .collect();
            (surface.clone(), quantiles(values))
        })
        .collect();
    let parameters: BTreeMap<_, _> = parameters
        .into_iter()
        .map(|(s, fields)| {
            (
                s,
                fields
                    .into_iter()
                    .map(|(k, v)| (k, quantiles(v)))
                    .collect::<BTreeMap<_, _>>(),
            )
        })
        .collect();
    let standardized_variance: BTreeMap<_, _> = descriptors
        .iter()
        .map(|(k, rows)| (k.clone(), participation(&standardized(rows))))
        .collect();
    let shape_variance: BTreeMap<_, _> = descriptors
        .iter()
        .map(|(k, rows)| {
            let centered: Vec<Vec<f64>> = rows
                .iter()
                .map(|row| {
                    let mean: [f64; 5] =
                        std::array::from_fn(|c| row.iter().skip(c).step_by(5).sum::<f64>() / 64.);
                    row.iter()
                        .enumerate()
                        .map(|(i, v)| v - mean[i % 5])
                        .collect()
                })
                .collect();
            (k.clone(), participation(&standardized(&centered)))
        })
        .collect();
    let variance: BTreeMap<_, _> = descriptors
        .iter()
        .map(|(k, v)| (k.clone(), participation(v)))
        .collect();
    let receipt = json!({"schema_version":3,"engine":bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,
        "build_provenance":bevy_zeroverse::provenance::capture_provenance(),
        "recipe_seed_range":[0,1023],"recipe_count":recipe_count,"unique_recipe_sha256":identities.len(),
        "map_seed_range":[0,map_seeds-1],"focused_finishes":focus,"map_set_count":records.len(),"unique_map_sha256":signatures.len(),
        "exact_map_duplicates":records.len()-signatures.len(),"weaving_topologies":topologies,
        "parameters":parameters,"hue_bins_30_degrees":hues,"nearest_map_descriptor_mae":nearest,"map_descriptor_variance":variance,"standardized_descriptor_variance":standardized_variance,"shape_only_standardized_variance":shape_variance,
        "maps":records,"scope":"Actual 256x256 substrate and 512x512 concrete/floor-atlas production maps and recorded programs. Descriptors use 8x8 means of linear RGB reflectance with the production base multiplier, perceptual roughness and RMS normal slope. Raw covariance rank retains mixed units. Standardized ranks use cohort per-coordinate standard deviations with a 1e-6 floor. Shape-only descriptors first remove each sample's spatial mean independently in its five channels. Chrome excludes its unbound normal map from fingerprints and descriptors. Neither rank is a real-image embedding or transfer qualification.",
        "structure_groups_per_ceramic_role":3,"structure_groups_per_upholstery_role":3,"structure_groups_per_timber_foliage_role":2,"finish_slots_per_role":12});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&receipt)?)?;
    println!(
        "1024 seeds / {} unique recipes; {} unique production map sets / {}, written {}",
        identities.len(),
        signatures.len(),
        records.len(),
        output.display()
    );
    Ok(())
}
