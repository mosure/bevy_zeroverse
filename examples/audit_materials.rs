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
fn main() -> anyhow::Result<()> {
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("out/material-audit.json".into()),
    );
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
                    ("vein_width", m.vein_width),
                    (
                        "aggregate_spacing_m",
                        r.period_m / tile_repeat / m.aggregate_cells[1] as f32,
                    ),
                ]);
            }
            if let Some(c) = &r.coating {
                fields.extend([
                    ("texture_mix", c.texture_mix),
                    ("gloss", c.gloss),
                    ("knockdown", c.knockdown),
                    ("roller", c.roller),
                    ("trowel", c.trowel),
                    ("pinholes", c.pinholes),
                    ("stipple_spacing_m", r.period_m / c.cells[1] as f32),
                ]);
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
    for seed in 0..32 {
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
            signatures.insert(fingerprint.clone());
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
            descriptors
                .entry(surface.clone())
                .or_default()
                .push(descriptor);
            records.push(json!({"seed":seed,"surface":surface,"structure_group":group,"floor_style":floor,"width":size,"normal_map_bound":r.surface!=Surface::Chrome,"sha256":fingerprint,
                    "albedo_mean_sd":albedo,"roughness_mean_sd":roughness,"normal_slope_rms":slope,"texture_base_color":r.texture_base_color(floor),"recipe":r}));
        }
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
    let variance: BTreeMap<_, _> = descriptors
        .iter()
        .map(|(k, v)| (k.clone(), participation(v)))
        .collect();
    let receipt = json!({"schema_version":2,"engine":bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,
        "build_provenance":bevy_zeroverse::provenance::capture_provenance(),
        "recipe_seed_range":[0,1023],"recipe_count":recipe_count,"unique_recipe_sha256":identities.len(),
        "map_seed_range":[0,31],"map_set_count":records.len(),"unique_map_sha256":signatures.len(),
        "exact_map_duplicates":records.len()-signatures.len(),"weaving_topologies":topologies,
        "parameters":parameters,"hue_bins_30_degrees":hues,"nearest_map_descriptor_mae":nearest,"map_descriptor_variance":variance,
        "maps":records,"scope":"Actual 256x256 substrate and 512x512 floor-atlas production maps and recorded programs. Descriptors use 8x8 means of linear RGB reflectance with the production base multiplier, perceptual roughness and RMS normal slope. Covariance rank uses the raw mixed descriptor units. Chrome excludes its unbound normal map from fingerprints and descriptors. This is not a real-image embedding or transfer qualification.",
        "structure_groups_per_upholstery_role":3,"structure_groups_per_timber_foliage_role":2,"finish_slots_per_role":12});
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
