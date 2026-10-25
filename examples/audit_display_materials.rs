//! Consecutive-seed appearance audit; no renderer/model inference in generation.
//! cargo run --example audit_display_materials -- --output out/display_rug/current
use anyhow::{ensure, Result};
use bevy_zeroverse::scene::procedural_indoor::materials::{
    art, boards, program, rugs, screens, Surface,
};
use clap::Parser;
use image::{imageops::FilterType, RgbaImage};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 256)]
    samples: usize,
    /// Comma-separated diagnostic subset; all categories remain the default.
    #[arg(long, value_delimiter = ',')]
    categories: Vec<String>,
}

fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

fn distances(samples: &[Vec<f32>]) -> Value {
    let mut nearest = vec![f64::INFINITY; samples.len()];
    for i in 0..samples.len() {
        for j in 0..i {
            let d = (samples[i]
                .iter()
                .zip(&samples[j])
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / samples[i].len() as f64)
                .sqrt();
            nearest[i] = nearest[i].min(d);
            nearest[j] = nearest[j].min(d);
        }
    }
    json!({"rmse_unit_range": {"min":percentile(nearest.clone(),0.),"p10":percentile(nearest.clone(),0.1),"median":percentile(nearest.clone(),0.5)},
        "nearest_below_0_02":nearest.iter().filter(|x| **x < 0.02).count(),"nearest":nearest})
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (16..=2048).contains(&args.samples),
        "samples must be 16..2048"
    );
    fs::create_dir_all(&args.output)?;
    let mut reports = Vec::new();
    let mut html = String::from("<!doctype html><meta charset=utf-8><title>Procedural surface audit</title><style>body{background:#171b23;color:#eef0f5;font:16px system-ui;margin:32px;max-width:1400px}img{max-width:100%}a{color:#a5ceff}section{margin:40px 0}p{max-width:85ch}</style><h1>Procedural surface audit</h1><p>Consecutive seeds. Color maps are diagnostic, not PBR renders. Blank and dormant samples remain in every denominator. <a href=report.json>Measurements and parameters</a></p>");
    for category in ["screen", "phone", "television", "whiteboard", "rug", "art"] {
        if !args.categories.is_empty() && !args.categories.iter().any(|c| c == category) {
            continue;
        }
        let mut hashes = BTreeSet::new();
        let mut thumbs = Vec::new();
        let mut structures = Vec::new();
        let mut active = Vec::new();
        let mut active_structures = Vec::new();
        let mut active_hashes = BTreeSet::new();
        let mut low_information = 0;
        let mut times = Vec::new();
        let mut rows = Vec::new();
        let mut grid = RgbaImage::new(8 * 160, 8 * 160);
        for seed in 0..args.samples as u64 {
            let start = Instant::now();
            let (pixels, size, parameters) = match category {
                "screen" => (screens::pixels(seed), 256, json!(screens::parameters(seed))),
                "phone" => (
                    screens::mobile_pixels(seed),
                    256,
                    json!(screens::parameters_for(seed, screens::DisplayKind::Phone)),
                ),
                "television" => (
                    screens::television_pixels(seed),
                    256,
                    json!(screens::parameters_for(
                        seed,
                        screens::DisplayKind::Television
                    )),
                ),
                "whiteboard" => (boards::pixels(seed), 256, json!(boards::parameters(seed))),
                "art" => (art::pixels(seed), 256, json!(art::parameters(seed))),
                _ => {
                    let r = rugs::recipe(
                        &program::sample(seed)[Surface::FabricAlt as usize],
                        rugs::slot(seed),
                    );
                    r.validate().map_err(anyhow::Error::msg)?;
                    let maps = r.maps(0);
                    let n = maps[0].width();
                    (
                        maps[0].data.as_ref().unwrap()[..(n * n * 4) as usize].to_vec(),
                        n,
                        json!(r),
                    )
                }
            };
            times.push(start.elapsed().as_secs_f64() * 1000.);
            ensure!(
                pixels.len() == (size * size * 4) as usize,
                "invalid map size"
            );
            ensure!(
                pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
                "non-opaque content"
            );
            let hash = format!("{:x}", Sha256::digest(&pixels));
            hashes.insert(hash.clone());
            let image = RgbaImage::from_raw(size, size, pixels).unwrap();
            let thumb = image::imageops::resize(&image, 16, 16, FilterType::Triangle);
            let rgb: Vec<_> = thumb
                .pixels()
                .flat_map(|p| p.0[..3].iter().map(|c| *c as f32 / 255.))
                .collect();
            let mut luma: Vec<_> = rgb
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722)
                .collect();
            let mean = luma.iter().sum::<f32>() / luma.len() as f32;
            luma.iter_mut().for_each(|v| *v -= mean);
            let quiet = parameters["dormant"].as_bool().unwrap_or(false)
                || parameters["blank"].as_bool().unwrap_or(false);
            if quiet {
                low_information += 1;
            } else {
                active.push(rgb.clone());
                active_structures.push(luma.clone());
                active_hashes.insert(hash.clone());
            }
            thumbs.push(rgb);
            structures.push(luma);
            if seed < 64 {
                let preview = image::imageops::resize(&image, 156, 156, FilterType::Triangle);
                image::imageops::replace(
                    &mut grid,
                    &preview,
                    (seed % 8 * 160) as i64,
                    (seed / 8 * 160) as i64,
                );
                if seed < 8 {
                    image.save(args.output.join(format!("{category}-{seed}.png")))?;
                }
            }
            rows.push(json!({"seed":seed,"hash":hash,"parameters":parameters,"mean_luma":mean}));
        }
        grid.save(args.output.join(format!("{category}.png")))?;
        reports.push(json!({"category":category,"samples":args.samples,"exact_duplicates":args.samples-hashes.len(),
            "low_frequency_rgb":distances(&thumbs),"mean_removed_luminance":distances(&structures),
            "dormant_or_blank":low_information,
            "active_supplement":{"samples":active.len(),"exact_duplicates":active.len()-active_hashes.len(),"low_frequency_rgb":distances(&active),"mean_removed_luminance":distances(&active_structures)},
            "synthesis_ms":{"p50":percentile(times.clone(),0.5),"p95":percentile(times,0.95)},"records":rows}));
        html.push_str(&format!("<section><h2>{category}</h2><p>First 64 seeds, left to right, top to bottom.</p><img src={category}.png></section>"));
        println!(
            "{category}: {} samples, {} exact duplicates",
            args.samples,
            args.samples - hashes.len()
        );
    }
    fs::write(args.output.join("index.html"), html)?;
    ensure!(!reports.is_empty(), "no recognized content categories");
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version":2,"provenance":bevy_zeroverse::provenance::capture_provenance(),
            "protocol":"Consecutive content seeds from zero; 16x16 area-filtered RGB and mean-removed luminance RMSE; all pairs, no sample rejection. Near-duplicate threshold 0.02 is a diagnostic, not a learned perceptual metric. Rug timing includes all three CPU maps and mip chains; display timing is content rasterization only. Warm process, monotonic wall clock.",
            "limitations":"Not a memorization or downstream transfer test. Low-information dormant/blank surfaces are valid. Raw content is not rendered PBR. Palette residency is bounded, so within-room instances can deliberately reuse finishes.",
            "categories":reports
        }))?,
    )?;
    Ok(())
}
