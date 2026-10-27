//! Current-state distribution and equal-metric-footprint production-map review.
use super::*;

fn distribution(mut xs: Vec<f64>) -> serde_json::Value {
    xs.sort_by(f64::total_cmp);
    let at = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
    json!({"min":at(0.),"p05":at(0.05),"median":at(0.5),"p95":at(0.95),"max":at(1.)})
}

// Read the real baked mip chain with bilinear repeat filtering. Every swatch
// shows the same 4 m square; showing one atlas per tile would hide scale changes.
fn physical_patch(maps: &[Image; 3], period: f32, size: u32) -> [Image; 3] {
    std::array::from_fn(|map| {
        let source = &maps[map];
        let footprint = 4. / period * source.width() as f32 / size as f32;
        let level = (footprint.log2().max(0.).round() as u32)
            .min(source.texture_descriptor.mip_level_count - 1);
        let n = (source.width() >> level) as i32;
        let offset: usize = (0..level)
            .map(|l| ((source.width() >> l).pow(2) * 4) as usize)
            .sum();
        let bytes = source.data.as_ref().unwrap();
        let mut data = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                let uv = [x, y].map(|p| {
                    ((p as f32 + 0.5) / size as f32 * 4. / period).fract() * n as f32 - 0.5
                });
                let sample = |dx: i32, dy: i32, c: usize| {
                    let px = (uv[0].floor() as i32 + dx).rem_euclid(n);
                    let py = (uv[1].floor() as i32 + dy).rem_euclid(n);
                    let v = bytes[offset + ((py * n + px) * 4) as usize + c] as f32 / 255.;
                    if map == 0 {
                        super::super::srgb_to_linear(v)
                    } else {
                        v
                    }
                };
                let [a, b] = uv.map(|v| v - v.floor());
                for c in 0..3 {
                    let v = sample(0, 0, c) * (1. - a) * (1. - b)
                        + sample(1, 0, c) * a * (1. - b)
                        + sample(0, 1, c) * (1. - a) * b
                        + sample(1, 1, c) * a * b;
                    let v = if map == 0 {
                        super::super::linear_to_srgb(v)
                    } else {
                        v
                    };
                    data.push((v * 255.).round() as u8);
                }
                data.push(255);
            }
        }
        let mut result = source.clone();
        result.texture_descriptor.size.width = size;
        result.texture_descriptor.size.height = size;
        result.texture_descriptor.mip_level_count = 1;
        result.data = Some(data);
        result
    })
}

#[test]
#[ignore = "bounded current floor scale/map study, CPU only; writes local out/floor_scale"]
fn floor_scale_qualification() {
    let dir = Path::new("out/floor_scale");
    std::fs::create_dir_all(dir).unwrap();
    let mut groups = Vec::new();
    let mut html = String::from("<!doctype html><meta charset='utf-8'><title>Floor scale qualification</title><style>body{background:#17212a;color:#eef3f5;font:16px system-ui;max-width:1200px;margin:32px auto}img{width:100%}section{margin:40px 0}p{line-height:1.5}</style><h1>Floor scales · current sampler</h1><p>Every patch covers 4 × 4 metres. Rows: albedo, shading normal, roughness, fixed GGX lighting diagnostic. These are filtered production maps, not Bevy room renders. Columns sample width quantiles; they do not indicate population frequencies.</p>");
    for style in 0..3 {
        let start = Instant::now();
        let recipes: Vec<_> = (0..4096)
            .map(|seed| {
                let r = program::sample_with_floor(seed, style)
                    .unwrap()
                    .remove(Surface::Floor as usize);
                r.validate().unwrap();
                r
            })
            .collect();
        let sampler_us = start.elapsed().as_secs_f64() * 1e6 / recipes.len() as f64;
        let mut layouts = BTreeMap::<String, usize>::new();
        let mut ordered: Vec<_> = recipes.iter().collect();
        let width = |r: &MaterialRecipe| r.period_m / r.flooring.as_ref().unwrap().cells[0] as f32;
        ordered.sort_by(|a, b| width(a).total_cmp(&width(b)));
        let mut gallery = RgbImage::new(6 * 192, 4 * 192);
        let mut samples = Vec::new();
        for (i, q) in [0.01, 0.20, 0.40, 0.60, 0.80, 0.99].into_iter().enumerate() {
            let r = ordered[((ordered.len() - 1) as f64 * q) as usize];
            let start = Instant::now();
            let maps = r.maps(style);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let (hash, roughness) = check_maps(r, style, &maps);
            if i == 0 {
                let replay: MaterialRecipe =
                    serde_json::from_slice(&serde_json::to_vec(r).unwrap()).unwrap();
                assert_eq!(hash, check_maps(&replay, style, &replay.maps(style)).0);
            }
            let patch = physical_patch(&maps, r.period_m, 192);
            image::imageops::replace(
                &mut gallery,
                &swatch(r, style, &patch, 192),
                i as i64 * 192,
                0,
            );
            samples.push(json!({"quantile":q,"recipe":r,"map_sha256":hash,"roughness_range":roughness,"cpu_maps_ms":ms}));
        }
        let name = ["timber", "carpet", "mineral"][style as usize];
        gallery.save(dir.join(format!("{name}.png"))).unwrap();
        html.push_str(&format!(
            "<section><h2>{name}</h2><img src='{name}.png'></section>"
        ));
        for r in &recipes {
            *layouts
                .entry(format!("{:?}", r.flooring.as_ref().unwrap().layout))
                .or_default() += 1;
        }
        let d = |f: fn(&MaterialRecipe) -> f64| distribution(recipes.iter().map(f).collect());
        groups.push(json!({"style":name,"samples":recipes.len(),"layouts":layouts,
            "period_m":d(|r| r.period_m as f64),
            "cell_width_m":d(|r| (r.period_m/r.flooring.as_ref().unwrap().cells[0] as f32) as f64),
            "cell_length_m":d(|r| (r.period_m/r.flooring.as_ref().unwrap().cells[1] as f32) as f64),
            "variable_stock_count":recipes.iter().filter(|r| r.flooring.as_ref().unwrap().stock.as_ref().unwrap().cut_jitter != [0.;2]).count(),
            "broadloom_count":if style==1 {recipes.iter().filter(|r| r.flooring.as_ref().unwrap().tile_definition == 0.).count()}else{0},
            "carpet_motif_m":if style==1 {d(|r| (r.period_m/r.carpet.as_ref().unwrap().repeats[0] as f32) as f64)}else{serde_json::Value::Null},
            "all_material_recipes_sampling_us_per_room":sampler_us,"gallery_samples":samples}));
    }
    std::fs::write(dir.join("index.html"), html).unwrap();
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({
        "schema":1,"provenance":crate::provenance::capture_provenance(),"groups":groups,
        "recipes_validated":12288,"map_triplets":18,"replayed_triplets":3,
        "maximum_texture_resolution":512,"physical_patch_metres":4,
        "limits":["Cell dimensions are installation units, not individual strip dimensions for basket/parquet/hex layouts",
        "Local CPU map diagnostics do not establish downstream transfer, hardware throughput or photographic realism"]
    })).unwrap()).unwrap();
}
