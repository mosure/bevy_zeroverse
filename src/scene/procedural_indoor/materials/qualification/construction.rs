//! Bounded counterfactual maps and distribution diagnostics. These measurements
//! deliberately distinguish physical descriptors from perceptual dataset size.
use super::*;
use crate::scene::procedural_indoor::materials::{construction as process, rugs};

pub(crate) fn descriptor(r: &MaterialRecipe, floor: u32) -> Vec<f64> {
    let mut sums = [0_f64; 8];
    let period = r.period_uv();
    for y in 0..12 {
        for x in 0..12 {
            let t = r.texel((x as f32 + 0.31) / 12., (y as f32 + 0.17) / 12., floor);
            for (i, c) in t.color.iter().enumerate() {
                sums[i] += *c as f64;
            }
            let l = t.color.iter().sum::<f32>() as f64 / 3.;
            sums[3] += l * l;
            sums[4] += t.roughness as f64;
            sums[5] += (t.roughness as f64).powi(2);
            sums[6] += t.height as f64 * 1000.;
            sums[7] += (t.height as f64 * 1000.).powi(2);
        }
    }
    let mut f: Vec<_> = sums.into_iter().map(|v| v / 144.).collect();
    f.extend([period.x as f64, period.y as f64]);
    f
}

fn statistics(features: &[Vec<f64>]) -> serde_json::Value {
    let n = features.len() as f64;
    let mean: Vec<_> = (0..10)
        .map(|i| features.iter().map(|f| f[i]).sum::<f64>() / n)
        .collect();
    let variance: Vec<_> = (0..10)
        .map(|i| {
            features
                .iter()
                .map(|f| (f[i] - mean[i]).powi(2))
                .sum::<f64>()
                / n
        })
        .collect();
    // Fixed physical bins, identical for both samplers. This is a diagnostic
    // occupancy, not a learned perceptual distance or an effective dataset size.
    let widths = [0.05, 0.05, 0.05, 0.025, 0.05, 0.05, 0.025, 0.01, 0.10, 0.10];
    let cells: BTreeSet<Vec<i64>> = features
        .iter()
        .map(|f| {
            f.iter()
                .zip(widths)
                .map(|(v, w)| (v / w).floor() as i64)
                .collect()
        })
        .collect();
    json!({"samples":features.len(),"mean":mean,"variance":variance,"occupied_cells":cells.len(),"bin_widths":widths})
}

fn recipe(seed: u64, surface: Surface, floor: u32, rug: bool, expanded: bool) -> MaterialRecipe {
    let mut recipes = if expanded {
        program::sample(seed)
    } else {
        program::sample_base(seed)
    };
    if expanded {
        program::floor_finish(&mut recipes, floor);
    } else {
        program::floor_finish_base(&mut recipes, floor);
    }
    let r = recipes[surface as usize].clone();
    if rug {
        rugs::recipe(&r, 0)
    } else {
        r
    }
}

pub(crate) fn construction_study() {
    let dir = Path::new("out/material_expansion/study");
    std::fs::create_dir_all(dir).unwrap();
    let mut categories: Vec<_> = program::SURFACES
        .into_iter()
        .filter_map(|s| process::labels(s, 0).map(|labels| (format!("{s:?}"), s, 0, false, labels)))
        .collect();
    for floor in [1, 2] {
        categories.push((
            format!("Floor-{floor}"),
            Surface::Floor,
            floor,
            false,
            process::labels(Surface::Floor, floor).unwrap(),
        ));
    }
    categories.push((
        "Rug".into(),
        Surface::FabricAlt,
        1,
        true,
        ["short heather pile", "cut pile", "woven pile", "dense felt"],
    ));
    let mut report = Vec::new();
    let mut html = String::from("<!doctype html><meta charset=utf-8><title>Material construction study</title><style>body{background:#111827;color:#edf2f7;font:16px system-ui;max-width:1050px;margin:35px auto}img{max-width:100%;border-radius:8px}section{margin:40px 0}p{line-height:1.6}.names{display:grid;grid-template-columns:repeat(4,1fr);width:768px;max-width:100%;gap:6px}</style><h1>Material construction space</h1><p>Same seed and geometry-independent pigments; four physical construction poles per category. Rows: production albedo, tangent normal, roughness and a fixed CPU GGX diagnostic. The diagnostic is not a Bevy render. All mip levels are checked. Local qualification only.</p>");
    for (name, surface, floor, rug, labels) in &categories {
        let old_features: Vec<_> = (0..128)
            .map(|s| descriptor(&recipe(s, *surface, *floor, *rug, false), *floor))
            .collect();
        let new_features: Vec<_> = (0..128)
            .map(|s| descriptor(&recipe(s, *surface, *floor, *rug, true), *floor))
            .collect();
        let mut counts = [0; 4];
        for seed in 0..128 {
            counts[recipe(seed, *surface, *floor, *rug, true)
                .construction
                .unwrap()
                .dominant()] += 1;
        }
        assert!(
            counts.iter().all(|n| *n >= 15),
            "unsampled construction: {name}/{counts:?}"
        );
        let old = recipe(4, *surface, *floor, *rug, false);
        let start = Instant::now();
        let old_maps = old.maps(*floor);
        let old_ms = start.elapsed().as_secs_f64() * 1000.;
        let (old_hash, _) = check_maps(&old, *floor, &old_maps);
        let mut sheet = RgbImage::new(192 * 4, 192 * 4);
        let mut records = Vec::new();
        let mut hashes = BTreeSet::new();
        for family in 0..4 {
            // Resolve onto the same previous recipe: changes cannot be explained
            // by a different random seed, unrelated palette or installation.
            let mut r = old.clone();
            let mut weights = [0.; 4];
            weights[family] = 1.;
            process::apply(
                &mut r,
                process::ConstructionSample {
                    weights,
                    scale: 1.,
                    variation: 0.6,
                },
                *floor,
            )
            .unwrap();
            let start = Instant::now();
            let maps = r.maps(*floor);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let (sha, roughness) = check_maps(&r, *floor, &maps);
            hashes.insert(sha.clone());
            assert_ne!(sha, old_hash, "unchanged maps {name}/{family}");
            if family == 0 {
                assert_eq!(sha, check_maps(&r, *floor, &r.maps(*floor)).0);
            }
            image::imageops::replace(
                &mut sheet,
                &swatch(&r, *floor, &maps, 192),
                family as i64 * 192,
                0,
            );
            records.push(json!({"construction":labels[family],"recipe":r,"sha256":sha,"roughness_range":roughness,"maps_ms":ms}));
        }
        assert_eq!(hashes.len(), 4, "collapsed maps {name}");
        sheet.save(dir.join(format!("{name}.png"))).unwrap();
        html.push_str(&format!(
            "<section><h2>{name}</h2><div class=names>{}</div><img src='{name}.png'></section>",
            labels
                .iter()
                .map(|s| format!("<span>{s}</span>"))
                .collect::<String>()
        ));
        report.push(json!({"category":name,"surface":surface,"floor":floor,"rug":rug,"pole_counts":counts,"previous_descriptor_statistics":statistics(&old_features),"expanded_descriptor_statistics":statistics(&new_features),"previous_maps_ms":old_ms,"distinct_conditioned_map_triplets":hashes.len(),"samples":records}));
        eprintln!("construction maps: {name}, 4 distinct alternatives, counts {counts:?}");
    }
    std::fs::write(dir.join("index.html"), html).unwrap();
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({
        "schema":1,"provenance":crate::provenance::capture_provenance(),"seed":4,
        "scope":"CPU material fields and production map mip chains; no learned embedding or effective dataset-size claim",
        "categories":categories.len(),"conditioned_map_triplets":categories.len()*4,"baseline_triplets":categories.len(),"replay_triplets":categories.len(),
        "descriptor_samples_per_category_per_sampler":128,"descriptor_fields":["mean_r","mean_g","mean_b","mean_luminance_squared","mean_roughness","mean_roughness_squared","mean_height_mm","mean_height_mm_squared","period_u_m","period_v_m"],
        "additional_atlas_slots":0,"maximum_map_resolution":512,"groups":report,
        "limits":["four distinct conditioned map sets are not proof of fourfold perceptual dataset size","descriptor occupancy depends on the stated bins","per-map CPU timing is diagnostic, not completed-room throughput","GPU parity and actual PBR renders are separate qualifications"]
    })).unwrap()).unwrap();
}
