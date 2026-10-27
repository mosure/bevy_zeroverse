//! Local matched floor qualification. Reference sampling retains the generator
//! 86 floor resolver; references are test-only and never exported by production.
use super::*;

fn reference(seed: u64, style: u32) -> MaterialRecipe {
    let mut rs = program::sample(seed);
    program::floor_finish_base(&mut rs, style);
    let mut r = rs.remove(Surface::Floor as usize);
    super::super::construction::sample(&mut r, style);
    r
}
fn stats(rows: &[Vec<f64>]) -> serde_json::Value {
    let n = rows.len() as f64;
    let d = rows[0].len();
    let mean: Vec<f64> = (0..d)
        .map(|i| rows.iter().map(|r| r[i]).sum::<f64>() / n)
        .collect();
    let cov: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..d)
                .map(|j| {
                    rows.iter()
                        .map(|r| (r[i] - mean[i]) * (r[j] - mean[j]))
                        .sum::<f64>()
                        / n
                })
                .collect()
        })
        .collect();
    let variance: Vec<f64> = (0..d).map(|i| cov[i][i]).collect();
    // Fixed measurement units, never normalized independently per population.
    let widths = [0.05, 0.05, 0.05, 0.025, 0.05, 0.05, 0.025, 0.01, 0.1, 0.1];
    let cells: BTreeSet<Vec<i64>> = rows
        .iter()
        .map(|r| {
            r.iter()
                .zip(widths)
                .map(|(v, w)| (v / w).floor() as i64)
                .collect()
        })
        .collect();
    let trace: f64 = variance.iter().sum();
    let squared: f64 = cov.iter().flatten().map(|v| v * v).sum();
    json!({"n":rows.len(),"mean":mean,"variance":variance,"occupied_cells":cells.len(),
        "bin_widths":widths,"covariance_participation_rank":trace*trace/squared.max(1e-30)})
}
fn distribution(mut xs: Vec<f64>) -> serde_json::Value {
    xs.sort_by(f64::total_cmp);
    let at = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
    json!({"min":at(0.),"p05":at(0.05),"median":at(0.5),"p95":at(0.95),"max":at(1.)})
}

fn spatial_descriptor(maps: &[Image; 3]) -> Vec<f64> {
    let n = maps[0].width() as usize;
    let stride = n / 8;
    let bytes = maps.each_ref().map(|m| m.data.as_ref().unwrap());
    let linear: [f64; 256] =
        std::array::from_fn(|i| super::super::srgb_to_linear(i as f32 / 255.) as f64);
    let mut out = Vec::new();
    for by in 0..8 {
        for bx in 0..8 {
            let mut sum = [0.; 5];
            for y in by * stride..(by + 1) * stride {
                for x in bx * stride..(bx + 1) * stride {
                    let i = (y * n + x) * 4;
                    for c in 0..3 {
                        sum[c] += linear[bytes[0][i + c] as usize];
                    }
                    sum[3] += bytes[2][i + 1] as f64 / 255.;
                    let l = (0..3)
                        .map(|c| linear[bytes[0][i + c] as usize])
                        .sum::<f64>()
                        / 3.;
                    sum[4] += l * l;
                }
            }
            let mean = sum.map(|v| v / (stride * stride) as f64);
            // Block reflectance, roughness and local luminance variation. No seed,
            // recipe fields or physical period are part of this image descriptor.
            out.extend_from_slice(&mean[..4]);
            let l = mean[..3].iter().sum::<f64>() / 3.;
            out.push((mean[4] - l * l).max(0.).sqrt());
        }
    }
    out
}

fn spatial_stats(rows: &[Vec<f64>]) -> serde_json::Value {
    let n = rows.len();
    let d = rows[0].len();
    let mean: Vec<_> = (0..d)
        .map(|i| rows.iter().map(|r| r[i]).sum::<f64>() / n as f64)
        .collect();
    let centered: Vec<Vec<_>> = rows
        .iter()
        .map(|r| r.iter().zip(&mean).map(|(v, m)| v - m).collect())
        .collect();
    let mut trace = 0.;
    let mut square = 0.;
    let mut nearest = vec![f64::INFINITY; n];
    for i in 0..n {
        for j in 0..n {
            let cov = centered[i]
                .iter()
                .zip(&centered[j])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / n as f64;
            if i == j {
                trace += cov;
            }
            square += cov * cov;
            if i != j {
                let distance = rows[i]
                    .iter()
                    .zip(&rows[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                nearest[i] = nearest[i].min((distance / d as f64).sqrt());
            }
        }
    }
    json!({"samples":n,"dimensions":d,"descriptor":"8x8 block linear RGB, roughness, luminance standard deviation",
        "variance_trace":trace,"covariance_participation_rank":trace*trace/square.max(1e-30),
        "nearest_rms_distance":distribution(nearest)})
}

#[test]
#[ignore = "local matched floor map/PBR/mip study; no GPU or training interaction"]
fn floor_diversity_qualification() {
    let dir = Path::new("out/floor_material_study");
    std::fs::create_dir_all(dir).unwrap();
    let map_samples =
        std::env::var("ZEROVERSE_FLOOR_MAP_SAMPLES").map_or(16, |s| s.parse::<usize>().unwrap());
    assert!((2..=64).contains(&map_samples));
    let mut groups = Vec::new();
    let mut html = String::from("<!doctype html><meta charset='utf-8'><title>Floor materials — local qualification</title><style>body{background:#14191e;color:#e6edf4;font:16px system-ui;margin:32px}img{width:100%;max-width:1024px}section{margin:3rem 0}h2{margin-bottom:.4rem}</style><h1>Floor materials</h1><p>Fixed seed sampling. Each four-row band: production albedo, tangent normal, roughness, CPU GGX diagnostic. This is not a Bevy room render.</p>");
    for style in 0..3 {
        let mut rows = [Vec::new(), Vec::new()];
        let mut parameters = [Vec::new(), Vec::new()];
        let mut times = [Vec::new(), Vec::new()];
        let mut spatial = [Vec::new(), Vec::new()];
        let mut hashes = [BTreeSet::new(), BTreeSet::new()];
        let mut recipes = Vec::new();
        let mut galleries: [RgbImage; 2] =
            std::array::from_fn(|_| RgbImage::new(8 * 128, map_samples.div_ceil(8) as u32 * 512));
        for seed in 0..1024 {
            let pair = [
                reference(seed, style),
                program::sample_with_floor(seed, style)
                    .unwrap()
                    .remove(Surface::Floor as usize),
            ];
            for version in 0..2 {
                let r = &pair[version];
                r.validate().unwrap();
                rows[version].push(descriptor(r, style));
                let cells = r.flooring.as_ref().unwrap().cells;
                parameters[version].push([
                    r.period_m as f64 / cells[0] as f64,
                    r.period_m as f64 / cells[1] as f64,
                    r.roughness as f64,
                ]);
            }
            if seed < map_samples as u64 {
                // Alternate timing order; no renderer, model or GPU initialization.
                for k in 0..2 {
                    let version = (k + seed as usize) % 2;
                    let r = &pair[version];
                    let start = Instant::now();
                    let maps = r.maps(style);
                    times[version].push(start.elapsed().as_secs_f64() * 1000.);
                    let (hash, _) = check_maps(r, style, &maps);
                    spatial[version].push(spatial_descriptor(&maps));
                    assert!(hashes[version].insert(hash));
                    if seed == 0 {
                        let copy: MaterialRecipe =
                            serde_json::from_slice(&serde_json::to_vec(r).unwrap()).unwrap();
                        let replay = copy.maps(style);
                        for i in 0..3 {
                            assert_eq!(maps[i].data, replay[i].data);
                        }
                    }
                    let sw = swatch(r, style, &maps, 128);
                    image::imageops::replace(
                        &mut galleries[version],
                        &sw,
                        (seed % 8 * 128) as i64,
                        (seed / 8 * 512) as i64,
                    );
                }
                recipes.push(json!({"seed":seed,"reference":pair[0],"current":pair[1]}));
            }
        }
        for (i, tag) in ["reference", "current"].into_iter().enumerate() {
            let filename = format!("floor-{style}-{tag}.png");
            galleries[i].save(dir.join(&filename)).unwrap();
            html.push_str(&format!(
                "<section><h2>{} / {tag}</h2><img src='{filename}'></section>",
                ["Timber", "Carpet", "Mineral"][style as usize]
            ));
        }
        let dims = |i: usize| {
            (0..3)
                .map(|d| distribution(parameters[i].iter().map(|p| p[d]).collect()))
                .collect::<Vec<_>>()
        };
        groups.push(json!({"style":style,"reference":stats(&rows[0]),"current":stats(&rows[1]),
            "nominal_width_length_roughness":{"reference":dims(0),"current":dims(1)},
            "cpu_map_ms":{"reference":distribution(times[0].clone()),"current":distribution(times[1].clone())},
            "spatial_map_variation":{"reference":spatial_stats(&spatial[0]),"current":spatial_stats(&spatial[1])},
            "unique_map_triplets":hashes.map(|h| h.len()),"recipes":recipes}));
    }
    std::fs::write(dir.join("index.html"), html).unwrap();
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({
        "schema":1,"provenance":crate::provenance::capture_provenance(),
        "reference_commit":"8860228b8267237ee9e943f992f03fea8eadb9aa",
        "sampler_pairs":3072,"full_map_triplets":map_samples*6,"replay_triplets":6,
        "texture_budget_unchanged":true,"max_texture_size":512,"groups":groups,
        "limits":["Physical descriptors and exact hashes are not perceptual dataset size or transfer utility",
            "10x is a design target, not an experimentally demonstrated count of distinct training samples",
            "CPU material timing is not completed-room or concurrent-training throughput",
            "Software WebGPU parity is qualified separately; no hardware GPU performance claim"]
    })).unwrap()).unwrap();
}
