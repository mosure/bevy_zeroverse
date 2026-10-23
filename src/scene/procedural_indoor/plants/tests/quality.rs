//! Physical foliage coverage is audited from triangles, independently of RGB,
//! colour seeds, texture resolution, camera placement and perceptual embeddings.
use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
use serde_json::json;

fn is_foliage(surface: Surface) -> bool {
    matches!(
        surface,
        Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
    )
}

fn fixture(variant: u32, seed: u64) -> IndoorObject {
    let mut o = audit::fixture(ObjectKind::Plant, seed);
    o.variant = variant;
    o.size = Vec3::new(
        0.9,
        match variant {
            4 => 0.75,
            7 => 0.55,
            8 | 9 => 1.1,
            11 => 0.8,
            13 => 0.95,
            16 => 1.0,
            _ => 1.3,
        },
        0.9,
    );
    o
}

fn triangles(mesh: &Assembly) -> Vec<[Vec3; 3]> {
    mesh.parts
        .iter()
        .filter(|((surface, _), _)| is_foliage(*surface))
        .flat_map(|(_, g)| {
            g.indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| t.map(|i| Vec3::from(g.positions[i as usize])))
        })
        .collect()
}

/// Fraction covered in the reserved canopy rectangle, rather than rescaling a
/// sparse twig's own small AABB to fill the diagnostic image. Backfaces count.
fn coverage(tris: &[[Vec3; 3]], o: &IndoorObject, axes: [usize; 2]) -> f32 {
    const N: usize = 64;
    let soil = pot_dimensions(o, &Growth::sample(o.seed)).1;
    let lo = Vec3::new(-o.size.x * 0.5, soil, -o.size.z * 0.5);
    let hi = Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5);
    let project = |p: Vec3| {
        Vec2::new(
            (p[axes[0]] - lo[axes[0]]) / (hi[axes[0]] - lo[axes[0]]),
            (p[axes[1]] - lo[axes[1]]) / (hi[axes[1]] - lo[axes[1]]),
        ) * N as f32
    };
    let mut occupied = [false; N * N];
    for t in tris {
        let p = t.map(project);
        let min = p
            .iter()
            .copied()
            .fold(Vec2::splat(f32::INFINITY), Vec2::min)
            .floor()
            .clamp(Vec2::ZERO, Vec2::splat((N - 1) as f32));
        let max = p
            .iter()
            .copied()
            .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max)
            .ceil()
            .clamp(Vec2::ZERO, Vec2::splat((N - 1) as f32));
        let determinant = (p[1] - p[0]).perp_dot(p[2] - p[0]);
        if determinant.abs() < 1e-7 {
            continue;
        }
        for y in min.y as usize..=max.y as usize {
            for x in min.x as usize..=max.x as usize {
                let d = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - p[0];
                let u = d.perp_dot(p[2] - p[0]) / determinant;
                let v = (p[1] - p[0]).perp_dot(d) / determinant;
                occupied[y * N + x] |= u >= 0. && v >= 0. && u + v <= 1.;
            }
        }
    }
    occupied.iter().filter(|&&v| v).count() as f32 / (N * N) as f32
}

#[test]
fn species_proportions_are_correlated_replayable_and_not_all_upright() {
    for tabletop in [false, true] {
        for variant in 0..SPECIES {
            let mut short = 0;
            let mut tall = 0;
            for seed in 0..64 {
                let size = sample_size(variant, tabletop, &mut stream(seed, 6470));
                assert_eq!(
                    size,
                    sample_size(variant, tabletop, &mut stream(seed, 6470))
                );
                let ratio = size.y / size.x.min(size.z);
                assert!((0.92..=1.08).contains(&(size.z / size.x)));
                short += usize::from(ratio < 1.4);
                tall += usize::from(ratio > 1.8);
                let mut o = fixture(variant, seed);
                o.size = size;
                let a = build_object(&o);
                let stats = audit::check(&o, &a)
                    .unwrap_or_else(|e| panic!("{variant}/{seed}/{size:?}: {e}"));
                assert!(stats.triangles < 20_000, "{variant}: {}", stats.triangles);
                if seed == 0 {
                    let b = build_object(&o);
                    for (key, p) in a.parts {
                        assert_eq!(p.positions, b.parts[&key].positions);
                        assert_eq!(p.normals, b.parts[&key].normals);
                        assert_eq!(p.uvs, b.parts[&key].uvs);
                        assert_eq!(p.indices, b.parts[&key].indices);
                    }
                }
            }
            if matches!(variant, 3 | 4 | 7 | 13 | 16) {
                assert!(
                    short >= 40,
                    "broad habit {variant} became vertical: {short}/64"
                );
            }
            if matches!(variant, 2 | 5 | 10) && !tabletop {
                assert!(
                    tall >= 48,
                    "upright habit {variant} lost its stature: {tall}/64"
                );
            }
        }
    }
}

#[test]
fn woody_and_vining_leaf_bases_attach_to_emitted_petiole_geometry() {
    for variant in [0, 6, 11, 12, 13, 16] {
        for seed in 0..16 {
            let o = fixture(variant, seed);
            let a = build_object(&o);
            let bark: Vec<_> = a
                .parts
                .iter()
                .filter(|((s, _), _)| *s == Surface::Bark)
                .flat_map(|(_, g)| g.positions.iter().copied().map(Vec3::from))
                .collect();
            for ((s, _), leaf) in &a.parts {
                if !is_foliage(*s) {
                    continue;
                }
                assert_eq!(leaf.positions.len() % 23, 0);
                for blade in leaf.positions.as_chunks::<23>().0 {
                    let p = Vec3::from(blade[0]);
                    assert!(
                        bark.iter().any(|&v| p.distance_squared(v) < 1e-10),
                        "floating leaf base {variant}/{seed}: {p:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn cascading_foliage_does_not_pass_through_pot_or_floor() {
    let mut below_rim = 0;
    for seed in 0..64 {
        let o = fixture(11, seed);
        let g = Growth::sample(seed);
        let (radius, ph) = pot_dimensions(&o, &g);
        let a = build_object(&o);
        for ((s, _), mesh) in &a.parts {
            if !is_foliage(*s) {
                continue;
            }
            for p in mesh.positions.iter().copied().map(Vec3::from) {
                assert!(p.y >= 0.);
                if p.y < ph - 0.012 {
                    below_rim += 1;
                    assert!(
                        p.xz().length() >= radius * 1.10,
                        "vine entered pot {seed}: {p:?}, radius={radius}"
                    );
                }
            }
        }
    }
    assert!(
        below_rim > 100,
        "cascading form must actually hang outside the rim"
    );
}

#[test]
fn broad_crowns_fill_horizontal_space_without_exceeding_geometry_budget() {
    for variant in [0, 6, 12, 13, 15, 16] {
        let mut coverage_sum = 0.;
        let mut area_sum = 0.;
        for seed in 0..16 {
            let o = fixture(variant, seed);
            let a = build_object(&o);
            let tris = triangles(&a);
            area_sum += tris
                .iter()
                .map(|t| (t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5)
                .sum::<f32>();
            coverage_sum += coverage(&tris, &o, [0, 2]);
        }
        eprintln!(
            "{}: projected reserve coverage {:.3}, leaf area {:.3} m2",
            species_name(variant),
            coverage_sum / 16.,
            area_sum / 16.
        );
        assert!(
            coverage_sum / 16. > 0.20,
            "too little crown coverage: {variant}"
        );
    }
}

#[test]
#[ignore = "local botanical shape/render qualification; writes out/plant_quality"]
fn plant_quality_receipt() {
    let dir = std::path::Path::new("out/plant_quality");
    std::fs::create_dir_all(dir).unwrap();
    let mut categories = Vec::new();
    let mut summaries = Vec::new();
    for variant in 0..SPECIES {
        let mut samples = Vec::new();
        let mut maps = Vec::new();
        let mut times = Vec::new();
        for seed in 0..16 {
            let o = fixture(variant, seed);
            let now = std::time::Instant::now();
            let a = build_object(&o);
            times.push(now.elapsed().as_secs_f64() * 1000.);
            let stats = audit::check(&o, &a).unwrap();
            let tris = triangles(&a);
            let cover = [[0, 1], [0, 2], [2, 1]].map(|axes| coverage(&tris, &o, axes));
            let area = tris
                .iter()
                .map(|t| (t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5)
                .sum::<f32>();
            maps.push(audit::descriptor(&o, &a));
            samples.push(json!({"object":o,"program":audit::program(&o),"triangles":stats.triangles,
                "leaf_area_m2":area,"reserve_leaf_coverage_xy_xz_zy":cover,"build_ms":times.last()}));
        }
        times.sort_by(f64::total_cmp);
        let means: Vec<f64> = (0..3)
            .map(|axis| {
                samples
                    .iter()
                    .map(|s| s["reserve_leaf_coverage_xy_xz_zy"][axis].as_f64().unwrap())
                    .sum::<f64>()
                    / 16.
            })
            .collect();
        summaries.push(
            json!({"species":species_name(variant),"coverage_xy_xz_zy":means,
            "build_ms_median":times[8],"build_ms_p95":times[15],
            "max_triangles":samples.iter().map(|s|s["triangles"].as_u64().unwrap()).max(),
            "shape_representatives_at_0015":audit::representatives(&maps,0.015).len()}),
        );
        categories.push(json!({"kind":species_name(variant),"samples":samples,"depth_descriptors":maps,"failures":[]}));
    }
    let report = json!({"schema_version":1,"build_provenance":crate::provenance::capture_provenance(),
        "method":"17 growth habits x 16 prespecified seeds; fixed habit envelopes and finish. Actual leaf triangles rasterized into 64-square orthographic canopy reserves. Cactus coverage is zero by construction. Shape coverage is diagnostic, not a dataset-independence or learning-utility estimate. Timings are local unoptimized-root CPU diagnostics, not production throughput.",
        "summaries":summaries,"categories":categories});
    std::fs::write(dir.join("audit.json"), serde_json::to_vec(&report).unwrap()).unwrap();
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&report["summaries"]).unwrap(),
    )
    .unwrap();
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&report["summaries"]).unwrap()
    );
}
