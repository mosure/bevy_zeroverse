//! Bounded, device-free study of actual meshes, not texture or seed cardinality.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview,
    layout::{IndoorLayout, IndoorManifest},
    plants, validation,
};
use image::{Rgb, RgbImage};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path, time::Instant};

fn hash(a: &Assembly) -> String {
    let mut hash = Sha256::new();
    for (key, g) in &a.parts {
        hash.update(format!("{key:?}"));
        for v in g.positions.iter().flatten() {
            hash.update(v.to_le_bytes());
        }
        for i in &g.indices {
            hash.update(i.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

#[test]
#[ignore = "local CPU mesh diversity qualification; writes out/furniture_diversity"]
fn plant_chair_table_population_qualification() {
    let dir = Path::new("out/furniture_diversity");
    std::fs::create_dir_all(dir).unwrap();
    let mut groups = Vec::new();
    let n = 192;
    for (category, count) in [
        ("plants", plants::SPECIES),
        ("chairs", chairs::FAMILIES),
        ("tables", 4),
    ] {
        let mut sheet = RgbImage::from_pixel(n * 8, n * count, Rgb([241, 243, 246]));
        for variant in 0..count {
            let kind = match category {
                "plants" => ObjectKind::Plant,
                "chairs" => ObjectKind::Chair,
                _ => [
                    ObjectKind::Table,
                    ObjectKind::Desk,
                    ObjectKind::CoffeeTable,
                    ObjectKind::SideTable,
                ][variant as usize],
            };
            let samples_count = if category == "tables" { 64 } else { 16 };
            let mut samples = Vec::new();
            let mut descriptors = Vec::new();
            let mut hashes = BTreeSet::new();
            for seed in 0..samples_count {
                let mut o = fixture(kind, seed);
                if category != "tables" {
                    o.variant = variant;
                }
                if category == "plants" {
                    o.size.y = match variant {
                        4 | 13 | 16 => 0.9,
                        7 => 0.55,
                        9 => 1.1,
                        _ => 1.7,
                    };
                } else if category == "chairs" {
                    o.size.y = if chairs::is_backless(&o) {
                        0.48
                    } else if variant == 7 {
                        0.79
                    } else if variant == 8 || (2..=4).contains(&variant) {
                        1.0
                    } else {
                        1.30
                    };
                    if variant == 8 {
                        o.size.x = 0.90;
                        o.size.z = 0.90;
                    }
                }
                let begin = Instant::now();
                let a = build_object(&o);
                let construction_ms = begin.elapsed().as_secs_f64() * 1000.;
                let check = check(&o, &a).unwrap_or_else(|e| panic!("{category}/{variant}: {e}"));
                assert!(
                    check.triangles
                        < if kind == ObjectKind::Plant {
                            20000
                        } else {
                            24000
                        }
                );
                let sha256 = hash(&a);
                assert_eq!(
                    sha256,
                    hash(&build_object(&o)),
                    "replay {category}/{variant}/{seed}"
                );
                hashes.insert(sha256.clone());
                descriptors.push(descriptor(&o, &a));
                samples.push(
                    json!({"seed":seed,"program":program(&o),"mesh":check,"sha256":sha256,
                    "construction_ms":construction_ms,"size_m":o.size}),
                );
                if seed < 8 {
                    let im = preview::orthographic(
                        &a,
                        Vec3::Y * o.size.y * 0.48,
                        o.size.length() * 1.04,
                        n,
                        Vec3::new(1.2, 0.9, -2.),
                    );
                    image::imageops::replace(
                        &mut sheet,
                        &im,
                        seed as i64 * n as i64,
                        variant as i64 * n as i64,
                    );
                }
            }
            let coverage: Vec<_> = DISTANCE_THRESHOLDS.iter().map(|&t|
                json!({"threshold":t,"representatives":representatives(&descriptors,t).len()})).collect();
            groups.push(json!({"category":category,"kind":kind,"variant":variant,
                "name":if category == "plants" { plants::species_name(variant).to_string() } else { format!("{kind:?}/{variant}") },
                "unique_mesh_hashes":hashes.len(),"coverage":coverage,"samples":samples}));
        }
        sheet.save(dir.join(format!("{category}.png"))).unwrap();
    }
    let mut rooms = Vec::new();
    for seed in 0..64 {
        let begin = Instant::now();
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        let manifest_ms = begin.elapsed().as_secs_f64() * 1000.;
        validation::validate_layout(&scene).unwrap();
        let begin = Instant::now();
        let mesh = validation::validate_geometry(&scene).unwrap();
        assert_eq!(mesh.degenerate_triangles, 0);
        let objects: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && (o.kind == ObjectKind::Plant
                        || o.kind == ObjectKind::Chair
                        || o.kind.has_table_outline())
            })
            .map(
                |o| json!({"kind":o.kind,"variant":o.variant,"size_m":o.size,"program":program(o)}),
            )
            .collect();
        rooms.push(json!({"seed":seed,"layout":scene.layout,"manifest_ms":manifest_ms,
            "validation_and_construction_ms":begin.elapsed().as_secs_f64()*1000.,"triangles":mesh.triangles,"objects":objects}));
    }
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({
        "generator_revision":crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
        "scope":"CPU triangle/semantic/clearance/replay checks and flat-shaded CPU previews; no GPU, photographic, throughput or training qualification",
        "metric":"six 24x24 first-surface depth maps; per-group fixed envelope, finish seed zero; ordered greedy distance thresholds; not dataset capacity or SigLIP volume",
        "gallery":"row = recorded variant/category, columns = seeds 0..7; role colors, no PBR",
        "groups":groups,"rooms":rooms
    })).unwrap()).unwrap();
    println!("688 meshes and 64 rooms qualified; {}", dir.display());
}
