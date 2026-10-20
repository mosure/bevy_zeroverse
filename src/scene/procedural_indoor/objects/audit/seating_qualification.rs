//! Fixed-envelope seating audit. Run explicitly; never part of capture startup.
use super::*;
use crate::scene::procedural_indoor::geometry::preview;
use image::{Rgb, RgbImage};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::PathBuf, time::Instant};

fn cases() -> Vec<(&'static str, ObjectKind, u32, Vec3)> {
    use ObjectKind::{Chair, Sofa};
    vec![
        ("task", Chair, 0, Vec3::new(0.68, 1.35, 0.68)),
        ("conference", Chair, 1, Vec3::new(0.68, 1.03, 0.68)),
        ("visitor", Chair, 2, Vec3::new(0.68, 1.03, 0.68)),
        ("dining", Chair, 3, Vec3::new(0.68, 1.03, 0.68)),
        ("shell", Chair, 4, Vec3::new(0.68, 1.03, 0.68)),
        ("executive", Chair, 9, Vec3::new(0.68, 1.42, 0.68)),
        ("lounge", Chair, 8, Vec3::new(0.96, 1.08, 1.02)),
        ("loveseat", Sofa, 0, Vec3::new(1.55, 0.92, 0.96)),
        ("sofa", Sofa, 0, Vec3::new(2.50, 1.04, 1.00)),
        ("sectional", Sofa, 0, Vec3::new(2.85, 0.94, 1.85)),
        ("recliner", Chair, 8, Vec3::new(0.92, 1.24, 1.50)),
    ]
}

// Active structural decisions only. Neither floats, seed IDs nor finish choices
// count as a different construction. The names also read the preceding program.
fn structural_signature(o: &IndoorObject) -> String {
    let p = if o.kind == ObjectKind::Sofa || o.variant == 8 {
        json!(seating::parameters(o))
    } else {
        json!(chairs::parameters(o))
    };
    let keys: &[&str] = if o.kind == ObjectKind::Sofa || o.variant == 8 {
        &[
            "modules",
            "chaise",
            "left_return",
            "arm_style",
            "base_style",
            "back_style",
            "seat_style",
            "recliner",
        ]
    } else {
        &[
            "back_construction",
            "armrests",
            "headrest",
            "arm_support",
            "spoke_count",
            "casters",
            "base_style",
            "seat_style",
            "headrest_style",
        ]
    };
    let mut active = serde_json::Map::new();
    for &key in keys {
        if (key == "left_return" && p["chaise"] != true)
            || (key == "arm_support" && p["armrests"] != true)
            || (key == "headrest_style" && p["headrest"] != true)
            || (matches!(key, "spoke_count" | "casters")
                && p["base_style"].as_u64().is_some_and(|s| s >= 3))
        {
            continue;
        }
        if let Some(v) = p.get(key) {
            // A bench over a one-module seat is the same realized construction.
            active.insert(
                key.to_owned(),
                if key == "seat_style" && p["modules"] == 1 && *v == 1 {
                    json!(0)
                } else {
                    v.clone()
                },
            );
        }
    }
    Value::Object(active).to_string()
}

#[test]
#[ignore = "bounded complete-room seating clearance and geometry qualification"]
fn seating_population_qualification() {
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation,
    };
    let dir = PathBuf::from("out/seating_space/current");
    std::fs::create_dir_all(&dir).unwrap();
    let mut rooms = Vec::new();
    for seed in 0..64 {
        let begin = Instant::now();
        let layout = [
            IndoorLayout::Mixed,
            IndoorLayout::Lounge,
            IndoorLayout::Conference,
            IndoorLayout::Coworking,
        ][seed as usize % 4];
        let scene = IndoorManifest::generate_with_humans(seed, layout, 0.70, 3, 0.55).unwrap();
        validation::validate_layout(&scene).unwrap();
        let check = validation::validate_geometry(&scene).unwrap();
        let objects: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| !o.neighbor && matches!(o.kind, ObjectKind::Chair | ObjectKind::Sofa))
            .map(|o| json!({"kind":o.kind,"size_m":o.size,"program":program(o)}))
            .collect();
        rooms.push(
            json!({"seed":seed,"layout":layout,"people":scene.humans.len(),
            "construction_validation_ms":begin.elapsed().as_secs_f64()*1000.,
            "triangles":check.triangles,"seating":objects}),
        );
    }
    std::fs::write(dir.join("rooms.json"),serde_json::to_vec_pretty(&json!({
        "scope":"CPU layout, human support, camera clearance, semantic geometry and triangle checks; no GPU runtime",
        "rooms":rooms
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "bounded CPU mesh audit and local seating contact sheets"]
fn seating_space_qualification() {
    let dir = PathBuf::from(
        std::env::var("ZEROVERSE_SEATING_REPORT")
            .unwrap_or_else(|_| "out/seating_space/current".into()),
    );
    std::fs::create_dir_all(&dir).unwrap();
    let mut groups = Vec::new();
    let n = 240;
    let mut sheet = RgbImage::from_pixel(n * 8, n * cases().len() as u32, Rgb([241, 243, 246]));
    for (row, (name, kind, variant, size)) in cases().into_iter().enumerate() {
        let seeds: Vec<u64> = (0..)
            .filter(|&seed| name != "recliner" || seating::recliner::selected(seed))
            .take(4096)
            .collect();
        let mut signatures = BTreeSet::new();
        for &seed in &seeds {
            let mut o = fixture(kind, seed);
            o.variant = variant;
            o.size = size;
            signatures.insert(structural_signature(&o));
        }
        let mut maps = Vec::new();
        let mut samples = Vec::new();
        let mut gallery = RgbImage::from_pixel(n * 8, n, Rgb([241, 243, 246]));
        for (column, &seed) in seeds.iter().take(96).enumerate() {
            let mut o = fixture(kind, seed);
            o.variant = variant;
            o.size = size;
            let start = Instant::now();
            let a = build_object(&o);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let check = check(&o, &a).unwrap_or_else(|e| panic!("{name}/{seed}: {e}"));
            // Multi-seat upholstery carries several cushions and pillows. Its
            // budget scales with seating capacity, not the single-chair limit.
            let budget = if kind == ObjectKind::Sofa {
                32000
            } else {
                24000
            };
            assert!(
                check.triangles < budget,
                "{name}/{seed}: triangle budget {check:?}"
            );
            if column < 8 {
                let b = build_object(&o);
                for (key, g) in &a.parts {
                    assert_eq!(g.positions, b.parts[key].positions);
                    assert_eq!(g.normals, b.parts[key].normals);
                    assert_eq!(g.indices, b.parts[key].indices);
                }
                let im = preview::orthographic(
                    &a,
                    Vec3::Y * size.y * 0.48,
                    size.length() * 1.04,
                    n,
                    Vec3::new(1.2, 0.8, -2.),
                );
                image::imageops::replace(
                    &mut sheet,
                    &im,
                    column as i64 * n as i64,
                    row as i64 * n as i64,
                );
                image::imageops::replace(&mut gallery, &im, column as i64 * n as i64, 0);
            }
            maps.push(descriptor(&o, &a));
            samples.push(json!({
                "seed":seed, "size_m":size, "mesh":check, "construction_ms":ms,
                "program": if kind == ObjectKind::Chair && variant == 8 {
                    json!(seating::parameters(&o))
                } else { program(&o) }
            }));
        }
        gallery.save(dir.join(format!("{name}.png"))).unwrap();
        let coverage: Vec<_> = [0.015, 0.03, 0.08, 0.15, 0.25]
            .into_iter()
            .map(|t| json!({"threshold":t,"representatives":representatives(&maps,t).len()}))
            .collect();
        let nearest: Vec<_> = maps
            .iter()
            .enumerate()
            .map(|(i, a)| {
                maps.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, b)| distance(a, b))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        println!(
            "{name}: {} structural signatures; {coverage:?}",
            signatures.len()
        );
        groups.push(json!({"category":name,"structural_samples":4096,
            "observed_structural_signatures":signatures.len(),"coverage":coverage,
            "nearest_shape_distances":nearest,"samples":samples}));
    }
    sheet.save(dir.join("seating.png")).unwrap();
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({
        "generator_revision": crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
        "method":"96 identical seeds per fixed-envelope category; six 24x24 first-surface depth maps; union-pixel normalized L1 distance; ordered greedy coverage",
        "limitations":"Structural signatures count observed active construction choices, not independent samples or perceptual volume. Flat CPU previews are not PBR. Timings exclude rendering, uploads and training.",
        "gallery":"rows follow groups; columns seeds 0..7 except recliners (first eight selected seeds)",
        "groups":groups
    })).unwrap()).unwrap();
}
