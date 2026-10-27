//! Local CPU qualification, deliberately independent of renderer/training GPUs.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview, layout::IndoorLayout, objects::Assembly, validation,
};
use image::{Rgb, RgbImage};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Instant,
};

fn fingerprint(a: &Assembly) -> String {
    let mut h = Sha256::new();
    for (key, g) in &a.parts {
        h.update(format!("{key:?}"));
        for p in g.positions.iter().flatten() {
            h.update(p.to_le_bytes());
        }
        for i in &g.indices {
            h.update(i.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

#[test]
#[ignore = "local CPU mezzanine/guardrail population study, writes out/mezzanine_diversity"]
fn mezzanine_population_qualification() {
    let dir = Path::new("out/mezzanine_diversity");
    std::fs::create_dir_all(dir).unwrap();
    let (mut scene, e) = tests::fixture();
    let mut samples = Vec::new();
    let mut hashes = BTreeSet::new();
    let mut representative = BTreeMap::new();
    let mut sheet = RgbImage::from_pixel(4 * 440, 3 * 440, Rgb([241, 243, 246]));
    let mut cells = Vec::new();
    for seed in 0..512 {
        scene.seed = seed;
        let begin = Instant::now();
        let m = Mezzanine::sample(&scene, &e).unwrap();
        let sample_ms = begin.elapsed().as_secs_f64() * 1000.;
        let begin = Instant::now();
        let mut a = Assembly::default();
        build(&mut a, &m);
        let mesh_ms = begin.elapsed().as_secs_f64() * 1000.;
        let triangles = tests::check(&a, &m, &e, scene.room_size);
        let hash = fingerprint(&a);
        hashes.insert(hash.clone());
        if seed < 16 {
            let mut again = Assembly::default();
            build(&mut again, &m);
            assert_eq!(hash, fingerprint(&again));
        }
        let c = m.construction.as_ref().unwrap();
        let row = match c.stair_structure {
            StairStructure::Solid => 0,
            StairStructure::ClosedStringer => 1,
            StairStructure::OpenStringer => 2,
        };
        let column = match c.guard.infill {
            super::super::guardrail::Infill::Glass => 0,
            super::super::guardrail::Infill::Pickets => 1,
            super::super::guardrail::Infill::Slats => 2,
            super::super::guardrail::Infill::Solid => 3,
        };
        if representative.insert((row, column), seed).is_none() {
            let (lo, hi) = a.bounds();
            let center = (lo + hi) * 0.5;
            let direction = if m.stair_axis == 0 {
                Vec3::new(if m.ascending_positive() { -1.5 } else { 1.5 }, 0.9, -1.2)
            } else {
                Vec3::new(-1.2, 0.9, if m.ascending_positive() { -1.5 } else { 1.5 })
            };
            let image =
                preview::orthographic(&a, center, (hi - lo).length() * 1.02, 440, direction);
            image
                .save(dir.join(format!("sample-{row}-{column}.png")))
                .unwrap();
            image::imageops::replace(&mut sheet, &image, column * 440, row * 440);
            cells.push(json!({"row":row,"column":column,"seed":seed,"stair":c.stair_structure,"infill":c.guard.infill,"width_m":m.stair_width(),"height_m":m.deck.height}));
        }
        samples.push(json!({"seed":seed,"plan":m,"triangles":triangles,"sha256":hash,"sampling_ms":sample_ms,"mesh_ms":mesh_ms}));
    }
    assert_eq!(representative.len(), 12);
    assert_eq!(hashes.len(), 512);
    sheet.save(dir.join("geometry.png")).unwrap();
    let mut rooms = Vec::new();
    let mut validated = 0;
    for seed in 0..256 {
        let begin = Instant::now();
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let generation_ms = begin.elapsed().as_secs_f64() * 1000.;
        validation::validate_layout(&scene).unwrap();
        let e = scene.envelope.as_ref().unwrap();
        let geometry = if e.mezzanine.is_some() {
            validated += 1;
            Some(validation::validate_geometry(&scene).unwrap().triangles)
        } else {
            None
        };
        rooms.push(json!({"seed":seed,"layout":scene.layout,"room_size":scene.room_size,"generation_ms":generation_ms,
            "mezzanine":e.mezzanine,"tier_guard":if e.tier_guards().is_empty(){None}else{e.tier_guard.as_ref()},"validated_room_triangles":geometry}));
    }
    assert!(validated >= 3);
    std::fs::write(dir.join("report.json"),serde_json::to_vec_pretty(&json!({
        "generator":crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
        "capture_identity":crate::CAPTURE_ENGINE_IDENTITY,
        "scope":"512 unconstrained tall-room fixtures plus 256 consecutive Mixed manifests; all admitted Mixed mezzanine rooms get full CPU geometry validation. Flat-shaded actual-mesh previews; no hardware GPU, PBR or downstream training qualification.",
        "unique_mesh_hashes":hashes.len(),"cells":cells,"samples":samples,"rooms":rooms,
        "limitations":"Hash uniqueness is not perceptual diversity or dataset capacity. Stair flights remain straight, optionally interrupted by an inline landing. Support/clearance volumes intentionally reserve spaces between open risers and guard infill."
    })).unwrap()).unwrap();
    let mut html=String::from("<!doctype html><meta charset='utf-8'><title>Mezzanine construction review</title><style>body{font:16px system-ui;background:#f1f3f6;color:#243341;margin:40px auto;max-width:1500px}h1{font-size:32px}section{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}article{background:white;border-radius:12px;overflow:hidden}img{width:100%}h2,p{margin:16px}small{display:block;margin:16px;color:#526171}a{color:#125c99}</style><h1>Mezzanine construction diversity</h1><p>Actual generated meshes. Rows: solid, closed-stringer, open-stringer stairs. Columns: glass, pickets, slats, solid infill. Dimensions, materials, bays, ascent and inline landings vary independently where structurally compatible.</p><p>CPU flat shading for geometry inspection; glass is shown opaque. These are not photometric renders or evidence of downstream generalization. <a href='report.json'>Parameters, timing and distribution report</a></p><section>");
    cells.sort_by_key(|v| (v["row"].as_i64().unwrap(), v["column"].as_i64().unwrap()));
    for cell in cells {
        html.push_str(&format!("<article><img src='sample-{}-{}.png'><h2>{} / {}</h2><small>Seed {} · stair {:.2} m · deck {:.2} m high</small></article>",cell["row"],cell["column"],cell["stair"].as_str().unwrap(),cell["infill"].as_str().unwrap(),cell["seed"],cell["width_m"].as_f64().unwrap(),cell["height_m"].as_f64().unwrap()));
    }
    html.push_str("</section>");
    std::fs::write(dir.join("index.html"), html).unwrap();
    println!(
        "512 meshes, 256 manifests, {validated} complete mezzanine rooms: {}",
        dir.display()
    );
}
