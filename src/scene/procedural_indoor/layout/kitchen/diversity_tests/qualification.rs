use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview,
    objects::{audit, kitchen},
};
use image::{Rgb, RgbImage};
use serde_json::json;
use std::{collections::BTreeMap, path::Path, time::Instant};

#[test]
#[ignore = "local CPU kitchen regularity and accepted-placement qualification"]
fn kitchen_regularity_population_qualification() {
    let out = Path::new("out/kitchen_regularity");
    std::fs::create_dir_all(out).unwrap();
    let mut objects = Vec::new();
    let mut sheet = RgbImage::from_pixel(4 * 360, 6 * 360, Rgb([241, 243, 246]));
    let mut html=String::from("<!doctype html><meta charset='utf-8'><title>Kitchen construction</title><style>body{font:16px system-ui;background:#f1f3f6;color:#243341;max-width:1500px;margin:32px auto}section{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}article{background:white;padding:12px;border-radius:12px}img{width:100%}</style><h1>Kitchen construction and placement</h1><p>Actual mesh geometry with CPU flat shading. Fixed chassis, closed doors and finish identity isolate construction variation. <a href='report.json'>Recorded inputs and measurements</a></p>");
    for (row, kind) in [
        ObjectKind::Cooktop,
        ObjectKind::Range,
        ObjectKind::Microwave,
        ObjectKind::Refrigerator,
        ObjectKind::Oven,
        ObjectKind::Dishwasher,
    ]
    .into_iter()
    .enumerate()
    {
        html.push_str(&format!("<h2>{kind:?}</h2><section>"));
        let mut descriptors = Vec::new();
        let mut timings = Vec::new();
        let mut records = Vec::new();
        for seed in 0..64 {
            let start = Instant::now();
            let (o, a, p) = kitchen::appliances::review_fixture(kind, seed);
            timings.push(start.elapsed().as_secs_f64() * 1000.);
            let check = audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            descriptors.push(audit::descriptor(&o, &a));
            records.push(json!({"seed":seed,"kitchen_state":p,"program":kitchen::appliances::metadata(&o,&p),"mesh":check}));
            if seed < 4 {
                let (lo, hi) = a.bounds();
                let img = preview::orthographic(
                    &a,
                    (lo + hi) * 0.5,
                    (hi - lo).length() * 1.05,
                    360,
                    Vec3::new(0.8, 0.75, 1.2),
                );
                let name = format!("{kind:?}-{seed}.png");
                img.save(out.join(&name)).unwrap();
                image::imageops::replace(&mut sheet, &img, seed as i64 * 360, row as i64 * 360);
                html.push_str(&format!(
                    "<article><img src='{name}'><p>Seed {seed}</p></article>"
                ));
            }
        }
        let coverage:Vec<_>=[0.005,0.015,0.03].map(|t|json!({"depth_difference_threshold":t,"representatives":audit::representatives(&descriptors,t).len()})).into();
        objects.push(json!({"kind":kind,"samples":64,"fixed_envelope_m":kitchen::fixture_size(kind),"geometry_coverage":coverage,"construction_ms":timings,"records":records}));
        html.push_str("</section>");
    }
    let mut rooms = Vec::new();
    let mut first = BTreeMap::new();
    for layout in [IndoorLayout::Kitchen, IndoorLayout::Kitchenette] {
        for seed in 0..64 {
            let start = Instant::now();
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.15)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            let time = start.elapsed().as_secs_f64() * 1000.;
            validation::validate_layout(&scene).unwrap();
            if seed < 16 {
                validation::validate_geometry(&scene).unwrap();
            }
            for group in &scene.assembly_diagnostics.kitchens {
                *first
                    .entry(format!("{:?}", scene.objects[group.units[0]].kind))
                    .or_insert(0usize) += 1;
            }
            rooms.push(json!({"seed":seed,"layout":layout,"manifest_ms":time,"object_count":scene.objects.len(),"groups":scene.assembly_diagnostics.kitchens,
                "objects":scene.objects.iter().filter(|o|o.kind.is_kitchen()&&!o.neighbor).collect::<Vec<_>>()}));
        }
    }
    sheet.save(out.join("gallery.png")).unwrap();
    html.push_str("<p>128 complete kitchen/kitchenette manifests, 32 complete-room geometry checks. Camera tests are separate. Depth-map coverage excludes color and material; it is not an embedding volume or a learning-utility estimate.</p>");
    std::fs::write(out.join("index.html"), html).unwrap();
    std::fs::write(out.join("report.json"),serde_json::to_vec_pretty(&json!({"generator":crate::scene::procedural_indoor::layout::GENERATOR_VERSION,"capture_identity":crate::CAPTURE_ENGINE_IDENTITY,"scope":"CPU actual-mesh qualification; no PBR or GPU throughput measurement", "objects":objects,"rooms":rooms,"first_unit_counts":first})).unwrap()).unwrap();
    println!("128 rooms; 384 fixed-envelope meshes; {}", out.display());
}
