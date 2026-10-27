use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview,
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use image::{Rgb, RgbImage};
use serde_json::json;
use std::{collections::BTreeMap, path::Path, time::Instant};

#[test]
#[ignore = "local CPU exposed-ceiling geometry and distribution qualification"]
fn exposed_ceiling_population_qualification() {
    let out = Path::new("out/ceiling_framing");
    std::fs::create_dir_all(out).unwrap();
    let mut records = Vec::new();
    let mut cells = Vec::new();
    let mut kinds = BTreeMap::new();
    let mut exposed = 0;
    let mut sheet = RgbImage::from_pixel(4 * 480, 2 * 480, Rgb([241, 243, 246]));
    for seed in 0..256 {
        let begin = Instant::now();
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let manifest_ms = begin.elapsed().as_secs_f64() * 1000.;
        validation::validate_layout(&scene).unwrap();
        let plan = scene.ceiling_installation.as_ref().unwrap();
        let begin = Instant::now();
        let mut a = Assembly::default();
        super::super::build(&mut a, &scene);
        let construction_ms = begin.elapsed().as_secs_f64() * 1000.;
        let triangles: usize = a.parts.values().map(|g| g.indices.len() / 3).sum();
        if let Some(frame) = &plan.framing {
            tests::check(frame);
            assert!(
                frame.connected().iter().all(|&c| c),
                "floating frame seed {seed}"
            );
            exposed += 1;
            for m in &frame.members {
                *kinds
                    .entry(format!("{:?}/{:?}", m.role, m.section))
                    .or_insert(0) += 1;
            }
            if cells.len() < 8 && frame.members.len() >= 12 {
                let (lo, hi) = a.bounds();
                let image = preview::orthographic(
                    &a,
                    (lo + hi) * 0.5,
                    (hi - lo).length() * 1.03,
                    480,
                    Vec3::new(0.8, -1.1, -1.2),
                );
                let index = cells.len() as i64;
                image
                    .save(out.join(format!("ceiling-{index}.png")))
                    .unwrap();
                image::imageops::replace(&mut sheet, &image, (index % 4) * 480, (index / 4) * 480);
                cells.push(json!({"index":index,"seed":seed,"members":frame.members.len(),"services":plan.modules.len(),"triangles":triangles}));
            }
            if seed < 64 {
                validation::validate_geometry(&scene).unwrap();
            }
        }
        let layered = plan
            .modules
            .iter()
            .enumerate()
            .map(|(i, a)| {
                plan.modules
                    .iter()
                    .skip(i + 1)
                    .filter(|b| {
                        overlap(a.bounds(), b.bounds(), 0.)
                            && !super::super::suspension::conflicts(a, b, 0.)
                    })
                    .count()
            })
            .sum::<usize>();
        records.push(json!({"seed":seed,"layout":scene.layout,"size":scene.room_size,"manifest_ms":manifest_ms,"ceiling_mesh_ms":construction_ms,"triangles":triangles,"layered_pairs":layered,"ceiling":plan}));
    }
    assert_eq!(cells.len(), 8);
    assert!(exposed > 40);
    sheet.save(out.join("gallery.png")).unwrap();
    std::fs::write(out.join("report.json"),serde_json::to_vec_pretty(&json!({"generator":crate::scene::procedural_indoor::layout::GENERATOR_VERSION,"capture_identity":crate::CAPTURE_ENGINE_IDENTITY,"scope":"256 CPU room manifests and ceiling meshes; exposed rooms in seeds 0..64 receive full scene geometry validation. Actual-mesh flat-shaded underside previews, no hardware GPU or photometric benchmark.","exposed":exposed,"members":kinds,"gallery":cells,"rooms":records})).unwrap()).unwrap();
    let mut html=String::from("<!doctype html><meta charset='utf-8'><title>Exposed ceilings</title><style>body{font:17px system-ui;background:#f1f3f6;color:#243341;max-width:1400px;margin:40px auto}section{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}article{background:white;border-radius:12px;overflow:hidden}img{width:100%}h2,p{margin:16px}</style><h1>Exposed ceiling structure</h1><p>Actual generated geometry viewed from below. The roof slab is omitted to reveal framing, depth layers and service routing. CPU flat shading; not a PBR render. <a href='report.json'>Complete recorded plans and timing</a></p><section>");
    for cell in cells {
        html.push_str(&format!("<article><img src='ceiling-{}.png'><h2>Seed {}</h2><p>{} frame members · {} installations</p></article>",cell["index"],cell["seed"],cell["members"],cell["services"]));
    }
    html.push_str("</section>");
    std::fs::write(out.join("index.html"), html).unwrap();
    println!(
        "256 rooms, {exposed} exposed ceiling programs; {}",
        out.display()
    );
}
