//! Reproducible local object/assembly receipt, consumed by review_objects.
use super::*;
use crate::scene::procedural_indoor::{geometry::preview, objects::Assembly};
use serde_json::json;
use std::time::Instant;

#[test]
#[ignore = "local kitchen/refreshment mesh audit and render input; writes out/refreshments"]
fn refreshment_geometry_and_assembly_receipt() {
    let dir = std::path::Path::new("out/refreshments");
    std::fs::create_dir_all(dir).unwrap();
    let mut categories = Vec::new();
    let mut summaries = Vec::new();
    let base = empty_workroom();
    for name in [
        "CoffeeMaker",
        "ServiceCounter",
        "KitchenIsland",
        "RefreshmentStation",
    ] {
        let kind = match name {
            "CoffeeMaker" => ObjectKind::CoffeeMaker,
            "KitchenIsland" => ObjectKind::KitchenCounter,
            _ => ObjectKind::ServiceCounter,
        };
        let mut samples = Vec::new();
        let mut maps = Vec::new();
        let mut times = Vec::new();
        let mut sheet = image::RgbImage::new(8 * 192, 192);
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            if name == "KitchenIsland" {
                objects::kitchen::configure_island(&mut o);
            }
            // Fixed envelopes separate shape variation from metric dimensions.
            o.size = match name {
                "CoffeeMaker" => Vec3::new(0.31, 0.37, 0.36),
                "KitchenIsland" => Vec3::new(1.9, 0.90, 1.05),
                _ => Vec3::new(2., 0.95, 0.78),
            };
            let mut scene = base.clone();
            scene.objects.push(o.clone());
            let now = Instant::now();
            if name == "RefreshmentStation" {
                scene.dress_refreshment_counter(
                    &o,
                    Some(if seed % 2 == 0 {
                        ObjectKind::CoffeeMaker
                    } else {
                        ObjectKind::DrinkDispenser
                    }),
                );
            }
            let mut mesh = Assembly::default();
            let mut triangles = 0;
            for part in &scene.objects {
                let mut a = objects::build_object(part);
                triangles += audit::check(part, &a).unwrap().triangles;
                let tf = part.transform();
                for g in a.parts.values_mut() {
                    for p in &mut g.positions {
                        *p = tf.transform_point(Vec3::from_array(*p)).to_array();
                    }
                    for n in &mut g.normals {
                        *n = (tf.rotation * Vec3::from_array(*n)).to_array();
                    }
                }
                mesh.append_assembly(a);
            }
            times.push(now.elapsed().as_secs_f64() * 1000.);
            let mut envelope = o.clone();
            if name == "RefreshmentStation" {
                envelope.size.y = mesh.bounds().1.y;
            }
            maps.push(audit::descriptor(&envelope, &mesh));
            if seed < 8 {
                let im = preview::orthographic(
                    &mesh,
                    Vec3::Y * envelope.size.y * 0.48,
                    envelope.size.length() * 1.04,
                    192,
                    Vec3::new(1.2, 0.9, 2.),
                );
                image::imageops::replace(&mut sheet, &im, seed as i64 * 192, 0);
            }
            samples.push(json!({"object":o,"attachments":&scene.objects[1..],"program":audit::program(&o),"triangles":triangles,"build_ms":times.last()}));
        }
        sheet
            .save(dir.join(format!("{name}-geometry.png")))
            .unwrap();
        let coverage: Vec<_> = audit::DISTANCE_THRESHOLDS
            .iter()
            .map(
                |&t| json!({"threshold":t,"representatives":audit::representatives(&maps,t).len()}),
            )
            .collect();
        times.sort_by(f64::total_cmp);
        summaries.push(json!({"category":name,"coverage":coverage,"build_ms_median":times[32],"build_ms_p95":times[60],"max_triangles":samples.iter().map(|s|s["triangles"].as_u64().unwrap()).max()}));
        categories
            .push(json!({"kind":name,"samples":samples,"depth_descriptors":maps,"failures":[]}));
    }
    let report = json!({"schema_version":1,"build_provenance":crate::provenance::capture_provenance(),"sample_count_per_category":64,"method":"fixed-envelope six orthographic depth maps; geometry coverage is a shape proxy, not statistical independence or SigLIP volume","summaries":summaries,"categories":categories});
    std::fs::write(dir.join("audit.json"), serde_json::to_vec(&report).unwrap()).unwrap();
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&report["summaries"]).unwrap()
    );
}
