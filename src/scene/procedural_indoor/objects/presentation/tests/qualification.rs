//! Local physical-construction audit, not a claim about downstream learning.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview,
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use image::{Rgb, RgbImage};
use serde_json::json;
use std::{path::Path, time::Instant};

#[test]
#[ignore = "local presentation equipment and accepted-placement qualification"]
fn presentation_equipment_population() {
    let out = Path::new("out/presentation_equipment");
    std::fs::create_dir_all(out).unwrap();
    let mut sheet = RgbImage::from_pixel(6 * 280, 3 * 280, Rgb([241, 243, 246]));
    let mut categories = Vec::new();
    for (row, kind) in KINDS.into_iter().enumerate() {
        let mut descriptors = Vec::new();
        let mut samples = Vec::new();
        for seed in 0..64 {
            let o = audit::fixture(kind, seed);
            let start = Instant::now();
            let a = build_object(&o);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let mesh = audit::check(&o, &a).unwrap();
            descriptors.push(audit::descriptor(&o, &a));
            samples.push(
                json!({"seed":seed,"program":audit::program(&o),"mesh":mesh,"construction_ms":ms}),
            );
            if seed < 6 {
                let img = preview::orthographic(
                    &a,
                    Vec3::Y * o.size.y * 0.5,
                    o.size.length() * 1.08,
                    280,
                    Vec3::new(0.7, 0.55, 1.4),
                );
                img.save(out.join(format!("{kind:?}-{seed}.png"))).unwrap();
                image::imageops::replace(&mut sheet, &img, seed as i64 * 280, row as i64 * 280);
                if kind == ObjectKind::Whiteboard {
                    let p = parameters(&o);
                    let center = (p.tray_lo + p.tray_hi) * 0.5;
                    // Tight physical tray view to make small accessories reviewable.
                    let tray = preview::orthographic(
                        &a,
                        Vec3::new(center.x, p.tray_base_top_m, center.y),
                        (p.tray_hi.x - p.tray_lo.x) * 1.12,
                        560,
                        Vec3::new(0.2, 1.0, 1.5),
                    );
                    tray.save(out.join(format!("tray-{seed}.png"))).unwrap();
                }
            }
        }
        let coverage:Vec<_>=audit::DISTANCE_THRESHOLDS.into_iter().map(|threshold|json!({"threshold":threshold,"representatives":audit::representatives(&descriptors,threshold).len()})).collect();
        categories.push(json!({"kind":kind,"fixed_envelope_m":audit::fixture(kind,0).size,"coverage":coverage,"samples":samples}));
    }
    let mut rooms = Vec::new();
    for (i, layout) in IndoorLayout::PROFILES.into_iter().enumerate() {
        for j in 0..4 {
            let seed = 100000 + (i * 4 + j) as u64;
            let start = Instant::now();
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 3, 0.15).unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.;
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            validation::validate_geometry(&scene)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            let objects: Vec<_> = scene
                .objects
                .iter()
                .filter(|o| !o.neighbor && KINDS.contains(&o.kind))
                .map(|o| json!({"object":o,"program":audit::program(o)}))
                .collect();
            rooms.push(json!({"layout":layout,"seed":seed,"manifest_ms":ms,"objects":objects}));
        }
    }
    sheet.save(out.join("gallery.png")).unwrap();
    std::fs::write(out.join("report.json"),serde_json::to_vec_pretty(&json!({
        "generator":crate::scene::procedural_indoor::layout::GENERATOR_VERSION,"capture_identity":crate::CAPTURE_ENGINE_IDENTITY,
        "method":"Fixed-envelope/finish actual meshes; six 24x24 first-surface depth maps, normalized union-pixel L1. CPU flat shading. Accepted placements exclude neighbors and are not visibility weighted. Distinct base seeds across all profiles.",
        "categories":categories,"rooms":rooms,"room_configuration":{"density":0.65,"cameras":3,"human_density":0.15}
    })).unwrap()).unwrap();
    println!(
        "192 meshes, 80 complete-room geometry/layout checks across all 20 profiles; {}",
        out.display()
    );
}
