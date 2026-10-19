//! Local CPU qualification only: no renderer, device, or external image/model assets.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    materials::{art, boards, screens},
    validation,
};
use image::{Rgb, RgbImage, RgbaImage};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path, time::Instant};

const KINDS: [ObjectKind; 11] = [
    ObjectKind::Elevator,
    ObjectKind::Sculpture,
    ObjectKind::InformationStand,
    ObjectKind::ExhibitionPanel,
    ObjectKind::NoticeBoard,
    ObjectKind::Whiteboard,
    ObjectKind::WallArt,
    ObjectKind::Monitor,
    ObjectKind::Display,
    ObjectKind::Phone,
    ObjectKind::Laptop,
];

#[test]
#[ignore = "bounded local CPU shape/content/room qualification; writes out/display_diversity"]
fn display_exhibit_population_qualification() {
    let dir = Path::new("out/display_diversity");
    std::fs::create_dir_all(dir).unwrap();
    let mut geometry = Vec::new();
    let mut sheets = RgbImage::from_pixel(8 * 160, KINDS.len() as u32 * 160, Rgb([241, 243, 246]));
    for (row, kind) in KINDS.into_iter().enumerate() {
        let fixed_size = fixture(kind, 0).size;
        let mut descriptors = Vec::new();
        let mut samples = Vec::new();
        for seed in 0..64 {
            let mut o = fixture(kind, seed);
            o.size = fixed_size;
            if kind == ObjectKind::WallArt {
                o.variant = (seed % 6) as u32;
            }
            let begin = Instant::now();
            let a = build_object(&o);
            let construction_ms = begin.elapsed().as_secs_f64() * 1000.;
            let mesh = check(&o, &a).unwrap_or_else(|e| panic!("{e}"));
            descriptors.push(descriptor(&o, &a));
            samples.push(json!({"seed":seed,"program":program(&o),"mesh":mesh,"construction_ms":construction_ms}));
            if seed < 8 {
                image::imageops::replace(
                    &mut sheets,
                    &thumbnail(&o, &a, 160),
                    seed as i64 * 160,
                    row as i64 * 160,
                );
            }
        }
        let coverage: Vec<_> = DISTANCE_THRESHOLDS
            .iter()
            .map(|&t| json!({"threshold":t,"representatives":representatives(&descriptors,t)}))
            .collect();
        let nearest: Vec<f32> = descriptors
            .iter()
            .enumerate()
            .map(|(i, a)| {
                descriptors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, b)| distance(a, b))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        geometry.push(
            json!({"kind":kind,"fixed_size_metres":fixed_size,"coverage":coverage,
            "nearest_distance":nearest,"samples":samples}),
        );
    }
    sheets.save(dir.join("geometry-contact.png")).unwrap();
    let mut textures = Vec::new();
    for kind in ["desktop", "phone", "television", "whiteboard", "art"] {
        let mut samples = Vec::new();
        let mut fingerprints = BTreeSet::new();
        let mut descriptors = Vec::new();
        let mut sheet = RgbaImage::new(1024, 1024);
        for seed in 0..64 {
            let begin = Instant::now();
            let (pixels, program) = match kind {
                "desktop" => (
                    screens::pixels(seed),
                    json!(screens::parameters_for(seed, screens::DisplayKind::Desktop)),
                ),
                "phone" => (
                    screens::mobile_pixels(seed),
                    json!(screens::parameters_for(seed, screens::DisplayKind::Phone)),
                ),
                "television" => (
                    screens::television_pixels(seed),
                    json!(screens::parameters_for(
                        seed,
                        screens::DisplayKind::Television
                    )),
                ),
                "whiteboard" => (boards::pixels(seed), json!(boards::parameters(seed))),
                _ => (art::pixels(seed), json!(art::parameters(seed))),
            };
            let synthesis_ms = begin.elapsed().as_secs_f64() * 1000.;
            assert_eq!(pixels.len(), 256 * 256 * 4);
            assert!(pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
            let hash = format!("{:x}", Sha256::digest(&pixels));
            fingerprints.insert(hash.clone());
            // Area means remove fine grain; this is a coarse RGB-distance proxy,
            // deliberately not labelled semantic embedding volume or capacity.
            let mut coarse = vec![0.; 16 * 16 * 3];
            for (i, p) in pixels.as_chunks::<4>().0.iter().enumerate() {
                let cell = (i / 256 / 16) * 16 + i % 256 / 16;
                for ch in 0..3 {
                    coarse[cell * 3 + ch] += p[ch] as f32 / (256. * 255.);
                }
            }
            descriptors.push(coarse);
            let image = RgbaImage::from_raw(256, 256, pixels).unwrap();
            let thumb =
                image::imageops::resize(&image, 128, 128, image::imageops::FilterType::Triangle);
            image::imageops::replace(
                &mut sheet,
                &thumb,
                (seed % 8) as i64 * 128,
                (seed / 8) as i64 * 128,
            );
            samples.push(json!({"seed":seed,"sha256":hash,"content_synthesis_ms":synthesis_ms,"program":program}));
        }
        sheet.save(dir.join(format!("{kind}-contact.png"))).unwrap();
        let nearest: Vec<f32> = descriptors
            .iter()
            .enumerate()
            .map(|(i, a)| {
                descriptors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, b)| distance(a, b))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        textures.push(json!({"kind":kind,"unique_pixel_hashes":fingerprints.len(),
            "nearest_16x16_mean_rgb_l1":nearest,"samples":samples}));
    }
    let mut rooms = Vec::new();
    for seed in 0..64 {
        let profile = if seed < 32 {
            IndoorLayout::Mixed
        } else {
            [
                IndoorLayout::ElevatorLanding,
                IndoorLayout::Gallery,
                IndoorLayout::Showroom,
                IndoorLayout::Conference,
            ][seed as usize % 4]
        };
        let begin = Instant::now();
        let room = IndoorManifest::generate_with_humans(seed, profile, 0.65, 3, 0.25).unwrap();
        let manifest_ms = begin.elapsed().as_secs_f64() * 1000.;
        validation::validate_layout(&room).unwrap();
        let begin = Instant::now();
        let mesh = validation::validate_geometry(&room).unwrap();
        assert_eq!(mesh.degenerate_triangles, 0, "seed {seed}");
        rooms.push(
            json!({"seed":seed,"requested_profile":profile,"layout":room.layout,
            "manifest_ms":manifest_ms,"mesh_validation_ms":begin.elapsed().as_secs_f64()*1000.,
            "objects":room.objects.len(),"triangles":mesh.triangles}),
        );
    }
    let report = json!({"generator_revision":crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
        "scope":"CPU meshes/content/semantics/layout only; no GPU render, SigLIP or downstream training qualification",
        "shape_metric":"ordered greedy coverage of six 24x24 depth maps; fixed size, finish seed zero; not dataset capacity",
        "geometry_contact":"CPU flat-shaded triangles, category rows in report order, seeds 0..7; not PBR",
        "content_contact":"production 256x256 base-color pixels, row-major seeds 0..63, downsampled for preview",
        "geometry":geometry,"textures":textures,"rooms":rooms});
    std::fs::write(
        dir.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "704 fixed-envelope meshes, 320 content maps, 64 rooms qualified; {}",
        dir.display()
    );
}

/// Flat-shaded CPU schematic for inspection of actual production triangles.
/// Deliberately no PBR, texture filtering, shadows or renderer-performance claims.
fn thumbnail(o: &IndoorObject, a: &Assembly, n: u32) -> RgbImage {
    crate::scene::procedural_indoor::geometry::preview::orthographic(
        a,
        Vec3::Y * o.size.y * 0.5,
        o.size.length() * 1.02,
        n,
        Vec3::new(0.65, 0.45, 2.),
    )
}
