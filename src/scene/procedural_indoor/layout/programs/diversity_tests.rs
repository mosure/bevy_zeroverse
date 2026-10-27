//! Realized-geometry and admitted-placement regressions, not raw RNG entropy.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview,
    objects::{self, audit},
    validation,
};
use serde_json::json;
use std::{collections::BTreeSet, time::Instant};

#[test]
fn lounge_and_perimeter_sampling_preserve_function_and_vary_accepted_objects() {
    let mut storage_sizes = BTreeSet::new();
    let mut plant_widths = BTreeSet::new();
    let mut seat_offsets = BTreeSet::new();
    let mut ottomans = 0;
    for layout in [
        IndoorLayout::Lounge,
        IndoorLayout::Reception,
        IndoorLayout::Breakroom,
        IndoorLayout::OpenOffice,
        IndoorLayout::Library,
    ] {
        for seed in 0..12 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.7, 0, 0.25).unwrap();
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            validation::validate_geometry(&scene)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            for o in scene.objects.iter().filter(|o| !o.neighbor) {
                match o.kind {
                    ObjectKind::Cabinet | ObjectKind::Bookcase => {
                        storage_sizes.insert(o.size.to_array().map(|v| (v * 100.).round() as i32));
                        assert_ne!(o.size, Vec3::new(1.50, 1.85, 0.36));
                    }
                    ObjectKind::Plant if o.support.is_none() => {
                        plant_widths.insert((o.size.x * 100.).round() as i32);
                    }
                    ObjectKind::Chair => {
                        if let Some(table) = o
                            .interaction_target
                            .and_then(|i| scene.objects.get(i))
                            .filter(|o| o.kind == ObjectKind::CoffeeTable)
                        {
                            let local = table.transform().rotation.inverse()
                                * (o.position - table.position);
                            seat_offsets.insert(
                                [local.x, local.z, o.yaw - table.yaw]
                                    .map(|v| (v * 20.).round() as i32),
                            );
                        }
                    }
                    ObjectKind::Ottoman => {
                        ottomans += 1;
                    }
                    _ => {}
                }
            }
            if seed == 0 {
                let replay =
                    IndoorManifest::generate_with_humans(seed, layout, 0.7, 0, 0.25).unwrap();
                assert_eq!(scene, replay);
            }
        }
    }
    assert!(
        storage_sizes.len() > 35,
        "storage sizes: {}",
        storage_sizes.len()
    );
    assert!(
        plant_widths.len() > 30,
        "plant widths: {}",
        plant_widths.len()
    );
    assert!(
        seat_offsets.len() > 20,
        "lounge seat offsets: {}",
        seat_offsets.len()
    );
    assert!(ottomans > 12, "ottomans: {ottomans}");
}

#[test]
#[ignore = "local mesh gallery and 80-room placement/CPU timing study"]
fn lounge_regularity_qualification() {
    let out = std::path::Path::new("out/lounge_regularity/current");
    std::fs::create_dir_all(out).unwrap();
    let mut descriptors = Vec::new();
    let mut meshes = Vec::new();
    let mut sheet = image::RgbImage::from_pixel(6 * 240, 2 * 240, image::Rgb([241, 243, 246]));
    for seed in 0..64 {
        let mut o = audit::fixture(ObjectKind::Ottoman, seed);
        o.finish_seed = None;
        let start = Instant::now();
        let a = objects::build_object(&o);
        let ms = start.elapsed().as_secs_f64() * 1000.;
        let check = audit::check(&o, &a).unwrap();
        descriptors.push(audit::descriptor(&o, &a));
        meshes.push(
            json!({"seed":seed,"program":audit::program(&o),"mesh":check,"construction_ms":ms}),
        );
        if seed < 12 {
            let img = preview::orthographic(
                &a,
                Vec3::Y * o.size.y * 0.48,
                o.size.length() * 0.98,
                240,
                Vec3::new(1.1, 0.7, 1.6),
            );
            image::imageops::replace(
                &mut sheet,
                &img,
                (seed % 6) as i64 * 240,
                (seed / 6) as i64 * 240,
            );
        }
    }
    let coverage: Vec<_> = audit::DISTANCE_THRESHOLDS.into_iter().map(|threshold|
        json!({"threshold":threshold,"representatives":audit::representatives(&descriptors,threshold).len()})).collect();
    let mut rooms = Vec::new();
    for layout in IndoorLayout::PROFILES {
        for seed in 0..4 {
            let density = [0., 0.35, 0.75, 1.][seed as usize];
            let start = Instant::now();
            let scene =
                IndoorManifest::generate_with_humans(seed, layout, density, 3, 0.25).unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.;
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            validation::validate_geometry(&scene)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            if matches!(layout, IndoorLayout::Lounge | IndoorLayout::Reception) && seed < 3 {
                use crate::annotation::schematic::{Overlay, RenderOptions, Schematic};
                Schematic::from_manifest(&scene, 0.)
                    .unwrap()
                    .write(
                        out.join(format!("{layout:?}-{seed}")),
                        RenderOptions {
                            labels: false,
                            ..Default::default()
                        },
                        Overlay::default(),
                    )
                    .unwrap();
            }
            let furniture: Vec<_> = scene
                .objects
                .iter()
                .filter(|o| {
                    !o.neighbor
                        && matches!(
                            o.kind,
                            ObjectKind::Ottoman
                                | ObjectKind::Sofa
                                | ObjectKind::Chair
                                | ObjectKind::CoffeeTable
                                | ObjectKind::Cabinet
                                | ObjectKind::Bookcase
                                | ObjectKind::Plant
                        )
                })
                .map(|o| json!({"object":o,"construction":audit::program(o)}))
                .collect();
            rooms.push(
                json!({"layout":layout,"seed":seed,"density":density,"manifest_ms":ms,
                "rejected_placements":scene.rejected_placements,"furniture":furniture}),
            );
        }
    }
    sheet.save(out.join("ottomans.png")).unwrap();
    std::fs::write(out.join("report.json"), serde_json::to_vec_pretty(&json!({
        "generator":GENERATOR_VERSION,"capture_identity":crate::CAPTURE_ENGINE_IDENTITY,
        "scope":"CPU geometry, exact room validation and admitted placement; not PBR, rendered visibility or training utility",
        "geometry_distance":"fixed-envelope six 24x24 orthographic first-surface depth maps, normalized union-pixel L1",
        "meshes":meshes,"coverage":coverage,"rooms":rooms,"cameras":3,"human_density":0.25
    })).unwrap()).unwrap();
}
