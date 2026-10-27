//! Bounded geometry/accepted-placement study; no renderer, GPU or model loads.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview,
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use image::{Rgb, RgbImage};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

// Small/medium product constructors. Complete bookcases have their own bay,
// transformed-mesh and placement tests; their many books exceed this mesh budget.
const KINDS: [ObjectKind; 31] = [
    ObjectKind::StorageBox,
    ObjectKind::CoatRack,
    ObjectKind::TrashCan,
    ObjectKind::PenHolder,
    ObjectKind::DeskOrganizer,
    ObjectKind::DocumentTray,
    ObjectKind::Bread,
    ObjectKind::SoapDispenser,
    ObjectKind::KitchenTowel,
    ObjectKind::CuttingBoard,
    ObjectKind::PowerStrip,
    ObjectKind::Cabinet,
    ObjectKind::Bench,
    ObjectKind::DrawerUnit,
    ObjectKind::Projector,
    ObjectKind::KnifeBlock,
    ObjectKind::Notebook,
    ObjectKind::Books,
    ObjectKind::MobileCart,
    ObjectKind::RoomDivider,
    ObjectKind::Kettle,
    ObjectKind::DeskFan,
    ObjectKind::Printer,
    ObjectKind::Dishware,
    ObjectKind::DryingRack,
    ObjectKind::UtensilHolder,
    ObjectKind::Speaker,
    ObjectKind::Headphones,
    ObjectKind::Microphone,
    ObjectKind::Stapler,
    ObjectKind::TapeDispenser,
];

#[test]
fn all_catalog_geometry_remains_valid_after_object_expansion() {
    let output = std::env::var_os("ZEROVERSE_OBJECT_REPORT").map(std::path::PathBuf::from);
    let mut categories = Vec::new();
    for kind in ObjectKind::ALL {
        let mut samples = Vec::new();
        let mut descriptors = Vec::new();
        for seed in 0..8 {
            let o = fixture(kind, seed);
            let start = Instant::now();
            let a = build_object(&o);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let mesh = check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            if output.is_some() {
                descriptors.push(descriptor(&o, &a));
                samples.push(json!({"seed":seed,"mesh":mesh,"build_ms":ms}));
            }
        }
        if output.is_some() {
            let coverage: Vec<_> = DISTANCE_THRESHOLDS.iter().map(|&t| json!({"threshold":t,"representatives":representatives(&descriptors,t).len()})).collect();
            categories.push(json!({"kind":kind,"coverage":coverage,"samples":samples}));
        }
    }
    if let Some(path) = output {
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("catalog.json"), serde_json::to_vec_pretty(&json!({"provenance":crate::provenance::capture_provenance(),"samples_per_category":8,"categories":categories,"method":"actual meshes; six fixed-envelope orthographic depth maps, 24x24 pixels each; CPU construction timing excludes descriptors and validation"})).unwrap()).unwrap();
    }
}

#[test]
fn object_regularity_fixed_envelopes_replay_and_vary() {
    let output = std::env::var_os("ZEROVERSE_OBJECT_REPORT").map(std::path::PathBuf::from);
    let mut sheet = output
        .as_ref()
        .map(|_| RgbImage::from_pixel(8 * 192, KINDS.len() as u32 * 192, Rgb([241, 243, 246])));
    let mut categories = Vec::new();
    for (row, kind) in KINDS.into_iter().enumerate() {
        let mut descriptors = Vec::new();
        let mut substrates = BTreeSet::new();
        let mut samples = Vec::new();
        for seed in 0..32 {
            let o = fixture(kind, seed);
            let start = Instant::now();
            let a = build_object(&o);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let mesh = check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            assert!(
                mesh.triangles < 30_000,
                "unexpected mesh expansion {kind:?}"
            );
            let replay = build_object(&o);
            assert_eq!(a.parts.len(), replay.parts.len());
            for (key, g) in &a.parts {
                let h = &replay.parts[key];
                assert_eq!(g.positions, h.positions);
                assert_eq!(g.normals, h.normals);
                assert_eq!(g.uvs, h.uvs);
                assert_eq!(g.indices, h.indices);
            }
            let set: BTreeSet<_> = a.parts.keys().map(|(s, _)| format!("{s:?}")).collect();
            substrates.insert(set);
            descriptors.push(descriptor(&o, &a));
            samples.push(json!({"seed":seed,"program":program(&o),"mesh":mesh,"build_ms":ms}));
            if let Some(sheet) = sheet.as_mut().filter(|_| seed < 8) {
                let im = preview::orthographic(
                    &a,
                    Vec3::Y * o.size.y * 0.46,
                    o.size.length() * 1.04,
                    192,
                    Vec3::new(1.1, 1.25, 1.6),
                );
                image::imageops::replace(sheet, &im, seed as i64 * 192, row as i64 * 192);
            }
        }
        let coverage: Vec<_> = DISTANCE_THRESHOLDS
            .iter()
            .map(
                |&t| json!({"threshold":t,"representatives":representatives(&descriptors,t).len()}),
            )
            .collect();
        assert!(
            representatives(&descriptors, 0.005).len() >= 8,
            "{kind:?} collapsed: {coverage:?}"
        );
        categories.push(json!({"kind":kind,"coverage":coverage,"substrate_combinations":substrates.len(),"samples":samples}));
    }
    if let Some(path) = output {
        std::fs::create_dir_all(&path).unwrap();
        sheet.unwrap().save(path.join("objects.png")).unwrap();
        std::fs::write(path.join("geometry.json"), serde_json::to_vec_pretty(&json!({"schema":1,"provenance":crate::provenance::capture_provenance(),"samples_per_kind":32,"method":"fixed-envelope six-view 24x24 depth coverage; no materials or UVs in shape distance; CPU flat-shaded contact sheet","categories":categories})).unwrap()).unwrap();
    }
}

#[test]
fn cabinet_contents_sample_usable_bays_without_center_or_axis_atoms() {
    let parent = fixture(ObjectKind::Cabinet, 0);
    let lo = Vec3::new(-0.23, 0.32, -0.18);
    let hi = Vec3::new(0.23, 0.65, 0.18);
    let mut accepted = 0;
    let mut centered = 0;
    let mut aligned = 0;
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for seed in 0..256 {
        let kind = if seed % 2 == 0 {
            ObjectKind::StorageBox
        } else {
            ObjectKind::Vase
        };
        let Some(o) = storage::bay_contents::item(&parent, seed, kind, lo, hi) else {
            continue;
        };
        accepted += 1;
        let a = build_object(&o);
        check(&o, &a).unwrap();
        for p in a.parts.values().flat_map(|g| &g.positions) {
            let p = o.transform().transform_point(Vec3::from_array(*p));
            assert!(
                p.cmpge(lo - Vec3::splat(0.0001)).all() && p.cmple(hi + Vec3::splat(0.0001)).all(),
                "{seed}: {p:?}"
            );
        }
        let q = (o.position.xz() - (lo + hi).xz() * 0.5) / (hi - lo).xz();
        min = min.min(q);
        max = max.max(q);
        centered += usize::from(q.abs().max_element() < 0.02);
        aligned += usize::from(
            (4. * o.yaw).sin().atan2((4. * o.yaw).cos()).abs() * 0.25 < 2_f32.to_radians(),
        );
    }
    assert!(
        accepted > 220 && centered < accepted / 10 && aligned < accepted / 8,
        "accepted={accepted}, centered={centered}, aligned={aligned}"
    );
    assert!(min.cmplt(Vec2::splat(-0.12)).all() && max.cmpgt(Vec2::splat(0.12)).all());
    if let Some(path) = std::env::var_os("ZEROVERSE_OBJECT_REPORT").map(std::path::PathBuf::from) {
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("cabinet-placement.json"), serde_json::to_vec_pretty(&json!({"proposed":256,"accepted":accepted,"centered_within_two_percent":centered,"axis_aligned_within_two_degrees":aligned,"support_local_min_fraction":min,"support_local_max_fraction":max,"all_transformed_vertices_contained":true})).unwrap()).unwrap();
    }
}

#[test]
fn expanded_object_sampling_preserves_room_supports_semantics_and_collections() {
    let mut seen = BTreeMap::<String, usize>::new();
    let mut group_counts = [0_usize; 3];
    let mut times = Vec::new();
    let mut storage_placements = Vec::new();
    for profile in IndoorLayout::PROFILES {
        for seed in 0..2 {
            let start = Instant::now();
            let scene = IndoorManifest::generate_with_humans(seed, profile, 0.75, 3, 0.25).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.);
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{profile:?}/{seed}: {e}"));
            validation::validate_geometry(&scene)
                .unwrap_or_else(|e| panic!("{profile:?}/{seed}: {e}"));
            for o in scene
                .objects
                .iter()
                .filter(|o| !o.neighbor && KINDS.contains(&o.kind))
            {
                *seen.entry(format!("{:?}", o.kind)).or_default() += 1;
                if matches!(
                    o.kind,
                    ObjectKind::StorageBox | ObjectKind::TrashCan | ObjectKind::CoatRack
                ) {
                    let a = build_object(o);
                    check(o, &a).unwrap();
                    storage_placements.push(json!({
                        "layout":profile,"room_seed":seed,"kind":o.kind,"seed":o.seed,
                        "size":o.size,"position":o.position,"yaw":o.yaw,"support":o.support,
                        "room_size":scene.room_size,"program":program(o),"mesh_bounds":a.bounds(),
                    }));
                }
            }
            for group in &scene.assembly_diagnostics.accepted {
                let seed = scene.objects[group.surface].seed;
                let count = group
                    .seats
                    .iter()
                    .filter(|&&i| scene.objects[i].finish_seed == Some(seed))
                    .count();
                group_counts[if count == 0 {
                    0
                } else if count == group.seats.len() {
                    2
                } else {
                    1
                }] += 1;
            }
        }
    }
    assert!(seen.len() >= 7, "realized categories {seen:?}");
    assert!(
        group_counts[1] > 0 && group_counts[2] > 0,
        "{group_counts:?}"
    );
    if let Some(path) = std::env::var_os("ZEROVERSE_OBJECT_REPORT").map(std::path::PathBuf::from) {
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("storage-room-placement.json"),
            serde_json::to_vec_pretty(&json!({
                "scope":"accepted non-neighbor scene objects; not image visibility weighted",
                "rooms":40,"profiles":IndoorLayout::PROFILES,"seeds":[0,1],
                "geometry_and_layout_validated":true,"items":storage_placements,
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(path.join("rooms.json"), serde_json::to_vec_pretty(&json!({"provenance":crate::provenance::capture_provenance(),"rooms":40,"profiles":IndoorLayout::PROFILES,"seeds":[0,1],"density":0.75,"cameras":3,"human_density":0.25,"accepted_objects":seen,"collection_finish_groups":{"all_independent":group_counts[0],"mixed":group_counts[1],"all_matching":group_counts[2]},"manifest_cpu_ms":times,"full_geometry_and_layout_validation":true})).unwrap()).unwrap();
    }
}
