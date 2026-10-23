use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    objects::{audit, build_object, storage},
    validation,
};
use std::collections::BTreeSet;

const KINDS: [ObjectKind; 3] = [
    ObjectKind::Bookcase,
    ObjectKind::Locker,
    ObjectKind::Banquette,
];

#[test]
fn varied_storage_and_banquettes_remain_bounded_semantic_and_replayable() {
    for kind in KINDS {
        for seed in 0..96 {
            let mut o = audit::fixture(kind, seed);
            o.size *= match seed % 3 {
                0 => Vec3::new(0.85, 1.15, 0.85),
                1 => Vec3::new(1.15, 0.85, 1.15),
                _ => Vec3::ONE,
            };
            let mesh = build_object(&o);
            let stats = audit::check(&o, &mesh).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            assert!(
                stats.triangles < 45_000,
                "{kind:?}/{seed}: {} triangles",
                stats.triangles
            );
            assert!(mesh.bounds().0.y >= -1e-5);
            let replay = build_object(&o);
            assert_eq!(mesh.parts.len(), replay.parts.len());
            for (key, g) in mesh.parts {
                let b = &replay.parts[&key];
                assert_eq!(g.positions, b.positions);
                assert_eq!(g.indices, b.indices);
                assert_eq!(g.uvs, b.uvs);
            }
        }
    }
}

#[test]
fn construction_distributions_do_not_collapse_at_a_fixed_envelope() {
    let mut backs = BTreeSet::new();
    let mut bases = BTreeSet::new();
    let mut fronts = BTreeSet::new();
    let mut tiers = BTreeSet::new();
    let mut open_doors = 0;
    let mut closed_doors = 0;
    let mut unaligned_shelves = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Banquette, seed);
        o.size = Vec3::new(2., 1.1, 0.75);
        let b = banquette::parameters(&o);
        backs.insert(b.back);
        bases.insert(b.base);
        o.kind = ObjectKind::Locker;
        o.size = Vec3::new(1.4, 1.8, 0.5);
        let l = education::parameters(&o);
        fronts.insert(l.front);
        for b in &l.bays {
            tiers.insert(b.levels.len());
            open_doors += b.open.iter().filter(|&&v| v).count();
            closed_doors += b.open.iter().filter(|&&v| !v).count();
        }
        o.kind = ObjectKind::Bookcase;
        let b = storage::bookcase::parameters(&o);
        unaligned_shelves += usize::from(b.bays.windows(2).any(|v| v[0].levels != v[1].levels));
    }
    assert_eq!(
        (backs.len(), bases.len(), fronts.len(), tiers.len()),
        (4, 3, 4, 4)
    );
    assert!(open_doors > 100 && closed_doors > open_doors * 2);
    assert!(
        unaligned_shelves > 100,
        "unaligned shelves: {unaligned_shelves}"
    );
}

#[test]
fn furnished_rooms_accept_varied_storage_and_seating() {
    let mut seen = BTreeSet::new();
    for profile in [
        IndoorLayout::Classroom,
        IndoorLayout::Cafe,
        IndoorLayout::Lounge,
        IndoorLayout::OpenOffice,
    ] {
        for seed in 0..16 {
            let scene = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.).unwrap();
            validation::validate_layout(&scene).unwrap();
            for o in &scene.objects {
                if KINDS.contains(&o.kind) {
                    seen.insert(format!("{:?}", o.kind));
                    audit::check(o, &build_object(o)).unwrap();
                }
            }
            if seed < 2 {
                validation::validate_geometry(&scene).unwrap();
            }
        }
    }
    assert_eq!(seen.len(), KINDS.len());
}

#[test]
#[ignore = "local 64-per-category shape and render qualification; writes out/repetition_review"]
fn repetition_geometry_receipt() {
    use serde_json::json;
    use std::time::Instant;
    let dir = std::path::Path::new("out/repetition_review");
    std::fs::create_dir_all(dir).unwrap();
    let mut categories = Vec::new();
    let mut summaries = Vec::new();
    for kind in KINDS {
        let fixed = match kind {
            ObjectKind::Bookcase => Vec3::new(1.2, 1.8, 0.40),
            ObjectKind::Locker => Vec3::new(1.4, 1.8, 0.5),
            _ => Vec3::new(2., 1.1, 0.75),
        };
        let mut samples = Vec::new();
        let mut maps = Vec::new();
        let mut times = Vec::new();
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            o.size = fixed;
            let now = Instant::now();
            let mesh = build_object(&o);
            times.push(now.elapsed().as_secs_f64() * 1000.);
            let stats = audit::check(&o, &mesh).unwrap();
            maps.push(audit::descriptor(&o, &mesh));
            samples.push(json!({"object":o,"program":audit::program(&o),"triangles":stats.triangles,"build_ms":times.last()}));
        }
        times.sort_by(f64::total_cmp);
        let coverage: Vec<_> = audit::DISTANCE_THRESHOLDS
            .iter()
            .map(
                |&t| json!({"threshold":t,"representatives":audit::representatives(&maps,t).len()}),
            )
            .collect();
        summaries.push(json!({"category":kind,"coverage":coverage,"build_ms_median":times[32],"build_ms_p95":times[60],"max_triangles":samples.iter().map(|s|s["triangles"].as_u64().unwrap()).max()}));
        categories.push(json!({"kind":format!("{kind:?}"),"samples":samples,"depth_descriptors":maps,"failures":[]}));
    }
    let report = json!({"schema_version":1,"build_provenance":crate::provenance::capture_provenance(),"sample_count_per_category":64,"method":"fixed envelopes and finish; six orthographic depth-map coverage at specified thresholds, not a statistical independence or learning utility estimate","summaries":summaries,"categories":categories});
    std::fs::write(dir.join("audit.json"), serde_json::to_vec(&report).unwrap()).unwrap();
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&report["summaries"]).unwrap()
    );
}
