use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorObject},
    validation,
};
use std::collections::BTreeSet;

const DOMAINS: [Domain; 18] = [
    Domain::SurfaceDressing,
    Domain::PublicDressing,
    Domain::WallOrder,
    Domain::WallArt,
    Domain::Displays,
    Domain::Whiteboards,
    Domain::Clocks,
    Domain::Electrical,
    Domain::WallEquipment,
    Domain::Pendants,
    Domain::WallGroups,
    Domain::ExitSigns,
    Domain::Alarms,
    Domain::Speakers,
    Domain::Detectors,
    Domain::StringLights,
    Domain::PullStations,
    Domain::PowerStandard,
];

#[test]
fn independent_proposals_replay_without_aliasing_counts_or_retry_draws() {
    for seed in [0, 1, 1_000_000_010, u64::MAX] {
        let mut keys = BTreeSet::new();
        for domain in DOMAINS {
            assert!(keys.insert(domain.rng(seed).random::<[u64; 4]>()));
            let mut previous = domain.proposal(seed, 0);
            for ordinal in [0, 1, 17, usize::MAX] {
                let expected = domain.proposal(seed, ordinal).random::<[u64; 4]>();
                for _ in 0..1024 {
                    let _ = previous.random::<u64>();
                }
                assert_eq!(
                    expected,
                    domain.proposal(seed, ordinal).random::<[u64; 4]>()
                );
                assert!(keys.insert(expected));
            }
        }
    }
}

fn correlation(pairs: &[[f64; 2]]) -> f64 {
    let n = pairs.len() as f64;
    let means = [0, 1].map(|i| pairs.iter().map(|p| p[i]).sum::<f64>() / n);
    let cov = pairs
        .iter()
        .map(|p| (p[0] - means[0]) * (p[1] - means[1]))
        .sum::<f64>();
    let vars = [0, 1].map(|i| pairs.iter().map(|p| (p[i] - means[i]).powi(2)).sum::<f64>());
    cov / (vars[0] * vars[1]).sqrt()
}

#[test]
fn decor_count_marginals_are_preserved_without_a_shared_category_budget() {
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    let kinds = [
        ObjectKind::WallArt,
        ObjectKind::Display,
        ObjectKind::Whiteboard,
        ObjectKind::Clock,
    ];
    let samples: Vec<_> = (0..4096)
        .map(|seed| {
            scene.seed = seed;
            kinds.map(|k| scene.wall_decoration_count(k) as f64)
        })
        .collect();
    for (i, mean) in [1.4, 1.4, 2.8, 1.4].into_iter().enumerate() {
        let measured = samples.iter().map(|v| v[i]).sum::<f64>() / samples.len() as f64;
        println!("proposal count {i}: mean={measured:.6}");
        assert!((measured - mean).abs() < 0.12, "{i}: {measured}");
        for j in 0..i {
            let pairs: Vec<_> = samples.iter().map(|p| [p[i], p[j]]).collect();
            println!("proposal counts {i}/{j}: r={:.6}", correlation(&pairs));
            assert!(
                correlation(&pairs).abs() < 0.06,
                "{i}/{j}: {}",
                correlation(&pairs)
            );
        }
    }
}

#[test]
fn artwork_is_neither_always_before_nor_after_alarms_or_electrical_fixtures() {
    for other in [
        WallPass::Alarms,
        WallPass::ExitSigns,
        WallPass::Electrical,
        WallPass::Groups,
    ] {
        let ahead = (0..2048)
            .filter(|&seed| {
                let order = wall_order(seed);
                assert_eq!(order, wall_order(seed));
                order.iter().position(|p| *p == WallPass::Art).unwrap()
                    < order.iter().position(|p| *p == other).unwrap()
            })
            .count();
        println!("art ahead of {other:?}: {ahead}/2048");
        assert!((900..1150).contains(&ahead), "{other:?}: {ahead}");
    }
}

fn objects_of(scene: &IndoorManifest, kind: ObjectKind) -> Vec<IndoorObject> {
    scene
        .objects
        .iter()
        .filter(|o| o.kind == kind)
        .map(|o| {
            let mut o = o.clone();
            o.id = 0; // Instance numbering is bookkeeping, not geometry or appearance.
            o
        })
        .collect()
}

#[test]
fn archived_rooms_sample_all_supported_attachment_walls() {
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    scene.envelope = None;
    let mut rng = Domain::WallArt.rng(412);
    let mut walls = BTreeSet::new();
    for _ in 0..256 {
        let (transform, span) = scene.attachment_wall(&mut rng);
        assert!(span > 0.);
        let normal = transform.rotation * bevy::prelude::Vec3::Z;
        walls.insert((
            (normal.x * 10.).round() as i32,
            (normal.z * 10.).round() as i32,
        ));
    }
    // Legacy +Z is an internal partition to the neighboring room; its openings
    // are not governed by the exterior facade attachment contract.
    assert_eq!(
        walls.len(),
        crate::scene::procedural_indoor::architecture::facade::FacadeSide::ALL.len()
    );
}

#[test]
fn no_op_decor_intervention_does_not_change_alarm_or_outlet_proposals() {
    let template =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    let mut admitted = [0; 2];
    for seed in 0..64 {
        let mut scene = template.clone();
        scene.seed = seed;
        scene.objects.clear();
        scene.rug_supports.clear();
        let mut changed = scene.clone();
        // Exercise full generation, then remove all geometry. It must not leave
        // a shared random cursor which reveals how many pictures were attempted.
        changed.compose_wall_groups();
        changed.wall_decoration_kind(ObjectKind::WallArt);
        changed.objects.clear();
        scene.furnish_wall_service(ObjectKind::FireAlarm);
        changed.furnish_wall_service(ObjectKind::FireAlarm);
        assert_eq!(
            objects_of(&scene, ObjectKind::FireAlarm),
            objects_of(&changed, ObjectKind::FireAlarm)
        );
        scene.wall_hardware();
        changed.wall_hardware();
        assert_eq!(
            objects_of(&scene, ObjectKind::WallOutlet),
            objects_of(&changed, ObjectKind::WallOutlet)
        );
        admitted[0] += objects_of(&scene, ObjectKind::FireAlarm).len();
        admitted[1] += objects_of(&scene, ObjectKind::WallOutlet).len();
    }
    assert!(admitted[0] > 40 && admitted[1] > 100, "{admitted:?}");
}

#[test]
fn reported_room_and_consecutive_scenes_keep_mesh_support_and_camera_contracts() {
    for seed in [1_000_000_010, 67_354_233, 67_331_663, 67_314_561]
        .into_iter()
        .chain(0..32)
    {
        let people = if seed % 4 == 0 { 0.3 } else { 0. };
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, people)
                .unwrap_or_else(|e| panic!("{seed}: {e}"));
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
        validation::validate_geometry(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
        scene.validate_public_programs().unwrap();
        assert!(scene.accepts_camera_geometry());
        assert_eq!(scene.cameras.len(), 3);
        if seed == 1_000_000_010 {
            assert_eq!(
                scene,
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, people)
                    .unwrap()
            );
        }
    }
}
