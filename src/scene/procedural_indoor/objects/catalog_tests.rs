use super::super::{
    items,
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn added_catalog_has_contact_valid_meshes_semantics_and_reserved_bounds() {
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let template = scene.objects[0].clone();
    scene.objects.clear();
    let floor = [
        (ObjectKind::SideTable, Vec3::new(0.42, 0.42, 0.39)),
        (ObjectKind::SideTable, Vec3::new(0.72, 0.66, 0.62)),
        (ObjectKind::MobileCart, Vec3::new(0.60, 0.74, 0.42)),
        (ObjectKind::DrawerUnit, Vec3::new(0.35, 0.51, 0.37)),
        (ObjectKind::Bench, Vec3::new(0.95, 0.46, 0.39)),
        (ObjectKind::Bench, Vec3::new(1.70, 0.87, 0.54)),
        (ObjectKind::Ottoman, Vec3::new(0.47, 0.35, 0.47)),
        (ObjectKind::RoomDivider, Vec3::new(0.90, 1.05, 0.35)),
        (ObjectKind::Cabinet, Vec3::new(0.64, 1.12, 0.32)),
        (ObjectKind::Bookcase, Vec3::new(0.70, 1.35, 0.34)),
    ];
    for seed in 0..64 {
        let mut support = template.clone();
        support.kind = ObjectKind::Cabinet;
        support.seed = seed;
        let program = items::parameters(&scene, &support);
        let mut rng = stream(seed, 9000);
        let cases = floor
            .into_iter()
            .chain((0..40).map(|_| program.item(&support, &mut rng)));
        for (kind, size) in cases {
            let mut object = template.clone();
            object.id = scene.objects.len();
            object.kind = kind;
            object.size = size;
            object.position = Vec3::ZERO;
            object.yaw = 0.;
            object.seed = seed;
            object.variant = seed as u32 % 3;
            object.support = None;
            object.interaction_target = None;
            let geometry = build_object(&object);
            let (lo, hi) = geometry.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(0.003))
                    .all()
                    && hi
                        .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.003))
                        .all(),
                "seed {seed} {kind:?}: {lo:?}..{hi:?}, size={size:?}"
            );
            assert!(lo.y.abs() < 0.003, "{kind:?} has no contact: {lo:?}");
            assert!(geometry
                .parts
                .keys()
                .all(|(_, label)| SemanticLabel::from_label(part_label(label)).is_some()));
            scene.objects.push(object);
        }
    }
    // Also inspect triangle areas, winding, normals, metric UVs and finiteness,
    // not just primitive bounds. Scene holds all min-size/profile combinations.
    validation::validate_geometry(&scene).unwrap();
    assert!(
        ObjectKind::ALL.len() <= 64,
        "placement coverage uses a u64 kind mask"
    );
}

#[test]
fn task_mixtures_are_replayable_and_cover_all_items_without_seed_identity() {
    let scene = IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut template = scene.objects[0].clone();
    template.kind = ObjectKind::Cabinet;
    template.size.y = 0.82;
    let mut kinds = BTreeSet::new();
    let mut tasks = BTreeSet::new();
    let mut programs = BTreeSet::new();
    for seed in 0..256 {
        template.seed = seed;
        template.kind = [
            ObjectKind::Desk,
            ObjectKind::Table,
            ObjectKind::Cabinet,
            ObjectKind::CoffeeTable,
            ObjectKind::MobileCart,
            ObjectKind::SideTable,
            ObjectKind::DrawerUnit,
        ][seed as usize % 7];
        let program = items::parameters(&scene, &template);
        assert!((program.task_weights.iter().sum::<f32>() - 1.).abs() < 1e-5);
        assert!(program
            .task_weights
            .iter()
            .all(|w| w.is_finite() && *w > 0.));
        assert_eq!(
            serde_json::to_value(&program).unwrap(),
            serde_json::to_value(items::parameters(&scene, &template)).unwrap()
        );
        programs.insert(program.task_weights.map(|w| (w * 20.) as u8));
        tasks.insert(program.dominant_task());
        let mut rng = stream(seed, 9010);
        for _ in 0..128 {
            kinds.insert(format!("{:?}", program.item(&template, &mut rng).0));
        }
    }
    assert_eq!(tasks.len(), 6);
    assert!(programs.len() >= 240);
    for kind in ObjectKind::ALL.into_iter().skip(42).take(13) {
        assert!(
            kinds.contains(&format!("{kind:?}")),
            "unreachable item {kind:?}"
        );
    }
}

#[test]
fn amenity_placements_replay_and_keep_activity_camera_support_and_human_constraints() {
    let mut occurrence = BTreeMap::<String, usize>::new();
    for seed in 0..128 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        validation::validate_layout(&scene).unwrap();
        if seed < 32 {
            validation::validate_geometry(&scene).unwrap();
        }
        let replay =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        assert_eq!(scene, replay);
        for object in scene.objects.iter().filter(|o| !o.neighbor) {
            *occurrence.entry(format!("{:?}", object.kind)).or_default() += 1;
        }
    }
    for kind in ObjectKind::ALL.into_iter().skip(36) {
        assert!(
            occurrence.get(&format!("{kind:?}")).copied().unwrap_or(0) >= 3,
            "insufficient accepted coverage for {kind:?}: {occurrence:?}"
        );
    }
    for seed in [
        202, 1_013_005, 43_084_584, 42_430_575, 43_218_880, 46_839_170, 47_586_113, 47_359_576,
        47_062_729, 49_309_986, 49_981_871,
    ] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        validation::validate_layout(&scene).unwrap();
        validation::validate_geometry(&scene).unwrap();
    }
}
