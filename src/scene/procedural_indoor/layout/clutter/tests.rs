use super::*;
use crate::scene::procedural_indoor::{objects, validation::validate_layout};
use std::{collections::BTreeSet, f32::consts::TAU};

#[test]
fn explicit_empty_and_dense_table_policies_share_physical_admission() {
    use crate::scene::procedural_indoor::items::distribution::ClutterSettings;
    let mut clear = 0;
    let mut busy = 0;
    for layout in [
        IndoorLayout::Conference,
        IndoorLayout::Cafe,
        IndoorLayout::Kitchen,
        IndoorLayout::Classroom,
    ] {
        for seed in 0..8 {
            for probability in [0., 1.] {
                let scene = IndoorManifest::generate_with_clutter(
                    seed,
                    layout,
                    0.75,
                    0,
                    0.4,
                    ClutterSettings {
                        probability,
                        minimum: 0.65,
                        maximum: 1.,
                        ..Default::default()
                    },
                )
                .unwrap();
                validate_layout(&scene)
                    .unwrap_or_else(|e| panic!("{layout:?}/{seed}/{probability}: {e}"));
                for support in scene.objects.iter().filter(|o| {
                    matches!(
                        o.kind,
                        ObjectKind::Desk
                            | ObjectKind::Table
                            | ObjectKind::CoffeeTable
                            | ObjectKind::SideTable
                    )
                }) {
                    let count = scene
                        .objects
                        .iter()
                        .filter(|o| {
                            o.support == Some(support.id)
                                && !(support.kind == ObjectKind::Desk
                                    && o.kind == ObjectKind::Laptop)
                                && !matches!(
                                    o.kind,
                                    ObjectKind::Cable
                                        | ObjectKind::Projector
                                        | ObjectKind::Monitor
                                        | ObjectKind::Keyboard
                                        | ObjectKind::Mouse
                                )
                        })
                        .count();
                    if probability == 0. {
                        assert_eq!(count, 0, "{layout:?}/{seed}: {:?}", support.kind);
                        clear += 1;
                    } else {
                        busy += usize::from(count >= 4);
                    }
                }
                let restored: IndoorManifest =
                    serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
                assert_eq!(scene.clutter, restored.clutter);
            }
        }
    }
    assert!(clear >= 20 && busy >= 10, "clear={clear} busy={busy}");
}

#[test]
fn loose_surface_programs_are_varied_bounded_supported_and_replayable() {
    let base =
        IndoorManifest::generate_with_humans(14, IndoorLayout::Conference, 0.7, 0, 0.).unwrap();
    let mut counts = Vec::new();
    let mut kinds = BTreeSet::new();
    let mut angles = BTreeSet::new();
    for seed in 0..128 {
        let mut scene = base.clone();
        let mut support = objects::audit::fixture(ObjectKind::Table, seed);
        support.id = 0;
        support.size = Vec3::new(1.5, 0.75, 2.8);
        support.yaw = 0.35;
        scene.objects = vec![support.clone()];
        scene.humans.clear();
        scene
            .program
            .as_mut()
            .unwrap()
            .domain
            .as_mut()
            .unwrap()
            .clutter = 0.75;
        let program = composition::parameters(&scene, &support);
        let mut replay = scene.clone();
        scene.dress_loose_surface(&support);
        replay.dress_loose_surface(&support);
        assert_eq!(scene.objects, replay.objects);
        assert!(scene.objects.len() <= program.target_items + 1);
        assert!(program.max_attempts <= 288);
        for o in &scene.objects[1..] {
            assert!(
                scene.prop_clear(o, &support, 0.012),
                "support clearance seed={seed}, object={o:?}"
            );
            assert!((o.position.y - support.size.y).abs() < 1e-5);
            kinds.insert(format!("{:?}", o.kind));
            angles.insert(((o.yaw - support.yaw).rem_euclid(TAU) * 12. / TAU) as u32);
        }
        counts.push(scene.objects.len() - 1);
    }
    counts.sort();
    assert!(counts[127] >= 18, "dense tables not generated: {counts:?}");
    assert!(
        counts[96] >= counts[32] + 5,
        "clutter too uniform: {counts:?}"
    );
    assert!(kinds.len() >= 20);
    assert_eq!(angles.len(), 12);
    println!("table qualification: count min/p25/median/p75/max={}/{}/{}/{}/{}, {} kinds, {} heading bins",counts[0],counts[32],counts[64],counts[96],counts[127],kinds.len(),angles.len());
}

#[test]
fn loose_clutter_density_zero_is_empty_and_area_increases_the_budget() {
    let mut scene =
        IndoorManifest::generate_with_humans(14, IndoorLayout::Conference, 0.7, 0, 0.).unwrap();
    let small = objects::audit::fixture(ObjectKind::Table, 91);
    let mut large = small.clone();
    large.size.x *= 2.;
    large.size.z *= 2.;
    scene
        .program
        .as_mut()
        .unwrap()
        .domain
        .as_mut()
        .unwrap()
        .clutter = 0.;
    scene.objects = vec![small.clone()];
    scene.dress_loose_surface(&small);
    assert_eq!(scene.objects.len(), 1);
    scene
        .program
        .as_mut()
        .unwrap()
        .domain
        .as_mut()
        .unwrap()
        .clutter = 0.4;
    let a = composition::parameters(&scene, &small);
    let b = composition::parameters(&scene, &large);
    assert!(b.target_items >= a.target_items * 3);
}

#[test]
fn clutter_population_preserves_room_layout_and_seated_person_clearance() {
    let mut tables = 0;
    let mut supported = 0;
    let mut occupied = 0;
    for layout in [
        IndoorLayout::Mixed,
        IndoorLayout::Conference,
        IndoorLayout::Cafe,
        IndoorLayout::Classroom,
    ] {
        for seed in 0..32 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.7, 0, 0.4).unwrap();
            validate_layout(&scene).unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            occupied += scene.humans.iter().filter(|h| h.chair.is_some()).count();
            for support in scene
                .objects
                .iter()
                .filter(|o| matches!(o.kind, ObjectKind::Table | ObjectKind::Desk))
            {
                tables += 1;
                for o in scene
                    .objects
                    .iter()
                    .filter(|o| o.support == Some(support.id))
                {
                    // Generation also has functional legacy content with 10 mm
                    // margins; validate the shared zero-overlap contract here.
                    assert!(scene.prop_clear(o, support, 0.), "{layout:?}/{seed}: {o:?}");
                    supported += 1;
                }
            }
        }
    }
    assert!(occupied > 10 && supported > 500 && tables > 100);
    println!("clutter rooms: 128, surfaces={tables}, supported={supported}, seated={occupied}");
}
