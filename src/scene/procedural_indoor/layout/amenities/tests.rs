use super::*;
use crate::scene::procedural_indoor::{
    objects::{self, audit},
    validation,
};
use std::collections::{BTreeMap, BTreeSet};
use std::f32::consts::{PI, TAU};

#[test]
fn accepted_amenities_use_multiple_edges_and_headings_and_replay() {
    let mut base =
        IndoorManifest::generate_with_humans(0, IndoorLayout::OpenOffice, 0.8, 0, 0.).unwrap();
    base.objects.clear();
    base.envelope = None;
    base.ceiling_installation = None;
    base.program = None;
    base.assembly_diagnostics = Default::default();
    base.rug_supports.clear();
    base.room_size = Vec3::new(18., 5., 18.);
    let mut counts: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    let mut headings: BTreeMap<String, BTreeSet<i32>> = BTreeMap::new();
    let mut first: BTreeSet<String> = BTreeSet::new();
    let mut records = Vec::new();
    for seed in 0..256 {
        let mut scene = base.clone();
        scene.seed = seed;
        let mut anchor = audit::fixture(
            if seed % 2 == 0 {
                ObjectKind::Desk
            } else {
                ObjectKind::Sofa
            },
            seed,
        );
        anchor.yaw = 0.63;
        scene.objects.push(anchor.clone());
        let mut replay = scene.clone();
        scene.furnish_amenities();
        replay.furnish_amenities();
        assert_eq!(scene.objects, replay.objects);
        if let Some(o) = scene.objects.get(1) {
            first.insert(format!("{:?}", o.kind));
        }
        let mut checked = base.clone();
        checked.objects.push(anchor.clone());
        for o in scene.objects.iter().skip(1) {
            assert!(
                checked.placement_clear(o, 0.06),
                "{seed}/{:?}: admission",
                o.kind
            );
            checked.objects.push(o.clone());
            let a = objects::build_object(o);
            audit::check(o, &a).unwrap_or_else(|e| panic!("{seed}/{:?}: {e}", o.kind));
            let pos = Quat::from_rotation_y(-anchor.yaw) * (o.position - anchor.position);
            let side = if (pos.x / anchor.size.x).abs() > (pos.z / anchor.size.z).abs() {
                usize::from(pos.x > 0.)
            } else {
                2 + usize::from(pos.z > 0.)
            };
            let key = format!("{:?}", o.kind);
            counts.entry(key.clone()).or_default()[side] += 1;
            let angle = (o.yaw - anchor.yaw + PI).rem_euclid(TAU) - PI;
            headings
                .entry(key)
                .or_default()
                .insert((angle * 10.).floor() as i32);
            records.push(serde_json::json!({"seed":seed,"anchor":anchor,"object":o,"local_position":pos,"relative_yaw":angle,"dominant_side":side}));
        }
    }
    assert_eq!(counts.len(), 6);
    assert_eq!(first.len(), 6, "budget ordering is biased");
    for (kind, sides) in &counts {
        assert!(
            sides.iter().filter(|n| **n > 3).count() >= 3,
            "{kind}: {sides:?}"
        );
        assert!(
            headings[kind].len() > 12,
            "{kind}: {} headings",
            headings[kind].len()
        );
    }
    println!("256 anchor contexts: accepted edges={counts:?}, first categories={first:?}");
    if let Some(dir) = std::env::var_os("ZEROVERSE_AMENITY_REPORT") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            std::path::Path::new(&dir).join("accepted-amenities.json"),
            serde_json::to_vec_pretty(&records).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn amenity_placements_preserve_complete_room_clearance() {
    for kind in [
        IndoorLayout::OpenOffice,
        IndoorLayout::Lounge,
        IndoorLayout::Workshop,
        IndoorLayout::Breakroom,
    ] {
        for seed in 0..16 {
            let scene = IndoorManifest::generate_with_humans(seed, kind, 0.85, 0, 0.25).unwrap();
            validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
        }
    }
}
