use super::*;
use crate::scene::procedural_indoor::{
    appearance::AppearanceSettings, layout::IndoorLayout, validation,
};

fn cafeteria() -> IndoorManifest {
    IndoorManifest::generate_with_humans(50800, IndoorLayout::Cafeteria, 0.65, 0, 0.25).unwrap()
}

fn controls(scene: &mut IndoorManifest) -> &mut Photometry {
    &mut scene
        .program
        .as_mut()
        .unwrap()
        .domain
        .as_mut()
        .unwrap()
        .photometry
}

#[test]
fn whole_circuits_switch_together_and_the_adjoining_bay_is_independent() {
    let mut scene = cafeteria();
    let mounts = architecture::fixture_positions(&scene);
    let primary = architecture::primary_fixture_count(&scene, &mounts);
    let wiring = scene
        .domain()
        .unwrap()
        .photometry
        .ceiling_circuits
        .as_ref()
        .unwrap();
    let primary_banks: std::collections::BTreeSet<_> =
        wiring.mount_circuits[..primary].iter().collect();
    assert!(primary_banks.len() < primary && primary_banks.len() >= 2);
    assert!(!wiring.mount_circuits[primary..].is_empty());
    assert!(wiring.mount_circuits[primary..]
        .iter()
        .all(|b| !primary_banks.contains(b)));
    for (i, _) in mounts.iter().enumerate() {
        let (_, flux) = architecture::fixture_photometry_at(&scene, i, &mounts);
        assert_eq!(flux > 0., wiring.for_mount(i).unwrap().dimmer > 0.);
    }
    for (fraction, expected) in [(0., false), (1., true)] {
        controls(&mut scene).active_fraction = fraction;
        let plan = CeilingCircuits::sample(&scene, &mounts, &[], 0.);
        assert!(plan.circuits.iter().all(|b| (b.dimmer > 0.) == expected));
    }
}

#[test]
fn partitions_and_mezzanine_levels_do_not_share_switches_across_the_wall_or_deck() {
    let scene = IndoorManifest::initial_mixed_fixture(72917, 0.65, 0, 0.25).unwrap();
    let deck = &scene
        .envelope
        .as_ref()
        .unwrap()
        .mezzanine
        .as_ref()
        .unwrap()
        .deck;
    let center = (deck.min + deck.max) * 0.5;
    let below = Vec3::new(center.x, deck.height - 0.3, center.y);
    let mounts = [below, below.with_y(scene.ceiling_height(center) - 0.12)];
    let plan = CeilingCircuits::sample(&scene, &mounts, &[], 0.);
    assert_ne!(plan.mount_circuits[0], plan.mount_circuits[1]);

    let mut scene = cafeteria();
    let zones = &mut scene.program.as_mut().unwrap().zones;
    let mut right = zones[0].clone();
    zones[0].max.x = 0.;
    right.min.x = 0.;
    zones.push(right);
    let mounts = [Vec3::new(-0.12, 3., 0.), Vec3::new(0.12, 3., 0.)];
    let plan = CeilingCircuits::sample(&scene, &mounts, &[], 0.);
    assert_ne!(plan.mount_circuits[0], plan.mount_circuits[1]);
}

#[test]
fn occupied_dining_tables_can_call_for_a_second_bank_without_forcing_it_in_daylight() {
    let mut scene = cafeteria();
    let mounts = architecture::fixture_positions(&scene);
    let primary = architecture::primary_fixture_count(&scene, &mounts);
    let live = |scene: &IndoorManifest, daylight: f32| {
        let plan = CeilingCircuits::sample(scene, &mounts, &[], daylight);
        (0..primary)
            .filter(|&i| plan.for_mount(i).unwrap().dimmer > 0.)
            .count()
    };
    let partial = live(&scene, 1000.);
    assert!(partial > 0 && partial < primary);
    assert_eq!(live(&scene, 0.), primary);
    scene.target_lux = 8.;
    assert_eq!(live(&scene, 0.), partial);
    scene.target_lux = 231.;
    scene.objects.clear();
    assert_eq!(live(&scene, 0.), partial);
}

#[test]
fn daylight_dimming_follows_glazing_and_stops_at_room_boundaries() {
    let mut scene = cafeteria();
    let zone = scene.program.as_ref().unwrap().zones[0].clone();
    let mut left = zone.clone();
    let mut right = zone;
    left.max.x = 0.;
    right.min.x = 0.;
    let zones = [left, right];
    let opening = DaylightOpening {
        a: Vec2::new(-6., -3.),
        b: Vec2::new(-6., 3.),
        reach_m: 4.,
        admitted_lux: 700.,
    };
    assert!(
        opening.signal(Vec2::new(-5., 0.), 0, &zones)
            > 2. * opening.signal(Vec2::new(-1., 5.), 0, &zones)
    );
    assert_eq!(opening.signal(Vec2::new(1., 0.), 1, &zones), 0.);
    let mounts = architecture::fixture_positions(&scene);
    let mut responsive = 0;
    let mut manual = 0;
    controls(&mut scene).active_fraction = 1.;
    for seed in 0..48 {
        scene.appearance = Some(AppearanceSettings {
            lighting_seed: Some(seed),
            ..default()
        });
        let dark = CeilingCircuits::sample(&scene, &mounts, &[], 0.);
        let bright = CeilingCircuits::sample(&scene, &mounts, std::slice::from_ref(&opening), 0.);
        assert_eq!(dark.mount_circuits, bright.mount_circuits);
        assert!(bright
            .circuits
            .iter()
            .zip(&dark.circuits)
            .all(|(b, d)| b.dimmer <= d.dimmer));
        if dark == bright {
            manual += 1;
        } else {
            responsive += 1;
        }
    }
    assert!(
        manual >= 8 && responsive >= 12,
        "{manual} manual, {responsive} responsive"
    );
}

#[test]
fn recorded_wiring_replays_and_legacy_cafeteria_preserves_its_photometry() {
    let mut scene = cafeteria();
    let mounts = architecture::fixture_positions(&scene);
    let outputs = |s: &IndoorManifest| {
        (0..mounts.len())
            .map(|i| architecture::fixture_photometry_at(s, i, &mounts))
            .collect::<Vec<_>>()
    };
    let replay = serde_json::from_slice(&serde_json::to_vec(&scene).unwrap()).unwrap();
    assert_eq!(outputs(&scene), outputs(&replay));
    assert_eq!(scene.ev100(), replay.ev100());
    let original = scene
        .domain()
        .unwrap()
        .photometry
        .ceiling_circuits
        .clone()
        .unwrap();
    scene
        .apply_appearance(AppearanceSettings {
            lighting_seed: Some(437),
            ..default()
        })
        .unwrap();
    let reseeded = scene
        .domain()
        .unwrap()
        .photometry
        .ceiling_circuits
        .as_ref()
        .unwrap();
    assert_eq!(original.mount_circuits, reseeded.mount_circuits);
    assert_ne!(original.circuits, reseeded.circuits);
    let mut legacy = serde_json::to_value(&replay).unwrap();
    legacy["program"]["domain"]["photometry"]
        .as_object_mut()
        .unwrap()
        .remove("ceiling_circuits");
    legacy["generator_version"] = 165.into();
    let mut legacy: IndoorManifest = serde_json::from_value(legacy).unwrap();
    legacy.meter_indoor_lighting();
    let p = &legacy.domain().unwrap().photometry;
    assert!(p.ceiling_circuits.is_none());
    // Independently recorded generator-165 manifest, seed 50800.
    assert!((p.meter.unwrap().electric_lux - 49.562626).abs() < 1e-4);
}

#[test]
fn malformed_wiring_is_rejected_before_capture() {
    let mut scene = cafeteria();
    controls(&mut scene)
        .ceiling_circuits
        .as_mut()
        .unwrap()
        .mount_circuits[0] = usize::MAX;
    assert!(validation::validate_layout(&scene)
        .unwrap_err()
        .contains("ceiling circuit"));
    let mut scene = cafeteria();
    controls(&mut scene)
        .ceiling_circuits
        .as_mut()
        .unwrap()
        .mount_circuits
        .pop();
    assert!(validation::validate_layout(&scene)
        .unwrap_err()
        .contains("installed mounts"));
}
