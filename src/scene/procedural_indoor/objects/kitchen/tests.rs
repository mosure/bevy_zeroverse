use super::*;

#[test]
fn island_mesh_has_a_raised_continuous_worktop_and_freestanding_base() {
    for seed in 0..64 {
        let mut o = super::super::audit::fixture(ObjectKind::KitchenCounter, seed);
        configure_island(&mut o);
        for height in [0.86, 0.90, 0.94] {
            align_worktop(&mut o, height);
            let p = island_parameters(&o).unwrap();
            let a = super::super::build_object(&o);
            let (lo, hi) = a.bounds();
            assert!(lo.y.abs() < 0.001 && (hi.y - height).abs() < 0.001);
            let expected_lo = Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5);
            let expected_hi = Vec3::new(o.size.x * 0.5, height, o.size.z * 0.5);
            assert!((expected_lo - lo).max(hi - expected_hi).max_element() < 0.003);
            for u in [-0.47, 0., 0.47] {
                for v in [-0.47, 0., 0.47] {
                    let hit = top_at(&a, Vec2::new(o.size.x * u, o.size.z * v));
                    assert!(
                        (hit - height).abs() < 0.001,
                        "seed {seed}: slab has a hole at {u}/{v}"
                    );
                }
            }
            let cabinet_positions: Vec<_> = a
                .parts
                .iter()
                .filter(|((_, label), _)| super::super::part_label(label) == "cabinet")
                .flat_map(|(_, g)| g.positions.iter().map(|p| Vec3::from_array(*p)))
                .collect();
            assert!(cabinet_positions.iter().any(|p| p.y < 0.002));
            assert!(cabinet_positions
                .iter()
                .any(|v| v.y > height - p.slab_thickness - 0.03));
            // End panels stop inside the slab, giving a visible horizontal lip.
            assert!(cabinet_positions
                .iter()
                .all(|v| v.x.abs() <= o.size.x * 0.5 - p.end_overhang + 0.001));
            assert!(a.parts.values().map(|g| g.indices.len() / 3).sum::<usize>() < 10_000);
        }
    }
}

#[test]
fn island_construction_varies_bases_bay_proportions_and_fronts() {
    let mut bases = std::collections::BTreeSet::new();
    let mut fronts = std::collections::BTreeSet::new();
    for seed in 0..128 {
        let mut o = super::super::audit::fixture(ObjectKind::KitchenCounter, seed);
        configure_island(&mut o);
        let p = island_parameters(&o).unwrap();
        bases.insert((p.raised_base, p.waterfall_ends));
        fronts.insert(
            (0..p.bays)
                .map(|i| island::is_drawer_bank(&o, i))
                .collect::<Vec<_>>(),
        );
        let sum = p.bay_weights.iter().sum::<f32>();
        let width = o.size.x - 2. * p.end_overhang;
        assert!(p.bay_weights.iter().all(|v| width * v / sum > 0.23));
        assert!(p
            .bay_weights
            .windows(2)
            .any(|v| (v[0] - v[1]).abs() > 0.001));
        let a = super::super::build_object(&o);
        super::super::audit::check(&o, &a).unwrap();
    }
    assert_eq!(bases.len(), 4);
    assert!(
        fronts.len() >= 15,
        "cabinet banks collapse into repeated alternating fronts"
    );
}

#[test]
fn island_admission_rejects_lowered_floor_and_steps_in_the_walking_apron() {
    use crate::scene::procedural_indoor::{
        envelope::FloorPatch,
        layout::{IndoorLayout, IndoorManifest},
    };
    let mut scene =
        IndoorManifest::generate_with_humans(27, IndoorLayout::Kitchenette, 0.65, 0, 0.).unwrap();
    let mut island = super::super::audit::fixture(ObjectKind::KitchenCounter, 1);
    configure_island(&mut island);
    let e = scene.envelope.as_mut().unwrap();
    e.floor_patches.clear();
    assert!(scene.island_floor_clear(&island, 0.));
    scene
        .envelope
        .as_mut()
        .unwrap()
        .floor_patches
        .push(FloorPatch {
            min: Vec2::splat(-3.),
            max: Vec2::splat(3.),
            height: -0.30,
        });
    island.position.y = -0.30;
    assert!(
        !scene.island_floor_clear(&island, 0.),
        "sunken island admitted"
    );
    island.position.y = 0.;
    scene.envelope.as_mut().unwrap().floor_patches[0] = FloorPatch {
        min: Vec2::new(island.size.x * 0.5 + 0.40, -0.1),
        max: Vec2::new(island.size.x * 0.5 + 0.42, 0.1),
        height: 0.15,
    };
    assert!(
        !scene.island_floor_clear(&island, 0.),
        "narrow step in apron admitted"
    );
}

#[test]
fn glazed_appliance_doors_keep_their_owner_semantic_class() {
    for kind in [
        ObjectKind::KitchenCupboard,
        ObjectKind::Pantry,
        ObjectKind::Range,
        ObjectKind::Oven,
        ObjectKind::Microwave,
        ObjectKind::Dishwasher,
        ObjectKind::Refrigerator,
    ] {
        let mut checked = 0;
        for seed in 0..128 {
            let o = super::super::audit::fixture(kind, seed);
            let a = super::super::build_object(&o);
            for (surface, label) in a.parts.into_keys() {
                if surface == Surface::GlassInterior {
                    assert_eq!(super::super::part_label(&label), kind.class_name());
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "{kind:?} never generated appliance glazing");
    }
}

#[test]
fn drying_towels_use_upper_bar_handles_with_clear_supported_folds() {
    let mut checked = 0;
    for seed in 0..128 {
        let o = super::super::audit::fixture(ObjectKind::KitchenCounter, seed);
        let p = parameters(&o);
        let a = super::super::build_object(&o);
        let towels: Vec<_> = a
            .parts
            .iter()
            .filter(|((_, label), _)| super::super::part_label(label) == "towel")
            .collect();
        assert_eq!(
            !towels.is_empty(),
            !p.counter_doors && matches!(p.handles, 0 | 2) && seed.is_multiple_of(3)
        );
        for (_, g) in towels {
            let (d, z) = body(&o, &p);
            let lo = g
                .positions
                .iter()
                .map(|v| Vec3::from_array(*v))
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let hi = g
                .positions
                .iter()
                .map(|v| Vec3::from_array(*v))
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            assert!(lo.y > o.size.y * 0.35 && hi.y > o.size.y * 0.70);
            assert!(lo.z > z + d * 0.5 + 0.026, "cloth intersects drawer front");
            assert!(hi.x.abs().max(lo.x.abs()) < if p.handles == 2 { 0.033 } else { 0.055 });
            checked += 1;
        }
    }
    assert!(checked > 5);
}

#[test]
fn kitchen_round_tubes_have_linear_mesh_cost_and_closed_seams() {
    let points: Vec<_> = (0..=32)
        .map(|i| {
            let t = TAU * (i % 32) as f32 / 32.;
            Vec3::new(t.cos() * 0.1, 0.1, t.sin() * 0.1)
        })
        .collect();
    let mut a = Assembly::default();
    pipe(&mut a, Surface::Metal, "stove", &points, 0.0012);
    let g = a.parts.values().next().unwrap();
    assert_eq!(g.indices.len() / 3, 32 * 24 * 2);
    assert_eq!(&g.positions[..25], &g.positions[32 * 25..33 * 25]);
}

// Intersect the actual triangle mesh with a downward ray, independent of the
// analytic placement envelope. This detects a cabinet top sealing a sink hole.
pub(super) fn top_at(a: &Assembly, point: Vec2) -> f32 {
    let mut top = f32::NEG_INFINITY;
    for g in a.parts.values() {
        for triangle in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = triangle.map(|id| Vec3::from_array(g.positions[id as usize]));
            let ab = b.xz() - a.xz();
            let ac = c.xz() - a.xz();
            let ap = point - a.xz();
            let det = ab.perp_dot(ac);
            if det.abs() < 1e-10 {
                continue;
            }
            let u = ap.perp_dot(ac) / det;
            let v = ab.perp_dot(ap) / det;
            if u >= -1e-6 && v >= -1e-6 && u + v <= 1. + 1e-6 {
                top = top.max(a.y + u * (b.y - a.y) + v * (c.y - a.y));
            }
        }
    }
    top
}

#[test]
fn bread_crust_markings_follow_the_actual_loaf_surface() {
    for seed in 0..64 {
        let o = super::super::audit::fixture(ObjectKind::Bread, seed);
        let mut a = super::super::build_object(&o);
        let key = a
            .parts
            .keys()
            .find(|(_, label)| label == "food#finish104")
            .unwrap()
            .clone();
        let scoring = a.parts.remove(&key).unwrap();
        assert!(!scoring.indices.is_empty());
        for point in &scoring.positions {
            let point = Vec3::from_array(*point);
            let gap = point.y - top_at(&a, point.xz());
            assert!(
                (0.0001..0.001).contains(&gap),
                "seed {seed}: crust gap {gap}"
            );
        }
    }
}

#[test]
fn sink_cutout_reaches_the_basin_in_the_actual_mesh() {
    let mut checked = 0;
    for seed in 0..128 {
        let o = super::super::audit::fixture(ObjectKind::KitchenSink, seed);
        let p = parameters(&o);
        if p.sink_dishes != 0 {
            continue;
        }
        let width = o.size.x * if p.draining_board { 0.59 } else { 0.78 };
        let cx = if p.draining_board {
            -o.size.x * 0.105
        } else {
            0.
        };
        let unit = width / p.basins as f32;
        let (d, z) = body(&o, &p);
        let mesh = super::super::build_object(&o);
        for basin in 0..p.basins {
            let point = Vec2::new(
                cx - width * 0.5 + unit * (basin as f32 + 0.65),
                z + d * 0.08,
            );
            let floor = o.size.y * (0.69 - 0.16);
            let hit = top_at(&mesh, point);
            assert!(
                (hit - (floor + 0.007)).abs() < 0.002,
                "seed {seed} basin {basin}: ray hit {hit}, expected floor {floor}"
            );
            checked += 1;
        }
    }
    assert!(checked > 30);
}

#[test]
fn aligned_worktops_preserve_body_depth_and_door_envelopes() {
    for kind in [
        ObjectKind::KitchenCounter,
        ObjectKind::KitchenSink,
        ObjectKind::KitchenCupboard,
        ObjectKind::Dishwasher,
        ObjectKind::Range,
        ObjectKind::Oven,
    ] {
        for seed in 0..64 {
            for height in [0.86, 0.94] {
                let mut o = super::super::audit::fixture(kind, seed);
                let depth = body(&o, &parameters(&o)).0;
                align_worktop(&mut o, height);
                assert!((depth - body(&o, &parameters(&o)).0).abs() < 1e-5);
                let (lo, hi) = super::super::build_object(&o).bounds();
                let expected_lo = Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5);
                let expected_hi = Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5);
                assert!(
                    (expected_lo - lo).max(hi - expected_hi).max_element() < 0.003,
                    "{kind:?} {seed} height {height}"
                );
            }
        }
    }
}

#[test]
fn range_burners_are_above_the_carcass_in_the_actual_mesh() {
    for seed in 0..64 {
        let o = super::super::audit::fixture(ObjectKind::Range, seed);
        let (depth, z) = body(&o, &parameters(&o));
        let point = Vec2::new(-(o.size.x - 0.018) * 0.22, z - depth * 0.22);
        let hit = top_at(&super::super::build_object(&o), point) / o.size.y;
        assert!(
            (0.945..0.98).contains(&hit),
            "seed {seed}: burner hidden by casing at {hit}"
        );
    }
}
#[test]
fn kitchen_objects_have_valid_semantics_and_fit_expanded_envelopes() {
    let mut failures = std::collections::BTreeMap::new();
    for kind in ObjectKind::ALL.into_iter().filter(|k| k.is_kitchen()) {
        for seed in 0..64 {
            let o = super::super::audit::fixture(kind, seed);
            let a = super::super::build_object(&o);
            let (lo, hi) = a.bounds();
            assert!(lo.is_finite() && hi.is_finite(), "{kind:?} {seed}");
            let expected_lo = Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5);
            let expected_hi = Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5);
            let over = (expected_lo - lo).max(hi - expected_hi).max_element();
            if over > 0.003 || lo.y > 0.003 {
                failures.entry(format!("{kind:?}")).or_insert_with(|| {
                    format!("seed {seed}: {lo:?}..{hi:?} vs {:?}, over {over}", o.size)
                });
            }
            for ((_, label), g) in &a.parts {
                assert!(
                    SemanticLabel::from_label(super::super::part_label(label)).is_some(),
                    "{label}"
                );
                assert!(g.positions.len() == g.normals.len() && g.positions.len() == g.uvs.len());
                assert!(g.normals.iter().all(|n| Vec3::from_array(*n).is_finite()));
                assert!(g.indices.iter().all(|&i| (i as usize) < g.positions.len()));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn kitchen_state_distributions_are_seeded_and_cover_conditional_features() {
    use std::collections::BTreeSet;
    let mut door = BTreeSet::new();
    let mut faucets = BTreeSet::new();
    let mut knobs = BTreeSet::new();
    let mut fuel = BTreeSet::new();
    let mut controls = BTreeSet::new();
    let mut soap = BTreeSet::new();
    let mut glasses = BTreeSet::new();
    let mut double = BTreeSet::new();
    let mut props = BTreeSet::new();
    for seed in 0..512 {
        let mut o = super::super::audit::fixture(ObjectKind::KitchenSink, seed);
        let p = parameters(&o);
        faucets.insert(p.faucet);
        knobs.insert(p.faucet_handles);
        soap.insert((p.hand_soap, p.dish_soap, p.sponge, p.sink_dishes > 0));
        o = super::super::audit::fixture(ObjectKind::Refrigerator, seed);
        let p = parameters(&o);
        door.insert(p.door_style);
        glasses.insert(p.viewport);
        let p = parameters(&super::super::audit::fixture(ObjectKind::Range, seed));
        fuel.insert(p.gas);
        controls.insert(p.controls);
        let p = parameters(&super::super::audit::fixture(ObjectKind::Oven, seed));
        double.insert(p.stacked_ovens);
        props.insert(
            serde_json::to_string(&metadata(&super::super::audit::fixture(
                ObjectKind::PaperTowel,
                seed,
            )))
            .unwrap(),
        );
    }
    assert_eq!(
        (
            door.len(),
            faucets.len(),
            knobs.len(),
            fuel.len(),
            controls.len(),
            glasses.len(),
            double.len()
        ),
        (4, 4, 3, 2, 3, 2, 2)
    );
    assert_eq!(soap.len(), 16);
    assert_eq!(props.len(), 2);
}
#[test]
fn kitchen_rooms_preserve_layout_and_actual_geometry_contracts() {
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation::{validate_geometry, validate_layout},
    };
    let mut failures = Vec::new();
    for layout in [IndoorLayout::Kitchenette, IndoorLayout::Kitchen] {
        for seed in 0..64 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.).unwrap();
            if let Err(e) =
                validate_layout(&scene).and_then(|_| validate_geometry(&scene).map(|_| ()))
            {
                failures.push(format!("{layout:?}: {e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures: {:#?}",
        failures.len(),
        &failures[..failures.len().min(12)]
    );
}

#[test]
fn kitchen_runs_fit_constrained_envelopes_with_people_and_cameras() {
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation::{validate_geometry, validate_layout},
    };
    for seed in [68, 254, 312, 365, 473] {
        for layout in [IndoorLayout::Kitchen, IndoorLayout::Kitchenette] {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 3, 0.25).unwrap();
            validate_layout(&scene).unwrap();
            validate_geometry(&scene).unwrap();
            assert_eq!(scene.cameras.len(), 3);
        }
    }
}

#[test]
fn kitchen_accepted_placements_cover_every_new_kind_and_replay() {
    use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
    let mut counts = std::collections::BTreeMap::new();
    let mut configurations = std::collections::BTreeMap::new();
    for seed in 0..256 {
        let layout = if seed % 2 == 0 {
            IndoorLayout::Kitchen
        } else {
            IndoorLayout::Kitchenette
        };
        let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.).unwrap();
        scene.validate_kitchens().unwrap();
        assert!(
            matches!(scene.floor_style, 0 | 2),
            "wet area sampled carpet"
        );
        for group in &scene.assembly_diagnostics.kitchens {
            *configurations
                .entry(group.configuration.clone())
                .or_insert(0) += 1;
            let floor = scene.objects[group.units[0]].position.y;
            for o in group
                .units
                .iter()
                .map(|&id| &scene.objects[id])
                .filter(|o| is_island(o))
            {
                assert!((o.position.y - floor).abs() < 0.001);
                assert!(scene.island_floor_clear(o, floor));
                assert!((o.size.y - group.worktop_height_metres).abs() < 0.001);
                assert_eq!(metadata(o)["installation"], "island");
            }
        }
        if seed < 8 {
            assert_eq!(
                scene,
                IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.).unwrap()
            );
        }
        for o in &scene.objects {
            *counts.entry(format!("{:?}", o.kind)).or_insert(0) += 1;
        }
    }
    for kind in ObjectKind::ALL.into_iter().filter(|k| k.is_kitchen()) {
        assert!(
            counts.get(&format!("{kind:?}")).copied().unwrap_or(0) >= 4,
            "under-sampled accepted {kind:?}: {counts:?}"
        );
    }
    for configuration in ["straight", "island", "parallel", "perpendicular_station"] {
        assert!(
            configurations.get(configuration).copied().unwrap_or(0) >= 2,
            "unreachable composition {configuration}: {configurations:?}"
        );
    }
}
