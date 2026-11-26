use super::*;
mod capacity;
mod cardio;
mod courtyard;
mod gymnasium;
mod hallway;
mod retail;
mod stair_hall;
mod venues;

#[test]
fn mounted_lights_and_hoods_vary_without_losing_their_hosts() {
    use std::collections::BTreeSet;
    let mut lamps = BTreeSet::new();
    let mut hoods = BTreeSet::new();
    let mut off_center = 0;
    let mut centered = 0;
    let mut saw_invalid = false;
    for profile in [IndoorLayout::Gallery, IndoorLayout::Kitchen] {
        for seed in 0..48 {
            let scene = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.).unwrap();
            scene.validate_program_mounts().unwrap();
            for mount in &scene.assembly_diagnostics.mounts {
                let lamp = &scene.objects[mount.object];
                let host = &scene.objects[mount.host];
                let clearance = lamp.position.y - host.position.y - host.size.y;
                if lamp.kind == ObjectKind::ExtractionHood {
                    use crate::scene::procedural_indoor::objects::{build_object, kitchen};
                    let (plate, footprint) =
                        kitchen::appliances::cooking_surface(host, &kitchen::parameters(host));
                    assert!((0.619..0.891).contains(&(lamp.position.y - plate.y)));
                    assert!(plate.xz().distance(lamp.position.xz()) < 0.001);
                    assert!((0.859..1.121).contains(&(lamp.size.z / footprint.y)));
                    assert!((0.999..1.221).contains(&(lamp.size.x / footprint.x)));
                    let mut nearest = f32::INFINITY;
                    for g in build_object(lamp).parts.values() {
                        for point in &g.positions {
                            let p = lamp.transform().transform_point(Vec3::from_array(*point));
                            let gap = scene.ceiling_height(p.xz()) - p.y;
                            assert!(gap > -0.001, "exhaust above roof at {seed}: {gap}");
                            nearest = nearest.min(gap);
                        }
                    }
                    assert!(nearest < 0.022, "unmounted exhaust at {seed}: {nearest}");
                    hoods.insert((lamp.size.z * 100.).round() as i32);
                } else {
                    assert!(clearance >= 0.949);
                    let p = host
                        .transform()
                        .compute_affine()
                        .inverse()
                        .transform_point3(lamp.position);
                    assert!(
                        p.x.abs() <= host.size.x * 0.29 + 0.001
                            && p.z.abs() <= host.size.z * 0.29 + 0.001
                    );
                    if p.xz().length() < 0.001 {
                        centered += 1
                    } else {
                        off_center += 1
                    }
                    lamps.insert((
                        (lamp.size.x * 100.).round() as i32,
                        (lamp.size.y * 100.).round() as i32,
                    ));
                    if !saw_invalid {
                        let mut invalid = scene.clone();
                        invalid.objects[mount.object].position += Vec3::X * scene.room_size.x;
                        assert!(invalid.validate_program_mounts().is_err());
                        saw_invalid = true;
                    }
                }
            }
        }
    }
    assert!(
        lamps.len() > 20 && centered > 0 && off_center > 10,
        "{lamps:?} {centered} {off_center}"
    );
    assert!(hoods.len() >= 4, "{hoods:?}");
}

#[test]
fn public_suites_realize_diverse_complete_adjacencies() {
    let mut pairs = std::collections::BTreeSet::new();
    for seed in 0..64 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::PublicSuite, 0.65, 0, 0.)
                .unwrap();
        scene.validate_public_programs().unwrap();
        let programs = scene.functional_programs();
        assert_eq!(programs.len(), 2);
        pairs.insert(format!(
            "{:?}/{:?}",
            programs[0].activity, programs[1].activity
        ));
        assert!(!scene.room_topology().open_connections.is_empty());
    }
    assert!(
        pairs.len() >= 20,
        "accepted suite adjacency collapsed: {pairs:?}"
    );
}
#[test]
fn public_programs_have_required_objects_clear_routes_and_replay() {
    for profile in IndoorLayout::PUBLIC_PROFILES
        .into_iter()
        .chain([IndoorLayout::PublicSuite])
        .filter(|p| p.is_public())
    {
        for seed in 0..16 {
            let s = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.).unwrap();
            s.validate_public_programs().unwrap();
            super::super::super::validation::validate_layout(&s).unwrap();
            let replay: IndoorManifest =
                serde_json::from_slice(&serde_json::to_vec(&s).unwrap()).unwrap();
            assert_eq!(s, replay);
        }
    }
}
#[test]
fn functional_receipts_reject_missing_anchors_and_blocked_routes() {
    let mut s =
        IndoorManifest::generate_with_humans(4, IndoorLayout::Gallery, 0.65, 0, 0.).unwrap();
    for o in &mut s.objects {
        if o.kind == ObjectKind::ExhibitPlinth {
            o.kind = ObjectKind::Desk;
        }
    }
    assert!(s.validate_public_programs().is_err());
    let mut s =
        IndoorManifest::generate_with_humans(4, IndoorLayout::Gallery, 0.65, 0, 0.).unwrap();
    let route = s.functional_programs()[0].reservations[0].clone();
    let id = s.functional_programs()[0].members[0];
    let center = (route.min + route.max) * 0.5;
    s.objects[id].position = Vec3::new(center.x, 0., center.y);
    assert!(s.validate_public_programs().is_err());
}
#[test]
fn lecture_supports_match_floor_query_and_keep_headroom() {
    for seed in 0..32 {
        let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::LectureHall, 0.65, 0, 0.)
            .unwrap();
        for p in support_regions(&s).into_iter().skip(1) {
            let center = (p.min + p.max) * 0.5;
            assert_eq!(s.floor_height(center), p.elevation_metres);
            assert!(s.ceiling_height(center) - p.elevation_metres >= 2.2);
        }
    }
}

#[test]
fn support_region_triangles_are_exact_and_level_aware() {
    for profile in [IndoorLayout::LectureHall, IndoorLayout::Mixed] {
        for seed in 0..16 {
            let s = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.).unwrap();
            let regions = support_regions(&s);
            let area: f32 = regions
                .iter()
                .filter(|r| !r.overhead)
                .flat_map(|r| r.triangles.iter())
                .map(|t| (t[1] - t[0]).perp_dot(t[2] - t[0]).abs() * 0.5)
                .sum();
            let envelope = s.envelope.as_ref().unwrap();
            let expected: f32 =
                super::super::super::envelope::polygon::triangles(&envelope.footprint)
                    .iter()
                    .map(|t| (t[1] - t[0]).perp_dot(t[2] - t[0]).abs() * 0.5)
                    .sum();
            assert!((area - expected).abs() < 0.001);
            for r in regions.iter().filter(|r| !r.overhead) {
                for t in &r.triangles {
                    // These regions describe architecture. The live support
                    // query also includes separately exported furniture rugs.
                    assert!(
                        (envelope.floor_height((t[0] + t[1] + t[2]) / 3.) - r.elevation_metres)
                            .abs()
                            < 1e-5,
                        "{profile:?} seed={seed} region={} level={} actual={} triangle={t:?}",
                        r.name,
                        r.elevation_metres,
                        envelope.floor_height((t[0] + t[1] + t[2]) / 3.)
                    );
                }
            }
        }
    }
}

#[test]
fn architectural_support_regions_preserve_the_floor_beneath_rugs() {
    use crate::scene::procedural_indoor::{envelope::polygon, layout::floor::area::RugSupport};
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    // Test the support-layer contract explicitly. Which furniture arrangements
    // leave room for an optional rug is a separate sampling decision.
    scene.rug_supports.clear();
    let regions = support_regions(&scene);
    let (region, triangle) = regions
        .iter()
        .filter(|region| !region.overhead)
        .flat_map(|region| {
            region
                .triangles
                .iter()
                .map(move |triangle| (region, triangle))
        })
        .find(|(region, triangle)| {
            (scene.floor_height((triangle[0] + triangle[1] + triangle[2]) / 3.)
                - region.elevation_metres)
                .abs()
                < 1e-5
        })
        .unwrap();
    scene.rug_supports.push(RugSupport {
        object: scene.objects.len(),
        outline: triangle.to_vec(),
        min: triangle.iter().copied().reduce(Vec2::min).unwrap(),
        max: triangle.iter().copied().reduce(Vec2::max).unwrap(),
        bottom: region.elevation_metres,
        top: region.elevation_metres + 0.007,
    });
    assert_eq!(regions, support_regions(&scene));
    for rug in &scene.rug_supports {
        let p = rug.outline.iter().copied().sum::<Vec2>() / rug.outline.len() as f32;
        assert!(rug.contains(p));
        assert_eq!(scene.floor_height(p), rug.top);
        let region = regions
            .iter()
            .find(|region| {
                !region.overhead && region.triangles.iter().any(|t| polygon::contains(t, p, 0.))
            })
            .unwrap();
        assert!((region.elevation_metres - rug.bottom).abs() < 1e-5);
        assert!(region.elevation_metres < scene.floor_height(p));
    }
}

#[cfg(feature = "human_motion")]
#[test]
fn public_motion_plans_use_level_floor_and_leave_static_people_supported() {
    use crate::human_motion::{planning, HumanMotionConfig};
    for profile in IndoorLayout::PUBLIC_PROFILES {
        let mut scene = IndoorManifest::generate_with_humans(16, profile, 0.65, 0, 0.8).unwrap();
        let config = HumanMotionConfig {
            fraction: 0.75,
            ..Default::default()
        };
        planning::prepare_scene(&mut scene, &config).unwrap();
        let (plans, _) = planning::plan(&scene, &config).unwrap();
        for plan in plans {
            for waypoint in plan.request.waypoints {
                // Current ARDY planner explicitly excludes stairs instead of silently
                // projecting a walking clip through raised audience tiers.
                assert!(scene.floor_height(waypoint.position.xz()).abs() < 0.001);
            }
        }
        super::super::super::humans::validate(&scene).unwrap();
    }
}

#[test]
fn rejected_training_elevator_landings_preserve_headroom_and_door_access() {
    use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};
    for (seed, density, people) in
        [63_069_510, 65_437_985, 69_608_412]
            .into_iter()
            .flat_map(|seed| {
                [0., 0.65, 1.].into_iter().flat_map(move |density| {
                    [0., 0.5, 1.].map(move |people| (seed, density, people))
                })
            })
    {
        // Replay the original Mixed draw, including the downstream RNG state
        // that produced these exact landing envelopes.
        let scene = IndoorManifest::initial_mixed_fixture(seed, density, 0, people).unwrap();
        assert_eq!(scene.layout, IndoorLayout::ElevatorLanding);
        validate_layout(&scene).unwrap();
        validate_geometry(&scene).unwrap();
        let program = &scene.functional_programs()[0];
        let elevators: Vec<_> = program
            .members
            .iter()
            .filter_map(|&id| {
                let o = &scene.objects[id];
                (o.kind == ObjectKind::Elevator).then_some(o)
            })
            .collect();
        assert!(!elevators.is_empty());
        for elevator in elevators {
            assert!(
                crate::scene::procedural_indoor::objects::public::elevator::car_height(elevator)
                    >= 2.50,
                "retain full-height car geometry"
            );
            assert!(elevator.size.x >= 2.05 && elevator.size.z >= 1.50);
            assert!(scene.placement_clear(elevator, 0.035));
            let front = elevator
                .transform()
                .transform_point(Vec3::new(0., 0., elevator.size.z * 0.5 + 0.5))
                .xz();
            assert!(program.reservations.iter().any(|route| {
                route.role == ReservationRole::DoorApproach
                    && front.cmpge(route.min).all()
                    && front.cmple(route.max).all()
            }));
        }
        assert!(
            program.rejected_candidates < 20_000,
            "bounded recovery budget"
        );
        let replay = IndoorManifest::initial_mixed_fixture(seed, density, 0, people).unwrap();
        assert_eq!(scene, replay);
    }
}

#[test]
fn rejected_cafe_cases_fit_narrow_bays_without_shrinking_products_or_aisles() {
    use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};
    for seed in [69_430_992, 69_531_153] {
        for density in [0., 0.65, 1.] {
            // Replay the catalog that produced the reported café, including its first RNG draw.
            let scene = IndoorManifest::initial_mixed_fixture(seed, density, 0, 0.5).unwrap();
            validate_layout(&scene).unwrap();
            validate_geometry(&scene).unwrap();
            let program = scene
                .functional_programs()
                .iter()
                .find(|p| p.activity == IndoorLayout::Cafe)
                .unwrap();
            let case = program
                .members
                .iter()
                .map(|&id| &scene.objects[id])
                .find(|o| o.kind == ObjectKind::DisplayCase)
                .unwrap();
            assert!((0.80..1.95).contains(&case.size.x));
            assert!((0.88..1.92).contains(&case.size.y));
            assert!((0.48..0.84).contains(&case.size.z));
            assert!(scene.placement_clear(case, 0.035));
            let aisle = &program.reservations[0];
            assert!(((aisle.max.x - aisle.min.x) - program.parameters.aisle_metres).abs() < 1e-5);
            assert!(program.rejected_candidates < 2_000);
        }
    }
}

#[test]
fn archived_truss_and_duct_landing_recovers_without_removing_ceiling_geometry() {
    use crate::scene::procedural_indoor::{
        architecture::ceiling::CeilingSystem,
        validation::{validate_geometry, validate_layout_for_generator},
    };
    // Retain the original generator-70 architecture independently of later
    // ceiling samplers. Only furnishings and their receipts were removed from
    // the rejected manifest, so this still exercises the narrow recovery bay.
    let mut scene: IndoorManifest =
        serde_json::from_str(include_str!("fixtures/elevator_truss_65437985.json")).unwrap();
    assert_eq!(scene.generator_version, 70);
    let ceiling = scene.ceiling_installation.clone().unwrap();
    for system in [CeilingSystem::Truss, CeilingSystem::RectangularDuct] {
        assert!(ceiling.modules.iter().any(|m| m.system == system));
    }
    assert!(scene.objects.is_empty());
    scene.furnish_public_programs();
    scene.validate_public_programs().unwrap();
    validate_layout_for_generator(&scene, 70).unwrap();
    validate_geometry(&scene).unwrap();
    assert_eq!(scene.ceiling_installation.as_ref().unwrap(), &ceiling);
    let elevators: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::Elevator)
        .collect();
    assert!(!elevators.is_empty());
    for elevator in elevators {
        assert!(scene.placement_clear(elevator, 0.035));
        assert!(elevator.size.x >= 2.05 && elevator.size.z >= 1.50);
        assert!(
            crate::scene::procedural_indoor::objects::public::elevator::car_height(elevator)
                >= 2.50
        );
    }
    assert!(scene.functional_programs()[0].rejected_candidates < 20_000);
}

#[test]
fn archived_landing_moves_aisle_instead_of_removing_roof_structure() {
    use crate::scene::procedural_indoor::validation::{
        validate_geometry, validate_layout_for_generator,
    };
    let mut scene: IndoorManifest =
        serde_json::from_str(include_str!("fixtures/elevator_circulation_69608412.json")).unwrap();
    let envelope = scene.envelope.clone();
    let ceiling = scene.ceiling_installation.clone();
    scene.furnish_public_programs();
    validate_layout_for_generator(&scene, 112).unwrap();
    validate_geometry(&scene).unwrap();
    assert_eq!(scene.envelope, envelope);
    assert_eq!(scene.ceiling_installation, ceiling);
    let p = &scene.functional_programs()[0];
    let aisle = &p.reservations[0];
    assert!(((aisle.max.x - aisle.min.x) - p.parameters.aisle_metres).abs() < 1e-5);
    assert!(((aisle.min.x + aisle.max.x) - (p.min.x + p.max.x)).abs() > 0.1);
    let floor = scene.floor_height((aisle.min + aisle.max) * 0.5);
    assert!(scene
        .structural_obstacles_with_ceiling(true)
        .iter()
        .all(|&(lo, hi)| lo.y >= floor + 2.0 || hi.y <= floor + 0.02 || !aisle.overlaps(lo, hi)));
    assert!(p.rejected_candidates < 20_000);
    let elevator = scene
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::Elevator)
        .unwrap();
    assert!(elevator.size.x >= 2.05 && elevator.size.z >= 1.50);
    assert!(
        crate::scene::procedural_indoor::objects::public::elevator::car_height(elevator) >= 2.50
    );
}
