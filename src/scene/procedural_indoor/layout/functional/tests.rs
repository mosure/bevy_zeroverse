use super::*;
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
    use crate::scene::procedural_indoor::envelope::polygon;
    let scene = IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    assert!(!scene.rug_supports.is_empty());
    let regions = support_regions(&scene);
    for rug in &scene.rug_supports {
        let p = (rug.min + rug.max) * 0.5;
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
fn rejected_training_elevator_landing_recovers_beside_low_duct() {
    use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};
    for density in [0., 0.65, 1.] {
        let scene =
            IndoorManifest::generate_with_humans(63_069_510, IndoorLayout::Mixed, density, 0, 0.)
                .unwrap();
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
        let replay =
            IndoorManifest::generate_with_humans(63_069_510, IndoorLayout::Mixed, density, 0, 0.)
                .unwrap();
        assert_eq!(scene, replay);
    }
}
