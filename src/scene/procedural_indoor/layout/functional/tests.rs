use super::*;
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
                    assert!(
                        (s.floor_height((t[0] + t[1] + t[2]) / 3.) - r.elevation_metres).abs()
                            < 1e-5,
                        "{profile:?} seed={seed} region={} level={} actual={} triangle={t:?}",
                        r.name,
                        r.elevation_metres,
                        s.floor_height((t[0] + t[1] + t[2]) / 3.)
                    );
                }
            }
        }
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
