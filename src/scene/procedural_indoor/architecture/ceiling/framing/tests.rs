use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use std::collections::BTreeSet;

pub(super) fn check(frame: &CeilingFraming) -> usize {
    frame.validate_integrity().unwrap();
    let bounds: Vec<_> = frame.bounds().collect();
    let a = frame.geometry();
    let mut triangles = 0;
    for ((_, label), g) in &a.parts {
        assert_eq!(label.split('#').next(), Some("ceiling"));
        for p in &g.positions {
            let p = Vec3::from(*p);
            assert!(
                p.is_finite()
                    && bounds
                        .iter()
                        .any(|(lo, hi)| p.cmpge(*lo - Vec3::splat(2e-5)).all()
                            && p.cmple(*hi + Vec3::splat(2e-5)).all()),
                "unreserved framing vertex {p:?}"
            );
        }
        assert!(g.uvs.iter().flatten().all(|v| v.is_finite()));
        assert!(g
            .normals
            .iter()
            .all(|n| (Vec3::from(*n).length() - 1.).abs() < 0.005));
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
            let normal = (b - a).cross(c - a);
            assert!(normal.length_squared() > 1e-16);
            let average: Vec3 = tri.iter().map(|&i| Vec3::from(g.normals[i as usize])).sum();
            assert!(normal.dot(average) > -1e-6, "inverted frame triangle");
        }
        triangles += g.indices.len() / 3;
    }
    assert!(triangles < 120_000);
    triangles
}

#[test]
fn roof_frames_replay_have_structural_depth_and_bound_actual_meshes() {
    let mut roles = BTreeSet::new();
    let mut sections = BTreeSet::new();
    let mut admitted = 0;
    let mut runs = 0;
    for seed in 0..96 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        validation::validate_layout(&scene).unwrap();
        let plan = scene.ceiling_installation.as_ref().unwrap();
        if let Some(frame) = &plan.framing {
            check(frame);
            assert!(
                frame.connected().iter().all(|&c| c),
                "floating frame seed {seed}"
            );
            if !frame.members.is_empty() {
                admitted += 1;
            }
            for m in &frame.members {
                roles.insert(format!("{:?}", m.role));
                sections.insert(format!("{:?}", m.section));
            }
            if seed < 24 {
                validation::validate_geometry(&scene).unwrap();
            }
        }
        for module in &plan.modules {
            assert!(module.suspension.is_some());
            module.validate_suspension().unwrap();
            for support in module.suspension.iter().flatten() {
                let actual_surface =
                    plan.attachment_height(support.start, scene.ceiling_height(support.start.xz()));
                assert!((support.end.y + 0.003 - actual_surface).abs() < 1e-5);
            }
            let boxes: Vec<_> = module.obstacles().collect();
            for g in module.geometry(&scene).parts.values() {
                for p in &g.positions {
                    let p = Vec3::from(*p);
                    assert!(
                        boxes
                            .iter()
                            .any(|(lo, hi)| p.cmpge(*lo - Vec3::splat(0.003)).all()
                                && p.cmple(*hi + Vec3::splat(0.003)).all()),
                        "unreserved ceiling mesh seed {seed}: {p:?}"
                    );
                }
            }
            runs += 1;
        }
        if seed < 8 {
            let replay =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            assert_eq!(scene.ceiling_installation, replay.ceiling_installation);
            assert_eq!(
                *plan,
                serde_json::from_str::<CeilingInstallation>(&serde_json::to_string(plan).unwrap())
                    .unwrap()
            );
        }
    }
    assert!(admitted > 12, "few structured ceilings: {admitted}");
    assert_eq!(roles.len(), 4);
    assert_eq!(sections.len(), 4);
    assert!(runs > 160);
}

fn fixture(section: FrameSection) -> FrameMember {
    let corners = [
        Vec2::new(-2., -0.15),
        Vec2::new(2., -0.15),
        Vec2::new(2., 0.15),
        Vec2::new(-2., 0.15),
    ];
    FrameMember {
        top: corners.map(|p| Vec3::new(p.x, 4. + p.x * 0.07 + p.y * 0.04, p.y)),
        depth: 0.5,
        role: FrameRole::Rafter,
        section,
        material: Surface::Metal,
        finish: 0,
        web_pitch: 1.,
    }
}

#[test]
fn suspension_hits_real_profiles_instead_of_empty_truss_volume() {
    for section in [
        FrameSection::Solid,
        FrameSection::ISection,
        FrameSection::Fluted,
        FrameSection::OpenWeb,
    ] {
        let m = fixture(section);
        for i in 0..20 {
            let p = m.point(0.37, 0.025 + i as f32 * 0.05, 4.);
            let expected = super::geometry::attachment(&m, p);
            let actual = m.attachment_y(p);
            assert_eq!(actual.is_some(), expected.is_some(), "{section:?}: {p:?}");
            if let (Some(a), Some(b)) = (actual, expected) {
                assert!((a - b).abs() < 0.0001, "{section:?}: {a} != {b}");
            }
        }
    }
    let m = fixture(FrameSection::OpenWeb);
    assert!(m.attachment_y(m.point(0.37, 0.5, 4.)).is_none());
    assert!(m.attachment_y(m.point(0.37, 0.22, 4.)).is_some());
    // The vertical end tubes project beyond the quad's ends. A hanger in
    // that strip must stop at the tube, not pass through it to the roof.
    for t in [-0.002, 1.002] {
        let p = m.point(t, 0.22, 4.);
        let expected = super::geometry::attachment(&m, p).unwrap();
        assert!((m.attachment_y(p).unwrap() - expected).abs() < 1e-5);
    }
    let mut thin = fixture(FrameSection::ISection);
    thin.depth = 0.012;
    check(&CeilingFraming {
        roof_material: Surface::Metal,
        members: vec![thin],
    });
}

#[test]
fn crossing_services_reserve_actual_hangers_and_keep_legacy_solid_reservations() {
    let scene = IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut lower = scene.ceiling_installation.unwrap().modules[0].clone();
    lower.system = CeilingSystem::Conduit;
    lower.center = Vec3::new(0., 2.6, 0.);
    lower.size = Vec3::new(4., 0.2, 0.3);
    lower.support_top_m = 4.;
    lower.suspension = Some(
        [-1.6, 1.6]
            .map(|x| Suspension {
                start: Vec3::new(x, 2.69, 0.1),
                end: Vec3::new(x, 3.99, 0.1),
            })
            .into(),
    );
    let mut upper = lower.clone();
    upper.center.y = 3.2;
    upper.size = Vec3::new(0.3, 0.2, 4.);
    upper.suspension = Some(
        [-1.6, 1.6]
            .map(|z| Suspension {
                start: Vec3::new(0.1, 3.29, z),
                end: Vec3::new(0.1, 3.99, z),
            })
            .into(),
    );
    lower.validate_suspension().unwrap();
    upper.validate_suspension().unwrap();
    assert!(overlap(lower.bounds(), upper.bounds(), 0.));
    assert!(!super::super::suspension::conflicts(&lower, &upper, 0.012));
    upper.center.x = 1.6;
    assert!(super::super::suspension::conflicts(&lower, &upper, 0.012));
    upper.center.x = 0.;
    lower.suspension = None;
    assert!(super::super::suspension::conflicts(&lower, &upper, 0.012));
    lower.suspension = Some(vec![]);
    lower.detail = None;
    assert!(super::super::suspension::conflicts(&lower, &upper, 0.012));
    let mut json = serde_json::to_value(&lower).unwrap();
    json.as_object_mut().unwrap().remove("suspension");
    assert!(serde_json::from_value::<CeilingModule>(json)
        .unwrap()
        .suspension
        .is_none());
    let legacy: CeilingInstallation =
        serde_json::from_str(r#"{"exposed_slab":true,"modules":[]}"#).unwrap();
    assert!(legacy.framing.is_none());
}

#[test]
fn frame_and_suspension_integrity_reject_nonfinite_or_unbounded_plans() {
    let mut frame = CeilingFraming {
        roof_material: Surface::Metal,
        members: vec![fixture(FrameSection::Solid)],
    };
    frame.validate_integrity().unwrap();
    frame.members[0].depth = f32::NAN;
    assert!(frame.validate_integrity().is_err());
    frame.members[0] = fixture(FrameSection::Solid);
    frame.members[0].top[1].x = f32::INFINITY;
    assert!(frame.validate_integrity().is_err());
    frame.members = vec![fixture(FrameSection::Solid); MAX_MEMBERS + 1];
    assert!(frame.validate_integrity().is_err());
}

#[test]
fn partial_bays_connect_secondary_members_to_the_roof() {
    for seed in [2, 5, 29, 46, 50, 66, 73, 92, 226, 251] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let plan = scene.ceiling_installation.as_ref().unwrap();
        plan.validate(&scene).unwrap();
        let frame = plan.framing.as_ref().unwrap();
        assert!(!frame.members.is_empty());
        assert!(frame.connected().iter().all(|&c| c), "seed {seed}");
        for m in frame
            .members
            .iter()
            .filter(|m| matches!(m.role, FrameRole::Bearing | FrameRole::RoofRib))
        {
            for p in m.top {
                assert!((scene.ceiling_height(p.xz()) - p.y - 0.001).abs() < 2e-5);
            }
        }
    }
    let mut frame = CeilingFraming {
        roof_material: Surface::Metal,
        members: vec![fixture(FrameSection::Solid)],
    };
    assert_eq!(frame.connected(), vec![false]);
    let mut root = fixture(FrameSection::Solid);
    root.role = FrameRole::Bearing;
    root.top.iter_mut().for_each(|p| p.y += root.depth);
    frame.members.push(root);
    assert_eq!(frame.connected(), vec![true, true]);
    frame.members[1].top.iter_mut().for_each(|p| p.y += 0.02);
    assert_eq!(frame.connected(), vec![false, true]);
}
