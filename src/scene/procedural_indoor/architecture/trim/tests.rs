use super::*;
use std::collections::BTreeSet;

#[test]
fn continuous_sections_cover_scale_profiles_and_absence() {
    let mut profiles = BTreeSet::new();
    let mut heights = Vec::new();
    let mut base_scales = Vec::new();
    let mut missing = [0; 2];
    for seed in 0..1024 {
        let p = TrimProgram::sample(seed);
        p.validate().unwrap();
        assert_eq!(p, TrimProgram::sample(seed));
        assert_eq!(
            p,
            serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap()
        );
        for (i, s) in [&p.base, &p.crown].into_iter().enumerate() {
            if let Some(s) = s {
                if i == 0 {
                    base_scales.push((s.height_m, s.depth_m));
                }
                profiles.insert(format!("{:?}", s.profile));
                heights.push(s.height_m);
            } else {
                missing[i] += 1;
            }
        }
    }
    heights.sort_by(f32::total_cmp);
    assert_eq!(profiles.len(), 5);
    assert!(missing[0] > 40 && missing[1] > 200);
    assert!(heights[heights.len() / 20] < 0.05);
    assert!(heights[heights.len() * 19 / 20] > 0.24);
    assert!(base_scales.iter().filter(|&&(h, _)| h > 0.30).count() > 80);
    assert!(base_scales.iter().filter(|&&(_, d)| d > 0.045).count() > 100);
    assert!(
        base_scales
            .iter()
            .filter(|&&(h, d)| h < 0.055 && d < 0.015)
            .count()
            > 60
    );
}

#[test]
fn deep_skirting_terminates_at_fitted_cabinets_but_not_clear_furniture() {
    use crate::scene::procedural_indoor::{
        layout::IndoorLayout,
        objects::{audit, ObjectKind},
    };
    let mut scene =
        IndoorManifest::generate_with_humans(3, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let section = TrimSection {
        height_m: 0.40,
        depth_m: 0.07,
        shoulder: 0.6,
        tip_fraction: 0.4,
        profile: TrimProfile::Stepped,
        surface: Surface::Paint,
        finish_slot: 0,
    };
    // A rotated real wall frame verifies the test is not axis-specific.
    let tf = Transform::from_xyz(2., 0., -3.).with_rotation(Quat::from_rotation_y(0.4));
    let mut cabinet = audit::fixture(ObjectKind::Cabinet, 3);
    cabinet.size = Vec3::new(1., 1.2, 0.5);
    cabinet.yaw = 0.4;
    cabinet.position = tf.transform_point(Vec3::new(0., 0., 0.295));
    scene.objects = vec![cabinet.clone()];
    let cutouts = fitted_openings(&scene, tf, &section, &[]);
    assert_eq!(cutouts.len(), 1);
    assert!(cutouts[0].min.x < -0.5 && cutouts[0].max.x > 0.5);
    let mut a = Assembly::default();
    build(
        &mut a,
        Transform::IDENTITY,
        Vec2::new(-2., 2.),
        Vec2::splat(0.006),
        &section,
        &cutouts,
        true,
    );
    for g in a.parts.values() {
        for t in g.indices.as_chunks::<3>().0 {
            let p = t
                .map(|i| Vec3::from(g.positions[i as usize]))
                .into_iter()
                .sum::<Vec3>()
                / 3.;
            assert!(p.x <= -0.505 || p.x >= 0.505);
        }
    }
    scene.objects[0].position = tf.transform_point(Vec3::new(0., 0., 0.40));
    assert!(fitted_openings(&scene, tf, &section, &[]).is_empty());
}

#[test]
fn sections_follow_roof_and_stop_at_glazing_without_degenerate_faces() {
    use crate::scene::procedural_indoor::architecture::facade::WindowOpening;
    let holes = [WindowOpening {
        glazing: None,
        min: Vec2::new(-0.4, 0.03),
        max: Vec2::new(0.6, 0.4),
        columns: 1,
        transom: 0.,
        transoms: vec![],
        operable: false,
    }];
    for seed in 0..128 {
        let p = TrimProgram::sample(seed);
        for section in p.base.iter().chain(&p.crown) {
            let mut a = Assembly::default();
            let level = Vec2::new(0.006, 0.10);
            build(
                &mut a,
                Transform::IDENTITY,
                Vec2::new(-2., 2.),
                level,
                section,
                &holes,
                true,
            );
            for g in a.parts.values() {
                for t in g.indices.as_chunks::<3>().0 {
                    let v = t.map(|i| Vec3::from(g.positions[i as usize]));
                    let n = (v[1] - v[0]).cross(v[2] - v[0]);
                    assert!(n.length_squared() > 1e-14);
                    for (i, p) in v.into_iter().enumerate() {
                        assert!(p.is_finite());
                        let base = level.x + (p.x + 2.) / 4. * (level.y - level.x);
                        assert!(p.y >= base - 1e-6 && p.y <= base + section.height_m + 1e-6);
                        assert!(p.z >= -0.00201 && p.z <= section.depth_m + 1e-6);
                        assert!(n.dot(Vec3::from(g.normals[t[i] as usize])) > 0.);
                    }
                    let c = (v[0] + v[1] + v[2]) / 3.;
                    assert!(!(c.x > -0.39999 && c.x < 0.59999 && c.y > 0.03001 && c.y < 0.39999));
                }
            }
        }
    }
}
