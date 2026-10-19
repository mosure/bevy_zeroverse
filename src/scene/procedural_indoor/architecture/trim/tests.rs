use super::*;
use std::collections::BTreeSet;

#[test]
fn continuous_sections_cover_scale_profiles_and_absence() {
    let mut profiles = BTreeSet::new();
    let mut heights = Vec::new();
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
