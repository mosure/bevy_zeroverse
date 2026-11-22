use super::*;
use crate::scene::procedural_indoor::{
    architecture::facade::glazing_at, materials::glazing::GlazingProgram,
};

#[test]
fn wide_tied_installations_have_valid_world_space_triangles() {
    use crate::scene::procedural_indoor::validation::validate_geometry;
    for seed in [75038, 75110] {
        let scene = IndoorManifest::generate_with_humans(seed, default(), 0.65, 3, 0.25).unwrap();
        let result = validate_geometry(&scene).unwrap();
        assert_eq!(result.degenerate_triangles, 0);
    }
}

#[test]
fn metered_curtains_match_projected_glazing_with_gathers_mullions_and_frit() {
    let size = Vec2::new(2.7, 2.5);
    let frit = (0..1024)
        .map(|s| GlazingProgram::sample(s, 0, false))
        .find(|g| g.treatment == 6)
        .unwrap();
    let mut worst = 0_f32;
    for seed in 0..3 {
        for tied in [false, true] {
            let mut f = facade(seed, Shade::Curtain, size, 0.);
            let p = f.treatment.as_mut().unwrap();
            p.stack = [Stack::Pair, Stack::Left, Stack::Right][seed as usize];
            p.tieback = if tied { 0.85 } else { 0. };
            f.openings[0].columns = 3;
            f.openings[0].transoms = vec![0.54];
            f.openings[0].operable = false;
            f.openings[0].glazing = (seed == 1).then_some(frit.clone());
            let mut previous = 1.;
            for coverage in [0., 0.28, 0.6, 0.88, 1.] {
                f.shade_coverage = coverage;
                let o = &f.openings[0];
                let estimated =
                    drapery::visible_area(&f, o, f.shade_coverage_at(o)).unwrap() / o.area();
                let mut a = Assembly::default();
                glazing_at(&mut a, Transform::IDENTITY, &f);
                let observed = frontal_clear::<301, 273>(&a, o.min, size);
                let error = (estimated - observed).abs();
                worst = worst.max(error);
                assert!(error < 0.028, "heading {seed}, tie {tied}, coverage {coverage}, estimate {estimated}, ray mask {observed}");
                assert!(observed < previous && estimated.is_finite());
                previous = observed;
            }
        }
    }
    println!("Maximum difference in fraction of rough aperture across 30 curtain states: {worst}");
}

#[test]
fn drawable_curtains_preserve_cut_coordinates_and_flow_while_ties_hold() {
    for seed in 0..18 {
        let mut f = facade(seed, Shade::Curtain, Vec2::new(3., 2.5), 0.3);
        let p = f.treatment.as_mut().unwrap();
        p.tieback = 0.;
        let cloth = p.fabric;
        let mut moving = Assembly {
            motion_policy: Some(IndoorMotionConfig {
                covering_fraction: 1.,
                ..default()
            }),
            ..default()
        };
        build(&mut moving, Transform::IDENTITY, &f, &f.openings[0]);
        assert_eq!(moving.joints.len(), 1, "seed {seed} lost its motion");
        let joint = &moving.joints[0];
        let key = joint.parts.iter().find(|k| k.0 == cloth).unwrap();
        let (start, end) = (&moving.parts[key], &joint.morph[key]);
        assert_eq!(start.indices, end.indices);
        assert_eq!(start.uvs, end.uvs);
        let travel = start
            .positions
            .iter()
            .zip(&end.positions)
            .map(|(a, b)| Vec3::from(*a).distance(Vec3::from(*b)))
            .fold(0., f32::max);
        assert!(travel > 0.12);
        for t in [0., 0.25, 0.5, 0.75, 1.] {
            for triangle in start.indices.as_chunks::<3>().0 {
                let v = triangle.map(|i| {
                    Vec3::from(start.positions[i as usize])
                        .lerp(Vec3::from(end.positions[i as usize]), t)
                });
                assert!(
                    (v[1] - v[0]).cross(v[2] - v[0]).length_squared() > 1e-20,
                    "seed={seed} t={t}"
                );
            }
        }
        f.treatment.as_mut().unwrap().tieback = 0.9;
        f.treatment.as_mut().unwrap().stack = Stack::Pair;
        let mut tied = Assembly {
            motion_policy: moving.motion_policy.clone(),
            ..default()
        };
        assert!(drapery::tied(
            f.treatment.as_ref().unwrap(),
            &f.openings[0],
            f.shade_coverage
        ));
        build(&mut tied, Transform::IDENTITY, &f, &f.openings[0]);
        assert!(tied.joints.is_empty() && !tied.parts.is_empty());
    }
}

#[test]
fn headings_share_hardware_across_a_wall_but_each_cut_hangs_independently() {
    let mut f = facade(13, Shade::Curtain, Vec2::new(2.8, 2.5), 0.5);
    f.treatment.as_mut().unwrap().stack = Stack::Pair;
    let p = f.treatment.as_ref().unwrap();
    let h = p.curtain_heading.unwrap();
    let opening = &f.openings[0];
    let mut next = opening.clone();
    next.min.x += 3.;
    next.max.x += 3.;
    let at = |o| {
        drapery::panels(p, o, h, 0.5, true)
            .map(|p| (p.phase, p.tie_at, p.cut_origin))
            .collect::<Vec<_>>()
    };
    let first = at(opening);
    assert_eq!(first, at(opening));
    assert_ne!(first[0], first[1]);
    assert_ne!(first, at(&next));
    for panel in drapery::panels(p, opening, h, 0.5, true) {
        let below_tie = panel.row_width(0.) - panel.row_width(panel.tie_at);
        assert!((0.0..=0.56 * panel.height * panel.tie_at).contains(&below_tie));
    }

    let mut archived = serde_json::to_value(&f).unwrap();
    archived["treatment"]
        .as_object_mut()
        .unwrap()
        .remove("curtain_heading");
    let legacy: Facade = serde_json::from_value(archived).unwrap();
    assert!(legacy.treatment.as_ref().unwrap().curtain_heading.is_none());
    assert!(drapery::visible_area(&legacy, &legacy.openings[0], legacy.shade_coverage).is_none());
    let mut old = Assembly::default();
    let mut current = Assembly::default();
    build(&mut old, Transform::IDENTITY, &legacy, &legacy.openings[0]);
    build(&mut current, Transform::IDENTITY, &f, opening);
    assert_ne!(
        old.parts.values().next().unwrap().positions,
        current.parts.values().next().unwrap().positions
    );
}
