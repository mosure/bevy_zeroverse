use super::*;
use crate::scene::procedural_indoor::{
    layout::{segment_hits_box, IndoorLayout},
    objects::Assembly,
};
use std::collections::BTreeSet;

pub(super) fn fixture() -> (IndoorManifest, EnvelopeProgram) {
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.room_size = Vec3::new(18., 7.3, 18.);
    scene.program = None;
    scene.envelope = None;
    scene.objects.clear();
    let e = EnvelopeProgram {
        wall_thickness_m: None,
        footprint: vec![
            Vec2::splat(-9.),
            Vec2::new(9., -9.),
            Vec2::splat(9.),
            Vec2::new(-9., 9.),
        ],
        soffits: vec![],
        tier_guard: None,
        ceiling_drop: Vec2::ZERO,
        floor_patches: vec![],
        pillars: vec![],
        mezzanine: None,
        walls: vec![],
    };
    (scene, e)
}

pub(super) fn check(a: &Assembly, m: &Mezzanine, e: &EnvelopeProgram, size: Vec3) -> usize {
    m.validate(e, size).unwrap();
    let mut e = e.clone();
    e.mezzanine = Some(m.clone());
    let bounds = e.structural_boxes(size);
    let mut count = 0;
    if m.construction.is_some() {
        let beams = m.beams();
        for (i, (lo, hi)) in beams.iter().enumerate() {
            for (a, b) in beams.iter().skip(i + 1) {
                assert!(
                    (hi.min(*b) - lo.max(*a)).min_element() < 1e-5,
                    "overlapping beam undersides"
                );
            }
            for (a, b) in m.columns() {
                assert!(
                    (hi.min(b) - lo.max(a)).min_element() < 1e-5,
                    "column must end below its beam"
                );
            }
        }
    }
    for ((_, label), g) in &a.parts {
        assert!(matches!(label.as_str(), "floor" | "other_structure"));
        assert!(g.positions.iter().flatten().all(|p| p.is_finite()));
        assert!(g.uvs.iter().flatten().all(|p| p.is_finite()));
        assert!(g
            .normals
            .iter()
            .all(|n| (Vec3::from(*n).length() - 1.).abs() < 0.003));
        for p in &g.positions {
            let p = Vec3::from(*p);
            assert!(
                bounds
                    .iter()
                    .any(|(lo, hi)| p.cmpge(*lo - Vec3::splat(1e-5)).all()
                        && p.cmple(*hi + Vec3::splat(1e-5)).all()),
                "unbounded {label}: {p:?}"
            );
        }
        for tri in g.indices.as_chunks::<3>().0 {
            let [p, q, r] = tri.map(|i| Vec3::from(g.positions[i as usize]));
            let cross = (q - p).cross(r - p);
            assert!(
                cross.length_squared() > 1e-16,
                "collapsed triangle {p:?} {q:?} {r:?}"
            );
            let n: Vec3 = tri.iter().map(|i| Vec3::from(g.normals[*i as usize])).sum();
            assert!(cross.dot(n) >= -1e-6, "inverted triangle");
        }
        count += g.indices.len() / 3;
    }
    assert!(count < 18000, "bounded detail budget: {count}");
    // Actual floor triangles agree with support queries at both noses and mid
    // treads, including the extended intermediate landing and reverse ascent.
    for (lo, hi) in m.treads() {
        for f in [0.02, 0.5, 0.98] {
            let p = lo.lerp(hi, f).with_y(hi.y + 0.05);
            let hit = a
                .parts
                .iter()
                .filter(|((_, label), _)| label == "floor")
                .flat_map(|(_, g)| {
                    g.indices.as_chunks::<3>().0.iter().filter_map(move |t| {
                        super::super::tests::ray_triangle(
                            p,
                            Vec3::NEG_Y,
                            t.map(|i| Vec3::from(g.positions[i as usize])),
                        )
                    })
                })
                .min_by(f32::total_cmp)
                .unwrap();
            assert!((hit - 0.05).abs() < 1e-4, "support mismatch {p:?}: {hit}");
            assert!(
                (m.stair_highest_in(p.xz() - Vec2::splat(0.001), p.xz() + Vec2::splat(0.001))
                    .unwrap()
                    - hi.y)
                    .abs()
                    < 1e-5
            );
        }
    }
    let mut join = (m.stair_min + m.stair_max) * 0.5;
    join[m.stair_axis] = if m.ascending_positive() {
        m.stair_max[m.stair_axis]
    } else {
        m.stair_min[m.stair_axis]
    };
    let center = Vec3::new(join.x, m.deck.height + 0.8, join.y);
    let axis = if m.stair_axis == 0 { Vec3::X } else { Vec3::Z };
    assert!(
        bounds.iter().all(|(lo, hi)| !segment_hits_box(
            center - axis * 0.2,
            center + axis * 0.2,
            *lo,
            *hi
        )),
        "guard blocks stair entry"
    );
    count
}

#[test]
fn mezzanine_meshes_support_openings_bounds_and_parameter_coverage() {
    let (mut scene, e) = fixture();
    let mut orientations = BTreeSet::new();
    let mut structures = BTreeSet::new();
    let mut infills = BTreeSet::new();
    let mut widths = BTreeSet::new();
    let mut landing = 0;
    let mut columns = BTreeSet::new();
    for seed in 0..128 {
        scene.seed = seed;
        let m = Mezzanine::sample(&scene, &e).unwrap();
        assert_eq!(Some(&m), Mezzanine::sample(&scene, &e).as_ref());
        assert_eq!(
            m,
            serde_json::from_str::<Mezzanine>(&serde_json::to_string(&m).unwrap()).unwrap()
        );
        let c = m.construction.as_ref().unwrap();
        orientations.insert((m.stair_axis, m.ascending_positive()));
        structures.insert(format!("{:?}", c.stair_structure));
        infills.insert(format!("{:?}", c.guard.infill));
        widths.insert((m.stair_width() * 100.) as u32);
        landing += usize::from(c.landing.is_some());
        columns.insert(m.columns().len());
        let mut a = Assembly::default();
        build(&mut a, &m);
        check(&a, &m, &e, scene.room_size);
    }
    assert_eq!(orientations.len(), 4);
    assert_eq!(structures.len(), 3);
    assert_eq!(infills.len(), 4);
    assert!(widths.len() > 45, "width coverage {widths:?}");
    assert!((20..90).contains(&landing));
    assert!(columns.len() >= 2);
}

#[test]
fn mezzanine_rejects_disconnected_stairs_invalid_landings_and_low_headroom() {
    let (scene, e) = fixture();
    let m = Mezzanine::sample(&scene, &e).unwrap();
    for edit in 0..8 {
        let mut m = m.clone();
        match edit {
            0 => m.stair_axis = 2,
            1 => m.steps = 0,
            2 => m.deck.min.x = f32::NAN,
            3 => {
                m.stair_min[m.stair_axis] += 0.1;
                m.stair_max[m.stair_axis] += 0.1;
            }
            4 => m.rail_height = f32::NAN,
            5 => {
                m.construction.as_mut().unwrap().landing = Some(Landing {
                    after_riser: 2,
                    extra_run: 1.,
                })
            }
            6 => m.construction.as_mut().unwrap().guard.post_pitch = f32::INFINITY,
            _ => m.construction.as_mut().unwrap().beam_depth = 1.,
        }
        assert!(
            m.validate(&e, scene.room_size).is_err(),
            "accepted invalid edit {edit}"
        );
    }
    let mut e = e.clone();
    let center = (m.deck.min + m.deck.max) * 0.5;
    e.soffits.push(FloorPatch {
        min: center - Vec2::splat(0.1),
        max: center + Vec2::splat(0.1),
        height: m.deck.height + 1.9,
    });
    assert!(m.validate(&e, scene.room_size).is_err());
}

#[test]
fn legacy_mezzanine_records_keep_their_construction_and_stair_oracle() {
    let m:Mezzanine=serde_json::from_value(serde_json::json!({"deck":{"min":[-3.,-2.],"max":[0.,2.],"height":2.88},
        "thickness":0.2,"stair_min":[0.,-0.54],"stair_max":[5.4,0.54],"stair_axis":0,"steps":18,"rail_height":1.1})).unwrap();
    assert!(m.construction.is_none());
    let mut a = Assembly::default();
    let mut b = Assembly::default();
    build(&mut a, &m);
    legacy::build(&mut b, &m);
    for (key, g) in &a.parts {
        let other = &b.parts[key];
        assert_eq!(g.positions, other.positions);
        assert_eq!(g.indices, other.indices);
        assert_eq!(g.uvs, other.uvs);
    }
    for (i, (lo, hi)) in m.treads().enumerate() {
        let run = (m.stair_max.x - m.stair_min.x) / m.steps as f32;
        assert_eq!(
            lo,
            Vec3::new(m.stair_max.x - (i + 1) as f32 * run, 0., m.stair_min.y)
        );
        assert_eq!(
            hi,
            Vec3::new(
                m.stair_max.x - i as f32 * run,
                m.deck.height * (i + 1) as f32 / m.steps as f32,
                m.stair_max.y
            )
        );
    }
    let (_, mut envelope) = fixture();
    envelope.mezzanine = Some(m);
    let mut value = serde_json::to_value(&envelope).unwrap();
    value.as_object_mut().unwrap().remove("tier_guard");
    assert!(serde_json::from_value::<EnvelopeProgram>(value)
        .unwrap()
        .tier_guard
        .is_none());
}

#[test]
fn guard_bays_share_corner_posts_and_have_valid_bounds() {
    use super::super::guardrail::*;
    let g = GuardSection::sample(88);
    let mut a = Assembly::default();
    let mut posts = Vec::new();
    for (start, end) in [
        (Vec3::ZERO, Vec3::X * 3.),
        (Vec3::X * 3., Vec3::X * 3. + Vec3::Z * 3.),
    ] {
        super::super::guardrail::build(
            &mut a,
            &g,
            GuardRun {
                start,
                end,
                height: 1.1,
                stair_rise: 0.,
            },
            |p| p.y,
            &mut posts,
        );
    }
    assert_eq!(posts.len(), 2 * (3. / g.post_pitch).ceil() as usize + 1);
    assert_eq!(
        posts
            .iter()
            .filter(|p| p.distance_squared(Vec3::X * 3.) < 1e-8)
            .count(),
        1
    );
}
