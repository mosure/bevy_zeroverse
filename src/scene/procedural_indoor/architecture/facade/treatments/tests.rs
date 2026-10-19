use super::*;
use crate::render::semantic::SemanticLabel;
use std::collections::{BTreeMap, BTreeSet};

const KINDS: [Shade; 6] = [
    Shade::Venetian,
    Shade::Roller,
    Shade::Vertical,
    Shade::Roman,
    Shade::Pleated,
    Shade::Curtain,
];

#[test]
fn split_coverings_reveal_the_window_as_they_retract() {
    // Intersect the actual cloth, not its AABB: folded/stacked covers must leave
    // usable sight lines and never retain a hidden full-width proxy surface.
    for kind in [Shade::Curtain, Shade::Vertical] {
        let mut visible = Vec::new();
        for coverage in [0., 0.55, 1.] {
            let mut f = facade(19, kind, Vec2::new(2.7, 2.5), coverage);
            f.shade_tilt = -0.65;
            f.treatment.as_mut().unwrap().stack = Stack::Pair;
            let mut a = Assembly::default();
            build(&mut a, Transform::IDENTITY, &f, &f.openings[0]);
            let mut hits = 0;
            for i in 0..160 {
                let origin = Vec3::new(-1.30 + 2.60 * (i as f32 + 0.5) / 160., 1.35, 0.25);
                let hit = a
                    .parts
                    .iter()
                    .filter(|((surface, _), _)| *surface == f.treatment.as_ref().unwrap().fabric)
                    .any(|(_, g)| {
                        g.indices.as_chunks::<3>().0.iter().any(|t| {
                            let [a, b, c] = t.map(|i| Vec3::from(g.positions[i as usize]));
                            let ab = b - a;
                            let ac = c - a;
                            let h = Vec3::NEG_Z.cross(ac);
                            let det = ab.dot(h);
                            if det.abs() < 1e-10 {
                                return false;
                            }
                            let s = origin - a;
                            let u = s.dot(h) / det;
                            let q = s.cross(ab);
                            let v = Vec3::NEG_Z.dot(q) / det;
                            let distance = ac.dot(q) / det;
                            u >= 0. && v >= 0. && u + v <= 1. && (0.0..0.25).contains(&distance)
                        })
                    });
                hits += usize::from(hit);
            }
            visible.push(hits);
        }
        assert!(
            visible[0] + 15 < visible[1] && visible[1] + 15 < visible[2],
            "{kind:?}: {visible:?}"
        );
    }
}

fn facade(seed: u64, kind: Shade, size: Vec2, coverage: f32) -> Facade {
    Facade {
        joinery: None,
        side: FacadeSide::Rear,
        openings: vec![WindowOpening {
            glazing: None,
            min: Vec2::new(-size.x * 0.5, 0.05),
            max: Vec2::new(size.x * 0.5, size.y + 0.05),
            columns: 1,
            transoms: Vec::new(),
            transom: 0.,
            operable: true,
        }],
        frame: Surface::Metal,
        frame_width: 0.03,
        frame_depth: 0.08,
        recess: 0.06,
        sill_projection: 0.1,
        treatment: Some(ShadeProgram::sample(seed).1),
        shade: kind,
        shade_coverage: coverage,
        shade_tilt: if seed.is_multiple_of(2) { -1.2 } else { 1.2 },
    }
}

#[test]
fn seeded_population_spans_coverings_materials_stacks_and_continuous_shapes() {
    let mut kinds = BTreeMap::<String, usize>::new();
    let mut combinations = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    for seed in 0..1024 {
        let (kind, p) = ShadeProgram::sample(seed);
        p.validate().unwrap();
        assert_eq!((kind, p.clone()), ShadeProgram::sample(seed));
        *kinds.entry(format!("{kind:?}")).or_default() += 1;
        combinations.insert(format!(
            "{:?}/{:?}/{}",
            p.stack, p.slat_material, p.finish_slot
        ));
        recipes.insert(serde_json::to_string(&p).unwrap());
        let f = facade(seed, kind, Vec2::splat(2.), 0.5);
        assert_eq!(
            f,
            serde_json::from_str::<Facade>(&serde_json::to_string(&f).unwrap()).unwrap()
        );
    }
    assert_eq!(kinds.len(), 7);
    assert!(kinds.values().all(|&n| n > 60), "{kinds:?}");
    assert!(combinations.len() > 95);
    assert_eq!(recipes.len(), 1024);
}

#[test]
fn coverings_stay_inside_aperture_and_reserved_depth_at_extreme_openings() {
    let mut max_triangles = 0;
    for seed in 0..16 {
        for kind in KINDS {
            for size in [
                Vec2::splat(0.35),
                Vec2::new(5.5, 0.5),
                Vec2::new(0.5, 5.2),
                Vec2::new(4.3, 3.2),
            ] {
                for coverage in [0., 0.5, 1.] {
                    let f = facade(seed, kind, size, coverage);
                    let o = &f.openings[0];
                    let mut a = Assembly::default();
                    build(&mut a, Transform::IDENTITY, &f, o);
                    let (lo, hi) = a.bounds();
                    assert!(lo.x>=o.min.x-1e-5 && hi.x<=o.max.x+1e-5 && lo.y>=o.min.y-1e-5 && hi.y<=o.max.y+1e-5 && lo.z>=0. && hi.z<=0.20,
                        "seed {seed}, {kind:?}, {size:?}, coverage={coverage}, bounds={lo:?}..{hi:?}");
                    let mut tris = 0;
                    for ((surface, label), g) in &a.parts {
                        assert!(variants::supports(*surface));
                        assert_eq!(
                            SemanticLabel::from_label(label.split('#').next().unwrap()),
                            Some(if kind == Shade::Curtain {
                                SemanticLabel::Curtain
                            } else {
                                SemanticLabel::Blinds
                            })
                        );
                        let mut unique = BTreeSet::new();
                        for n in &g.normals {
                            assert!((Vec3::from(*n).length() - 1.).abs() < 1e-4);
                        }
                        for uv in &g.uvs {
                            assert!(Vec2::from(*uv).is_finite());
                        }
                        for t in g.indices.as_chunks::<3>().0 {
                            let p = t.map(|i| Vec3::from(g.positions[i as usize]));
                            let cross = (p[1] - p[0]).cross(p[2] - p[0]);
                            let n: Vec3 =
                                t.iter().map(|&i| Vec3::from(g.normals[i as usize])).sum();
                            assert!(
                                cross.length_squared() > 1e-20 && cross.dot(n) >= -1e-9,
                                "{seed}/{kind:?}, size={size:?}, coverage={coverage}, {label}/{surface:?}: invalid triangle, p={p:?}, n={n:?}, dot={}, area={}", cross.dot(n), cross.length_squared()
                            );
                            let mut vertices =
                                p.map(|v| v.to_array().map(|x| (x * 1e6).round() as i64));
                            vertices.sort();
                            assert!(
                                unique.insert(vertices),
                                "{seed}/{kind:?}: coincident triangles"
                            );
                            tris += 1;
                        }
                    }
                    assert!(a.parts.len() <= 4);
                    max_triangles = max_triangles.max(tris);
                }
            }
        }
    }
    assert!(
        max_triangles < 7500,
        "bounded installation complexity: {max_triangles}"
    );
}

#[test]
fn installed_hardware_blocks_sash_sweep_even_when_retracted_and_legacy_replays() {
    let mut f = facade(5, Shade::Curtain, Vec2::splat(2.), 0.);
    assert!(!f.allows_open_sash());
    f.shade = Shade::None;
    assert!(f.allows_open_sash());
    f.treatment = None;
    f.shade = Shade::Roller;
    let encoded = serde_json::to_string(&f).unwrap();
    assert!(!encoded.contains("treatment"));
    let f: Facade = serde_json::from_str(&encoded).unwrap();
    assert!(f.treatment.is_none() && f.allows_open_sash());
    for field in [
        "slat_pitch",
        "fold_pitch",
        "fold_depth",
        "fullness",
        "hem_sag",
        "split",
        "tieback",
    ] {
        let mut v = serde_json::to_value(ShadeProgram::sample(2).1).unwrap();
        v[field] = serde_json::json!(-10.);
        assert!(serde_json::from_value::<ShadeProgram>(v)
            .unwrap()
            .validate()
            .is_err());
    }
}
