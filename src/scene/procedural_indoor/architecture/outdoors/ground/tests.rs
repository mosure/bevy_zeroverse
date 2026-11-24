use super::super::qualification::check;
use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

fn scene(seed: u64) -> IndoorManifest {
    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap()
}

// A downward geometry ray restricted to ground finishes. Return all front-facing
// surfaces, including visible bedding and hidden material below a paving slab.
fn hits(a: &Assembly, point: Vec2) -> Vec<(f32, Surface)> {
    let mut hits = Vec::new();
    for ((surface, label), g) in &a.parts {
        if label.split('#').next() != Some("floor") {
            continue;
        }
        for &[a, b, c] in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = [a, b, c].map(|i| Vec3::from_array(g.positions[i as usize]));
            let u = (b - a).xz();
            let v = (c - a).xz();
            let det = u.perp_dot(v);
            if det >= -1e-9 {
                continue;
            }
            let p = point - a.xz();
            let x = p.perp_dot(v) / det;
            let z = u.perp_dot(p) / det;
            if x >= 0. && z >= 0. && x + z <= 1. {
                hits.push((a.y + x * (b.y - a.y) + z * (c.y - a.y), *surface));
            }
        }
    }
    hits.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    hits
}

#[test]
fn clipped_courses_keep_metric_winding_materials_and_joint_support() {
    let frame = Frame::new(Vec2::ZERO, 0.);
    let rect = Rect {
        lo: Vec2::new(-2.103, -1.672),
        hi: Vec2::new(2.103, 1.672),
    };
    let soil = Rect {
        lo: Vec2::new(-0.481, -0.773),
        hi: Vec2::new(0.654, 0.201),
    };
    let mut bonds = std::collections::BTreeSet::new();
    let mut r = stream(12, 13);
    for seed in 0..24 {
        let program = Paving::sample(seed, rect);
        bonds.insert(program.bond as u32);
        let mut a = Assembly::default();
        frame.base(
            a.part(Surface::Concrete, "floor"),
            rect,
            -0.08,
            PATIO - program.relief,
            Some(soil),
        );
        program.lay(
            frame,
            a.part(program.surface, &label(program.surface, "floor", seed)),
            rect,
            PATIO,
            Some(soil),
        );
        frame.top(a.part(Surface::Soil, "floor"), soil, -0.014, Vec2::ZERO);
        let tris = check(&a);
        assert!(tris < 1_200);
        assert!(program.panel_count() <= 96);
        if program.surface == Surface::Terracotta {
            assert!(program.module(rect).max_element() <= 0.60);
        }
        let mut panels = Vec::new();
        program.panels(rect, |r, _| panels.push(r));
        assert_eq!(program.panel_count(), panels.len());
        let points = (0..60)
            .map(|_| rect.lo + rect.size() * Vec2::new(r.random(), r.random()))
            .chain(
                panels
                    .iter()
                    .map(|p| Vec2::new(p.lo.x - program.joint * 0.25, (p.lo.y + p.hi.y) * 0.5)),
            );
        for point in points {
            let inside = |r: Rect| point.cmpge(r.lo).all() && point.cmple(r.hi).all();
            let found = hits(&a, point);
            let (height, finish) = found[0];
            let expected = if inside(soil) {
                assert_eq!(finish, Surface::Soil);
                assert_eq!(found.len(), 1, "hidden paving through root bed");
                -0.014
            } else if panels.iter().any(|r| inside(*r)) {
                PATIO
            } else {
                PATIO - program.relief
            };
            assert!(
                (height - expected).abs() < 1e-5,
                "{seed}, {point:?}: {found:?}"
            );
            if found.len() > 1 {
                assert!(height - found[1].0 > 0.001, "coincident surface {found:?}");
            }
        }
    }
    assert_eq!(bonds.len(), 3);
}

#[test]
fn every_rotated_courtyard_root_terminates_in_soil_and_bench_feet_on_paving() {
    let scene = scene(58106);
    for seed in 0..48 {
        let p = Parcel {
            center: Vec2::new(8.3, -4.6),
            size: Vec2::new(2.3 + 0.071 * seed as f32, 2.5 + 0.051 * seed as f32),
            yaw: seed as f32 * 0.21,
            kind: 2,
            seed,
        };
        let frame = Frame::new(p.center, p.yaw);
        let mut plants = garden_plants(&scene, &p);
        let bed = Bed::fit(&p, frame, &plants).unwrap();
        let mut a = Assembly::default();
        patio(&mut a, &p, &mut plants);
        garden(&mut a, &p, &plants, scene.generator_version);
        check(&a);
        let bark = &a.parts[&(Surface::Bark, "plant".into())];
        let roots: Vec<_> = bark
            .positions
            .iter()
            .map(|v| Vec3::from_array(*v))
            .filter(|p| (p.y - bed.elevation).abs() < 1e-5)
            .collect();
        assert!(roots.len() >= plants.len());
        for root in roots {
            let (y, s) = hits(&a, root.xz())[0];
            assert_eq!(s, Surface::Soil, "seed {seed}");
            assert!((root.y - y).abs() < 1e-5);
        }
        for plant in plants {
            assert_eq!(hits(&a, plant.position.xz())[0].1, Surface::Soil);
        }
        assert!(bed.outer.hi.y < p.size.y * 0.32 - 0.095);
        let rotation = Quat::from_rotation_y(p.yaw);
        for x in [-0.30, 0., 0.30] {
            let foot = p.center + (rotation * Vec3::new(x * p.size.x, 0., 0.32 * p.size.y)).xz();
            let (height, s) = hits(&a, foot)[0];
            assert!(matches!(s, Surface::Concrete | Surface::Terracotta));
            assert!((PATIO - 0.015..=PATIO + 1e-5).contains(&height));
        }
    }
}

#[test]
fn forecourts_follow_openings_and_leave_prior_manifests_untouched() {
    let mut covered = 0;
    let mut without = 0;
    for seed in 58100..58228 {
        let mut s = scene(seed);
        let d = District::sample(seed);
        let w = window_openings(&s);
        let mut count = 0;
        surrounds(&s, &w, &d, |r, c| {
            if let Some(c) = c {
                assert!((c.rect.lo - r.lo).min_element() > 0.025);
                assert!((r.hi - c.rect.hi).min_element() > 0.025);
                assert!(c.paving.panel_count() <= 96);
                count += 1;
            }
        });
        if count > 0 {
            covered += 1;
        } else {
            without += 1;
        }
        surrounds(&s, &[], &d, |_, c| assert!(c.is_none()));
        s.generator_version = FIRST_VERSION - 1;
        surrounds(&s, &w, &d, |_, c| assert!(c.is_none()));
    }
    assert!(
        covered > 80 && (6..46).contains(&without),
        "{covered} rooms with paving, {without} open grounds"
    );
}
