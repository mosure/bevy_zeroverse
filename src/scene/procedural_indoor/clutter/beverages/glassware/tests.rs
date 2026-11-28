use super::*;
use crate::scene::procedural_indoor::{
    clutter::beverages::test_geometry::{top_surface_at, triangles},
    objects::{self, audit},
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn catalog_glasses_have_one_closed_shell_an_open_mouth_and_seated_water() {
    let mut forms = BTreeSet::new();
    let mut fills = [0; 2];
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::DrinkingGlass, seed);
        let p = Glass::for_object(&o);
        let a = objects::build_object(&o);
        let mesh = audit::check(&o, &a).unwrap();
        forms.insert(p.form as usize);
        assert!(mesh.triangles < 2100 && mesh.max_envelope_overrun_m < 1e-6);
        let (lo, hi) = a.bounds();
        assert_eq!(lo.y, 0.);
        assert!((hi.y - o.size.y).abs() < 1e-6);
        let mut edges = BTreeMap::new();
        let mut wet_points = BTreeSet::new();
        let key = |p: Vec3| p.to_array().map(|v| if v == 0. { 0 } else { v.to_bits() });
        for ((surface, label), g) in &a.parts {
            assert_eq!(objects::part_label(label), "drink_container");
            if *surface != Surface::ContainerGlass {
                continue;
            }
            if label == tableware::IMMERSED {
                wet_points.extend(g.positions.iter().map(|&p| key(Vec3::from_array(p))));
            }
            for tri in triangles(g) {
                let [a, b, c] = tri.map(key);
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    assert_ne!(u, v);
                    let (edge, side) = if u < v { ((u, v), 0) } else { ((v, u), 1) };
                    edges.entry(edge).or_insert([0; 2])[side] += 1;
                }
            }
        }
        assert!(
            edges.values().all(|use_count| *use_count == [1, 1]),
            "shell at seed {seed}"
        );
        let (surface_y, surface) = top_surface_at(&a, Vec2::new(0.0007, 0.0013)).unwrap();
        fills[usize::from(p.water_depth_fraction.is_some())] += 1;
        if let Some(depth) = p.water_depth_fraction {
            assert_eq!(surface, Surface::Liquid);
            let level = p.floor_m + depth * (o.size.y - p.wall_m * 0.4 - p.floor_m);
            assert!((surface_y - level).abs() <= 0.00073);
            let water = &a.parts[&(Surface::Liquid, "drink_container".into())];
            assert!(triangles(water).all(|[a, b, c]| (b - a).cross(c - a).y > 0.));
            let edge_y = water
                .positions
                .iter()
                .map(|v| v[1])
                .fold(f32::NEG_INFINITY, f32::max);
            for &v in water.positions.iter().filter(|p| p[1] == edge_y) {
                assert!(
                    wet_points.contains(&key(Vec3::from_array(v))),
                    "water meets wall at {seed}"
                );
            }
        } else {
            assert!(wet_points.is_empty());
            assert_eq!(surface, Surface::ContainerGlass);
            assert!((surface_y - p.floor_m).abs() < 1e-6);
        }
    }
    assert_eq!(forms.len(), 5);
    assert!(fills.into_iter().all(|count| count > 20));
}

#[test]
fn lot_keeps_manufacturing_constant_across_independently_served_glasses() {
    let mut o = audit::fixture(ObjectKind::DrinkingGlass, 29);
    let first = serde_json::to_value(Glass::for_object(&o)).unwrap();
    let mut levels = BTreeSet::new();
    for seed in 80..112 {
        o.seed = seed;
        let mut next = serde_json::to_value(Glass::for_object(&o)).unwrap();
        levels.insert(next["water_depth_fraction"].to_string());
        next["water_depth_fraction"] = first["water_depth_fraction"].clone();
        assert_eq!(next, first);
    }
    assert!(levels.len() > 12);
}
