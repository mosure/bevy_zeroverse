use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn housing_and_image_are_one_closed_oriented_surface_across_the_stock_range() {
    let mut fits = BTreeSet::new();
    let mut curved = 0;
    for kind in [ObjectKind::Monitor, ObjectKind::Display] {
        for seed in 0..96 {
            let mut o = audit::fixture(kind, seed);
            o.size = if kind == ObjectKind::Monitor {
                [Vec3::new(0.29, 0.56, 0.19), Vec3::new(1.02, 0.37, 0.22)][seed as usize % 2]
            } else {
                [Vec3::new(0.55, 1.27, 0.037), Vec3::new(2.25, 1.26, 0.092)][seed as usize % 2]
            };
            let p = display_parameters(&o);
            let d = geometry_parameters(&o, &p);
            let stock = PanelSection::sample(&o, &d);
            fits.insert(stock.glass_fit);
            curved += usize::from(d.curvature_radius_m > 0.);
            let mut a = Assembly::default();
            build(&mut a, &o, &p, &d);
            assert_eq!(a.parts.len(), 2);
            let mut edges = BTreeMap::new();
            for ((_, label), g) in &a.parts {
                assert_eq!(label, o.kind.class_name());
                for tri in g.indices.as_chunks::<3>().0 {
                    let vertices = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let [a, b, c] = vertices;
                    let face = (b - a).cross(c - a);
                    let [u, v, w] = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
                    assert!((v - u).perp_dot(w - u).abs() > 1e-12, "{kind:?}/{seed}: UV");
                    assert!(
                        tri.iter().all(|&i| {
                            let n = Vec3::from_array(g.normals[i as usize]);
                            n.is_normalized() && face.dot(n) > 0.
                        }),
                        "{kind:?}/{seed}: winding"
                    );
                    let key = vertices
                        .map(|p| p.to_array().map(|v| if v == 0. { 0 } else { v.to_bits() }));
                    for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                        let (a, b) = (key[i], key[j]);
                        let edge = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                        edge.0 += 1;
                        edge.1 += if a < b { 1 } else { -1 };
                    }
                }
            }
            assert!(
                edges.values().all(|v| *v == (2, 0)),
                "{kind:?}/{seed}: open joint"
            );
        }
    }
    assert_eq!(fits.len(), 3);
    assert!(curved > 25);
}

#[test]
fn a_ray_through_the_active_image_hits_only_glass_and_the_rear_housing() {
    for seed in 0..80 {
        let o = audit::fixture(ObjectKind::Monitor, seed);
        let p = display_parameters(&o);
        let d = geometry_parameters(&o, &p);
        let stock = PanelSection::sample(&o, &d);
        let mut a = Assembly::default();
        build(&mut a, &o, &p, &d);
        let inv = d.panel_transform.compute_affine().inverse();
        for sample in [
            Vec2::new(-0.465, 0.411),
            Vec2::new(-0.223, -0.184),
            Vec2::new(0.019, 0.148),
            Vec2::new(0.437, -0.419),
        ] {
            let xy = sample
                * Vec2::new(
                    d.width_m - 2. * d.bezel_m,
                    d.height_m * (1. - p.chin_fraction) - 2. * d.bezel_m,
                )
                + Vec2::Y * (0.5 * d.height_m * p.chin_fraction);
            let mut hits = Vec::new();
            for ((surface, _), g) in &a.parts {
                for tri in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = tri
                        .map(|i| inv.transform_point3(Vec3::from_array(g.positions[i as usize])));
                    let ex = b.xy() - a.xy();
                    let ey = c.xy() - a.xy();
                    let delta = xy - a.xy();
                    let det = ex.perp_dot(ey);
                    if det.abs() > 1e-10 {
                        let u = delta.perp_dot(ey) / det;
                        let v = ex.perp_dot(delta) / det;
                        if u >= 0. && v >= 0. && u + v <= 1. {
                            hits.push((a.z + u * (b.z - a.z) + v * (c.z - a.z), *surface));
                        }
                    }
                }
            }
            hits.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(hits.len(), 2, "seed {seed}, {xy:?}: {hits:?}");
            assert_eq!(hits[0].1, p.casing);
            assert_eq!(hits[1].1, Surface::Screen);
            let r = d.curvature_radius_m;
            let sag = if r > 0. {
                xy.x * xy.x / (r + (r * r - xy.x * xy.x).sqrt())
            } else {
                0.
            };
            assert!((hits[1].0 - (sag + 0.5 * d.panel_m + stock.glass_offset_m)).abs() < 0.0008);
        }
    }
}

#[test]
fn emitted_content_covers_the_complete_image_in_reading_order() {
    for kind in [ObjectKind::Display, ObjectKind::Monitor] {
        for seed in 0..32 {
            let o = audit::fixture(kind, seed);
            let p = display_parameters(&o);
            let d = geometry_parameters(&o, &p);
            let inv = d.panel_transform.compute_affine().inverse();
            let a = build_object(&o);
            let g = a
                .parts
                .iter()
                .find(|((s, _), _)| matches!(s, Surface::Screen | Surface::Television))
                .unwrap()
                .1;
            let width = d.width_m - 2. * d.bezel_m;
            let height = d.height_m * (1. - p.chin_fraction) - 2. * d.bezel_m;
            let arc = |x: f32| {
                let r = d.curvature_radius_m;
                if r > 0. {
                    r * (x / r).asin()
                } else {
                    x
                }
            };
            for (point, uv) in g.positions.iter().zip(&g.uvs) {
                let local = inv.transform_point3(Vec3::from_array(*point));
                let u = 0.5 + arc(local.x) / (2. * arc(width * 0.5));
                let v = 0.5 - (local.y - d.height_m * p.chin_fraction * 0.5) / height;
                assert!((uv[0] - u).abs() < 2e-5 && (uv[1] - v).abs() < 2e-5);
            }
        }
    }
}
