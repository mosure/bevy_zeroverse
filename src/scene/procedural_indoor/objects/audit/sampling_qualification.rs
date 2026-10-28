//! Shape diversity is measured from actual meshes, not unused parameter ranges.
use super::*;
#[test]
fn paired_switch_buttons_have_disjoint_front_faces() {
    let mut checked = 0;
    for seed in 0..128 {
        let o = fixture(ObjectKind::LightSwitch, seed);
        let p = super::super::wall::parameters(&o);
        if p.style != 5 {
            continue;
        }
        let s = o.size * Vec3::new(p.case_fraction.x, p.case_fraction.y, 1.);
        let a = build_object(&o);
        let mut caps = 0;
        for ((surface, _), g) in &a.parts {
            if *surface != Surface::Plastic {
                continue;
            }
            for triangle in g.indices.as_chunks::<3>().0 {
                let vertices = triangle.map(|i| Vec3::from_array(g.positions[i as usize]));
                if !vertices
                    .iter()
                    .all(|v| (v.z - (s.z * 0.48 + 0.002)).abs() < 1e-6)
                {
                    continue;
                }
                for gang in 0..p.gangs {
                    let x = (gang as f32 - (p.gangs - 1) as f32 * 0.5) * s.x / p.gangs as f32;
                    for lower in [true, false] {
                        let y = s.y * if lower { 0.33 } else { 0.67 };
                        if vertices
                            .iter()
                            .any(|v| (v.x - x).abs() < 1e-6 && (v.y - y).abs() < 1e-6)
                        {
                            assert!(
                                vertices.iter().all(|v| if lower {
                                    v.y < s.y * 0.5 - 0.001
                                } else {
                                    v.y > s.y * 0.5 + 0.001
                                }),
                                "overlapping switch buttons: {seed}"
                            );
                            caps += 1;
                        }
                    }
                }
            }
        }
        assert!(caps >= p.gangs * 40, "missing button caps: {seed}");
        checked += 1;
    }
    assert!(checked >= 12);
}

#[test]
fn small_products_have_replayable_noncollapsed_geometry() {
    for kind in [
        ObjectKind::SodaCan,
        ObjectKind::Sponge,
        ObjectKind::WallOutlet,
        ObjectKind::LightSwitch,
        ObjectKind::CuttingBoard,
        ObjectKind::WallArt,
    ] {
        let mut descriptors = Vec::new();
        for seed in 0..64 {
            let o = fixture(kind, seed);
            let a = build_object(&o);
            let b = build_object(&o);
            check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            if kind == ObjectKind::SodaCan {
                for ((surface, _), g) in &a.parts {
                    if *surface == Surface::Art {
                        for axis in 0..2 {
                            let lo = g
                                .uvs
                                .iter()
                                .map(|uv| uv[axis])
                                .fold(f32::INFINITY, f32::min);
                            let hi = g
                                .uvs
                                .iter()
                                .map(|uv| uv[axis])
                                .fold(f32::NEG_INFINITY, f32::max);
                            assert!(lo.abs() < 1e-5 && (hi - 1.).abs() < 1e-5);
                        }
                    }
                }
            }
            for (key, g) in &a.parts {
                let h = &b.parts[key];
                assert_eq!(g.positions, h.positions);
                assert_eq!(g.indices, h.indices);
                assert_eq!(g.normals, h.normals);
                assert_eq!(g.uvs, h.uvs);
            }
            descriptors.push(descriptor(&o, &a));
        }
        let coverage = representatives(&descriptors, 0.005).len();
        assert!(coverage >= 8, "{kind:?}: {coverage}");
        println!("product coverage {kind:?}: {coverage}/64 at 0.005");
    }
}
#[test]
fn board_support_rectangles_lie_on_real_top_triangles_and_exclude_handles() {
    use super::super::kitchen::supplies;
    for seed in 0..64 {
        let o = fixture(ObjectKind::CuttingBoard, seed);
        let p = supplies::board_parameters(&o);
        let a = build_object(&o);
        let (lo, hi) = supplies::board_top_bounds(&o);
        for q in [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)] {
            let q = q.lerp((lo + hi) * 0.5, 0.015);
            let on_top = a.parts.values().any(|g| {
                g.indices.as_chunks::<3>().0.iter().any(|t| {
                    let v = [
                        Vec3::from_array(g.positions[t[0] as usize]),
                        Vec3::from_array(g.positions[t[1] as usize]),
                        Vec3::from_array(g.positions[t[2] as usize]),
                    ];
                    if v.iter().any(|v| (v.y - o.size.y).abs() > 1e-5) {
                        return false;
                    }
                    let (a, b, c) = (v[0].xz(), v[1].xz(), v[2].xz());
                    let det = (b - a).perp_dot(c - a);
                    if det.abs() < 1e-10 {
                        return false;
                    }
                    let u = (q - a).perp_dot(c - a) / det;
                    let v = (b - a).perp_dot(q - a) / det;
                    u >= -1e-5 && v >= -1e-5 && u + v <= 1. + 1e-5
                })
            });
            assert!(on_top, "{seed}: unsupported top point {q:?}");
        }
        if p.handle {
            let q = Vec2::new(o.size.x * 0.46, 0.);
            assert!(!supplies::board_supports(
                &o,
                q - Vec2::splat(0.005),
                q + Vec2::splat(0.005),
                0.
            ));
        }
    }
}
#[test]
fn outlet_port_tracks_case_and_electrical_standard() {
    for seed in 0..64 {
        for standard in 0..3 {
            let mut o = fixture(ObjectKind::WallOutlet, seed);
            o.variant = standard;
            let p = super::super::wall::parameters(&o);
            assert_eq!(p.style as u32 % 3, standard);
            let port = super::super::wall::outlet_port_local(&o);
            let a = build_object(&o);
            let (lo, hi) = a.bounds();
            assert!(port.x >= lo.x && port.x <= hi.x && port.y >= lo.y && port.y <= hi.y);
            // Dark socket apertures straddle the connector center; they must not
            // be stranded at the old nominal plate center after resizing.
            let holes: Vec<_> = a
                .parts
                .iter()
                .filter(|((s, _), _)| *s == Surface::Ink)
                .flat_map(|(_, g)| g.positions.iter())
                .map(|v| Vec3::from_array(*v))
                .filter(|v| (v.y - port.y).abs() < 0.02 && (v.z - port.z).abs() < 0.003)
                .collect();
            assert!(holes.iter().any(|q| q.x < port.x) && holes.iter().any(|q| q.x > port.x));
        }
    }
}
