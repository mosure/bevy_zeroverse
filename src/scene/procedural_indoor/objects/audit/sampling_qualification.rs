//! Shape diversity is measured from actual meshes, not unused parameter ranges.
use super::*;

#[test]
fn installed_shelves_and_bench_slats_follow_their_cut_direction() {
    use super::super::{services::bench, storage};
    let mut observed = [0; 4];
    for seed in 0..32 {
        for kind in [ObjectKind::Bookcase, ObjectKind::Cabinet, ObjectKind::Bench] {
            let mut o = fixture(kind, seed);
            o.finish_seed = None; // sample real finishes instead of the fixed audit swatch
                                  // Pick an exposed board with a known elevation, independent of mesh order.
            let (height, along, group) = match kind {
                ObjectKind::Bookcase => {
                    if storage::bookcase::parameters(&o).shelves != Surface::Wood {
                        continue;
                    }
                    (o.size.y, Vec3::X, 0)
                }
                ObjectKind::Cabinet => {
                    if storage::parameters(&o).shelves != Surface::Wood {
                        continue;
                    }
                    (o.size.y, Vec3::X, 1)
                }
                _ => {
                    let p = bench::parameters(&o);
                    if !p.slats {
                        continue;
                    }
                    (
                        p.seat_height,
                        if p.longitudinal { Vec3::X } else { Vec3::Z },
                        if p.longitudinal { 2 } else { 3 },
                    )
                }
            };
            let assembly = build_object(&o);
            let mut faces = 0;
            for ((s, _), g) in &assembly.parts {
                if *s != Surface::Wood {
                    continue;
                }
                for t in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = t.map(|i| Vec3::from(g.positions[i as usize]));
                    if [a, b, c].iter().any(|p| (p.y - height).abs() > 1e-5)
                        || (b - a).cross(c - a).y < 0.004
                    {
                        continue;
                    }
                    let [ua, ub, uc] = t.map(|i| g.uvs[i as usize][0]);
                    let fibre = ((c - a) * (ub - ua) - (b - a) * (uc - ua)).normalize();
                    assert!(fibre.dot(along).abs() > 0.9999, "{kind:?} {seed}");
                    faces += 1;
                }
            }
            assert!(faces >= 2);
            observed[group] += 1;
        }
    }
    assert!(observed.iter().all(|&n| n > 0), "{observed:?}");
}

#[test]
fn toggle_levers_project_within_the_plate_envelope_and_dimmers_travel_the_slot() {
    let mut toggle_states = [0; 2];
    let mut split = false;
    let mut slider_extremes = [false; 2];
    for seed in 0..160 {
        let mut o = fixture(ObjectKind::LightSwitch, seed);
        o.size.x = 0.225; // exercise multi-gang products
        let p = super::super::wall::parameters(&o);
        if !matches!(p.style, 1 | 4) {
            continue;
        }
        let a = build_object(&o);
        check(&o, &a).unwrap();
        assert!(a.parts.keys().all(|(s, _)| *s != Surface::Paper));
        let case = o.size * Vec3::new(p.case_fraction.x, p.case_fraction.y, 1.);
        let pitch = case.x / p.gangs as f32;
        let threshold = if p.style == 1 {
            case.z * 0.44
        } else {
            case.z * 0.48 + 0.0028
        };
        let mut direction = None;
        for gang in 0..p.gangs {
            let x = (gang as f32 - (p.gangs - 1) as f32 * 0.5) * pitch;
            let mut y_min = f32::INFINITY;
            let mut y_max = f32::NEG_INFINITY;
            for v in a
                .parts
                .iter()
                .filter(|((s, _), _)| *s == Surface::Plastic)
                .flat_map(|(_, g)| &g.positions)
                .filter(|v| (v[0] - x).abs() < pitch * 0.48 && v[2] > threshold)
            {
                y_min = y_min.min(v[1]);
                y_max = y_max.max(v[1]);
                if p.style == 1 {
                    assert!(v[2] <= case.z * 0.5);
                }
            }
            assert!(y_min <= y_max, "no visible actuator tip");
            let travel = (y_min + y_max - case.y) * 0.5;
            if p.style == 1 {
                assert!(travel.abs() > 0.0015, "toggle remains flat");
                let side = usize::from(travel > 0.);
                toggle_states[side] += 1;
                split |= direction.replace(side).is_some_and(|prev| prev != side);
            } else {
                let fraction = travel / (case.y * p.control_fraction.y);
                assert!(fraction.abs() <= 0.276, "slider extends past its track");
                if fraction.abs() > 0.18 {
                    slider_extremes[usize::from(travel > 0.)] = true;
                }
            }
        }
    }
    assert!(toggle_states.iter().all(|&n| n > 8) && split);
    assert_eq!(slider_extremes, [true; 2]);
}
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
