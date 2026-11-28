use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};
#[test]
fn laundry_pockets_have_an_open_housing_and_a_real_drum_behind_the_glass() {
    let mut stacked = 0;
    let mut laden = 0;
    let mut clear = 0;
    for kind in [ObjectKind::WashingMachine, ObjectKind::TumbleDryer] {
        for seed in 0..48 {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = Some(seed);
            let m = Machine::for_object(&o);
            let a = build_object(&o);
            let check = audit::check(&o, &a).unwrap();
            // A stack has two complete door, barrel and control assemblies.
            assert!(check.triangles <= 3_150 * m.pockets.len());
            stacked += usize::from(m.pockets.len() == 2);
            for p in &m.pockets {
                // A ray above the settled load must see glass first, then
                // the far drum. No painted front or false solid door cap.
                let xy = Vec2::new(0.035 * p.radius_m, p.center_y_m + 0.30 * p.radius_m);
                let mut window = f32::NEG_INFINITY;
                let mut opaque = f32::NEG_INFINITY;
                for ((surface, label), g) in &a.parts {
                    assert!(
                        part_label(label) == kind.class_name() || part_label(label) == "clothes"
                    );
                    for tri in g.indices.as_chunks::<3>().0 {
                        let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                        let (u, v) = (b.xy() - a.xy(), c.xy() - a.xy());
                        let det = u.perp_dot(v);
                        if det.abs() < 1e-9 {
                            continue;
                        }
                        let d = xy - a.xy();
                        let (s, t) = (d.perp_dot(v) / det, u.perp_dot(d) / det);
                        if s >= 0. && t >= 0. && s + t <= 1. {
                            let z = a.z + s * (b.z - a.z) + t * (c.z - a.z);
                            if *surface == Surface::ContainerGlass {
                                window = window.max(z)
                            } else {
                                opaque = opaque.max(z)
                            }
                        }
                    }
                }
                assert!(
                    (m.front_z_m - window).abs() < 0.04
                        && (opaque - (m.front_z_m - p.depth_m)).abs() < 0.018,
                    "{seed} {p:?} {window} {opaque}"
                );
                assert_eq!(
                    a.parts.keys().any(|(_, l)| part_label(l) == "clothes"),
                    m.pockets.iter().any(|p| p.load_fraction > 0.)
                );
                if p.load_fraction > 0. {
                    laden += 1
                } else {
                    clear += 1
                }
            }
        }
    }
    assert!(stacked > 8 && laden > 24 && clear > 22);
}
