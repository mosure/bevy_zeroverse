use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, equipment::wires};
use std::collections::BTreeSet;

#[test]
fn tools_rest_on_the_actual_floor_and_lip_without_piercing_the_cup_or_one_another() {
    use writing::WritingEnd::*;

    let mut counts = BTreeSet::new();
    let mut products = BTreeSet::new();
    let mut pitches = BTreeSet::new();
    let mut requested = 0;
    let mut admitted = 0;
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::PenHolder, seed);
        o.size = [
            Vec3::new(0.065, 0.21, 0.065),
            Vec3::new(0.115, 0.12, 0.115),
            Vec3::new(0.105, 0.18, 0.084),
            Vec3::new(0.08, 0.15, 0.112),
        ][seed as usize % 4];
        let p = parameters(&o);
        let cup = Cup::new(&o, &p);
        let pens = packing::settle(&o, &cup);
        counts.insert(pens.len());
        requested += p.requested_pens;
        admitted += pens.len();
        assert!(pens.len() <= p.requested_pens);
        audit::check(&o, &build_object(&o)).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let mut shell = Assembly::default();
        cup.draw(&mut shell, "stationery");
        let shell = &shell.parts.values().next().unwrap();
        for (i, pen) in pens.iter().enumerate() {
            products.insert(pen.upper_end);
            let tf = pen.transform();
            let first_y = tf.translation.y;
            let last_y = tf.transform_point(Vec3::Y * pen.tool.length).y;
            assert!(match pen.upper_end {
                PencilPoint => pen.tool.mechanism == 0 && last_y > first_y,
                Ballpoint => pen.tool.mechanism == 1 && last_y > first_y,
                MarkerCap => pen.tool.mechanism == 2 && first_y > last_y,
                Eraser => pen.tool.eraser && first_y > last_y,
                PushButton => pen.tool.mechanism == 1 && first_y > last_y,
                BarrelEnd =>
                    (pen.tool.mechanism == 0 && !pen.tool.eraser && first_y > last_y)
                        || (pen.tool.mechanism == 2 && last_y > first_y),
            });
            pitches.insert(pen.inclination_rad().to_degrees().round() as u32);
            for other in &pens[..i] {
                assert!(pen.bounds.separate(other.bounds));
            }
            let mut mesh = Assembly::default();
            pen.tool
                .draw(&mut mesh, pen.transform(), "stationery", "stationery");
            assert!(
                (mesh.bounds().0.y - p.floor_m).abs() < 1e-6,
                "floor {seed}/{i}"
            );
            let mut nearest = f32::INFINITY;
            let check = |pt: Vec3| {
                assert!(cup.edges.iter().all(|e|e.normal.dot(pt.xz())
                    <= e.distance*cup.inner_scale(pt.y)+1e-6),"wall {seed}/{i} {pt}");
            };
            for g in mesh.parts.values() {
                for tri in g.indices.as_chunks::<3>().0 {
                    let vs = tri.map(|j| Vec3::from(g.positions[j as usize]));
                    // A whole triangle must lie in one convex reservation.
                    // Disjoint reservations then imply disjoint rendered meshes.
                    assert!([Some(pen.bounds.barrel), pen.bounds.clip]
                        .into_iter()
                        .flatten()
                        .any(|c| vs.iter().all(|pt| wires::segment_distance_squared(
                            *pt, *pt, c.start, c.end
                        ) <= f64::from(c.radius + 1e-6).powi(2))));
                    for j in 0..3 {
                        let (a, b) = (vs[j], vs[(j + 1) % 3]);
                        if a.y <= cup.height {
                            check(a);
                        }
                        if (a.y < cup.height) != (b.y < cup.height) {
                            let cross = a.lerp(b, (cup.height - a.y) / (b.y - a.y));
                            check(cross);
                            nearest = nearest.min(cross.distance(pen.contact));
                        }
                    }
                }
            }
            assert!(
                nearest < 2e-6,
                "missing barrel contact {seed}/{i}: {nearest}"
            );
            let touches_shell = shell.indices.as_chunks::<3>().0.iter().any(|tri| {
                let v = tri.map(|j| Vec3::from(shell.positions[j as usize]));
                (0..3).any(|j| {
                    let (a, b) = (v[j], v[(j + 1) % 3]);
                    a.y == cup.height
                        && b.y == cup.height
                        && wires::segment_distance_squared(pen.contact, pen.contact, a, b) < 4e-12
                })
            });
            assert!(touches_shell, "missing rim contact {seed}/{i}");
        }
    }
    assert!(admitted * 100 > requested * 75, "{admitted}/{requested}");
    assert_eq!(products.len(), 6);
    assert!(counts.contains(&0) && counts.len() >= 9 && pitches.len() > 17);
}

#[test]
fn furnishing_finish_edits_preserve_the_settled_poses_and_individual_tool_geometry() {
    let mut varied = false;
    for seed in 0..32 {
        let mut o = audit::fixture(ObjectKind::PenHolder, seed);
        let original = contents(&o, &parameters(&o));
        assert_eq!(
            serde_json::to_value(&original).unwrap(),
            serde_json::to_value(contents(&o, &parameters(&o))).unwrap()
        );
        o.finish_seed = Some(9845);
        let recolored = contents(&o, &parameters(&o));
        assert_eq!(recolored.len(), original.len());
        for (a, b) in original.iter().zip(recolored) {
            assert_eq!(
                (a.base, a.top, a.contact, a.rotation),
                (b.base, b.top, b.contact, b.rotation)
            );
            assert_eq!(
                (a.tool.length, a.tool.radius, a.tool.mechanism),
                (b.tool.length, b.tool.radius, b.tool.mechanism)
            );
            varied |= a.finish_slot != b.finish_slot || a.tool.finish != b.tool.finish;
        }
    }
    assert!(varied);
}
