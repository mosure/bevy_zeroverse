use super::*;
use crate::scene::procedural_indoor::{
    layout::floor::area::RugSupport, materials::rugs, objects::audit,
};
use std::collections::BTreeMap;

#[test]
fn metric_binding_is_watertight_and_has_valid_uvs_on_every_face() {
    for seed in 0..32 {
        for cut in [Cut::Straight, Cut::Chamfered, Cut::Rounded, Cut::Oval] {
            for size in [
                Vec3::new(4.2, 0.005, 0.58),
                Vec3::new(0.45, 0.008, 3.8),
                Vec3::new(2.2, 0.010, 2.2),
            ] {
                let mut o = audit::fixture(ObjectKind::Rug, seed);
                o.size = size;
                o.variant = cut as u32;
                let mut a = Assembly::default();
                build(&mut a, &o);
                let stats = audit::check(&o, &a).unwrap();
                assert_eq!(stats.batches, 2);
                let contour = outline(&o);
                let support = RugSupport::from_object(&o);
                assert_eq!(support.outline, contour);
                let program = parameters(&o).unwrap();
                let mut edges = BTreeMap::new();
                let mut binding_widths = 0;
                for ((_, label), g) in &a.parts {
                    let binding = rugs::key(label).unwrap().1;
                    for tri in g.indices.as_chunks::<3>().0 {
                        let points = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                        let uv = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
                        assert!(
                            (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-12,
                            "{cut:?}: collapsed UV triangle"
                        );
                        let normal = (points[1] - points[0]).cross(points[2] - points[0]);
                        assert!(normal.dot(Vec3::from_array(g.normals[tri[0] as usize])) > 0.);
                        for i in 0..3 {
                            let from = points[i].to_array().map(|v| (v * 1e7).round() as i32);
                            let to = points[(i + 1) % 3]
                                .to_array()
                                .map(|v| (v * 1e7).round() as i32);
                            let (key, sign) = if from < to {
                                ((from, to), 1)
                            } else {
                                ((to, from), -1)
                            };
                            let e = edges.entry(key).or_insert((0, 0));
                            e.0 += 1;
                            e.1 += sign;
                        }
                        if binding && normal.y > 0. && points.iter().all(|p| p.y == size.y) {
                            let outer: Vec<_> = points
                                .iter()
                                .filter(|p| contour.contains(&p.xz()))
                                .collect();
                            if let [a, b] = outer.as_slice() {
                                let inside =
                                    points.iter().find(|p| !contour.contains(&p.xz())).unwrap();
                                let dist = (a.xz() - b.xz())
                                    .normalize()
                                    .perp_dot(inside.xz() - b.xz())
                                    .abs();
                                assert!(
                                    (dist - program.binding_width_m).abs() < 2e-6,
                                    "{cut:?} edge width {dist}"
                                );
                                binding_widths += 1;
                            }
                        }
                    }
                }
                assert_eq!(binding_widths, contour.len());
                assert!(
                    edges.values().all(|&e| e == (2, 0)),
                    "open edge or overlapping face"
                );
            }
        }
    }
}

#[test]
fn cut_and_binding_replay_independently_of_dye_and_legacy_outlines() {
    let mut cuts = [0; 4];
    let mut bindings = [0; 3];
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Rug, seed);
        let first = audit::program(&o);
        let p = parameters(&o).unwrap();
        cuts[p.cut as usize - Cut::Straight as usize] += 1;
        bindings[p.binding as usize] += 1;
        let shape = outline(&o);
        o.finish_seed = Some(seed.wrapping_add(19));
        assert_eq!(outline(&o), shape);
        assert_eq!(audit::program(&o), first);
        assert!(first["binding_width_m"].is_number());
        o.variant = (seed % 4) as u32;
        assert!(parameters(&o).is_none());
        assert_eq!(outline(&o), legacy::outline(&o));
        assert!(audit::program(&o)["rug_exponent"].is_number());
    }
    assert!(cuts.iter().all(|&n| n >= 12) && bindings.iter().all(|&n| n >= 10));
}
