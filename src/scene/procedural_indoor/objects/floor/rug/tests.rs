use super::*;
use crate::scene::procedural_indoor::{
    layout::floor::area::RugSupport, materials::rugs, objects::audit,
};
use std::collections::BTreeMap;

#[test]
fn metric_binding_is_watertight_and_has_valid_uvs_on_every_face() {
    for seed in 0..32 {
        for cut in [
            Cut::Straight,
            Cut::Chamfered,
            Cut::Rounded,
            Cut::Oval,
            Cut::Capsule,
            Cut::Asymmetric,
        ] {
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
    let mut cuts = [0; 6];
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

#[test]
fn runner_proportions_change_the_cut_mix_without_an_axis_preference() {
    let compact = Vec3::new(2., 0.008, 2.);
    let long = Vec3::new(4.2, 0.008, 0.75);
    let mut counts = [[0; 6]; 2];
    for seed in 0..2048 {
        for (i, size) in [compact, long].into_iter().enumerate() {
            let cut = Cut::sample(seed, size);
            assert_eq!(cut, Cut::sample(seed, Vec3::new(size.z, size.y, size.x)));
            counts[i][cut as usize - Cut::Straight as usize] += 1;
        }
    }
    for group in counts {
        assert_eq!(group.iter().max(), Some(&group[0]));
        assert!(group.into_iter().all(|n| n > 30));
    }
    let capsule = Cut::Capsule as usize - Cut::Straight as usize;
    let asymmetric = Cut::Asymmetric as usize - Cut::Straight as usize;
    assert!(counts[1][capsule] > counts[0][capsule] * 2);
    assert!(counts[0][asymmetric] > counts[1][asymmetric] * 2);
}

#[test]
fn the_cut_supports_furniture_only_inside_its_rotated_footprint() {
    use crate::scene::procedural_indoor::envelope::polygon;
    for seed in 0..64 {
        for cut in [Cut::Capsule, Cut::Asymmetric] {
            let mut o = audit::fixture(ObjectKind::Rug, seed);
            o.variant = cut as u32;
            o.yaw = 0.63;
            o.position = Vec3::new(-3., 0.10, 7.);
            let contour = outline(&o);
            for i in 0..contour.len() {
                let [a, b, c] = [i, i + 1, i + 2].map(|i| contour[i % contour.len()]);
                assert!(
                    (b - a).perp_dot(c - b) > 0.,
                    "{cut:?} reversed or redundant edge"
                );
            }
            let support = RugSupport::from_object(&o);
            let point = |xz: Vec2| o.transform().transform_point(Vec3::new(xz.x, 0., xz.y));
            let corner = point(o.size.xz() * 0.499);
            let pad = Vec3::splat(0.04);
            assert!(support.supports(o.position - pad, o.position + pad));
            assert!(
                !support.contains(corner.xz()) && !support.supports(corner - pad, corner + pad)
            );
            let area = polygon::area(&contour);
            assert!(area > o.size.x * o.size.z * 0.55 && area < o.size.x * o.size.z);
        }
    }
}
