use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn pads_are_closed_shells_with_metric_charts_inside_the_existing_clearance() {
    let mut stocks = BTreeSet::new();
    let mut gauges = (f32::MAX, f32::MIN);
    for seed in 0..224 {
        let o = audit::fixture(ObjectKind::Chair, seed);
        let p = parameters(&o);
        assert!(p.arm_pad(&o, INTRODUCED - 1).is_none());
        let Some(pad) = p.arm_pad(&o, INTRODUCED) else {
            continue;
        };
        stocks.insert(pad.stock.name());
        gauges = (gauges.0.min(pad.gauge_m), gauges.1.max(pad.gauge_m));
        let x = (o.size.x * p.seat_width_fraction).min(0.56) * 0.5 + 0.037;
        let (lo, hi) = *clearance_boxes(&o).last().unwrap();
        for side in [-1., 1.] {
            let mut a = Assembly::default();
            pad.build(&mut a, &p, side * x);
            let check = audit::check(&o, &a).unwrap();
            assert!(check.triangles < 200);
            let (min, max) = a.bounds();
            assert!(min.cmpge(lo).all() && max.cmple(hi).all());
            let mut edges = BTreeMap::new();
            for ((surface, label), g) in &a.parts {
                assert_eq!(label, "chair");
                assert!(*surface == pad.face || *surface == pad.lower_surface());
                for tri in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
                    assert!(
                        (b - a).perp_dot(c - a).abs() > 1e-11,
                        "collapsed texture chart"
                    );
                    let vertices = tri.map(|i| g.positions[i as usize].map(f32::to_bits));
                    for k in 0..3 {
                        let (a, b) = (vertices[k], vertices[(k + 1) % 3]);
                        let (forward, reverse) =
                            edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                        if a < b {
                            *forward += 1;
                        } else {
                            *reverse += 1;
                        }
                    }
                }
            }
            assert!(edges.values().all(|&n| n == (1, 1)), "open or doubled edge");
            let hits = super::super::tests::vertical_intersections(
                a.parts.values(),
                Vec2::new(side * x, pad.offset_z_m),
            );
            assert_eq!(hits.len(), 2);
            assert!((hits[1] - hits[0] - pad.gauge_m).abs() < 1e-5);
        }
    }
    assert_eq!(stocks.len(), 4);
    assert!(gauges.0 < 0.016 && gauges.1 > 0.042);
}

#[test]
fn all_six_carriers_reach_the_underside_without_breaking_the_upper_skin() {
    let o = audit::fixture(ObjectKind::Chair, 19);
    let mut p = parameters(&o);
    p.armrests = true;
    let x = 0.245;
    for width in [0.024, 0.052, 0.072] {
        for length in [0.16, 0.24, 0.33] {
            p.arm_pad_width_m = width;
            p.arm_pad_length = length;
            for support in 0..6 {
                p.arm_support = support;
                let mut pad = p.arm_pad(&o, INTRODUCED).unwrap();
                for sign in [-1., 1.] {
                    pad.offset_z_m = sign * 0.009;
                    pad.sweep_m = sign * width * 0.105;
                    let mut rest = Assembly::default();
                    let mut carrier = Assembly::default();
                    pad.build(&mut rest, &p, x);
                    upper::arm_support(&mut carrier, &p, x, Some(pad));
                    audit::check(&o, &carrier).unwrap();
                    let post = length * 0.18 + pad.offset_z_m;
                    let stations = match support {
                        0 | 2 => [post, post],
                        3 => [-0.31 * length, 0.31 * length],
                        5 => [-0.21 * length, 0.21 * length],
                        _ => [-0.04 * length, 0.04 * length],
                    };
                    for z in stations {
                        let at = Vec2::new(x, z);
                        let skins =
                            super::super::tests::vertical_intersections(rest.parts.values(), at);
                        let metal =
                            super::super::tests::vertical_intersections(carrier.parts.values(), at);
                        let crest = *metal.last().unwrap();
                        assert!(skins.len()==2 && crest>=skins[0]-0.0001 && crest<skins[1]-0.002,
                            "support {support}, length {length}, width {width}, z {z}, carrier {crest}, pad {skins:?}");
                    }
                    // All vertices entering the pad must be inside its actual silhouette.
                    for g in carrier.parts.values() {
                        for &v in &g.positions {
                            if v[1] > pad.underside(&p) + 0.00005 {
                                let skin = super::super::tests::vertical_intersections(
                                    rest.parts.values(),
                                    Vec2::new(v[0], v[2]),
                                );
                                assert!(skin.len()==2 && v[1]<skin[1], "exposed bearing: style {support}, width {width}, length {length}, v {v:?}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn shared_pad_tooling_allows_replacements_and_independent_adjustment() {
    let mut tooling = BTreeSet::new();
    let mut replacements = BTreeSet::new();
    let mut adjustments = BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 3;
        o.interaction_target = Some(27);
        let p = parameters(&o);
        let Some(pad) = p.arm_pad(&o, INTRODUCED) else {
            continue;
        };
        assert_eq!(p.arm_pad(&o, INTRODUCED), Some(pad));
        if pad.table_order {
            tooling.insert(pad.gauge_m.to_bits());
        } else {
            replacements.insert(pad.gauge_m.to_bits());
        }
        adjustments.insert(pad.offset_z_m.to_bits());
    }
    assert_eq!(tooling.len(), 1);
    assert!(replacements.len() > 12 && adjustments.len() > 112);
}
