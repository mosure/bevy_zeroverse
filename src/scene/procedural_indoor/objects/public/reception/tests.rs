use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};
use std::collections::BTreeSet;

// Trace a vertical ray into the completed triangles. This checks that layout contact
// heights and ledge limits describe the actual construction, including both wing hands.
fn highest_surface(a: &Assembly, point: Vec2) -> f32 {
    a.parts
        .values()
        .flat_map(|g| {
            g.indices.as_chunks::<3>().0.iter().filter_map(|tri| {
                let vertex = |i: usize| Vec3::from_array(g.positions[tri[i] as usize]);
                let [a, b, c] = [vertex(0), vertex(1), vertex(2)];
                if (a.y - b.y).abs() > 1e-5 || (a.y - c.y).abs() > 1e-5 {
                    return None;
                }
                let s = (b - a).xz();
                let t = (c - a).xz();
                let off = point - a.xz();
                let det = s.perp_dot(t);
                if det.abs() < 1e-8 {
                    return None;
                }
                let u = off.perp_dot(t) / det;
                let v = s.perp_dot(off) / det;
                (u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001).then_some(a.y)
            })
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

#[test]
fn reception_products_replay_with_physically_distinct_contact_surfaces() {
    let mut combinations = BTreeSet::new();
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::ReceptionDesk, seed);
        let p = Counter::sample(&o);
        let a = build_object(&o);
        let checked = audit::check(&o, &a).unwrap();
        assert!(checked.triangles < 6500);
        assert!(a.parts.keys().all(|(_, l)| part_label(l) == "counter"));
        let ledge = p.ledge(&o);
        let rear = p.worktop(&o);
        let mut sites = vec![(ledge.center(), o.size.y), (rear.center(), p.worktop_m)];
        if let Some(wing) = p.signing_wing(&o) {
            sites.push((Vec2::new(wing.center().x, ledge.center().y), p.worktop_m));
        }
        for (site, y) in sites {
            let e = Vec2::splat(0.036);
            assert!(p.supports(&o, y, site - e, site + e, 0.01));
            assert!(!p.supports(&o, y - 0.09, site - e, site + e, 0.));
            assert!(
                (highest_surface(&a, site) - y).abs() < 1e-4,
                "seed {seed}: {site:?} at {y}"
            );
        }
        assert!(!p.supports(
            &o,
            p.worktop_m,
            ledge.center() - 0.04,
            ledge.center() + 0.04,
            0.
        ));
        assert!(!p.supports(&o, o.size.y, rear.center() - 0.04, rear.center() + 0.04, 0.));
        let hand = (p.signing_wing_m > 0.) as i8 - (p.signing_wing_m < 0.) as i8;
        combinations.insert((hand, p.fascia as u8));
    }
    assert_eq!(combinations.len(), 9);
}
