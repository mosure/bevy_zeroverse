use super::*;
use crate::scene::procedural_indoor::{
    materials::{finish_slot, registered_finish},
    objects::{self, audit},
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn folded_napkins_have_two_woven_faces_an_open_fold_and_a_sealed_hem() {
    let mut designs = BTreeSet::new();
    for seed in 0..120 {
        let o = audit::fixture(ObjectKind::TableNapkin, seed);
        let p = Napkin::for_object(&o);
        let a = objects::build_object(&o);
        let check = audit::check(&o, &a).unwrap();
        assert!(
            check.max_envelope_overrun_m < 1e-6,
            "hem outside bounds at {seed}: {check:?}"
        );
        assert_eq!(check.triangles, 568);
        assert_eq!(a.parts.len(), 1);
        let ((surface, label), g) = a.parts.iter().next().unwrap();
        let at = |layer: usize, row: usize| {
            Vec3::from_array(g.positions[(layer * cloth::ROWS + row) * cloth::COLS + 5])
        };
        let thickness = |row| at(0, row).distance(at(1, row));
        assert!((at(0, 2).z - at(0, 0).z - p.hem_m).abs() < 1e-6);
        assert!(thickness(1) > 1.6 * thickness(3) && thickness(2) < 1.18 * thickness(3));
        assert_eq!(objects::part_label(label), o.kind.class_name());
        assert_eq!(finish_slot(*surface, label), Some(p.weave.slot()));
        assert!(registered_finish(*surface, p.weave.slot()));
        assert!(label.ends_with("#uv_local"));
        designs.insert((p.fold as u8, p.weave as u8));
        let (lo, hi) = a.bounds();
        assert_eq!(lo.y, 0.);
        assert!((hi.y - o.size.y).abs() < 2e-6);
        let mut edges = BTreeMap::new();
        let mut layers = Vec::new();
        let mut end_blocked = false;
        for indices in g.indices.as_chunks::<3>().0 {
            let pts = indices.map(|i| Vec3::from_array(g.positions[i as usize]));
            let keys = pts.map(|p| p.to_array().map(|v| if v == 0. { 0 } else { v.to_bits() }));
            for j in 0..3 {
                let (a, b) = (keys[j], keys[(j + 1) % 3]);
                let (edge, side) = if a < b { ((a, b), 0) } else { ((b, a), 1) };
                edges.entry(edge).or_insert([0; 2])[side] += 1;
            }
            if let Some(y) = crossing(
                pts,
                |p| p.xz(),
                |p| p.y,
                Vec2::new(0.006 * o.size.x, 0.017 * o.size.z),
            ) {
                layers.push(y);
            }
            end_blocked |=
                crossing(pts, |p| p.xy(), |p| p.z, Vec2::new(0., 0.26 * o.size.y)).is_some();
        }
        assert!(
            edges.values().all(|s| *s == [1, 1]),
            "open or overlapping edge at {seed}"
        );
        layers.sort_by(f32::total_cmp);
        if p.fold == Fold::Tent {
            assert!(!end_blocked && layers.len() == 2 && layers[0] > 0.50 * o.size.y);
        } else {
            assert_eq!(layers.len(), 4, "two distinct folded leaves at {seed}");
            assert!(layers[2] - layers[1] > 0.0003);
        }
    }
    assert_eq!(designs.len(), 12);
}

/// Intersection with a triangle viewed in the supplied plane. The orthogonal
/// coordinate distinguishes two fabric leaves and the space between them.
fn crossing(
    pts: [Vec3; 3],
    plane: impl Fn(Vec3) -> Vec2,
    depth: impl Fn(Vec3) -> f32,
    q: Vec2,
) -> Option<f32> {
    let [a, b, c] = pts.map(plane);
    let (ab, ac, aq) = (b - a, c - a, q - a);
    let det = ab.perp_dot(ac);
    if det.abs() < 1e-12 {
        return None;
    }
    let (u, v) = (aq.perp_dot(ac) / det, ab.perp_dot(aq) / det);
    (u >= 0. && v >= 0. && u + v <= 1.).then(|| {
        let [a, b, c] = pts.map(depth);
        a + (b - a) * u + (c - a) * v
    })
}
