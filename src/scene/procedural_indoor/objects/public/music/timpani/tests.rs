use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
use std::collections::BTreeSet;

#[test]
fn concert_kettles_are_grounded_and_the_head_and_hoop_share_one_bearing_edge() {
    let mut diameters = BTreeSet::new();
    let mut stocks = BTreeSet::new();
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::Timpani, seed);
        let t = Timpani::sample(&o);
        let a = build_object(&o);
        let check = audit::check(&o, &a).unwrap();
        assert!(check.min_y_m.abs() < 0.0001 && check.batches <= 6);
        assert!(a.parts.values().map(|g| g.indices.len() / 3).sum::<usize>() < 3600);
        assert!((t.head_radius_m - f32::from(t.nominal_inches) * 0.0127).abs() < 0.00001);
        let g = &a.parts[&(Surface::Plastic, drum_stock::HEAD_LABEL.to_owned())];
        let h = t.head_height_m;
        let c = t.bowl_center_z_m;
        let on_rim = |v: [f32; 3]| {
            (v[1] - h).abs() < 0.00001
                && (Vec2::new(v[0], v[2] - c).length() - t.head_radius_m).abs() < 0.00001
        };
        let head_edge: Vec<_> = g.positions.iter().filter(|&&v| on_rim(v)).collect();
        assert!(head_edge.len() >= 40);
        assert!(head_edge.iter().all(|point| a
            .parts
            .iter()
            .filter(|((s, _), _)| *s == Surface::Chrome)
            .any(|(_, g)| g.positions.contains(point))));
        for tri in g.indices.as_chunks::<3>().0 {
            let [p0, p1, p2] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            if [p0, p1, p2].iter().all(|p| (p.y - h).abs() < 0.00001) {
                assert!((p1 - p0).cross(p2 - p0).y > 0.);
            }
        }
        let mut contacts: Vec<Vec2> = Vec::new();
        for g in a.parts.values() {
            for v in &g.positions {
                let p = Vec3::from_array(*v);
                if p.y.abs() < 0.0001 && !contacts.iter().any(|q| p.xz().distance(*q) < 0.1) {
                    contacts.push(p.xz());
                }
            }
        }
        assert_eq!(contacts.len(), 3);
        let (surface, label) = drum_stock::bowl_label(t.bowl_stock);
        let bowl = &a.parts[&(surface, label.to_owned())];
        for ids in bowl.indices.as_chunks::<3>().0 {
            let [p0, p1, p2] = ids.map(|j| Vec3::from_array(bowl.positions[j as usize]));
            let normals = ids.map(|j| Vec3::from_array(bowl.normals[j as usize]));
            assert!((p1 - p0).cross(p2 - p0).dot(normals.into_iter().sum()) > 0.);
        }
        diameters.insert(t.nominal_inches);
        stocks.insert(t.bowl_stock as usize);
    }
    assert_eq!(diameters, BTreeSet::from(SIZES));
    assert_eq!(stocks.len(), 4);
}
