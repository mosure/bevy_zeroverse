use super::*;
use crate::scene::procedural_indoor::{
    items::sizing,
    objects::{audit, build_object},
};
use std::collections::BTreeSet;

#[test]
fn stationery_meshes_replay_and_contact_support_across_metric_sizes() {
    for kind in [ObjectKind::Notepad, ObjectKind::Pencil] {
        for seed in 0..128 {
            let mut o = audit::fixture(kind, seed);
            o.size = sizing::loose(kind, &mut stream(seed, 9876)).unwrap();
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            let (lo, hi) = a.bounds();
            assert!(lo.y.abs() < 0.00005, "{kind:?}/{seed}: support {lo:?}");
            assert!(lo
                .cmpge(Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5) - Vec3::splat(0.00005))
                .all());
            assert!(hi
                .cmple(Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5) + Vec3::splat(0.00005))
                .all());
            let replay = build_object(&o);
            for (key, g) in a.parts {
                let other = &replay.parts[&key];
                assert_eq!(g.positions, other.positions);
                assert_eq!(g.indices, other.indices);
                assert_eq!(g.uvs, other.uvs);
            }
        }
    }
}
#[test]
fn pad_top_is_one_complete_document_without_coplanar_underlay() {
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::Notepad, seed);
        let a = build_object(&o);
        let g = a
            .parts
            .iter()
            .find(|((s, _), _)| *s == Surface::PrintedPaper)
            .unwrap()
            .1;
        let lo = g
            .uvs
            .iter()
            .map(|p| Vec2::from_array(*p))
            .reduce(Vec2::min)
            .unwrap();
        let hi = g
            .uvs
            .iter()
            .map(|p| Vec2::from_array(*p))
            .reduce(Vec2::max)
            .unwrap();
        assert!(lo.cmpge(Vec2::ZERO).all() && lo.cmplt(Vec2::splat(0.005)).all());
        assert!(hi.cmple(Vec2::ONE).all() && hi.cmpgt(Vec2::splat(0.995)).all());
        let top = g.positions[0][1];
        for ((surface, _), other) in &a.parts {
            if *surface != Surface::Paper {
                continue;
            }
            for tri in other.indices.as_chunks::<3>().0 {
                assert!(
                    !tri.iter()
                        .all(|&i| (other.positions[i as usize][1] - top).abs() < 1e-6
                            && other.normals[i as usize][1] > 0.999),
                    "duplicate top at {seed}"
                );
            }
        }
    }
}
#[test]
fn stationery_sampling_varies_construction_without_changing_it_with_finish() {
    let mut bindings = BTreeSet::new();
    let mut mechanisms = BTreeSet::new();
    let mut heights = BTreeSet::new();
    let mut radii = BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Notepad, seed);
        let p = pad::parameters(&o);
        bindings.insert(p.binding);
        heights.insert((p.height / o.size.y * 100.) as u32);
        o.finish_seed = Some(999);
        let q = pad::parameters(&o);
        assert_eq!(p.span, q.span);
        assert_eq!(p.height, q.height);
        assert_eq!(p.binding, q.binding);
        o = audit::fixture(ObjectKind::Pencil, seed);
        let p = writing::parameters(&o);
        mechanisms.insert(p.mechanism);
        radii.insert((p.radius / o.size.x * 1000.) as u32);
        o.finish_seed = Some(999);
        let q = writing::parameters(&o);
        assert_eq!(p.length, q.length);
        assert_eq!(p.radius, q.radius);
        assert_eq!(p.clip, q.clip);
    }
    assert_eq!(bindings.len(), 5);
    assert_eq!(mechanisms.len(), 3);
    assert!(heights.len() > 40 && radii.len() > 90);
}
