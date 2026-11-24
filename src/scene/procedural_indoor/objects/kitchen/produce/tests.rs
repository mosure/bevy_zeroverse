use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::BTreeSet;

#[test]
fn packed_fruit_rests_on_the_floor_and_clears_the_actual_carrier_and_other_fruit() {
    let mut kinds = BTreeSet::new();
    let mut carriers = BTreeSet::new();
    let mut counts = BTreeSet::new();
    for seed in 0..512 {
        let o = audit::fixture(ObjectKind::FruitBowl, seed);
        let arrangement = Arrangement::sample(&o);
        let v = &arrangement.vessel;
        carriers.insert(v.kind as usize);
        counts.insert(arrangement.fruit.len());
        let mut complete = Assembly::default();
        v.build(&mut complete, &o);
        assert!(complete.bounds().0.y.abs() < 1e-6);
        for piece in &arrangement.fruit {
            let support_y = super::super::tests::top_at(&complete, piece.center_m);
            assert!((support_y - v.floor_m).abs() < 1e-5, "seed {seed}");
        }
        for (i, piece) in arrangement.fruit.iter().enumerate() {
            kinds.insert(piece.variety as usize);
            for other in &arrangement.fruit[..i] {
                assert!(
                    piece.center_m.distance(other.center_m)
                        > piece.clearance_radius() + other.clearance_radius()
                );
            }
            let mut single = Assembly::default();
            piece.build(&mut single, v.floor_m);
            assert!((single.bounds().0.y - v.floor_m).abs() < 1e-6);
            for part in single.parts.values() {
                for vertex in &part.positions {
                    let p = Vec3::from_array(*vertex);
                    if p.y <= v.rim_y_m {
                        let radius = (p.xz() / v.rim_axes_m).length();
                        assert!(
                            radius <= v.clear_fraction(p.y - v.floor_m),
                            "seed {seed}, {p:?}"
                        );
                    }
                }
            }
            piece.build(&mut complete, v.floor_m);
        }
        audit::check(&o, &complete).unwrap();
    }
    assert_eq!((kinds.len(), carriers.len()), (5, 3));
    assert!(counts.contains(&0) && counts.contains(&1) && counts.last() >= Some(&5));
}

#[test]
fn changing_the_finish_preserves_grocery_sizes_arrangement_and_support() {
    let mut changed_skins = 0;
    for seed in 100..228 {
        let mut o = audit::fixture(ObjectKind::FruitBowl, seed);
        let original = Arrangement::sample(&o);
        o.finish_seed = Some(seed ^ 0x63ef_112d);
        let refinished = Arrangement::sample(&o);
        assert_eq!(original.vessel.kind, refinished.vessel.kind);
        assert_eq!(original.vessel.rim_axes_m, refinished.vessel.rim_axes_m);
        assert_eq!(original.vessel.floor_m, refinished.vessel.floor_m);
        assert_eq!(original.fruit.len(), refinished.fruit.len());
        for (p, q) in original.fruit.iter().zip(&refinished.fruit) {
            changed_skins += usize::from(p.skin != q.skin);
            assert_eq!(
                (p.variety, p.center_m, p.axes_m),
                (q.variety, q.center_m, q.axes_m)
            );
            assert_eq!(
                (p.height_m, p.yaw, p.phase, p.lean_m, p.lobing, p.stem_m),
                (q.height_m, q.yaw, q.phase, q.lean_m, q.lobing, q.stem_m)
            );
        }
        let metadata = super::super::metadata_for_version(&o, INTRODUCED);
        assert_eq!(metadata["fruit_count"], refinished.fruit.len());
        assert_eq!(metadata["basket"], refinished.vessel.is_basket());
        assert_eq!(metadata["fruit"], serde_json::json!(refinished.fruit));
        let old = props::program(&o);
        assert_eq!(
            super::super::metadata_for_version(&o, INTRODUCED - 1),
            serde_json::json!({"basket":old.basket, "fruit_count":old.count})
        );
    }
    assert!(changed_skins > 60);
}
