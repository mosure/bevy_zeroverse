use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    motion::IndoorMotionConfig,
    validation,
};

#[test]
fn convectors_replay_fit_their_bounds_and_remain_static_obstacles() {
    let mut styles = std::collections::BTreeSet::new();
    for seed in 0..128 {
        let o = super::super::super::audit::fixture(ObjectKind::BaseboardHeater, seed);
        let p = parameters(&o);
        styles.insert((format!("{:?}", p.surface), p.hydronic));
        let a = super::super::super::build_object(&o);
        super::super::super::audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
        assert!(a
            .parts
            .keys()
            .all(|(_, label)| label.split('#').next() == Some("appliance")));
        assert_eq!(o.size, dimensions(seed));
        let motion = super::super::super::build_object_with_motion(
            &o,
            Some(&IndoorMotionConfig {
                appliance_fraction: 1.,
                ..default()
            }),
        );
        assert!(motion.joints.is_empty());
        assert_eq!(
            a.parts.values().map(|g| g.positions.len()).sum::<usize>(),
            motion
                .parts
                .values()
                .map(|g| g.positions.len())
                .sum::<usize>()
        );
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-5 && hi.y <= o.size.y + 1e-5);
    }
    assert_eq!(styles.len(), 6);
}

#[test]
fn placed_heaters_are_optional_supported_annotated_and_reject_collisions() {
    let mut count = 0;
    let mut rooms = 0;
    for seed in 0..128 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let units: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::BaseboardHeater)
            .collect();
        rooms += usize::from(!units.is_empty());
        count += units.len();
        for o in units {
            assert!(o.solid && o.support.is_none() && !o.neighbor);
            assert!(scene.placement_clear(o, 0.));
            assert_eq!(o.kind.class_name(), "appliance");
            let mut overlap = o.clone();
            overlap.id = usize::MAX;
            assert!(!scene.placement_clear(&overlap, 0.));
        }
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
    }
    assert!((8..70).contains(&rooms), "{rooms} rooms, {count} heaters");
    assert!(count >= rooms && count <= 4 * rooms);
}
