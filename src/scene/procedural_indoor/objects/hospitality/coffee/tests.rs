use super::*;
use crate::scene::procedural_indoor::objects::{self, audit};

#[test]
fn brewers_replay_inside_scaled_envelopes_with_appliance_semantics() {
    let mut styles = std::collections::BTreeSet::new();
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::CoffeeMaker, seed);
        styles.insert(format!("{:?}", parameters(&o).brewer));
        for scale in [
            Vec3::ONE,
            Vec3::new(0.75, 1.25, 0.85),
            Vec3::new(1.30, 0.85, 1.10),
        ] {
            o.size = dimensions(seed) * scale;
            let a = objects::build_object(&o);
            let check = audit::check(&o, &a).unwrap();
            assert!(check.triangles < 6000, "seed={seed} {check:?}");
            assert!(a.bounds().0.y >= -1e-6);
            assert!(a
                .parts
                .keys()
                .all(|(_, key)| objects::part_label(key) == "appliance"));
            let replay = objects::build_object(&o);
            assert_eq!(a.parts.len(), replay.parts.len());
            for (key, g) in a.parts {
                let b = &replay.parts[&key];
                assert_eq!(g.positions, b.positions);
                assert_eq!(g.normals, b.normals);
                assert_eq!(g.uvs, b.uvs);
                assert_eq!(g.indices, b.indices);
            }
        }
    }
    assert_eq!(styles.len(), 4);
}

#[test]
fn brewer_dimensions_follow_the_constructed_machine_family() {
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::CoffeeMaker, seed);
        let s = dimensions(seed);
        match parameters(&o).brewer {
            Brewer::Capsule => assert!((0.17..0.26).contains(&s.x)),
            Brewer::Espresso => assert!((0.28..0.46).contains(&s.x)),
            Brewer::Automatic => assert!((0.25..0.37).contains(&s.x)),
            Brewer::Filter => assert!((0.22..0.33).contains(&s.x)),
        }
    }
}
