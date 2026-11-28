use super::*;
use crate::scene::procedural_indoor::{
    motion::{IndoorMotionConfig, MotionKind},
    objects::{audit, build_object, build_object_with_motion, part_label},
};
use std::collections::BTreeSet;

#[test]
fn rolling_stock_has_floor_contact_an_open_rim_and_supported_semantic_loads() {
    let mut combinations = BTreeSet::new();
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::LaundryCart, seed);
        let p = LaundryCart::for_object(&o);
        let a = build_object(&o);
        let check = audit::check(&o, &a).unwrap();
        assert!(
            check.min_y_m.abs() < 1e-5 && check.max_envelope_overrun_m < 1e-5,
            "{seed} {check:?}"
        );
        assert!(check.triangles < 12500 && check.batches <= 10);
        assert!(!o.kind.is_surface());
        let expected = match p.load {
            Load::Empty => None,
            Load::LooseLaundry => Some("clothes"),
            Load::FoldedLinen => Some("towel"),
        };
        let mut lowest = f32::INFINITY;
        for ((_, label), g) in &a.parts {
            let class = part_label(label);
            if class == LABEL {
                continue;
            }
            assert_eq!(Some(class), expected);
            for &[x, y, z] in &g.positions {
                lowest = lowest.min(y);
                assert!(
                    y < p.rim_height_m && x.abs() < p.base_half_m.x && z.abs() < p.base_half_m.y
                );
            }
        }
        if expected.is_some() {
            let floor = match p.basket {
                Basket::Wire => {
                    p.base_height_m + 2.85 * (p.wire_pitch_m * 0.047).clamp(0.0024, 0.0036)
                }
                Basket::Canvas => p.base_height_m + 0.0032,
            };
            assert!(
                (lowest - floor).abs() < 1e-5,
                "{seed}: contact {lowest} vs {floor}"
            );
        } else {
            assert!(lowest.is_infinite());
        }
        // Across the top opening there is no sheet/box lid. Rail hardware
        // only crosses at the sides and, on tall carts, well above the basket.
        for g in a.parts.values() {
            for &t in g.indices.as_chunks::<3>().0 {
                let v = t.map(|i| Vec3::from(g.positions[i as usize]));
                let center = (v[0] + v[1] + v[2]) / 3.;
                assert!(
                    !((center.y - p.rim_height_m).abs() < 0.028
                        && center.x.abs() < p.rim_half_m.x * 0.55
                        && center.z.abs() < p.rim_half_m.y * 0.55)
                );
            }
        }
        assert_eq!(p.hanging_rail, o.size.y > 1.1);
        combinations.insert(format!("{:?}/{:?}/{}", p.basket, p.load, p.hanging_rail));
    }
    assert_eq!(combinations.len(), 9);
}

#[test]
fn single_cart_joint_includes_every_caster_liner_and_garment_batch() {
    let policy = IndoorMotionConfig {
        cart_fraction: 1.,
        ..default()
    };
    let off = IndoorMotionConfig {
        cart_fraction: 0.,
        ..default()
    };
    for seed in 0..12 {
        let o = audit::fixture(ObjectKind::LaundryCart, seed);
        let fixed = build_object(&o);
        assert!(
            fixed.joints.is_empty() && build_object_with_motion(&o, Some(&off)).joints.is_empty()
        );
        let moving = build_object_with_motion(&o, Some(&policy));
        assert_eq!(moving.joints.len(), 1);
        assert_eq!(moving.joints[0].kind, MotionKind::Cart);
        assert_eq!(moving.joints[0].parts.len(), fixed.parts.len());
        assert_eq!(moving.parts.len(), fixed.parts.len());
        assert!(moving
            .parts
            .iter()
            .all(|(k, g)| moving.joints[0].parts.contains(k) && !g.indices.is_empty()));
    }
}
