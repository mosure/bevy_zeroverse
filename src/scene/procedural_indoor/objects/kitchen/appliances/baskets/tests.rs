use super::*;
use crate::scene::procedural_indoor::{envelope::ray_triangle, objects::audit};
use std::collections::BTreeSet;

fn metal_hit(g: &Geometry, x: f32, z: f32, height: f32) -> Option<f32> {
    let eye = Vec3::new(x, height, z);
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            ray_triangle(
                eye,
                Vec3::NEG_Y,
                t.map(|i| Vec3::from(g.positions[i as usize])),
            )
        })
        .reduce(f32::min)
        .map(|d| height - d)
}

#[test]
fn dishes_rest_on_rendered_cross_wires_and_keep_their_slots_and_clearances() {
    let mut checked = 0;
    let mut pitches = BTreeSet::new();
    for seed in 0..70 {
        let o = audit::fixture(ObjectKind::Dishwasher, seed);
        let p = parameters(&o);
        for basket in interior::dishwasher_baskets(&o, &p)
            .into_iter()
            .filter(|b| !b.cutlery)
        {
            let old = slots(&basket, true);
            let old_plates = old.plates.clone();
            let wash = WashBasket::sample(old, &basket, true);
            let mut metal = Geometry::default();
            wash.wirework(&mut metal, Vec3::ZERO);
            assert_eq!(wash.rack.plates.len(), old_plates.len());
            pitches.insert(((wash.cross_z[1] - wash.cross_z[0]) * 1000.).round() as i32);
            for (dish, previous) in wash.rack.plates.iter().zip(old_plates) {
                assert_eq!(
                    (dish.slot, &dish.dish, dish.rotation(), dish.position.x),
                    (
                        previous.slot,
                        &previous.dish,
                        previous.rotation(),
                        previous.position.x
                    )
                );
                let mut rendered = Assembly::default();
                dishware::build(
                    &mut rendered,
                    dish.transform(),
                    &dish.dish,
                    dish.finish_seed,
                );
                let (lo, hi) = rendered.bounds();
                let expected = 4. * wash.radius;
                assert!((lo.y - expected).abs() < 2e-6 && hi.y <= wash.rack.size.y + 2e-6);
                assert!(lo.x >= wash.rack.rail_x[dish.slot] + wash.radius);
                assert!(hi.x <= wash.rack.rail_x[dish.slot + 1] - wash.radius);
                assert!(
                    lo.z >= -wash.rack.size.z * HALF_DEPTH && hi.z <= wash.rack.size.z * HALF_DEPTH
                );
                // Use the actual dish vertices and triangles from the rendered wire bed.
                let touches = rendered
                    .parts
                    .values()
                    .flat_map(|g| &g.positions)
                    .filter(|v| v[1] <= lo.y + 2e-6)
                    .any(|v| {
                        metal_hit(&metal, v[0], v[2], lo.y + 0.001)
                            .is_some_and(|y| (y - lo.y).abs() < 2e-5)
                    });
                assert!(touches, "unsupported plate {seed}/{}", dish.slot);
                checked += 1;
            }
        }
    }
    assert!(checked > 180 && pitches.len() >= 8);
}

#[test]
fn cutlery_occupies_separate_supported_lanes_and_keeps_its_semantics() {
    let mut heads = BTreeSet::new();
    let mut counts = BTreeSet::new();
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::Dishwasher, seed);
        let p = parameters(&o);
        for mut basket in interior::dishwasher_baskets(&o, &p)
            .into_iter()
            .filter(|b| b.cutlery)
        {
            basket.position = Vec3::ZERO;
            let mut wash = WashBasket::sample(slots(&basket, true), &basket, true);
            assert!(WashBasket::sample(slots(&basket, false), &basket, false)
                .utensils
                .is_empty());
            assert!(wash.rack.plates.is_empty());
            counts.insert(wash.utensils.len());
            let mut bed = Geometry::default();
            wash.wirework(&mut bed, basket.position);
            let mut lanes = BTreeSet::new();
            for utensil in std::mem::take(&mut wash.utensils) {
                let lane = utensil.lane;
                assert!(lanes.insert(lane));
                let item = IndoorObject {
                    seed: utensil.seed,
                    finish_seed: Some(basket.seed),
                    ..o.clone()
                };
                heads.insert(flatware::parameters_at(&item, INTRODUCED).tines);
                wash.utensils.push(utensil);
                let mut built = Assembly::default();
                wash.build(&mut built, &o, &basket, INTRODUCED);
                wash.utensils.clear();
                built
                    .parts
                    .retain(|(_, label), _| label.starts_with("kitchen_utensil#"));
                assert!(!built.parts.is_empty());
                assert!(built.parts.keys().all(|(s, _)| matches!(
                    s,
                    Surface::Chrome | Surface::Metal | Surface::Plastic
                )));
                let (lo, hi) = built.bounds();
                let crown = 4. * wash.radius;
                assert!(lo.x > wash.rack.rail_x[lane] + wash.radius);
                assert!(hi.x < wash.rack.rail_x[lane + 1] - wash.radius);
                assert!(
                    lo.z > -wash.rack.size.z * HALF_DEPTH && hi.z < wash.rack.size.z * HALF_DEPTH
                );
                assert!(lo.y >= crown - 0.0001 && hi.y < basket.size.y);
                assert!(
                    built
                        .parts
                        .values()
                        .flat_map(|g| &g.positions)
                        .filter(|v| v[1] < crown + 0.0001)
                        .any(|v| metal_hit(&bed, v[0], v[2], crown + 0.001)
                            .is_some_and(|y| (y - v[1]).abs() < 0.0001)),
                    "flatware misses the runner {seed}/{lane}"
                );
            }
        }
    }
    assert_eq!(heads, BTreeSet::from([0, 3, 4]));
    assert!(counts.contains(&0) && counts.contains(&5) && counts.len() >= 5);
}

#[test]
fn wire_openings_are_empty_in_the_mesh_and_archived_trays_remain_solid() {
    let mut kinds = BTreeSet::new();
    for seed in 0..32 {
        let o = audit::fixture(ObjectKind::Dishwasher, seed);
        let p = parameters(&o);
        for basket in interior::dishwasher_baskets(&o, &p) {
            if kinds.contains(&basket.cutlery) {
                continue;
            }
            let slots = slots(&basket, false);
            let mut legacy = Assembly::default();
            supplies::rack::build_with_label(
                &mut legacy,
                Transform::IDENTITY,
                &slots,
                basket.seed,
                "dishwasher",
            );
            let wash = WashBasket::sample(slots, &basket, false);
            let mut bed = Geometry::default();
            wash.wirework(&mut bed, Vec3::ZERO);
            let height = wash.rack.size.y;
            let mut open = 0;
            for i in 0..23 {
                for j in 0..23 {
                    let x = ((i as f32 + 0.37) / 23. - 0.5) * wash.rack.size.x * 0.85;
                    let z = ((j as f32 + 0.61) / 23. - 0.5) * wash.rack.size.z * 0.85;
                    open += usize::from(metal_hit(&bed, x, z, height).is_none());
                    assert!(legacy
                        .parts
                        .values()
                        .any(|g| metal_hit(g, x, z, height).is_some()));
                }
            }
            assert!((200..500).contains(&open), "{open}/529 open rays");
            assert!(bed.indices.len() / 3 < 2600);
            kinds.insert(basket.cutlery);
        }
        assert!(metadata_for_version(&o, &p, INTRODUCED - 1)["baskets"][0]
            .get("wash_grid")
            .is_none());
        let current = metadata_for_version(&o, &p, INTRODUCED);
        assert!(
            current["baskets"][0]["wash_grid"]["cross_wires_z_m"]
                .as_array()
                .unwrap()
                .len()
                >= 9
        );
        if kinds.len() == 2 {
            break;
        }
    }
    assert_eq!(kinds.len(), 2);
}
