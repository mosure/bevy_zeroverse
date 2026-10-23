use super::*;
use crate::scene::procedural_indoor::{objects::audit, validation};
use std::collections::BTreeSet;
mod qualification;

fn empty_workroom() -> IndoorManifest {
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    scene.envelope = None;
    scene.ceiling_installation = None;
    scene.program = None;
    scene.room_size = Vec3::new(20., 5., 20.);
    scene
}

#[test]
fn refreshment_equipment_mix_is_not_a_fixed_pair_or_fixed_left_right_template() {
    let base = empty_workroom();
    let mut paired = 0;
    let mut count = 0;
    let mut x_bins = BTreeSet::new();
    let mut z_bins = BTreeSet::new();
    let mut occupancies = BTreeSet::new();
    for seed in 0..128 {
        let mut scene = base.clone();
        let mut counter = audit::fixture(ObjectKind::ServiceCounter, seed);
        counter.yaw = 0.37;
        counter.position = Vec3::new(1., 0.15, -1.);
        scene.objects.push(counter.clone());
        let required = if seed % 2 == 0 {
            ObjectKind::CoffeeMaker
        } else {
            ObjectKind::DrinkDispenser
        };
        scene.dress_refreshment_counter(&counter, Some(required));
        let mut replay = base.clone();
        replay.objects.push(counter.clone());
        replay.dress_refreshment_counter(&counter, Some(required));
        assert_eq!(scene.objects, replay.objects);
        let primary = scene
            .objects
            .iter()
            .find(|o| o.support == Some(0) && o.kind == required)
            .expect("mandatory service equipment must fit");
        let local = counter
            .transform()
            .compute_affine()
            .inverse()
            .transform_point3(primary.position);
        x_bins.insert((local.x / counter.size.x * 10.).floor() as i32);
        z_bins.insert((local.z / counter.size.z * 10.).floor() as i32);
        paired += usize::from(
            [ObjectKind::CoffeeMaker, ObjectKind::DrinkDispenser]
                .iter()
                .all(|k| scene.objects.iter().any(|o| o.kind == *k)),
        );
        occupancies.insert(scene.objects.len());
        for o in scene.objects.iter().skip(1) {
            assert_eq!(o.support, Some(0));
            assert!((o.position.y - counter.position.y - counter.size.y).abs() < 1e-5);
            assert!(scene.prop_clear(o, &counter, 0.0119), "{seed}/{:?}", o.kind);
            audit::check(o, &objects::build_object(o)).unwrap();
            count += 1;
        }
    }
    assert!((16..100).contains(&paired), "paired={paired}/128");
    assert!(
        x_bins.len() >= 7 && z_bins.len() >= 3,
        "x={x_bins:?} z={z_bins:?}"
    );
    assert!(occupancies.len() >= 5, "occupancies={occupancies:?}");
    eprintln!("128 refreshment worktops: {paired} coffee/dispenser pairs, {count} supported items, x/z bins={}/{}, occupancy counts={occupancies:?}", x_bins.len(), z_bins.len());
}

#[test]
fn kitchen_and_dining_programs_keep_required_equipment_and_valid_geometry() {
    for profile in [
        IndoorLayout::Kitchen,
        IndoorLayout::Kitchenette,
        IndoorLayout::Cafe,
        IndoorLayout::Cafeteria,
    ] {
        for seed in 0..16 {
            let scene = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.).unwrap();
            validation::validate_layout(&scene).unwrap();
            for o in &scene.objects {
                if o.kind.is_kitchen()
                    || matches!(
                        o.kind,
                        ObjectKind::CoffeeMaker
                            | ObjectKind::Kettle
                            | ObjectKind::DrinkDispenser
                            | ObjectKind::ServiceCounter
                    )
                {
                    audit::check(o, &objects::build_object(o))
                        .unwrap_or_else(|error| panic!("{profile:?}/{seed}: {error}"));
                }
            }
            if seed < 2 {
                validation::validate_geometry(&scene).unwrap();
            }
        }
    }
}

#[test]
fn cart_refreshments_use_varied_realistic_supported_placements() {
    let base = empty_workroom();
    let mut counts = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    let mut x_bins = BTreeSet::new();
    let mut active = 0;
    for seed in 0..128 {
        let mut scene = base.clone();
        let mut cart = audit::fixture(ObjectKind::MobileCart, seed);
        cart.size = Vec3::new(0.76, 0.84, 0.49);
        cart.yaw = 0.41;
        scene.objects.push(cart.clone());
        scene.appliance_group(&cart);
        active += usize::from(scene.objects.len() > 1);
        counts.insert(scene.objects.len() - 1);
        for item in scene.objects.iter().skip(1) {
            assert!(
                scene.prop_clear(item, &cart, 0.0119),
                "{seed}/{:?}",
                item.kind
            );
            assert_eq!(item.support, Some(cart.id));
            audit::check(item, &objects::build_object(item)).unwrap();
            kinds.insert(format!("{:?}", item.kind));
            let local = cart
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(item.position);
            x_bins.insert((local.x / cart.size.x * 10.).floor() as i32);
        }
        let mut replay = base.clone();
        replay.objects.push(cart.clone());
        replay.appliance_group(&cart);
        assert_eq!(scene.objects, replay.objects);
    }
    assert!((30..110).contains(&active), "active={active}");
    assert!(
        counts.len() >= 4 && kinds.len() >= 6 && x_bins.len() >= 5,
        "counts={counts:?}, kinds={kinds:?}, x_bins={x_bins:?}"
    );
    eprintln!("128 carts: active={active}, counts={counts:?}, kinds={kinds:?}, x_bins={x_bins:?}");
}

#[test]
fn optional_cart_assemblies_respect_scene_clutter_thinning() {
    let mut base = empty_workroom();
    base.program = IndoorManifest::generate_with_humans(0, IndoorLayout::Kitchen, 0.65, 0, 0.)
        .unwrap()
        .program;
    let mut counts = [0; 2];
    for seed in 0..128 {
        let mut cart = audit::fixture(ObjectKind::MobileCart, seed);
        cart.size = Vec3::new(0.76, 0.84, 0.49);
        for (i, clutter) in [0., 1.].into_iter().enumerate() {
            let mut scene = base.clone();
            scene
                .program
                .as_mut()
                .unwrap()
                .domain
                .as_mut()
                .unwrap()
                .clutter = clutter;
            scene.objects.push(cart.clone());
            scene.appliance_group(&cart);
            counts[i] += usize::from(scene.objects.len() > 1);
        }
    }
    assert!(counts[0] > 0 && counts[0] * 4 < counts[1], "{counts:?}");
}
