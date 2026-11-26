use super::{
    stock::{dress_carts, CartJob, ShelfLoad, INTRODUCED},
    *,
};
use crate::scene::procedural_indoor::{
    layout::IndoorLayout,
    motion::IndoorMotionConfig,
    objects::{audit, kitchen::contents},
};
use std::collections::BTreeSet;

#[test]
fn fitted_equipment_steers_inventory_and_keeps_unrelated_objects_independent() {
    let mut installed_beverage = 0;
    let mut workshop_beverage = 0;
    let mut kitchen_service = 0;
    let mut program = crate::scene::procedural_indoor::program::IndoorProgram::sample(
        3,
        Vec3::new(6., 3., 5.),
        IndoorLayout::Kitchen,
        0.72,
    );
    program.zones[0].activity = IndoorLayout::Kitchen;
    for seed in 0..192 {
        let mut cart = audit::fixture(ObjectKind::MobileCart, seed);
        cart.id = 7;
        let mut appliance = audit::fixture(ObjectKind::Kettle, seed + 1);
        appliance.id = 21;
        appliance.support = Some(cart.id);
        let mut items = [cart, appliance];
        let original = items[0].variant;
        dress_carts(&mut items, IndoorLayout::Workshop, None);
        let installed = items[0].variant;
        installed_beverage +=
            usize::from(CartJob::read(&items[0], INTRODUCED) == CartJob::Beverage);
        assert_eq!(installed & !stock::JOB_MASK, original);
        items.swap(0, 1);
        dress_carts(&mut items, IndoorLayout::Workshop, None);
        assert_eq!(items[1].variant, installed);
        items[0].support = None;
        dress_carts(&mut items, IndoorLayout::Workshop, None);
        workshop_beverage += usize::from(CartJob::read(&items[1], INTRODUCED) == CartJob::Beverage);
        dress_carts(&mut items, IndoorLayout::Kitchen, None);
        kitchen_service +=
            usize::from(CartJob::read(&items[1], INTRODUCED) == CartJob::TableService);
        let kitchen = items[1].variant;
        dress_carts(&mut items, IndoorLayout::Workshop, Some(&program));
        assert_eq!(items[1].variant, kitchen, "cart in a kitchenette zone");
        items[0].kind = ObjectKind::Printer;
        items[0].support = Some(items[1].id);
        dress_carts(&mut items, IndoorLayout::Kitchen, None);
        assert_eq!(items[1].variant, original);
    }
    assert!(
        (132..187).contains(&installed_beverage),
        "{installed_beverage}"
    );
    assert!((1..30).contains(&workshop_beverage), "{workshop_beverage}");
    assert!((65..135).contains(&kitchen_service), "{kitchen_service}");
}

#[test]
fn cart_service_cargo_is_grounded_inside_clear_bays_with_reproducible_finishes() {
    let mut styles = BTreeSet::new();
    let mut empty = 0;
    let mut multi = 0;
    let mut plate_stacks = 0;
    for job in [CartJob::Beverage, CartJob::TableService] {
        for seed in 0..128 {
            let mut cart = audit::fixture(ObjectKind::MobileCart, seed);
            let sizes = [[0.45, 0.58, 0.32], [0.91, 1.02, 0.59], [0.60, 0.76, 0.43]];
            cart.size = Vec3::from_array(sizes[seed as usize % sizes.len()]);
            job.write(&mut cart);
            let p = parameters(&cart);
            let mesh = build_object(&cart);
            audit::check(&cart, &mesh).unwrap();
            assert!((mesh.bounds().1.y - cart.size.y).abs() < 1e-5);
            for level in 0..p.shelf_tops.len() - 1 {
                let (lo, hi) = p.bay(cart.size, level);
                let sample = || match ShelfLoad::sample(&cart, &p, level, INTRODUCED) {
                    ShelfLoad::Service(items) => items,
                    ShelfLoad::Utility(_) => unreachable!(),
                };
                let items = sample();
                assert_eq!(items, sample());
                assert!(items.len() <= 3);
                empty += usize::from(items.is_empty());
                multi += usize::from(items.len() > 1);
                for (index, item) in items.iter().enumerate() {
                    let footprint = item.footprint();
                    assert!(footprint.inside(lo.xz(), hi.xz(), 0.00799));
                    assert!(items[..index]
                        .iter()
                        .all(|other| !footprint.overlaps(other.footprint(), 0.00799)));
                    let mut built = Assembly::default();
                    contents::build(&mut built, std::slice::from_ref(item));
                    let (bottom, top) = built.bounds();
                    assert!((bottom.y - lo.y).abs() < 1e-5 && top.y < hi.y);
                    assert!(bottom
                        .xz()
                        .cmpge(footprint.center - footprint.half - Vec2::splat(1e-5))
                        .all());
                    assert!(top
                        .xz()
                        .cmple(footprint.center + footprint.half + Vec2::splat(1e-5))
                        .all());
                    for key in built.parts.keys() {
                        assert!(mesh.parts.contains_key(key), "lost cargo surface {key:?}");
                    }
                    styles.insert((item.pantry, item.style));
                    plate_stacks += usize::from(!item.pantry && item.style == 0 && item.count > 1);
                }
            }
        }
    }
    assert_eq!(styles.len(), 11);
    assert!(empty > 50 && multi > 25 && plate_stacks > 10);
}

#[test]
fn cart_stock_follows_rigid_motion_and_archives_keep_utility_geometry() {
    let policy = IndoorMotionConfig {
        cart_fraction: 1.,
        ..Default::default()
    };
    for seed in 0..24 {
        let mut cart = audit::fixture(ObjectKind::MobileCart, seed);
        let old = build_object_at_version(&cart, None, INTRODUCED - 1);
        CartJob::Beverage.write(&mut cart);
        let replay = build_object_at_version(&cart, None, INTRODUCED - 1);
        assert_eq!(
            old.parts.keys().collect::<Vec<_>>(),
            replay.parts.keys().collect::<Vec<_>>()
        );
        for (key, mesh) in &old.parts {
            let restored = &replay.parts[key];
            assert_eq!(mesh.positions, restored.positions);
            assert_eq!(mesh.indices, restored.indices);
            assert_eq!(mesh.uvs, restored.uvs);
        }
        let still = build_object(&cart);
        let moving = build_object_at_version(&cart, Some(&policy), INTRODUCED);
        assert_eq!(moving.joints.len(), 1);
        assert_eq!(moving.joints[0].parts.len(), still.parts.len());
        let vertices = |a: &Assembly| {
            let mut v: Vec<_> = a
                .parts
                .values()
                .flat_map(|g| &g.positions)
                .map(|p| p.map(f32::to_bits))
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(vertices(&still), vertices(&moving));
    }
}
