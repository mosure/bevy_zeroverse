use super::*;
use crate::scene::procedural_indoor::{
    objects::{
        self,
        kitchen::{dishware, flatware},
    },
    program::relationships::{Participant, Relationship},
    validation,
};
use std::cell::OnceCell;
use std::collections::BTreeSet;

#[test]
fn dining_service_reuses_the_same_tabletop_clearance_decisions() {
    let mut decisions = [0; 2];
    for seed in 71800..71808 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Restaurant, 0.65, 0, 0.)
                .unwrap();
        for table in scene.objects.iter().filter(|o| scene.is_dining_table(o)) {
            let outline = objects::tables::outline(table);
            for (index, kind) in [
                ObjectKind::Dishware,
                ObjectKind::Flatware,
                ObjectKind::DrinkingGlass,
                ObjectKind::TableNapkin,
            ]
            .into_iter()
            .enumerate()
            {
                let mut prop = objects::audit::fixture(kind, seed);
                prop.id = scene.objects.len();
                prop.support = Some(table.id);
                prop.yaw = table.yaw + index as f32 * 0.59;
                for u in [-0.54, -0.30, 0., 0.32, 0.49] {
                    for v in [-0.41, 0., 0.55] {
                        prop.position = table.transform().transform_point(Vec3::new(
                            table.size.x * u,
                            table.size.y,
                            table.size.z * v,
                        ));
                        let cached = scene.orient_and_check_prop(&mut prop, table, Some(&outline));
                        assert_eq!(cached, scene.prop_clear(&prop, table, 0.012));
                        decisions[usize::from(cached)] += 1;
                    }
                }
            }
        }
    }
    assert!(decisions.into_iter().all(|n| n > 100));
}

#[test]
fn napkins_belong_to_served_places_and_leave_clear_tables_and_work_areas() {
    let mut folds = BTreeSet::new();
    let mut empty = 0;
    let mut count = 0;
    for seed in 71800..71928 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Restaurant, 0.65, 0, 0.35)
                .unwrap();
        validation::validate_layout(&scene).unwrap();
        let graph = scene.functional_graph().unwrap();
        let mut room_count = 0;
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::TableNapkin)
        {
            let table = &scene.objects[o.support.unwrap()];
            let chair = &scene.objects[o.interaction_target.unwrap()];
            assert!(scene.valid_place_setting(o, chair) && scene.prop_clear(o, table, 0.012));
            assert!(scene.is_dining_table(table) && !o.solid);
            assert!(scene.objects.iter().any(|d| d.support == o.support
                && d.interaction_target == o.interaction_target
                && dishware::served_style(d).is_some()));
            assert!(graph
                .edges
                .iter()
                .any(|e| e.source == Participant::Object(o.id)
                    && e.relation == Relationship::Serves
                    && e.target == Participant::Object(chair.id)));
            assert!((o.position.y - table.position.y - table.size.y).abs() < 1e-5);
            let program = items::composition::parameters(&scene, table);
            let local = table
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(o.position)
                .xz();
            let half = super::super::super::placement::rotated_half(o.size.xz(), o.yaw - table.yaw);
            assert!(!(local - program.clear_center)
                .abs()
                .cmplt(program.clear_half + half)
                .all());
            folds.insert(o.variant & 3);
            room_count += 1;
        }
        count += room_count;
        empty += usize::from(room_count == 0);
    }
    eprintln!("restaurant napkins={count}, rooms without={empty}, folds={folds:?}");
    assert!(
        empty > 0 && count > 20,
        "napkins={count}, rooms without={empty}"
    );
    assert_eq!(folds.len(), 3);
}

#[test]
fn public_suite_serves_diners_without_turning_every_spare_slot_into_a_cup() {
    let scene =
        IndoorManifest::generate_with_humans(57306, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
    let mut seated_tables = 0;
    let mut settings = 0;
    for table in scene.objects.iter().filter(|o| scene.is_dining_table(o)) {
        let seats = scene.dining_places(table, &OnceCell::new()).len();
        if seats == 0 {
            continue;
        }
        seated_tables += 1;
        let props = scene.objects.iter().filter(|o| o.support == Some(table.id));
        assert!(props.clone().filter(|o| is_drink(o.kind)).count() <= 2 * seats + 1);
        settings += props
            .filter(|o| dishware::served_style(o).is_some())
            .count();
    }
    assert!(seated_tables >= 2 && settings > 0);
}

#[test]
fn dining_service_uses_actual_seats_supports_and_semantic_meshes() {
    let mut served = 0;
    let mut unserved = 0;
    let mut plates = 0;
    let mut bowls = 0;
    let mut utensils = [0; 3];
    let mut headings = BTreeSet::new();
    let (mut booths, mut waiting_benches, mut meals_at_booths) = (0, 0, 0);
    for layout in [
        IndoorLayout::Cafe,
        IndoorLayout::Cafeteria,
        IndoorLayout::PublicSuite,
    ] {
        for seed in 400..440 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.72, 0, 0.).unwrap();
            validation::validate_layout(&scene).unwrap();
            let graph = scene.functional_graph().unwrap();
            for bench in scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::Banquette)
            {
                let table = graph
                    .edges
                    .iter()
                    .find(|e| {
                        e.source == Participant::Object(bench.id)
                            && e.relation == Relationship::SeatsAt
                    })
                    .map(|e| e.target);
                assert_eq!(table, bench.interaction_target.map(Participant::Object));
                booths += usize::from(table.is_some());
                waiting_benches += usize::from(table.is_none());
            }
            for table in scene.objects.iter().filter(|o| o.kind == ObjectKind::Table) {
                let mut meals = 0;
                let mut drinks = 0;
                let seats = scene.dining_places(table, &OnceCell::new());
                for o in scene.objects.iter().filter(|o| o.support == Some(table.id)) {
                    drinks += usize::from(is_drink(o.kind));
                    let served_dish = dishware::served_style(o);
                    if served_dish.is_none() && o.kind != ObjectKind::Flatware {
                        continue;
                    }
                    assert!(scene.is_dining_table(table));
                    assert!(
                        scene.prop_clear(o, table, 0.012),
                        "{seed}/{layout:?}: {o:?}"
                    );
                    assert!((o.position.y - table.position.y - table.size.y).abs() < 1e-5);
                    let mesh = objects::build_object(o);
                    objects::audit::check(o, &mesh).unwrap();
                    let data = objects::kitchen::metadata(o);
                    if let Some(style) = served_dish {
                        let seat = &scene.objects[o.interaction_target.unwrap()];
                        assert_eq!(seat.interaction_target, Some(table.id));
                        assert!(graph
                            .edges
                            .iter()
                            .any(|e| e.source == Participant::Object(o.id)
                                && e.relation == Relationship::Serves
                                && e.target == Participant::Object(seat.id)));
                        let dish = seat
                            .transform()
                            .compute_affine()
                            .inverse()
                            .transform_point3(o.position);
                        if seat.kind == ObjectKind::Banquette {
                            meals_at_booths += 1;
                            // Independent physical check in the real cushion frame.
                            assert!(
                                dish.z > 0.44 * seat.size.z && dish.z < 0.10 * seat.size.z + 0.901
                            );
                            assert!(dish.x.abs() <= seat.size.x * 0.48);
                        } else {
                            assert!(seat.position.xz().distance(o.position.xz()) <= 0.901);
                        }
                        assert_eq!(data["count"], 1);
                        assert_eq!(data["style"], style);
                        meals += 1;
                        plates += usize::from(style == 0);
                        bowls += usize::from(style == 1);
                    } else {
                        let p = flatware::parameters(o);
                        utensils[match p.tines {
                            0 => 0,
                            3 => 1,
                            4 => 2,
                            _ => unreachable!(),
                        }] += 1;
                        assert_eq!(data["tines"], p.tines);
                        assert!(mesh
                            .parts
                            .keys()
                            .all(|(_, label)| label.starts_with("kitchen_utensil#")));
                    }
                    headings.insert((o.yaw.rem_euclid(2. * PI) * 8. / PI) as u32);
                }
                assert!(meals <= seats.len());
                if scene.is_dining_table(table) && !seats.is_empty() {
                    assert!(
                        drinks <= 2 * seats.len() + 1,
                        "{seed}/{layout:?}: {drinks} drinks for {} diners",
                        seats.len()
                    );
                    served += usize::from(meals > 0);
                    unserved += usize::from(meals == 0);
                }
            }
        }
    }
    eprintln!("seated dining tables served={served} unserved={unserved} plates={plates} bowls={bowls} spoons/3-/4-tine forks={utensils:?}");
    assert!(served > 12 && unserved > 12 && plates > 8 && bowls > 8);
    assert!(utensils.into_iter().all(|n| n > 0) && headings.len() >= 10);
    assert!(booths > 0 && waiting_benches > 0 && meals_at_booths > 0);
}

#[test]
fn glassware_is_admitted_on_dining_tops_with_real_footprints_and_shared_stock() {
    let mut glasses = 0;
    let mut empty_rooms = 0;
    let mut shared = 0;
    for profile in [
        IndoorLayout::Restaurant,
        IndoorLayout::Cafe,
        IndoorLayout::Cafeteria,
    ] {
        for seed in 71800..71824 {
            let scene = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.35).unwrap();
            validation::validate_layout(&scene).unwrap();
            let mut lots = std::collections::BTreeMap::new();
            for o in scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::DrinkingGlass)
            {
                glasses += 1;
                let table = &scene.objects[o.support.unwrap()];
                assert!(scene.is_dining_table(table));
                assert!((o.position.y - table.position.y - table.size.y).abs() < 1e-5);
                assert!(scene.prop_clear(o, table, 0.012));
                assert!(!o.solid && o.size.x == o.size.z);
                let p = items::composition::parameters(&scene, table);
                let pos = table
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(o.position)
                    .xz();
                let half =
                    super::super::super::placement::rotated_half(o.size.xz(), o.yaw - table.yaw);
                assert!(!(pos - p.clear_center)
                    .abs()
                    .cmplt(p.clear_half + half)
                    .all());
                if let Some(other) = lots.insert(o.finish_seed.unwrap(), (o.variant, o.size)) {
                    assert_eq!(other, (o.variant, o.size));
                    shared += 1;
                }
            }
            empty_rooms += usize::from(lots.is_empty());
        }
    }
    assert!(
        glasses > 25 && empty_rooms > 0 && shared > 4,
        "{glasses} {empty_rooms} {shared}"
    );
}
