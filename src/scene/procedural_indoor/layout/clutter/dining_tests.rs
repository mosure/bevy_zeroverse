use super::*;
use crate::scene::procedural_indoor::{
    objects::{
        self,
        kitchen::{dishware, flatware},
    },
    validation,
};
use std::collections::BTreeSet;

#[test]
fn public_suite_serves_diners_without_turning_every_spare_slot_into_a_cup() {
    let scene =
        IndoorManifest::generate_with_humans(57306, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
    let mut seated_tables = 0;
    let mut settings = 0;
    for table in scene.objects.iter().filter(|o| scene.is_dining_table(o)) {
        let seats = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(table.id))
            .count();
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
    for layout in [
        IndoorLayout::Cafe,
        IndoorLayout::Cafeteria,
        IndoorLayout::PublicSuite,
    ] {
        for seed in 400..440 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.72, 0, 0.).unwrap();
            validation::validate_layout(&scene).unwrap();
            for table in scene.objects.iter().filter(|o| o.kind == ObjectKind::Table) {
                let mut meals = 0;
                let mut drinks = 0;
                let seats: Vec<_> = scene
                    .objects
                    .iter()
                    .filter(|o| {
                        o.kind == ObjectKind::Chair && o.interaction_target == Some(table.id)
                    })
                    .collect();
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
                        assert!(seats
                            .iter()
                            .any(|s| s.position.xz().distance(o.position.xz()) <= 0.901));
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
}
