use super::*;
use crate::scene::procedural_indoor::{
    objects::public::laundry::Machine, validation::validate_layout,
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn laundry_banks_share_stock_have_usable_fronts_and_vary_between_rooms() {
    let mut directions = BTreeSet::new();
    let mut capacities = BTreeSet::new();
    let mut pocket_counts = BTreeSet::new();
    let mut workloads = BTreeSet::new();
    let mut shared_stock = false;
    let mut furnished = false;
    let mut rooms_with_folding_tables = 0;
    for seed in 0..72 {
        let density = [0., 0.65, 1.][seed as usize % 3];
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Laundromat, density, 0, 0.25)
                .unwrap();
        validate_layout(&scene).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let p = &scene.functional_programs()[0];
        assert_eq!(p.capacity_unit, CapacityUnit::LaundryMachines);
        let mut kinds = BTreeSet::new();
        let mut stock = BTreeMap::new();
        rooms_with_folding_tables += usize::from(
            p.members
                .iter()
                .any(|&id| scene.objects[id].kind == ObjectKind::LaundryFoldingTable),
        );
        for o in p.members.iter().map(|&id| &scene.objects[id]) {
            match o.kind {
                ObjectKind::WashingMachine | ObjectKind::TumbleDryer => {
                    kinds.insert(o.kind as usize);
                    directions.insert((o.yaw.sin().round() as i32, o.yaw.cos().round() as i32));
                    let model = Machine::for_object(o);
                    if o.kind == ObjectKind::TumbleDryer {
                        pocket_counts.insert(model.pockets.len());
                    }
                    let spec = (
                        o.size.to_array().map(f32::to_bits),
                        model.finish,
                        model.hinge_right,
                        model.dial,
                    );
                    if let Some(same_line) =
                        stock.insert((o.kind as usize, o.finish_seed.unwrap()), spec)
                    {
                        assert_eq!(same_line, spec);
                        shared_stock = true;
                    }
                    for pocket in &model.pockets {
                        workloads.insert(pocket.load_fraction > 0.);
                    }
                    for x in [-0.495, 0., 0.495] {
                        for d in [0.01, 0.50, 1.04] {
                            let q = o.transform().transform_point(Vec3::new(
                                o.size.x * x,
                                0.8,
                                o.size.z * 0.5 + d,
                            ));
                            assert!(p
                                .reservations
                                .iter()
                                .any(|r| r.role == ReservationRole::EquipmentAccess
                                    && q.xz().cmpge(r.min).all()
                                    && q.xz().cmple(r.max).all()));
                            assert!(scene.objects.iter().filter(|a| a.solid && !a.neighbor).all(
                                |a| {
                                    let (lo, hi) = a.bounds();
                                    (q.cmplt(lo) | q.cmpgt(hi)).any()
                                }
                            ));
                        }
                    }
                }
                ObjectKind::KitchenTowel => {
                    let table = &scene.objects[o.support.unwrap()];
                    assert_eq!(table.kind, ObjectKind::LaundryFoldingTable);
                    assert!((o.position.y - table.position.y - table.size.y).abs() < 0.002);
                    furnished = true;
                }
                _ => {}
            }
        }
        assert_eq!(kinds.len(), 2);
        let actual = p
            .members
            .iter()
            .filter(|&&id| kinds.contains(&(scene.objects[id].kind as usize)))
            .count();
        assert_eq!(actual, p.accepted_capacity);
        assert!(actual <= p.requested_capacity);
        capacities.insert(actual);
    }
    assert_eq!(directions.len(), 4);
    assert_eq!(pocket_counts, BTreeSet::from([1, 2]));
    assert_eq!(workloads.len(), 2);
    assert!(capacities.len() >= 4 && shared_stock && furnished);
    assert!(
        (40..72).contains(&rooms_with_folding_tables),
        "{rooms_with_folding_tables}"
    );
}

#[test]
fn laundry_validation_rejects_a_bank_without_loading_access() {
    let mut s =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Laundromat, 0.65, 0, 0.).unwrap();
    s.validate_public_programs().unwrap();
    s.assembly_diagnostics.programs[0]
        .reservations
        .retain(|r| r.role != ReservationRole::EquipmentAccess);
    let error = s.validate_public_programs().unwrap_err();
    assert!(error.contains("loading or folding aisle"), "{error}");
}
