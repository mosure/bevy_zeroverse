use super::*;
use crate::scene::procedural_indoor::{
    materials,
    objects::{audit, build_object, part_label, public::retail},
    validation::{validate_geometry, validate_layout},
};
use std::collections::BTreeSet;

#[test]
fn retail_stock_fits_both_display_formats_and_replays_with_registered_finishes() {
    let mut seen = BTreeSet::new();
    for seed in 0..56 {
        for kind in [
            ObjectKind::GroceryDisplay,
            ObjectKind::ApparelDisplay,
            ObjectKind::CheckoutCounter,
        ] {
            let o = audit::fixture(kind, seed);
            let m = build_object(&o);
            let check = audit::check(&o, &m).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            assert!(check.min_y_m.abs() < 0.001 && check.max_envelope_overrun_m < 0.001);
            assert!(
                check.batches <= 20 && check.triangles < 56_000,
                "{kind:?}/{seed}: {check:?}"
            );
            let form = retail::parameters(&o);
            let key = serde_json::to_value(form).unwrap();
            seen.insert(key.as_object().unwrap().keys().next().unwrap().clone());
            let mut prints = BTreeSet::new();
            let mut classes = BTreeSet::new();
            let replay = (seed < 2).then(|| build_object(&o));
            for ((surface, name), g) in &m.parts {
                classes.insert(part_label(name));
                if let Some(slot) = materials::finish_slot(*surface, name) {
                    assert!(
                        materials::registered_finish(*surface, slot),
                        "{surface:?}/{slot}"
                    );
                    if matches!(part_label(name), "food" | "drink_container")
                        && name.ends_with("#uv_local")
                    {
                        prints.insert((*surface, slot));
                    }
                }
                if let Some(replay) = &replay {
                    let other = &replay.parts[&(*surface, name.clone())];
                    assert_eq!(g.positions, other.positions);
                    assert_eq!(g.uvs, other.uvs);
                    assert_eq!(g.indices, other.indices);
                }
            }
            match form {
                retail::Installation::Gondola { .. } | retail::Installation::Refrigerated(_) => {
                    assert_eq!(prints, retail::print_finishes(&o).into_iter().collect())
                }
                retail::Installation::Produce { .. } => assert!(classes.contains("food")),
                retail::Installation::Rail { .. } | retail::Installation::Folded { .. } => {
                    assert!(classes.contains("clothes"))
                }
                retail::Installation::Checkout { .. } | retail::Installation::Register(_) => {
                    assert!(classes.contains("control_panel"))
                }
            }
        }
    }
    assert_eq!(seen.len(), 6);
}

#[test]
fn retail_routes_connect_each_shopping_front_to_a_shared_aisle_and_keep_the_till_usable() {
    for profile in [IndoorLayout::GroceryStore, IndoorLayout::ClothingStore] {
        for seed in 0..12 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                profile,
                [0., 0.65, 1.][seed as usize % 3],
                3,
                0.25,
            )
            .unwrap();
            validate_layout(&scene).unwrap();
            let p = &scene.functional_programs()[0];
            assert_eq!(p.capacity_unit, CapacityUnit::MerchandisingFixtures);
            assert!(p.accepted_capacity >= 2 && p.requested_capacity >= p.accepted_capacity);
            let spine = &p.reservations[0];
            let cross = &p.reservations[1];
            assert!(spine.min.x >= p.min.x && cross.min.y >= p.min.y);
            let columns = scene.columns();
            for r in &p.reservations {
                assert!(
                    columns
                        .iter()
                        .all(|&(l, h)| l.y >= 1.90 || h.y <= 0.001 || !r.overlaps(l, h)),
                    "{profile:?}/{seed} column obstructs {r:?}"
                );
                if r.role == ReservationRole::Browsing {
                    let touches = |a: &Reservation| {
                        r.min.cmple(a.max + Vec2::splat(0.001)).all()
                            && r.max.cmpge(a.min - Vec2::splat(0.001)).all()
                    };
                    assert!(
                        touches(spine) || touches(cross),
                        "{profile:?}/{seed}: isolated stock"
                    );
                }
            }
            let tills: Vec<_> = p
                .members
                .iter()
                .map(|&id| &scene.objects[id])
                .filter_map(retail::checkout::checkout_station)
                .collect();
            assert!((1..=3).contains(&tills.len()));
            let staffed = tills
                .iter()
                .filter(|t| t.form != retail::checkout::CheckoutForm::SelfCheckout)
                .count();
            assert_eq!(staffed, 1);
            assert_eq!(
                p.reservations
                    .iter()
                    .filter(|r| r.role == ReservationRole::Browsing)
                    .count(),
                p.accepted_capacity + tills.len()
            );
            assert_eq!(
                p.reservations
                    .iter()
                    .filter(|r| r.role == ReservationRole::StaffAccess)
                    .count(),
                staffed
            );
            if seed < 2 {
                validate_geometry(&scene).unwrap();
            }
        }
    }
}

#[test]
fn retail_receipt_rejects_a_missing_customer_path() {
    let mut room =
        IndoorManifest::generate_with_humans(13, IndoorLayout::ClothingStore, 0.65, 0, 0.).unwrap();
    let routes = &mut room.assembly_diagnostics.programs[0].reservations;
    routes.remove(
        routes
            .iter()
            .position(|r| r.role == ReservationRole::Browsing)
            .unwrap(),
    );
    assert!(room
        .validate_public_programs()
        .unwrap_err()
        .contains("missing sales front"));
}

#[test]
fn retail_cold_cases_replay_from_the_manifest_and_open_into_a_reserved_sales_front() {
    let mut with = 0;
    let mut without = 0;
    let mut frontage = [false; 4];
    for seed in 56100..56132 {
        let room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::GroceryStore, 0.65, 0, 0.)
                .unwrap();
        let program = &room.functional_programs()[0];
        let mut count = 0;
        for o in program.members.iter().map(|&id| &room.objects[id]) {
            let Some(case) = retail::refrigerated_case(o) else {
                continue;
            };
            let front = o
                .transform()
                .transform_point(Vec3::new(0., 0., o.size.z * 0.5 + 0.36))
                .xz();
            assert!(
                program
                    .reservations
                    .iter()
                    .any(|r| r.role == ReservationRole::Browsing
                        && front.cmpge(r.min).all()
                        && front.cmple(r.max).all()),
                "{seed}/{}",
                o.id
            );
            let (sin, cos) = o.yaw.sin_cos();
            frontage[if cos.abs() >= sin.abs() {
                usize::from(cos > 0.)
            } else {
                2 + usize::from(sin > 0.)
            }] = true;
            let archived: IndoorObject =
                serde_json::from_slice(&serde_json::to_vec(o).unwrap()).unwrap();
            assert_eq!(
                serde_json::to_value(case).unwrap(),
                serde_json::to_value(retail::refrigerated_case(&archived).unwrap()).unwrap()
            );
            let mut ambient = archived;
            ambient.variant &= !retail::REFRIGERATED;
            assert!(matches!(
                retail::parameters(&ambient),
                retail::Installation::Gondola { .. }
            ));
            count += 1;
        }
        if count == 0 {
            without += 1;
        } else {
            with += 1;
        }
    }
    assert!(with > 0 && without > 0 && frontage.into_iter().all(|v| v));
}

#[test]
fn fitting_rooms_keep_each_entrance_on_the_shopping_aisle() {
    use crate::scene::procedural_indoor::objects::public::fitting_room;
    let mut sizes = BTreeSet::new();
    let mut removed = false;
    for seed in 57100..57148 {
        let mut room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::ClothingStore, 0.65, 3, 0.25)
                .unwrap();
        validate_layout(&room).unwrap();
        assert!(room.cameras.iter().all(|c| room.camera_curve_clear(c)));
        let p = &room.functional_programs()[0];
        let routes: Vec<_> = p
            .reservations
            .iter()
            .filter(|r| r.role == ReservationRole::FittingAccess)
            .collect();
        let booths: Vec<_> = p
            .members
            .iter()
            .filter_map(|&id| {
                let o = &room.objects[id];
                fitting_room(o).map(|s| (o, s))
            })
            .collect();
        assert!(booths.len() <= 1);
        assert_eq!(booths.len(), routes.len());
        assert_eq!(
            p.accepted_capacity,
            p.members
                .iter()
                .filter(|&&id| room.objects[id].kind == ObjectKind::ApparelDisplay)
                .count()
        );
        let Some((o, f)) = booths.first() else {
            sizes.insert(0);
            continue;
        };
        sizes.insert(f.bays.len());
        assert!(o.solid);
        let r = routes[0];
        for b in &f.bays {
            for x in [b.left + 0.015, b.right - 0.015] {
                let q = o
                    .transform()
                    .transform_point(Vec3::new(x, 0., o.size.z * 0.5 + 0.49))
                    .xz();
                assert!(q.cmpge(r.min).all() && q.cmple(r.max).all());
            }
        }
        if !removed {
            let axis = usize::from(o.yaw.sin().abs() > o.yaw.cos().abs());
            let mid = o.position.xz()[axis];
            let clipped = room.assembly_diagnostics.programs[0]
                .reservations
                .iter_mut()
                .find(|r| r.role == ReservationRole::FittingAccess)
                .unwrap();
            clipped.max[axis] = mid + 0.02;
            assert!(room
                .validate_public_programs()
                .unwrap_err()
                .contains("fitting room approach"));
            room.assembly_diagnostics.programs[0]
                .reservations
                .retain(|r| r.role != ReservationRole::FittingAccess);
            assert!(room
                .validate_public_programs()
                .unwrap_err()
                .contains("fitting room approach"));
            removed = true;
        }
    }
    assert_eq!(sizes, BTreeSet::from([0, 1, 2]));
}

#[test]
fn retail_checkout_banks_serve_their_own_shop_and_keep_attendant_access() {
    use retail::checkout::{checkout_station, CheckoutForm::*};
    for profile in [IndoorLayout::GroceryStore, IndoorLayout::ClothingStore] {
        let mut staffed_forms = BTreeSet::new();
        let mut populations = BTreeSet::new();
        let mut modified = false;
        for seed in 67300..67364 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, profile, 0.78, 0, 0.).unwrap();
            let p = &scene.functional_programs()[0];
            let stations: Vec<_> = p
                .members
                .iter()
                .map(|&i| &scene.objects[i])
                .filter_map(|o| checkout_station(o).map(|d| (o, d)))
                .collect();
            let staffed: Vec<_> = stations
                .iter()
                .filter(|(_, d)| d.form != SelfCheckout)
                .collect();
            assert_eq!(staffed.len(), 1);
            staffed_forms.insert(staffed[0].1.form.tag());
            populations.insert(stations.len());
            for (o, d) in &stations {
                assert_eq!(d.apparel_assortment, profile == IndoorLayout::ClothingStore);
                assert!(staffed[0].0.position.xz().distance(o.position.xz()) <= 6.201);
            }
            if !modified {
                if let Some((o, _)) = stations.iter().find(|(_, d)| d.form == SelfCheckout) {
                    let id = o.id;
                    scene.objects[id].variant = retail::checkout::CheckoutForm::self_service_tag(
                        profile != IndoorLayout::ClothingStore,
                    );
                    assert!(scene
                        .validate_public_programs()
                        .unwrap_err()
                        .contains("checkout form"));
                    modified = true;
                }
            }
        }
        assert!(
            modified && populations.contains(&1) && populations.iter().any(|&n| n >= 2),
            "{profile:?} {populations:?}"
        );
        if profile == IndoorLayout::GroceryStore {
            assert_eq!(
                staffed_forms,
                BTreeSet::from([ConveyorLane.tag(), BasketRegister.tag()])
            );
            assert!(populations.contains(&3));
        } else {
            assert_eq!(staffed_forms, BTreeSet::from([ApparelWrap.tag()]));
        }
    }
}
