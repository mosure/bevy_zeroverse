use super::*;
use crate::scene::procedural_indoor::objects::{audit, desktop::ports, equipment};
use std::collections::BTreeSet;

#[test]
fn custom_socket_budgets_keep_the_dock_and_bound_packing_work() {
    let o = audit::fixture(ObjectKind::Laptop, 21);
    let mut p = parameters(&o);
    p.port_count = 0;
    assert_eq!(p.io_bank(&o).unwrap().ports.len(), 1);
    p.port_count = MAX_PORTS;
    let fitted = serde_json::to_value(p.io_bank(&o).unwrap()).unwrap();
    p.port_count = u32::MAX;
    assert_eq!(
        serde_json::to_value(p.io_bank(&o).unwrap()).unwrap(),
        fitted
    );
}

#[test]
fn every_fitted_connector_uses_an_existing_device_material_batch() {
    for seed in 0..64 {
        let mut o = audit::fixture(ObjectKind::Laptop, seed);
        let fitted = crate::scene::procedural_indoor::objects::build_object(&o);
        o.variant &= !FITTED_IO;
        let archived = crate::scene::procedural_indoor::objects::build_object(&o);
        assert!(fitted
            .parts
            .keys()
            .all(|key| archived.parts.contains_key(key)));
    }
}

#[test]
fn device_banks_fit_the_case_and_replay_across_finishes_and_world_poses() {
    let mut counts = BTreeSet::new();
    let mut masks = BTreeSet::new();
    let mut left = 0;
    let mut kinds = BTreeSet::new();
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::Laptop, seed);
        // Include the short, narrow end of the supported installation envelope.
        if seed % 3 == 0 {
            o.size = Vec3::new(0.27, 0.19, 0.24);
        }
        let p = parameters(&o);
        let deck = Deck::fit(&o, &p);
        let bank = p.io_bank(&o).unwrap();
        let half = 0.5 * o.size.x;
        counts.insert(bank.ports.len());
        let mut mask = 0;
        let primary = bank.ports[0];
        assert_eq!(primary.connector, Connector::UsbC);
        assert_eq!(primary.center_m, charge_port(&o));
        assert_eq!(charge_normal(&o), Vec3::X * primary.center_m.x.signum());
        left += usize::from(primary.center_m.x < 0.);
        for (i, q) in bank.ports.iter().enumerate() {
            let span = q.connector.dimensions().0;
            kinds.insert(q.connector as u8);
            mask |= if q.center_m.x < 0. { 1 } else { 2 };
            assert_eq!(q.center_m.x.abs(), half);
            let bevel = p.corner_m.min(0.002);
            assert!(q.center_m.y - span.y * 0.5 > 0.003 + bevel + 0.00012);
            assert!(q.center_m.y + span.y * 0.5 < 0.003 + p.chassis_m - bevel - 0.00012);
            assert!(q.center_m.z - span.x * 0.5 > deck.rear + p.corner_m + 0.001);
            assert!(q.center_m.z + span.x * 0.5 < deck.rear + deck.depth - p.corner_m - 0.001);
            for other in &bank.ports[..i] {
                assert!(
                    other.center_m.x != q.center_m.x
                        || (other.center_m.z - q.center_m.z).abs()
                            > 0.0049 + (span.x + other.connector.dimensions().0.x) * 0.5
                );
            }
        }
        masks.insert(mask);
        o.finish_seed = Some(!seed);
        o.yaw = 1.73;
        o.position = Vec3::new(-3.1, 0.74, 4.12);
        let archived: IndoorObject =
            serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&bank).unwrap(),
            serde_json::to_value(parameters(&archived).io_bank(&archived).unwrap()).unwrap()
        );
        let world = o.transform().transform_point(primary.center_m);
        let normal = o.transform().rotation * Vec3::X * primary.center_m.x.signum();
        assert_eq!(equipment::power_port(&o, true), Some(world));
        assert_eq!(equipment::power_port_normal(&o, true), Some(normal));
        assert_eq!(
            ports::data_ports(&o, &audit::fixture(ObjectKind::Monitor, seed)).unwrap()[0],
            (world, normal)
        );
    }
    assert!((60..135).contains(&left));
    assert_eq!(masks, BTreeSet::from([1, 2, 3]));
    assert_eq!(kinds.len(), 4);
    assert_eq!(counts, BTreeSet::from([1, 2, 3, 4, 5, 6]));
}

#[test]
fn actual_side_rays_hit_the_housing_contact_and_well_at_different_depths() {
    // Orthographic first hit on actual triangles, projected into Z/Y. Test the
    // two sides with both metal and polymer housings.
    fn hit(a: &Assembly, origin: Vec3, side: f32) -> (Surface, f32) {
        a.parts
            .iter()
            .flat_map(|((s, _), g)| {
                g.indices.as_chunks::<3>().0.iter().filter_map(move |idx| {
                    let [a, b, c] = idx.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let edge = b.zy() - a.zy();
                    let other = c.zy() - a.zy();
                    let ray = origin.zy() - a.zy();
                    let det = edge.perp_dot(other);
                    if det.abs() < 1e-12 {
                        return None;
                    }
                    let u = ray.perp_dot(other) / det;
                    let v = edge.perp_dot(ray) / det;
                    let t = (origin.x - (a.x + u * (b.x - a.x) + v * (c.x - a.x))) * side;
                    (u >= -1e-6 && v >= -1e-6 && u + v <= 1.000001 && t >= 0.).then_some((*s, t))
                })
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
    }
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::Laptop, seed);
        let p = parameters(&o);
        let deck = Deck::fit(&o, &p);
        let bank = p.io_bank(&o).unwrap();
        let stock = Housing::for_object(&o);
        let mut a = Assembly::default();
        palmrest::build(
            &mut a,
            &o,
            &p,
            &stock,
            deck.depth,
            deck.rear + deck.depth * 0.5,
            Some(&bank),
        );
        audit::check(&o, &a).unwrap();
        for port in bank.ports {
            let (span, depth) = port.connector.dimensions();
            let side = port.center_m.x.signum();
            let eye = port.center_m + Vec3::new(side * 0.020, 0., 0.00007);
            let (s, t) = hit(&a, eye + Vec3::Y * (0.5 * span.y + 0.00010), side);
            assert!(
                s == stock.surface && (t - 0.020).abs() < 2e-6,
                "missing housing at {seed}/{port:?}: {s:?}, {t}"
            );
            let (s, t) = hit(&a, eye - Vec3::Y * (0.31 * span.y), side);
            assert!(
                s == Surface::Ink && (t - (0.020 + depth)).abs() < 2e-6,
                "blocked pocket {seed}/{port:?}: {s:?}, {t}"
            );
            if port.connector != Connector::Audio {
                let shift = if port.connector == Connector::UsbA {
                    span.y * 0.20
                } else {
                    0.
                };
                let (s, t) = hit(&a, eye + Vec3::Y * shift, side);
                assert!(
                    s == Surface::Plastic && (0.0207..0.022).contains(&t),
                    "missing contact {seed}/{port:?}: {s:?}, {t}"
                );
            }
        }
    }
}

#[test]
fn unmarked_archived_laptops_keep_the_old_power_endpoint() {
    let mut o = audit::fixture(ObjectKind::Laptop, 33);
    o.variant &= !FITTED_IO;
    let p = parameters(&o);
    let deck = Deck::fit(&o, &p);
    assert!(p.io_bank(&o).is_none());
    assert_eq!(charge_normal(&o), Vec3::X);
    assert_eq!(
        charge_port(&o),
        Vec3::new(
            o.size.x * 0.5 + 0.001,
            p.chassis_m * 0.55 + 0.003,
            deck.rear + deck.depth * 0.17
        )
    );
}
