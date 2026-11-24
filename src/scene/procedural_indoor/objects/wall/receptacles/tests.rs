use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};

// First outward triangle hit by a ray from +Z, irrespective of material. A
// decorative dark rectangle over a solid face would fail this depth test.
fn front(mesh: &Assembly, p: Vec2) -> f32 {
    let mut front = f32::NEG_INFINITY;
    for g in mesh.parts.values() {
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let (u, v, q) = ((b - a).xy(), (c - a).xy(), p - a.xy());
            let area = u.perp_dot(v);
            if area <= 1e-12 {
                continue;
            }
            let (s, t) = (q.perp_dot(v) / area, u.perp_dot(q) / area);
            if s >= -1e-5 && t >= -1e-5 && s + t <= 1.00001 {
                front = front.max(a.z + s * (b.z - a.z) + t * (c.z - a.z));
            }
        }
    }
    front
}

#[test]
fn installed_contacts_meet_the_rendered_face_and_open_into_real_pockets() {
    let mut blank_positions = std::collections::BTreeSet::new();
    let mut counts = std::collections::BTreeSet::new();
    let mut recess_range = (f32::INFINITY, 0_f32);
    for seed in 0..96 {
        for standard in 0..3 {
            let mut o = audit::fixture(ObjectKind::WallOutlet, seed);
            o.variant = installed_variant(standard);
            let mut r = stream(seed, 7773);
            o.size = Vec3::new(
                r.random_range(0.063..0.28),
                r.random_range(0.091..0.143),
                r.random_range(0.016..0.026),
            );
            let bank = parameters(&o).bank.unwrap();
            let mesh = build_object(&o);
            audit::check(&o, &mesh).unwrap_or_else(|e| panic!("{seed}/{standard}: {e}"));
            let (lo, hi) = mesh.bounds();
            assert!(lo.z >= -o.size.z * 0.5 && hi.z <= o.size.z * 0.5);
            assert!(lo.y >= -1e-6 && hi.y <= o.size.y);
            counts.insert((standard, bank.rows, bank.gangs));
            assert!(bank.socket_local(bank.socket_count()).is_none());
            if let Some(slot) = bank.blank_slot {
                blank_positions.insert(slot);
                let p = bank.slot_center(slot);
                assert!((front(&mesh, p) - (bank.face_z_m - 0.00045)).abs() < 2e-6);
                assert!((0..bank.socket_count())
                    .all(|i| (bank.socket_local(i).unwrap().x - p.x).abs()
                        > bank.module_size.x * 0.5));
            }
            if standard == 1 {
                recess_range.0 = recess_range.0.min(bank.well_depth_m);
                recess_range.1 = recess_range.1.max(bank.well_depth_m);
            }
            let slot_offset = match bank.face {
                OutletFace::Polarized => Vec2::new(-0.00635, 0.003),
                OutletFace::RoundWell => Vec2::new(0.0093, 0.),
                OutletFace::ThreePin => Vec2::new(0.0111, -0.004),
            } * bank.contact_scale;
            let expected_depth = (if bank.face == OutletFace::RoundWell {
                0.0016
            } else {
                0.0017
            }) * bank.contact_scale;
            o.finish_seed = Some(4123);
            let alt = parameters(&o).bank.unwrap();
            assert_eq!(alt.socket_count(), bank.socket_count());
            for index in 0..bank.socket_count() {
                let port = bank.socket_local(index).unwrap();
                assert_eq!(alt.socket_local(index), Some(port));
                let face = front(&mesh, port.xy());
                let bore = front(&mesh, port.xy() + slot_offset);
                assert!(
                    (face - port.z).abs() < 2e-6 && (face - bore - expected_depth).abs() < 2e-6,
                    "{seed}/{standard}/{index}: socket {port:?}, face {face}, pocket {bore}"
                );
                if bank.face == OutletFace::Polarized {
                    let k = bank.contact_scale;
                    let center = port.xy() - Vec2::Y * 0.009 * k;
                    let corner = Vec2::splat(0.0023 * k);
                    assert!((front(&mesh, center + corner) - face).abs() < 2e-6);
                    assert!((front(&mesh, center - corner) - bore).abs() < 2e-6);
                }
                if bank.face == OutletFace::RoundWell {
                    let lip = front(&mesh, port.xy() + Vec2::X * 0.0205 * bank.contact_scale);
                    assert!((lip - face - bank.well_depth_m).abs() < 2e-6);
                }
            }
        }
    }
    assert_eq!(blank_positions.len(), 3);
    assert!(recess_range.1 - recess_range.0 > 0.003);
    for gangs in 1..=3 {
        for standard in 0..3 {
            assert!(counts.contains(&(standard, 1, gangs)));
        }
        assert!(counts.contains(&(0, 2, gangs)));
    }
    assert_eq!(counts.len(), 12);
}

#[test]
fn archived_wall_variants_keep_their_original_connector_coordinates() {
    for standard in 0..3 {
        let mut plate = audit::fixture(ObjectKind::WallOutlet, 65);
        plate.variant = standard;
        let p = parameters(&plate);
        assert!(p.bank.is_none());
        let old = outlet_port_local(&plate);
        assert_eq!(
            old.y,
            plate.size.y * p.case_fraction.y * p.receptacle_height
        );
        assert_eq!(old.z, plate.size.z * 0.52 + 0.001);
    }
}
