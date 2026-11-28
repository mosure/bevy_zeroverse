use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};
use std::collections::BTreeSet;

#[test]
fn waiting_banks_fit_their_envelope_with_real_seats_and_floor_contacts() {
    let mut stock = BTreeSet::new();
    let mut counts = BTreeSet::new();
    let mut tables = BTreeSet::new();
    for seed in 0..120 {
        let o = audit::fixture(ObjectKind::BeamSeating, seed);
        let p = BeamSeating::for_object(&o);
        let a = build_object(&o);
        let mesh = audit::check(&o, &a).unwrap();
        assert!(
            mesh.min_y_m.abs() < 1e-6 && mesh.max_envelope_overrun_m < 1e-5,
            "{seed}: {mesh:?}"
        );
        assert!(
            mesh.batches <= 5 && mesh.triangles < 10000,
            "{seed}: {mesh:?}"
        );
        assert!((0.563..0.652).contains(&p.pitch_m) && p.seat_count() >= 2);
        assert!(a.parts.keys().all(|(_, label)| part_label(label) == LABEL));
        let pads = &a.parts[&(Surface::Rubber, LABEL.to_owned())];
        let on_floor = pads.positions.iter().filter(|p| p[1].abs() < 1e-6);
        for side in [-1., 1.] {
            for end in [-1., 1.] {
                assert!(on_floor
                    .clone()
                    .any(|v| side * v[0] > 0.25 && end * v[2] > 0.14));
            }
        }
        let skin = &a.parts[&(
            p.facing,
            format!("{LABEL}#finish{}#uv_local", p.finish_slot),
        )];
        // Separate cuts of the same batch agree on tooling and texel scale
        // while showing different regions of the grain or weave.
        let cut = skin.uvs.len() / p.seat_count();
        assert!(cut > 3 && skin.uvs.len().is_multiple_of(p.seat_count()));
        for (i, copy) in skin.uvs.chunks_exact(cut).enumerate().skip(1) {
            let delta = Vec2::from(copy[0]) - Vec2::from(skin.uvs[0]);
            assert!(delta.length_squared() > 1e-8, "{seed}, cut {i}");
            assert!(copy
                .iter()
                .zip(&skin.uvs[..cut])
                .all(
                    |(&uv, &original)| (Vec2::from(uv) - Vec2::from(original) - delta).length()
                        < 3e-6
                ));
        }
        for bay in 0..p.bays {
            let x = p.center_x(bay);
            let near = |v: &&[f32; 3]| (v[0] - x).abs() < p.seat_width_m * 0.49;
            let has_back = skin
                .positions
                .iter()
                .filter(near)
                .any(|v| v[1] > o.size.y - 0.04 && v[2] < 0.);
            let has_seat = skin
                .positions
                .iter()
                .filter(near)
                .any(|v| (v[1] - p.seat_height_m).abs() < 0.032 && v[2] > 0.04);
            assert_eq!(
                (has_back, has_seat),
                (Some(bay) != p.table_bay, Some(bay) != p.table_bay)
            );
        }
        stock.insert(format!("{:?}/{:?}", p.stock, p.armrests));
        counts.insert(p.bays);
        tables.insert(p.table_bay);
    }
    assert_eq!(stock.len(), 9);
    assert_eq!(counts, BTreeSet::from([2, 3, 4, 5]));
    assert!(tables.len() >= 4);
}

#[test]
fn waiting_shell_stays_closed_and_inside_at_extreme_tooling_dimensions() {
    for depth in [0.575, 0.755] {
        for height in [0.77, 0.985] {
            let mut o = audit::fixture(ObjectKind::BeamSeating, 0);
            o.size = Vec3::new(1.22, height, depth);
            let mut p = BeamSeating::for_object(&o);
            p.stock = SeatStock::BentPlywood;
            p.facing = Surface::Wood;
            for width in [0.453, 0.581] {
                for seat_height in [0.414, 0.472] {
                    for wrap in [0.018, 0.045] {
                        for thickness in [0.009, 0.015] {
                            p.seat_width_m = width;
                            p.seat_height_m = seat_height;
                            p.wrap_m = wrap;
                            p.shell_thickness_m = thickness;
                            let a = shell::seat(&o, p, FITTED_BACK_VERSION);
                            let result = audit::check(&o, &a).unwrap();
                            assert!(result.max_envelope_overrun_m < 1e-6);
                            let key = (
                                p.facing,
                                format!("{LABEL}#finish{}#uv_local", p.finish_slot),
                            );
                            let skin = &a.parts[&key];
                            // Every physical edge of the face, rear skin and folded
                            // rim has precisely two incident triangles.
                            let mut edges = std::collections::BTreeMap::new();
                            for t in skin.indices.as_chunks::<3>().0 {
                                let pts = t.map(|i| skin.positions[i as usize].map(f32::to_bits));
                                for (mut a, mut b) in
                                    [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])]
                                {
                                    if a > b {
                                        std::mem::swap(&mut a, &mut b);
                                    }
                                    *edges.entry((a, b)).or_insert(0) += 1;
                                }
                            }
                            assert!(edges.values().all(|&count| count == 2));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn reclined_upholstery_fits_tall_shallow_banks_with_maximum_wrap() {
    let mut o = audit::fixture(ObjectKind::BeamSeating, 71);
    o.size = Vec3::new(1.22, 0.985, 0.575);
    let mut p = BeamSeating::for_object(&o);
    p.stock = SeatStock::Upholstered;
    p.facing = Surface::Fabric;
    p.seat_height_m = 0.414;
    p.wrap_m = 0.045;
    let old = shell::seat(&o, p, FITTED_BACK_VERSION - 1);
    assert!(old.bounds().0.z < -o.size.z * 0.5 - 0.01);
    let new = shell::seat(&o, p, FITTED_BACK_VERSION);
    assert!(audit::check(&o, &new).unwrap().max_envelope_overrun_m < 1e-6);
    assert_eq!(
        old.parts.keys().collect::<Vec<_>>(),
        new.parts.keys().collect::<Vec<_>>()
    );
    for (key, g) in &old.parts {
        assert_eq!(g.indices, new.parts[key].indices);
    }
}
