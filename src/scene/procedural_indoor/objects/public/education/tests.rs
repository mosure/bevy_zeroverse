use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};

// Independent +Z ray/triangle intersection in door space.
fn front_depth(g: &Geometry, x: f32, y: f32) -> Option<f32> {
    let mut front = None;
    for tri in g.indices.as_chunks::<3>().0 {
        let at = |i: usize| Vec3::from(g.positions[tri[i] as usize]);
        let a = at(0);
        let b = at(1) - a;
        let c = at(2) - a;
        let d = Vec2::new(x, y) - a.truncate();
        let det = b.truncate().perp_dot(c.truncate());
        if det.abs() < 1e-12 {
            continue;
        }
        let u = d.perp_dot(c.truncate()) / det;
        let v = b.truncate().perp_dot(d) / det;
        if u >= 0. && v >= 0. && u + v <= 1. {
            let z = a.z + u * b.z + v * c.z;
            front = Some(front.map_or(z, |f: f32| f.max(z)));
        }
    }
    front
}

#[test]
fn punched_and_formed_stock_have_open_throats_and_folded_returns() {
    let mut families = std::collections::BTreeSet::new();
    for seed in 0..112 {
        let mut r = stream(seed, 731);
        let v = VentStock::sample(&mut r);
        let size = Vec3::new(
            r.random_range(0.18..0.52),
            r.random_range(0.27..1.70),
            r.random_range(0.014..0.023),
        );
        let rows = r.random_range(2..=6);
        let mut g = Geometry::default();
        v.build(&mut g, size, rows);
        families.insert((v.upper, v.lower, v.hooded));
        let centers: Vec<_> = v.centers(size.y, rows).collect();
        assert!(centers.windows(2).all(|p| p[1] - p[0] > v.slot_height_m));
        for y in centers {
            let x = size.x * 0.043;
            assert!(front_depth(&g, x, y - 0.44 * v.slot_height_m).is_none());
            let hood = front_depth(&g, x, y + 0.20 * v.slot_height_m);
            if v.hooded {
                assert!(hood.is_some_and(|z| z > size.z * 0.5 + 0.001));
            } else {
                assert!(hood.is_none());
            }
            assert!((front_depth(&g, size.x * 0.44, y).unwrap() - size.z * 0.5).abs() < 2e-6);
        }
        assert!(g.indices.len() / 3 < 1300);
    }
    assert_eq!(families.len(), 6);
}

#[test]
fn installed_banks_fit_the_admitted_volume_and_glazed_latches_land_on_stiles() {
    let mut fronts = std::collections::BTreeSet::new();
    let mut hands = std::collections::BTreeSet::new();
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::Locker, seed);
        o.size = super::super::dimensions(ObjectKind::Locker, seed);
        let p = parameters(&o);
        let fit = p.fittings.as_ref().unwrap();
        fronts.insert(p.front);
        let mesh = build_object(&o);
        audit::check(&o, &mesh).unwrap_or_else(|e| panic!("{seed}: {e}"));
        if let Some(label) = fit.case_label() {
            assert!(mesh
                .parts
                .keys()
                .any(|(s, k)| *s == Surface::Metal && k.starts_with(&label)));
        }
        let old = parameters_at(&o, FITTED_VERSION - 1);
        assert!(old.fittings.is_none());
        for (i, b) in p.bays.iter().enumerate() {
            let side = fit.handing.latch_side(i);
            hands.insert(side > 0.);
            if p.front != 2 {
                continue;
            }
            for (row, limits) in b.levels.windows(2).enumerate() {
                let lo = limits[0] + if row > 0 { p.panel_m * 0.5 } else { 0. };
                let hi = limits[1]
                    - if row + 1 < b.angles.len() {
                        p.panel_m * 0.5
                    } else {
                        0.
                    };
                let mut closed = p.clone();
                closed.bays[i].angles[row] = 0.;
                let mut leaf = Assembly::default();
                fit.door(&mut leaf, &o, &closed, i, row, (lo, hi), "locker");
                let w = b.width - 0.006;
                let stile = fit.stile_width_m.min(w * 0.17);
                let center = b.left + b.width * 0.5;
                let y = fit.handle_y(lo, hi);
                let latch = center + side * (w - stile) * 0.5;
                // Offset from the small latch while still over its structural stile.
                let solid = &leaf.parts[&(p.finish, "locker".to_owned())];
                let glass = &leaf.parts[&(Surface::ContainerGlass, "locker".to_owned())];
                assert!(front_depth(solid, latch, y + 0.07).is_some());
                assert!(front_depth(glass, latch, y).is_none());
                assert!(front_depth(glass, center, y).is_some());
                assert!(front_depth(solid, center, y).is_none());
                // There is no open slit between the panes and their frames.
                let join = center + side * (w * 0.5 - stile);
                assert!(front_depth(glass, join - side * 0.0004, y).is_some());
                assert!(front_depth(solid, join + side * 0.0017, y).is_some());
            }
        }
    }
    assert_eq!(fronts.len(), 4);
    assert_eq!(hands.len(), 2);
}

#[test]
fn bound_volumes_and_stacks_stay_inside_their_own_shelf() {
    let mut filled = 0;
    for seed in 0..80 {
        let mut o = audit::fixture(ObjectKind::Locker, seed);
        o.size = super::super::dimensions(ObjectKind::Locker, seed);
        let p = parameters(&o);
        for (bay, b) in p.bays.iter().enumerate() {
            for (row, limits) in b.levels.windows(2).enumerate() {
                let lo = limits[0] + if row > 0 { p.panel_m * 0.5 } else { 0. };
                let hi = limits[1]
                    - if row + 1 < b.angles.len() {
                        p.panel_m * 0.5
                    } else {
                        0.
                    };
                let mut a = Assembly::default();
                contents::build(&mut a, &o, &p, bay, row, lo, hi);
                if a.parts.is_empty() {
                    continue;
                }
                filled += 1;
                let (min, max) = a.bounds();
                assert!((min.y - lo).abs() < 0.001 && max.y < hi);
                assert!(
                    min.x >= b.left + 0.002 && max.x <= b.left + b.width - 0.002,
                    "shelf side in {seed}: {min:?} {max:?}"
                );
                assert!(
                    min.z >= -o.size.z * 0.5 + p.panel_m
                        && max.z <= -o.size.z * 0.5 + p.body_depth_m - 0.002
                );
            }
        }
    }
    assert!(filled > 80);
}
