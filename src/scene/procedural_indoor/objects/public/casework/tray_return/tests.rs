use super::*;
use crate::scene::procedural_indoor::{
    geometry::Geometry,
    objects::{audit, build_object},
};

// Intersect the submitted triangles with a vertical line. Return the height
// and the sign of the face normal without consulting the product dimensions.
fn crossings(g: &Geometry, point: Vec2) -> Vec<(f32, bool)> {
    let mut hits = Vec::new();
    for tri in g.indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]).as_dvec3());
        let e = b - a;
        let f = c - a;
        let q = point.as_dvec2() - a.xz();
        let det = e.xz().perp_dot(f.xz());
        if det.abs() < 1e-13 {
            continue;
        }
        let u = q.perp_dot(f.xz()) / det;
        let v = e.xz().perp_dot(q) / det;
        if u >= -1e-7 && v >= -1e-7 && u + v <= 1. + 1e-7 {
            hits.push(((a.y + u * e.y + v * f.y) as f32, det < 0.));
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    hits.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && a.1 == b.1);
    hits
}

#[test]
fn mirrored_runners_form_closed_folded_sheets_with_a_flat_bearing() {
    let o = audit::fixture(ObjectKind::TrayReturn, 17);
    let station = parameters(&o).tray_station.unwrap();
    for side in [-1., 1.] {
        let mut a = Assembly::default();
        station.runner(&mut a, o.size, Vec3::ZERO, side);
        let g = a.parts.values().next().unwrap();
        let mut edges = std::collections::BTreeMap::new();
        for tri in g.indices.as_chunks::<3>().0 {
            let p = tri.map(|i| g.positions[i as usize].map(f32::to_bits));
            for (lo, hi) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
                let counts = edges.entry((lo.min(hi), lo.max(hi))).or_insert([0; 2]);
                counts[usize::from(lo > hi)] += 1;
            }
        }
        assert!(edges.values().all(|&uses| uses == [1, 1]));
        assert!(g.indices.len() / 3 <= 64);
        let flange = crossings(g, Vec2::new(-side * station.guide_clearance_m, 0.01));
        assert_eq!(flange.len(), 2);
        assert!(!flange[0].1 && (flange[0].0 + frame::WEB).abs() < 1e-6);
        assert!(flange[1].1 && flange[1].0.abs() < 1e-6);
    }
}

#[test]
fn installed_ware_contacts_its_actual_shelves_or_rim_guides() {
    let mut supports = [0; 2];
    let mut doubles = 0;
    let mut nested = 0;
    let mut tubs = 0;
    let mut empty = 0;
    for seed in 0..144 {
        let mut o = audit::fixture(ObjectKind::TrayReturn, seed);
        if seed % 3 == 1 {
            o.size *= Vec3::new(0.85, 1.15, 0.85);
        }
        let case = parameters(&o);
        let s = case.tray_station.as_ref().unwrap();
        assert_eq!(case.frame, Surface::Metal);
        assert!(
            matches!(case.finish, Surface::Metal | Surface::Plastic)
                || (s.enclosed && case.finish == Surface::Wood)
        );
        let built = build_object(&o);
        audit::check(&o, &built).unwrap();
        let mut supports_only = Assembly::default();
        s.frame(&mut supports_only, &o, &case);
        let pan = s.ware.mesh();
        doubles += usize::from(s.bay_centers_m.len() == 2);
        empty += usize::from(s.loads.is_empty());
        tubs += usize::from(s.ware.kind == WareKind::BusTub);
        for load in &s.loads {
            let rot = Quat::from_rotation_y(load.yaw);
            let y = s.bearings_m[load.level];
            let half = s.ware.size_m * 0.5;
            let top = load.origin.y
                + (load.count - 1) as f32 * s.ware.nesting_pitch_m()
                + s.ware.height_m;
            assert!(
                top < s
                    .bearings_m
                    .get(load.level + 1)
                    .copied()
                    .unwrap_or(o.size.y)
                    - s.deck_m
                    + 1e-5
            );
            let (x, contact) = match s.bearing {
                Bearing::Shelf => {
                    nested += usize::from(load.count > 1);
                    supports[0] += 1;
                    ((half.x - s.ware.lip_m - s.ware.flare_m) * 0.55, 0.)
                }
                Bearing::RimGuides => {
                    supports[1] += 1;
                    assert_eq!(load.count, 1);
                    (half.x - s.ware.lip_m * 0.30, s.ware.under_lip_m())
                }
            };
            for sign in [-1., 1.] {
                let local = Vec2::new(sign * x, 0.003);
                let point = (load.origin + rot * Vec3::new(local.x, 0., local.y)).xz();
                assert!(crossings(&pan, local)
                    .iter()
                    .any(|&(h, up)| !up && (h - contact).abs() < 3e-6));
                assert!(
                    supports_only.parts.values().any(|g| crossings(g, point)
                        .iter()
                        .any(|&(h, up)| up && (h - y).abs() < 3e-6)),
                    "unseated {:?} at seed {seed}, bay {}, level {}",
                    s.bearing,
                    load.bay,
                    load.level
                );
            }
            for p in &pan.positions {
                let v = load.origin + rot * Vec3::from(*p);
                assert!(v.x.abs() <= o.size.x * 0.5 + 1e-6 && v.z.abs() <= o.size.z * 0.5 + 1e-6);
            }
        }
        assert!(built.parts.keys().all(|(_, l)| matches!(
            crate::scene::procedural_indoor::objects::part_label(l),
            "dishware" | "tray_return"
        )));
        assert!(parameters_at(&o, INTRODUCED - 1).tray_station.is_none());
    }
    assert!(
        supports.iter().all(|n| *n > 50) && nested > 20 && doubles > 18 && empty > 6 && tubs > 16
    );
}

#[test]
fn nested_shells_clear_one_another_and_leave_an_open_well() {
    let mut seen = 0;
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::TrayReturn, seed);
        let case = parameters(&o);
        let w = case.tray_station.unwrap().ware;
        if w.kind != WareKind::ServingTray {
            continue;
        }
        let g = w.mesh();
        let center = crossings(&g, Vec2::new(0.001, -0.002));
        assert_eq!(center.len(), 2);
        assert!(!center[0].1 && center[1].1);
        assert!((center[1].0 - w.nesting_pitch_m()).abs() < 1e-6);
        // Include flared sidewalls, rounded corners and the bearing lip.
        for x in 0..36 {
            for z in [0.07, 0.56, 0.88, 0.965] {
                let pt = w.size_m * 0.5 * Vec2::new((x as f32 + 0.38) / 36., z);
                let hits = crossings(&g, pt);
                if let (Some(low), Some(high)) = (hits.first(), hits.last()) {
                    assert!(
                        high.0 - low.0 <= w.nesting_pitch_m() + 3e-6,
                        "nested surfaces intersect at {pt:?} in product {seed}: {hits:?}, pitch {}",
                        w.nesting_pitch_m()
                    );
                }
            }
        }
        seen += 1;
    }
    assert!(seen >= 32);
}
