use super::*;
use crate::scene::procedural_indoor::materials::glazing::GlazingProgram;

fn facade(seed: u64) -> Facade {
    let mut j = WindowJoinery::sample(seed);
    j.column_spread = 0.1;
    let mut glass = GlazingProgram::sample(seed, 0, true);
    glass.opening = 1.;
    glass.treatment = 0;
    Facade {
        joinery: Some(j),
        side: FacadeSide::Rear,
        openings: vec![WindowOpening {
            glazing: Some(glass),
            min: Vec2::new(-1.6, 0.34),
            max: Vec2::new(1.6, 3.85),
            columns: 4,
            transom: 0.22,
            transoms: vec![0.52, 0.77],
            operable: true,
        }],
        frame: Surface::Metal,
        frame_width: 0.05,
        frame_depth: 0.08,
        recess: 0.10,
        sill_projection: 0.,
        treatment: None,
        shade: Shade::None,
        shade_coverage: 0.,
        shade_tilt: 0.,
    }
}

#[test]
fn fitted_banks_mix_reachable_operating_lights_and_fixed_transoms() {
    let mut counts = [0; 4];
    let mut bank_counts = std::collections::BTreeSet::new();
    for seed in 0..96 {
        let mut f = facade(seed);
        let bank = f.pane_installations(&f.openings[0]);
        assert_eq!(bank.len(), 16);
        let operating: Vec<_> = bank.iter().filter(|p| p.handle.is_some()).collect();
        assert!(!operating.is_empty());
        bank_counts.insert(operating.len());
        assert!(bank.iter().any(|p| p.opening_fraction == 1.));
        for p in &bank {
            if p.min.y > HIGH_HANDLE_M || p.max.y < LOW_HANDLE_M {
                assert_eq!(p.hinge, SashHinge::Fixed);
            }
            if let Some(h) = p.handle {
                assert!((LOW_HANDLE_M..=HIGH_HANDLE_M).contains(&h.y));
                assert!(h.cmpgt(p.min).all() && h.cmplt(p.max).all());
                let index = match p.hinge {
                    SashHinge::LeftCasement => 0,
                    SashHinge::RightCasement => 1,
                    SashHinge::TopAwning => 2,
                    SashHinge::BottomHopper => 3,
                    SashHinge::Fixed => panic!("fixed light with hardware"),
                };
                counts[index] += 1;
                let (lo, hi) = p.bounds(true);
                let width = fitted_rail_width(f.joinery.as_ref().unwrap(), hi - lo);
                let edge = match p.hinge {
                    SashHinge::LeftCasement => hi.x - h.x,
                    SashHinge::RightCasement => h.x - lo.x,
                    SashHinge::TopAwning => h.y - lo.y,
                    SashHinge::BottomHopper => hi.y - h.y,
                    _ => unreachable!(),
                };
                assert!((edge - width * 0.5).abs() < 1e-6);
            } else {
                assert_eq!(p.swing_radians, 0.);
            }
        }
        f.shade = Shade::Curtain;
        for (a, b) in bank.iter().zip(f.pane_installations(&f.openings[0])) {
            assert_eq!((a.hinge, a.handle), (b.hinge, b.handle));
            assert_eq!((b.swing_radians, b.opening_fraction), (0., 0.));
        }
    }
    assert!(counts.into_iter().all(|n| n >= 10));
    assert!(bank_counts.len() >= 4);
}

#[test]
fn ineligible_banks_and_fixed_facades_have_no_moving_or_detached_hardware() {
    let mut f = facade(23);
    let impossible = [
        (Vec2::new(-1., 2.3), Vec2::new(1., 3.8), 2),
        (Vec2::new(-1.5, 0.2), Vec2::new(1.5, 3.4), 1),
        (Vec2::new(-0.5, 0.04), Vec2::new(0.5, 0.58), 1),
    ];
    for (lo, hi, cols) in impossible {
        let o = &mut f.openings[0];
        (o.min, o.max, o.columns, o.transom, o.transoms) = (lo, hi, cols, 0., vec![]);
        assert!(f
            .pane_installations(&f.openings[0])
            .iter()
            .all(|p| p.handle.is_none() && p.swing_radians == 0.));
        let mut a = Assembly::default();
        glazing_at(&mut a, Transform::IDENTITY, &f);
        assert!(a.parts.keys().all(|(_, label)| label != "window#hardware"));
    }
    let mut f = facade(23);
    f.openings[0].operable = false;
    assert!(f
        .pane_installations(&f.openings[0])
        .iter()
        .all(|p| p.hinge == SashHinge::Fixed));
    f.openings[0].operable = true;
    f.joinery
        .as_mut()
        .unwrap()
        .sash_stock
        .as_mut()
        .unwrap()
        .operating_share = 0.;
    assert!(f
        .pane_installations(&f.openings[0])
        .iter()
        .all(|p| p.hinge == SashHinge::Fixed));
}

#[test]
fn opening_sweep_keeps_hardware_seated_and_glass_in_its_own_light() {
    let wall = Transform::from_xyz(1.4, 0.3, -3.2).with_rotation(Quat::from_rotation_y(0.7));
    let mut orientations = std::collections::BTreeSet::new();
    for seed in 0..16 {
        let mut f = facade(seed);
        f.openings[0].glazing.as_mut().unwrap().treatment = 6;
        let o = &f.openings[0];
        let j = f.joinery.as_ref().unwrap();
        let stock = j.sash_stock.as_ref().unwrap();
        let glass = o.glazing.as_ref().unwrap();
        let face = 0.006 + j.sash_depth_m * 0.5;
        let back = (0.006 - j.sash_depth_m * 0.5).min(-glass.thickness_m);
        for p in f
            .pane_installations(o)
            .into_iter()
            .filter(|p| p.handle.is_some())
        {
            for fraction in [0., 0.1, 0.4, 0.7, 1.] {
                let mut p = p;
                p.set_opening(fraction, stock.swing_limit_m);
                let mut a = Assembly::default();
                super::super::pane::build(&mut a, wall, &f, o, p);
                let pose = p.pose(&f, face, back);
                let mut mount_depth = f32::INFINITY;
                let mut furthest = f32::NEG_INFINITY;
                let mut glass_inside = 0_f32;
                let mut glass_outside = 0_f32;
                for ((surface, label), g) in &a.parts {
                    let mut fronts = 0;
                    for (v, n) in g.positions.iter().zip(&g.normals) {
                        let local = wall.rotation.inverse() * (Vec3::from(*v) - wall.translation);
                        assert!(
                            local.truncate().cmpge(p.min - 0.00003).all()
                                && local.truncate().cmple(p.max + 0.00003).all(),
                            "{seed}: {p:?} vertex {local:?}"
                        );
                        let closed = pose.rotation.inverse() * (local - pose.translation);
                        if label == "window#hardware" {
                            mount_depth = mount_depth.min(closed.z);
                            furthest = furthest.max(closed.z);
                        }
                        if *surface == Surface::Glass {
                            let material_normal =
                                pose.rotation.inverse() * wall.rotation.inverse() * Vec3::from(*n);
                            if material_normal.z > 0.9999 {
                                assert!(closed.z.abs() < 2e-5);
                                fronts += 1;
                                glass_inside = glass_inside.max(local.z + f.recess);
                                glass_outside = glass_outside.min(local.z + f.recess);
                            }
                            assert!((-glass.thickness_m - 2e-5..=2e-5).contains(&closed.z));
                        }
                        if label == "window#frit" {
                            assert!((0.00047..0.00113).contains(&closed.z));
                        }
                    }
                    if *surface == Surface::Glass {
                        assert_eq!(fronts, 4, "extra or missing glazed surface");
                    }
                }
                assert!((mount_depth - (face - 0.0006)).abs() < 2e-5);
                assert!((furthest - (face + stock.grip_projection_m + 0.006)).abs() < 2e-5);
                if fraction == 1. {
                    orientations.insert(format!("{:?}", p.hinge));
                    if p.hinge == SashHinge::TopAwning {
                        assert!(glass_outside < -0.035);
                    } else {
                        assert!(glass_inside > 0.035);
                    }
                    assert!(glass_inside.max(-glass_outside) <= stock.swing_limit_m + 0.002);
                }
            }
        }
    }
    assert_eq!(orientations.len(), 4);
}

#[test]
fn clear_glass_meter_uses_the_built_sash_and_beads() {
    let mut f = facade(18);
    f.joinery
        .as_mut()
        .unwrap()
        .sash_stock
        .as_mut()
        .unwrap()
        .operating_share = 1.;
    let o = &f.openings[0];
    let mut actual = 0.;
    for p in f.pane_installations(o) {
        let mut a = Assembly::default();
        super::super::pane::build(&mut a, Transform::IDENTITY, &f, o, p);
        let glass = &a
            .parts
            .iter()
            .find(|((s, _), _)| *s == Surface::Glass)
            .unwrap()
            .1;
        let front: Vec<_> = glass
            .positions
            .iter()
            .zip(&glass.normals)
            .filter(|(_, n)| n[2] > 0.9)
            .map(|(p, _)| Vec3::from(*p))
            .collect();
        let width = front[0].distance(front[1]);
        let height = front[0].distance(front[2]);
        let j = f.joinery.as_ref().unwrap();
        let (lo, hi) = p.bounds(true);
        let rail = if p.handle.is_some() {
            fitted_rail_width(j, hi - lo)
        } else {
            j.seal_width_m * 2.
        };
        let bead = j.seal_width_m.min(rail * 0.3);
        actual += (width - 2. * bead) * (height - 2. * bead) * p.swing_radians.cos();
    }
    assert!((actual - super::super::pane::visible_pane_area(&f, o)).abs() < 2e-5);
}

#[test]
fn archives_without_sash_stock_keep_the_historical_midpoint_and_tilt() {
    let f = facade(31);
    let serialized = serde_json::to_value(&f).unwrap();
    let restored: Facade = serde_json::from_value(serialized.clone()).unwrap();
    assert_eq!(
        f.pane_installations(&f.openings[0]),
        restored.pane_installations(&restored.openings[0])
    );
    let mut old = serialized;
    old["joinery"].as_object_mut().unwrap().remove("sash_stock");
    let replay: Facade = serde_json::from_value(old).unwrap();
    assert!(replay.joinery.as_ref().unwrap().sash_stock.is_none());
    for p in replay.pane_installations(&replay.openings[0]) {
        assert_eq!(p.hinge, SashHinge::BottomHopper);
        assert_eq!(p.handle.unwrap().y, (p.min.y + p.max.y) * 0.5);
        assert_eq!(
            p.swing_radians,
            (0.14 / (p.max.y - p.min.y)).min(0.2).asin()
        );
    }
    let mut stock = f.joinery.unwrap().sash_stock.unwrap();
    stock.grip_projection_m = f32::NAN;
    assert!(stock.validate().is_err());
}

#[test]
fn exterior_hinge_leaf_reaches_shallow_rail_around_deep_glass() {
    for seed in 0..64 {
        let mut f = facade(seed);
        f.joinery.as_mut().unwrap().sash_depth_m = 0.018;
        f.openings[0].glazing.as_mut().unwrap().thickness_m = 0.018;
        let o = &f.openings[0];
        let Some(mut p) = f
            .pane_installations(o)
            .into_iter()
            .find(|p| p.hinge == SashHinge::TopAwning)
        else {
            continue;
        };
        p.set_opening(1., 0.14);
        let pose = p.pose(&f, 0.015, -0.018);
        let mut a = Assembly::default();
        super::super::pane::build(&mut a, Transform::IDENTITY, &f, o, p);
        let (_, hi) = p.bounds(true);
        // Only the attached leaf reaches this far inboard from the pin.
        let depths = a.parts[&(Surface::Metal, "window#hinge".into())]
            .positions
            .iter()
            .map(|v| pose.rotation.inverse() * (Vec3::from(*v) - pose.translation))
            .filter(|v| v.y < hi.y - 0.011)
            .map(|v| v.z);
        let (min, max) = depths.fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), z| {
            (min.min(z), max.max(z))
        });
        assert!(min < -0.018 && max > -0.003, "{min} {max}");
        return;
    }
    panic!("expected a top-hung light");
}
