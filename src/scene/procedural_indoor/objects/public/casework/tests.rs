use super::*;
use crate::scene::procedural_indoor::objects::audit;

#[test]
fn rear_leaves_have_real_glazing_thickness_and_separate_bypass_tracks() {
    let mut seen = [0; 3];
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::DisplayCase, seed);
        o.size = super::super::dimensions(o.kind, seed);
        o.variant = if seed % 2 == 0 { 0 } else { 16 };
        let p = parameters(&o);
        let g = p.service_glazing.unwrap();
        let cabinet = service_glazing::Cabinet::new(o.size, &p, g);
        let mut enclosure = Assembly::default();
        cabinet.enclosure(&mut enclosure, &p, g);
        let glass = &enclosure.parts[&(Surface::ContainerGlass, g.pane.label().into())];
        let probe = Vec2::new(
            0.009,
            o.size.y * (p.base_fraction + (1. - p.base_fraction) * 0.41),
        );
        // Intersect a ray through the actual polygons in the overlap of a
        // bypass pair, away from the seam in a hinged pair.
        let mut crossings = vec![];
        for tri in glass.indices.as_chunks::<3>().0 {
            let v = tri.map(|k| Vec3::from(glass.positions[k as usize]));
            if v[0].z > 0. || (v[0].z - v[1].z).abs() + (v[1].z - v[2].z).abs() > 1e-6 {
                continue;
            }
            let t = v.map(|p| p.truncate());
            let edge = [0, 1, 2].map(|i| (t[(i + 1) % 3] - t[i]).perp_dot(probe - t[i]));
            if edge.iter().all(|&d| d >= 0.) || edge.iter().all(|&d| d <= 0.) {
                crossings.push(v[0].z);
            }
        }
        crossings.sort_by(f32::total_cmp);
        crossings.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        assert_eq!(
            crossings.len(),
            if g.door == CaseDoor::Sliding { 4 } else { 2 }
        );
        for pane in crossings.as_chunks::<2>().0 {
            assert!((pane[1] - pane[0] - g.pane.metres()).abs() < 1e-5);
        }
        if g.door == CaseDoor::Sliding {
            assert!((crossings[2] - crossings[1] - 0.003).abs() < 1e-5);
        }
        assert!(cabinet.shelf.0.y >= crossings.last().unwrap() + 0.009);
        for &level in &p.levels {
            let shelf = cabinet.shelf_height(level, o.size.y);
            let mut cleats = Assembly::default();
            cabinet.shelf_supports(&mut cleats, p.frame, shelf, g.pane.metres());
            let (lo, hi) = cleats.bounds();
            assert!((hi.y - shelf + cabinet.shelf_thickness * 0.5).abs() < 1e-6);
            assert!(lo.x < cabinet.shelf.0.x && hi.x > cabinet.shelf.1.x);
            assert!(cleats
                .parts
                .values()
                .flat_map(|p| &p.positions)
                .any(|v| v[0] > 0. && v[0] < cabinet.shelf.1.x));
            let mut nose = Assembly::default();
            let pane_end = cabinet.shelf_front(&mut nose, shelf);
            if let Some(edge) = p.shelf_edge {
                assert_eq!(p.structure, 1);
                let (lower, upper) = nose.bounds();
                assert!((upper.z - cabinet.shelf.1.y).abs() < 1e-6);
                assert!((pane_end + edge.depth_m - upper.z).abs() < 1e-6);
                assert!((upper.y - shelf - cabinet.shelf_thickness * 0.5).abs() < 1e-6);
                assert!((lower.x - cabinet.shelf.0.x).abs() < 1e-6);
            } else {
                assert!(nose.parts.is_empty());
                assert_eq!(pane_end, cabinet.shelf.1.y);
            }
        }
        seen[match g.door {
            CaseDoor::Hinged => 0,
            CaseDoor::PairedHinged => 1,
            CaseDoor::Sliding => 2,
        }] += 1;
        assert!(
            audit::check(&o, &super::super::super::build_object(&o)).is_ok(),
            "{seed}"
        );
    }
    assert!(seen.iter().all(|&n| n >= 5), "{seen:?}");
}

#[test]
fn bakery_stock_is_supported_inside_the_reserved_tray_and_clear_of_the_next_shelf() {
    let bounds = (Vec2::new(-0.51, -0.26), Vec2::new(0.51, 0.21));
    let y = 0.72;
    let mut stocks = [0usize; 4];
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::DisplayCase, seed);
        let headroom = if seed % 2 == 0 { 0.06 } else { 0.12 };
        let mut a = Assembly::default();
        let Some(f) = stocking::legacy_bakery_tray(&mut a, &o, 0, y, headroom, bounds) else {
            continue;
        };
        assert!(f.inside(bounds.0, bounds.1, 0.014));
        let (lo, hi) = a.bounds();
        assert!((lo.y - y).abs() < 2e-6 && hi.y <= y + headroom + 1e-6);
        let inv = Quat::from_rotation_y(-f.yaw);
        let local = |v: &[f32; 3]| inv * (Vec3::from(*v) - Vec3::new(f.center.x, y, f.center.y));
        let mut pan_size = Vec2::ZERO;
        let mut minimum_food = f32::INFINITY;
        for ((surf, l), geo) in &a.parts {
            if l == "dishware" {
                stocks[usize::from(*surf == Surface::Wood)] += 1;
                for v in &geo.positions {
                    pan_size = pan_size.max(local(v).xz().abs());
                }
            }
        }
        for ((_, label), g) in &a.parts {
            if label.starts_with("food#") {
                for v in &g.positions {
                    let p = local(v);
                    assert!(p.xz().abs().cmplt(pan_size - Vec2::splat(0.003)).all());
                    minimum_food = minimum_food.min(p.y);
                }
            }
            if label == "food#finish103" {
                stocks[2] += 1;
            }
            if label == "food#finish104" {
                stocks[3] += 1;
            }
        }
        let board = a
            .parts
            .keys()
            .any(|(s, l)| *s == Surface::Wood && l == "dishware");
        assert!(if board {
            (0.00799..0.01401).contains(&minimum_food)
        } else {
            (minimum_food - 0.003).abs() < 2e-6
        });
    }
    assert!(stocks.iter().all(|&n| n > 10), "{stocks:?}");
}
