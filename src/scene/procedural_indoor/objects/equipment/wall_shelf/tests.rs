use super::*;
use crate::scene::procedural_indoor::{
    envelope::ray_triangle,
    layout::{IndoorLayout, IndoorManifest},
    objects::audit,
};
use std::collections::{BTreeSet, HashMap};

fn hit(a: &Assembly, p: Vec3, dir: Vec3) -> Option<f32> {
    a.parts
        .values()
        .flat_map(|g| {
            g.indices.as_chunks::<3>().0.iter().filter_map(|t| {
                ray_triangle(p, dir, t.map(|i| Vec3::from_array(g.positions[i as usize])))
            })
        })
        .min_by(f32::total_cmp)
}

#[test]
fn wall_shelf_support_matches_rays_at_shaped_ends_and_folded_undersides() {
    let mut shapes = BTreeSet::new();
    let mut cut_points = 0;
    let mut fold_count = 0;
    for seed in 0..108 {
        let mut o = audit::fixture(ObjectKind::WallShelf, seed);
        o.size = match seed % 3 {
            0 => Vec3::new(0.42, 0.13, 0.17),
            1 => Vec3::new(1.70, 0.38, 0.40),
            _ => o.size,
        };
        let p = parameters(&o);
        let d = p.deck.as_ref().unwrap();
        let half = o.size.xz() * 0.5;
        let a = build_object(&o);
        let report = audit::check(&o, &a).unwrap();
        assert!(report.max_envelope_overrun_m < 1e-5);
        shapes.insert((format!("{:?}", d.stock), format!("{:?}", d.ends[0].cut)));
        let stations = d.stations(half).collect::<Vec<_>>();
        for pair in stations.windows(2) {
            let x = 0.5 * (pair[0].0 + pair[1].0);
            let front = 0.5 * (pair[0].1 + pair[1].1);
            assert!((d.front_at(half, x) - front).abs() < 1e-6);
            for inset in [0.0004, -0.0004] {
                let q = Vec2::new(x, front - d.nose_m - inset);
                assert_eq!(d.supports(&o, q, Vec2::ZERO, 0., 0.), inset > 0.);
                let t = hit(&a, Vec3::new(q.x, o.size.y + 0.02, q.y), Vec3::NEG_Y);
                assert_eq!(
                    t.is_some_and(|v| (v - 0.02).abs() < 1e-6),
                    inset > 0.,
                    "{seed} {q:?} {t:?}"
                );
            }
            if front < half.y - 0.001 {
                cut_points += 1;
                if d.stock == ShelfStock::FoldedSteel
                    && p.bracket_x_m
                        .iter()
                        .all(|b| (x - b).abs() > p.strap_width_m)
                {
                    let origin = Vec3::new(x, 0., front - d.nose_m - 0.55 * d.return_m);
                    let underside = hit(&a, origin, Vec3::Y).unwrap();
                    assert!(
                        (underside - (o.size.y - d.gauge_m)).abs() < 1e-5,
                        "{seed} {origin:?}"
                    );
                }
            }
        }
        if d.stock == ShelfStock::FoldedSteel {
            // At midspan, away from brackets, look up first into the cavity and
            // then at the returned hem. The two depths differ by the fascia drop.
            let x = [0., half.x * 0.32, -half.x * 0.32]
                .into_iter()
                .find(|x| {
                    p.bracket_x_m
                        .iter()
                        .all(|b| (x - b).abs() > p.strap_width_m)
                })
                .unwrap();
            let z = d.front_at(half, x);
            for (at, height) in [
                (0., o.size.y - d.gauge_m),
                (z - d.nose_m - d.return_m * 0.57, o.size.y - d.depth_m),
            ] {
                assert!(
                    (hit(&a, Vec3::new(x, 0., at), Vec3::Y).unwrap() - height).abs() < 1e-5,
                    "{seed}"
                );
            }
            fold_count += 1;
        }
    }
    assert_eq!(shapes.len(), 9);
    assert!(cut_points > 100 && fold_count > 10);
}

#[test]
fn wall_shelf_deck_is_closed_and_wood_grain_runs_along_the_span() {
    // Isolate the deck from its hardware. Adjacent material groups must share
    // each geometric boundary exactly once in each direction.
    for seed in 0..72 {
        let o = audit::fixture(ObjectKind::WallShelf, seed);
        let mut p = parameters(&o);
        p.bracket_x_m.clear();
        let d = p.deck.as_ref().unwrap();
        let mut a = Assembly::default();
        mesh::build(&mut a, &o, &p, d);
        let mut edges = HashMap::new();
        for g in a.parts.values() {
            for t in g.indices.as_chunks::<3>().0 {
                let coords = t.map(|i| g.positions[i as usize].map(|v| (v * 1e6).round() as i32));
                for (a, b) in [
                    (coords[0], coords[1]),
                    (coords[1], coords[2]),
                    (coords[2], coords[0]),
                ] {
                    let (key, dir) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                    let e = edges.entry(key).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += dir;
                }
                if d.stock == ShelfStock::RoutedTimber
                    && t.iter()
                        .all(|&i| (g.positions[i as usize][1] - o.size.y).abs() < 1e-6)
                {
                    for ab in t.windows(2) {
                        let (a, b) = (ab[0] as usize, ab[1] as usize);
                        let xyz =
                            Vec3::from_array(g.positions[a]) - Vec3::from_array(g.positions[b]);
                        let uv = Vec2::from_array(g.uvs[a]) - Vec2::from_array(g.uvs[b]);
                        assert!((uv - Vec2::new(xyz.z, -xyz.x)).length() < 2e-6);
                    }
                }
            }
        }
        assert!(
            edges.values().all(|&v| v == (2, 0)),
            "open or overlapping deck at seed {seed}"
        );
        let mut refinish = o.clone();
        refinish.finish_seed = Some(!seed);
        assert_eq!(
            serde_json::to_value(d).unwrap(),
            serde_json::to_value(ShelfDeck::sample(&refinish)).unwrap()
        );
    }
}

#[test]
fn wall_shelf_rotated_props_use_the_bearing_surface_on_archive_and_live_paths() {
    let mut room =
        IndoorManifest::generate_with_humans(42, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    room.objects.clear();
    let mut shelf = (0..128)
        .map(|seed| audit::fixture(ObjectKind::WallShelf, seed))
        .find(|o| {
            let d = ShelfDeck::sample(o);
            d.ends[1].setback_m > 0.05 && d.ends[1].width_m > 0.06
        })
        .unwrap();
    shelf.id = 0;
    shelf.yaw = 0.72;
    shelf.position = Vec3::new(0., 1.25, 0.);
    room.objects.push(shelf.clone());
    let mut prop = audit::fixture(ObjectKind::Notepad, 4);
    prop.id = 1;
    prop.support = Some(0);
    prop.size = Vec3::new(0.028, 0.009, 0.015);
    let half = shelf.size.xz() * 0.5;
    let d = ShelfDeck::sample(&shelf);
    let x = half.x - 0.027;
    let mut checked = 0;
    for turn in [-0.78, 0., 0.51] {
        prop.yaw = shelf.yaw + turn;
        for z in [half.y - 0.021, 0.] {
            prop.position = shelf
                .transform()
                .transform_point(Vec3::new(x, shelf.size.y, z));
            room.generator_version = DECK_VERSION - 1;
            assert!(room.prop_clear(&prop, &shelf, 0.));
            room.generator_version = DECK_VERSION;
            let accepted = room.prop_clear(&prop, &shelf, 0.);
            assert_eq!(accepted, z == 0.);
            assert_eq!(
                accepted,
                d.supports(&shelf, Vec2::new(x, z), prop.size.xz(), turn, 0.)
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 6);
}
