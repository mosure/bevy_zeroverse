use super::*;
use crate::scene::procedural_indoor::objects::{audit, part_label};

fn ray_hits(g: &Geometry, q: Vec2) -> bool {
    g.indices.as_chunks::<3>().0.iter().any(|tri| {
        let t = tri.map(|i| Vec3::from(g.positions[i as usize]).xz());
        let edge = [0, 1, 2].map(|i| (t[(i + 1) % 3] - t[i]).perp_dot(q - t[i]));
        (t[1] - t[0]).perp_dot(t[2] - t[0]).abs() > 1e-10
            && (edge.iter().all(|&x| x >= 0.) || edge.iter().all(|&x| x <= 0.))
    })
}

#[test]
fn portions_fit_real_shelves_trays_and_neighbors_with_an_open_bagel_and_supported_liner() {
    let mut recipes = [0; 4];
    let mut layout = [0; 2];
    let mut empty = 0;
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::DisplayCase, seed);
        o.variant = 0;
        let case = parameters(&o);
        let cabinet = service_glazing::Cabinet::new(o.size, &case, case.service_glazing.unwrap());
        for b in case.bakery_batches(&o, VERSION) {
            let shelf = cabinet.stock_space(&o, &case, b.level);
            assert!(b
                .tray
                .footprint()
                .inside(shelf.bounds.0, shelf.bounds.1, 0.014));
            assert_eq!(b.shelf_y_m, shelf.y);
            layout[usize::from(b.arranged_in_rows)] += 1;
            if b.portions.is_empty() {
                empty += 1;
            }
            for (i, &p) in b.portions.iter().enumerate() {
                recipes[b.recipe as usize] += 1;
                assert!((p.center_m.abs() + p.radius_m)
                    .cmple(b.tray.size_m * 0.5 - 0.0079)
                    .all());
                assert!(p.height_m + b.tray.deck_m < shelf.headroom);
                assert!(b.portions[..i]
                    .iter()
                    .all(|other| p.center_m.distance(other.center_m) + 1e-6
                        >= p.radius_m + other.radius_m + 0.004));
                let mut a = Assembly::default();
                p.build(&mut a, b.recipe, Transform::IDENTITY);
                let (lo, hi) = a.bounds();
                assert!(lo.y.abs() < 1e-6 && hi.y <= p.height_m + 1e-6);
                let mut saw_liner = false;
                for ((_, name), g) in &a.parts {
                    assert!(g
                        .positions
                        .iter()
                        .all(|v| Vec3::from(*v).xz().length() <= p.radius_m + 1e-6));
                    assert!(g.uvs.iter().flatten().all(|v| v.is_finite()));
                    if part_label(name) == "paper" {
                        saw_liner = true;
                        let top = g.positions.iter().map(|v| v[1]).fold(0f32, f32::max);
                        assert!((0.53..0.68).contains(&(top / p.height_m)));
                        let crust = &a.parts[&(
                            Surface::Terracotta,
                            crate::scene::procedural_indoor::materials::baking::Finish::Crust
                                .label(VERSION)
                                .into(),
                        )];
                        for v in g.positions.iter().filter(|v| v[1] == top) {
                            assert!(ray_hits(crust, Vec3::from(*v).xz()));
                        }
                    } else {
                        assert_eq!(part_label(name), "food");
                        if b.recipe == Bake::Bagel {
                            assert!(!ray_hits(g, Vec2::ZERO));
                            assert!(ray_hits(g, Vec2::X * p.radius_m * 0.62));
                        }
                    }
                }
                assert_eq!(saw_liner, b.recipe == Bake::Muffin);
            }
        }
        assert!(case.bakery_batches(&o, VERSION - 1).is_empty());
        o.variant = 16;
        assert!(case.bakery_batches(&o, VERSION).is_empty());
    }
    assert!(recipes.into_iter().all(|n| n > 25), "{recipes:?}");
    assert!(empty > 3 && layout.into_iter().all(|n| n > 20));
}

#[test]
fn shelf_stock_replays_independently_of_cosmetic_finish_when_support_is_fixed() {
    let bounds = (Vec2::new(-0.58, -0.23), Vec2::new(0.58, 0.26));
    for seed in 0..64 {
        let mut o = audit::fixture(ObjectKind::DisplayCase, seed);
        let run = |o: &IndoorObject| {
            serde_json::to_value(BakeryBatch::fit(o, 1, 0.62, 0.09, bounds)).unwrap()
        };
        let original = run(&o);
        o.finish_seed = Some(!seed);
        assert_eq!(original, run(&o));
        assert!(BakeryBatch::fit(&o, 1, 0.62, 0.04, bounds).is_none());
    }
}

#[test]
fn historical_food_cases_and_exhibition_contents_replay_their_meshes() {
    use crate::scene::procedural_indoor::objects::build_object_at_version;
    // Recorded from the previous generator's varied-dimension, varied-finish catalog.
    for (seed, triangles) in [(0, 22223), (1, 11789), (5, 10204), (21, 33260)] {
        let mut o = audit::fixture(ObjectKind::DisplayCase, seed);
        o.finish_seed = Some(seed);
        let old = build_object_at_version(&o, None, VERSION - 1);
        assert_eq!(audit::check(&o, &old).unwrap().triangles, triangles);
        assert!(old
            .parts
            .keys()
            .all(|(_, label)| !label.starts_with("food#finish117")));
        o.variant = 16;
        let gallery = build_object_at_version(&o, None, VERSION);
        let earlier = build_object_at_version(&o, None, VERSION - 1);
        assert_eq!(gallery.parts.len(), earlier.parts.len());
        for (key, mesh) in gallery.parts {
            let reference = &earlier.parts[&key];
            assert_eq!(mesh.positions, reference.positions);
            assert_eq!(mesh.normals, reference.normals);
            assert_eq!(mesh.uvs, reference.uvs);
            assert_eq!(mesh.indices, reference.indices);
        }
    }
}
