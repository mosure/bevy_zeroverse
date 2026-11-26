use super::*;
use crate::scene::procedural_indoor::{layout::ObjectKind, objects::audit};
use std::collections::BTreeSet;

#[test]
fn erect_fronds_can_show_a_vertical_leaf_face() {
    let leaflet = leaflet::Blade {
        root: Vec3::ZERO,
        delta: Vec3::Z * 0.12,
        normal: Vec3::X,
        arch: 0.,
        half_width: 0.013,
        fold: 0.,
        twist: 0.,
        distal_side: 1.,
    };
    for kind in [Kind::Palm, Kind::Fern] {
        let mut mesh = Geometry::default();
        leaflet.write(&mut mesh, kind);
        assert!(mesh
            .normals
            .iter()
            .all(|&n| Vec3::from(n).dot(Vec3::X) > 0.999));
    }
}

#[test]
fn leaf_bases_land_on_the_mesh_path_with_staggered_ranks_and_registered_veins() {
    let space = Envelope {
        soil: 0.24,
        top: 1.1,
        radius: 0.45,
        spread: 0.36,
    };
    for (variant, vertices) in [(1, 14), (4, 20)] {
        for seed in 0..24 {
            let mut o = audit::fixture(ObjectKind::Plant, seed);
            o.variant = variant;
            let g = Growth::sample(o.seed);
            let habit = Habit::sample(&o, &g).unwrap();
            let mut rng = stream(o.seed, 87);
            let crown = Vec3::new(
                0.04,
                space.soil + habit.crown_elevation * (space.top - space.soil),
                0.02,
            );
            for age in [0.05, 0.90] {
                let frond = Frond::sample(
                    crown,
                    radial(g.shoot_phase),
                    age,
                    space,
                    variant == 1,
                    &mut rng,
                );
                let mut geo = Geometry::default();
                frond.add_leaves(&mut geo, &habit, space, &mut rng);
                let mut nodes = Vec::new();
                for (pos, uv) in geo
                    .positions
                    .chunks_exact(vertices)
                    .zip(geo.uvs.chunks_exact(vertices))
                {
                    let p = Vec3::from(pos[0]);
                    assert!(frond.points.windows(2).any(|w| {
                        let d = w[1] - w[0];
                        let t = (p - w[0]).dot(d) / d.length_squared();
                        p.distance(w[0] + d * t.clamp(0., 1.)) < 1e-6
                    }));
                    assert_eq!(uv[0], [0.5, 0.]);
                    assert_eq!(uv[vertices - 1], [0.5, 1.]);
                    assert!(uv.iter().flatten().all(|x| (0.0..=1.0).contains(x)));
                    for section in uv[1..vertices - 1].as_chunks::<3>().0 {
                        assert!(section[0][0] < 0.5 && section[1][0] == 0.5 && section[2][0] > 0.5);
                    }
                    nodes.push(p);
                }
                assert_eq!(nodes.len() % 2, 1);
                assert!(nodes[..nodes.len() - 1]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|pair| pair[0].distance_squared(pair[1]) > 1e-10));
                assert!(
                    Vec3::from(*geo.positions.last().unwrap()).distance(frond.points[7]) < 1e-6
                );
                assert!(geo.normals.iter().all(|n| Vec3::from(*n).is_normalized()));
                assert!(geo
                    .positions
                    .iter()
                    .map(|&p| Vec3::from(p))
                    .all(|p| p.y >= space.soil
                        && p.y <= space.top
                        && p.xz().length() < space.radius));
            }
        }
    }
}

#[test]
fn sampled_crowns_change_foliage_at_the_version_boundary_and_preserve_pots() {
    for variant in [1, 4] {
        let mut crowns = BTreeSet::new();
        let mut counts = BTreeSet::new();
        for seed in 0..40 {
            let mut o = audit::fixture(ObjectKind::Plant, seed);
            o.variant = variant;
            let habit = Habit::sample(&o, &Growth::sample(o.seed)).unwrap();
            crowns.insert(habit.crowns);
            counts.insert(habit.fronds);
            assert_eq!(habit.crown_elevation > 0.20, variant == 1);
            let mut old = Assembly::default();
            let mut new = Assembly::default();
            build_at_version(&mut old, &o, INTRODUCED - 1);
            build_at_version(&mut new, &o, INTRODUCED);
            assert!(!old.parts.keys().any(|(s, _)| *s == Surface::Succulent));
            assert!(new.parts.keys().any(|(s, _)| *s == Surface::Succulent));
            for (key, geo) in &old.parts {
                if matches!(key.0, Surface::Bark | Surface::Leaf | Surface::LeafLight) {
                    continue;
                }
                let saved = &new.parts[key];
                assert_eq!(geo.positions, saved.positions);
                assert_eq!(geo.normals, saved.normals);
                assert_eq!(geo.indices, saved.indices);
                assert_eq!(geo.uvs, saved.uvs);
            }
            let stats = audit::check(&o, &new).unwrap();
            assert!(stats.triangles < 20_000);
        }
        assert_eq!(crowns, BTreeSet::from([1, 2, 3]));
        assert!(counts.len() >= 5);
    }
}
