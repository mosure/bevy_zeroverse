use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn first_hit(g: &Geometry, x: f32) -> Option<f32> {
    let sample = Vec2::new(x, 0.073);
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|tri| {
            let [a, b, c] = [0, 1, 2].map(|i| Vec3::from_array(g.positions[tri[i] as usize]));
            let (ab, ac, ap) = (b.xz() - a.xz(), c.xz() - a.xz(), sample - a.xz());
            let det = ab.perp_dot(ac);
            if det.abs() < 1e-9 {
                return None;
            }
            let (u, v) = (ap.perp_dot(ac) / det, ab.perp_dot(ap) / det);
            (u >= 0. && v >= 0. && u + v <= 1.).then_some(a.y + u * (b.y - a.y) + v * (c.y - a.y))
        })
        .min_by(f32::total_cmp)
}

#[test]
fn cut_boards_are_closed_metric_solids_supported_by_the_runner() {
    let pitch = Vec2::new(0.6, 0.44);
    let base = 0.025;
    let bearing = base + 0.004 - 0.00035;
    let mut families = BTreeSet::new();
    for seed in 0..128 {
        let tile = CeilingTile::sample(seed, 2);
        assert!(tile.valid());
        families.insert((tile.edge as u8, tile.facing as u8));
        let mut g = Geometry::default();
        board(&mut g, tile, Vec2::ZERO, pitch, base, &mut stream(seed, 9));
        let full = pitch.x * 0.5 - 0.002 - tile.clearance_m;
        let flange = pitch.x * 0.5 - tile.flange_width_m * 0.5;
        assert!(full - flange > 0.002);
        let bottom = bearing - tile.drop_m;
        assert!((first_hit(&g, 0.).unwrap() - bottom).abs() < 1e-6);
        assert!((first_hit(&g, (flange + full) * 0.5).unwrap() - bearing).abs() < 1e-6);
        assert!(first_hit(&g, full + 0.0005).is_none());
        if tile.edge == TileEdge::BeveledTegular {
            let neck = flange - tile.clearance_m;
            let middle = neck - tile.drop_m * 0.65 * 0.5;
            assert!((first_hit(&g, middle).unwrap() - (bearing + bottom) * 0.5).abs() < 2e-6);
        }
        let mut incidence = BTreeMap::new();
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = [0, 1, 2].map(|i| Vec3::from_array(g.positions[tri[i] as usize]));
            let normal = (b - a).cross(c - a);
            for &i in tri {
                assert!(normal.dot(Vec3::from_array(g.normals[i as usize])) > 1e-9);
            }
            for (p, q) in [(a, b), (b, c), (c, a)] {
                let (p, q) = (
                    p.to_array().map(f32::to_bits),
                    q.to_array().map(f32::to_bits),
                );
                let edge = incidence.entry((p.min(q), p.max(q))).or_insert((0, 0));
                edge.0 += 1;
                edge.1 += if p < q { 1 } else { -1 };
            }
        }
        assert!(incidence.values().all(|&v| v == (2, 0)));
        assert_eq!(
            g.indices.len() / 3,
            if tile.edge == TileEdge::SquareLayIn {
                12
            } else {
                28
            }
        );
        let mut another_cut = Geometry::default();
        board(
            &mut another_cut,
            tile,
            Vec2::ZERO,
            pitch,
            base,
            &mut stream(seed, 10),
        );
        assert_eq!(g.positions, another_cut.positions);
        assert_ne!(g.uvs, another_cut.uvs);
        let first_edge = Vec2::from_array(g.uvs[1]) - Vec2::from_array(g.uvs[0]);
        assert!(
            (first_edge.length()
                - Vec3::from_array(g.positions[0]).distance(Vec3::from_array(g.positions[1])))
            .abs()
                < 1e-6
        );
    }
    assert_eq!(families.len(), 9);
}

#[test]
fn anchors_and_archived_grids_replay_without_building_tiles_to_plan_hangers() {
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    let room =
        IndoorManifest::generate_with_humans(75243, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    // The test fixture requires a tile module, independent of its room profile.
    let mut module = room.ceiling_installation.as_ref().unwrap().modules[0].clone();
    module.system = CeilingSystem::SuspendedTiles;
    let archived: CeilingShape =
        serde_json::from_str(r#"{"kind":"Grid","aspect":1.35,"recess_m":0.017}"#).unwrap();
    module.detail = Some(CeilingDetail {
        finish: 2,
        shape: archived,
    });
    let f = Frame::new(&module, 2);
    let mut assembled = Assembly::default();
    let original = build(&mut assembled, &f, 1.35, Infill::ArchivedTile(0.017), false);
    let replay = module.geometry(&room);
    assert!(
        replay
            .parts
            .contains_key(&(Surface::Ceiling, "ceiling".to_string()))
            || replay
                .parts
                .contains_key(&(Surface::Ceiling, f.label.clone()))
    );
    assert!(!replay
        .parts
        .keys()
        .any(|(_, key)| key == acoustic_tile::GRID_LABEL));
    assert_eq!(original.len(), 4);
    for infill in [
        Infill::ArchivedTile(0.017),
        Infill::Mineral {
            base: 0.017 + 0.009,
            stock: CeilingTile::sample(1, 2),
        },
    ] {
        let mut planned = Assembly::default();
        let anchors = build(&mut planned, &f, 1.35, infill, true);
        assert!(planned.parts.is_empty());
        assert_eq!(original, anchors);
    }
}
