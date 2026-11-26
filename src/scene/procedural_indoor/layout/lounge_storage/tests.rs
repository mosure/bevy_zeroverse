use super::*;
use crate::scene::procedural_indoor::{
    envelope::polygon,
    items::distribution::ClutterSettings,
    objects::{self, audit, Assembly},
};
use std::collections::BTreeSet;

fn fixture() -> IndoorManifest {
    let mut s = IndoorManifest::generate_with_humans(3, IndoorLayout::Lounge, 0.65, 0, 0.).unwrap();
    s.objects.clear();
    s.humans.clear();
    s.ceiling_installation = None;
    s.clutter = ClutterSettings {
        probability: 1.,
        minimum: 1.,
        maximum: 1.,
        ..default()
    };
    s
}

fn table(seed: u64) -> IndoorObject {
    let mut o = audit::fixture(ObjectKind::CoffeeTable, seed);
    o.id = 0;
    o.size = Vec3::new(1.32, 0.47, 0.67);
    o.position = Vec3::new(-0.2, -0.17, 0.25);
    o.yaw = 0.79;
    o
}

#[test]
fn lower_shelf_resolves_physical_contact_independent_of_top_and_world_elevation() {
    let mut s = fixture();
    let (table, deck) = (0..128)
        .find_map(|i| {
            let t = table(i);
            let d = tables::parameters(&t).lounge_shelf(&t, INTRODUCED)?;
            (d.deck == tables::ShelfDeck::FoldedTray && d.width_m > 0.26).then_some((t, d))
        })
        .unwrap();
    let mut item = audit::fixture(ObjectKind::Notepad, 3);
    item.id = 1;
    item.support = Some(0);
    item.solid = false;
    item.size = Vec3::new(0.12, 0.019, 0.15);
    item.yaw = table.yaw + 0.49;
    item.position = table.transform().transform_point(Vec3::new(
        deck.center_xz_m.x,
        deck.top_y_m,
        deck.center_xz_m.y,
    ));
    let mut top = item.clone();
    top.id = 2;
    top.position.y = table.position.y + table.size.y;
    s.objects.extend([table.clone(), item.clone(), top.clone()]);
    assert_eq!(s.support_plane(&item), Some(SupportPlane::LowerShelf));
    assert_eq!(s.support_plane(&top), Some(SupportPlane::Top));
    let mut cache = std::collections::HashMap::new();
    for (object, expected) in [(&item, true), (&top, true)] {
        assert_eq!(s.prop_clear(object, &table, 0.01), expected);
        assert_eq!(
            s.prop_clear_cached(object, &table, 0.01, Some(&mut cache)),
            expected
        );
    }
    assert_eq!(cache.len(), 1);
    let mut invalid = item.clone();
    invalid.position.y += 0.012;
    assert_eq!(s.support_plane(&invalid), None);
    assert!(!s.prop_clear(&invalid, &table, 0.));
    invalid = item.clone();
    invalid.size.y = deck.underside_y_m - deck.top_y_m + 0.001;
    assert!(!s.prop_clear(&invalid, &table, 0.));
    invalid = item.clone();
    let (_, edge) = deck.storage_bounds();
    invalid.position = table
        .transform()
        .transform_point(Vec3::new(edge.x, deck.top_y_m, edge.y));
    assert!(!s.prop_clear(&invalid, &table, 0.));
    let mut peer = item.clone();
    peer.id = 3;
    s.objects.push(peer);
    assert!(!s.prop_clear(&item, &table, 0.));
    s.objects[3].support = None;
    assert!(!s.prop_clear(&item, &table, 0.));
    s.generator_version = INTRODUCED - 1;
    assert_eq!(s.support_plane(&item), None);
    assert_eq!(s.support_plane(&top), Some(SupportPlane::Top));
}

// Intersect the built faces themselves. The check is independent of the
// placement rectangle, nominal dimensions and analytic slat-gap criterion.
fn contact_faces(mesh: &Assembly, y: f32, outward_y: f32) -> Vec<[Vec3; 3]> {
    mesh.parts
        .values()
        .flat_map(|g| {
            g.indices.as_chunks::<3>().0.iter().filter_map(move |t| {
                let p = t.map(|i| Vec3::from_array(g.positions[i as usize]));
                let normal = (p[1] - p[0]).cross(p[2] - p[0]);
                (normal.y * outward_y > 1e-10 && p.iter().all(|v| (v.y - y).abs() < 0.00002))
                    .then_some(p)
            })
        })
        .collect()
}

// Clip the two horizontal meshes against one another in table-local metres.
fn balanced_contact(deck: &[[Vec3; 3]], base: &[[Vec3; 3]], center: Vec3, rotation: Quat) -> bool {
    let mut angles = Vec::new();
    for triangle in deck {
        let mut board = triangle.map(|p| p.xz());
        if polygon::area(&board) < 0. {
            board.swap(1, 2);
        }
        for foot in base {
            let mut patch = foot.map(|p| (center + rotation * p).xz()).to_vec();
            for (a, b) in polygon::edges(&board) {
                let n = (b - a).perp();
                patch = polygon::clip(&patch, n, n.dot(a));
                if patch.len() < 3 {
                    break;
                }
            }
            if polygon::area(&patch).abs() > 1e-10 {
                angles.extend(patch.iter().map(|p| (*p - center.xz()).to_angle()));
            }
        }
    }
    // The centre lies strictly inside the convex hull of the actual contact
    // patches iff their vertices occupy more than a semicircle about it.
    angles.sort_unstable_by(f32::total_cmp);
    angles.first().zip(angles.last()).is_some_and(|(a, b)| {
        std::f32::consts::TAU + a - b < std::f32::consts::PI
            && angles
                .windows(2)
                .all(|v| v[1] - v[0] < std::f32::consts::PI)
    })
}

#[test]
fn shelf_stock_has_rendered_bearing_replays_and_obeys_clutter_settings() {
    let mut s = fixture();
    let mut styles = BTreeSet::new();
    let mut stock = BTreeSet::new();
    let mut empty = 0;
    let mut multiple = 0;
    let mut slatted = 0;
    for seed in 0..256 {
        let t = table(seed);
        let Some(deck) = tables::parameters(&t).lounge_shelf(&t, s.generator_version) else {
            continue;
        };
        s.objects = vec![t.clone()];
        let mut replay = s.clone();
        s.furnish_lounge_shelves();
        replay.furnish_lounge_shelves();
        assert_eq!(s.objects, replay.objects);
        empty += usize::from(s.objects.len() == 1);
        multiple += usize::from(s.objects.len() > 2);
        let bearing = contact_faces(&objects::build_manifest_object(&s, &t), deck.top_y_m, 1.);
        for item in &s.objects[1..] {
            styles.insert(deck.deck.name());
            stock.insert((
                item.kind.class_name(),
                item.kind == ObjectKind::Books && literature::parameters(item).volumes == 1,
            ));
            assert_eq!(s.support_plane(item), Some(SupportPlane::LowerShelf));
            assert!(s.prop_clear(item, &t, PLACEMENT_MARGIN_M - 1e-5));
            let mesh = objects::build_manifest_object(&s, item);
            audit::check(item, &mesh).unwrap();
            let center = t
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(item.position);
            let base = contact_faces(&mesh, 0., -1.);
            let rotation = Quat::from_rotation_y(item.yaw - t.yaw);
            assert!(
                balanced_contact(&bearing, &base, center, rotation),
                "unbalanced {:?} on {:?}, seed={seed}, size={:?}",
                item.kind,
                deck.deck,
                item.size
            );
            assert!(!balanced_contact(
                &bearing,
                &base,
                center + Vec3::X * 2.,
                rotation
            ));
            if deck.deck == tables::ShelfDeck::SpacedSlats {
                slatted += 1;
            }
        }
        replay.objects.truncate(1);
        replay.clutter.probability = 0.;
        replay.furnish_lounge_shelves();
        assert_eq!(replay.objects.len(), 1);
    }
    assert_eq!(styles.len(), 3);
    assert_eq!(stock.len(), 4);
    assert!(
        empty >= 8 && multiple >= 3 && slatted >= 8,
        "{empty}, {multiple}, {slatted}"
    );
}
