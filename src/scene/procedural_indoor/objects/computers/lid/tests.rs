use super::*;
use crate::scene::procedural_indoor::{motion::IndoorMotionConfig, objects::audit};
use std::collections::{BTreeMap, BTreeSet};

fn fixture(seed: u64) -> IndoorObject {
    let mut o = audit::fixture(ObjectKind::Laptop, seed);
    let mut r = stream(seed, 41);
    o.size = Vec3::new(
        r.random_range(0.25..0.44),
        r.random_range(0.205..0.34),
        r.random_range(0.225..0.42),
    );
    o
}

#[test]
fn active_face_and_optics_tile_one_closed_oriented_lid() {
    let mut cameras = BTreeSet::new();
    let mut narrow_chin = 0;
    let mut broad_chin = 0;
    for seed in 0..384 {
        let o = fixture(seed);
        let mut p = parameters(&o);
        // Also exercise the thinnest manufactured rails on a deep, generously
        // rounded shell, where independently sized openings could cross.
        if seed % 4 == 1 {
            p.corner_m = 0.008;
            p.bezel_m = 0.004;
            p.lid_m = 0.011;
        }
        let deck = Deck::fit(&o, &p);
        let face = p.lid_face(&o, INTRODUCED).unwrap();
        assert!(p.lid_face(&o, INTRODUCED - 1).is_none());
        cameras.insert(face.camera.map(|c| c.window as u8));
        narrow_chin += usize::from(face.chin_m < face.top_bezel_m);
        broad_chin += usize::from(face.chin_m > 1.6 * face.top_bezel_m);
        let mut a = Assembly::default();
        let lid = Transform::from_xyz(0., p.chassis_m + 0.004, deck.rear)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2 - p.hinge_radians));
        build(&mut a, &o, &p, &Housing::for_object(&o), deck, lid);
        audit::check(&o, &a).unwrap();
        let mut edges = BTreeMap::new();
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let vertices = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let [a, b, c] = vertices;
                let n = (b - a).cross(c - a);
                assert!(
                    tri.iter()
                        .all(|&i| n.dot(Vec3::from_array(g.normals[i as usize])) > 0.),
                    "seed {seed}: winding"
                );
                let [a, b, c] = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
                assert!((b - a).perp_dot(c - a).abs() > 1e-13, "seed {seed}: UV");
                let key =
                    vertices.map(|p| p.to_array().map(|x| if x == 0. { 0 } else { x.to_bits() }));
                for (a, b) in [(key[0], key[1]), (key[1], key[2]), (key[2], key[0])] {
                    let entry = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                    entry.0 += 1;
                    entry.1 += if a < b { 1 } else { -1 };
                }
            }
        }
        assert!(
            edges.values().all(|&edge| edge == (2, 0)),
            "seed {seed}: open or overlapping edge"
        );
        let local = lid.compute_affine().inverse();
        // The full front projection (including the sloping reveals) covers the
        // rear footprint plus its bevel, exactly once. A buried front plate fails.
        let mut front_area = 0.;
        let mut back_area = 0.;
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = tri.map(|i| {
                    local
                        .transform_point3(Vec3::from_array(g.positions[i as usize]))
                        .truncate()
                });
                let area = (b - a).perp_dot(c - a) * 0.5;
                front_area += area.max(0.);
                back_area -= area.min(0.);
            }
        }
        let outline = palmrest::rounded_outline(
            Vec2::new(o.size.x, deck.height) * 0.5,
            p.corner_m,
            Vec2::ZERO,
        );
        let area = (0..outline.len())
            .map(|i| outline[i].perp_dot(outline[(i + 1) % outline.len()]) * 0.5)
            .sum::<f32>();
        assert!(
            (front_area - area).abs() < 2e-6 && (back_area - area).abs() < 2e-6,
            "seed {seed}: {front_area} {back_area} {area}"
        );
        let used: BTreeSet<_> = a
            .parts
            .keys()
            .filter_map(|(surface, label)| {
                crate::scene::procedural_indoor::materials::finish_slot(*surface, label)
                    .and_then(LidStock::from_slot)
                    .map(LidStock::slot)
            })
            .collect();
        let expected = face.stocks().map(LidStock::slot).collect();
        assert_eq!(used, expected);
    }
    assert_eq!(cameras.len(), 5);
    assert!(narrow_chin > 15 && broad_chin > 40);
}

#[test]
fn replay_preserves_the_old_screen_and_the_moving_leaf_uses_the_whole_new_front() {
    for seed in 0..32 {
        let o = fixture(seed);
        let p = parameters(&o);
        let deck = Deck::fit(&o, &p);
        let hinge = Vec3::new(0., p.chassis_m + 0.004, deck.rear);
        let inv = Transform::from_translation(hinge)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2 - p.hinge_radians))
            .compute_affine()
            .inverse();
        let face = p.lid_face(&o, INTRODUCED).unwrap();
        let motion = IndoorMotionConfig {
            prop_fraction: 1.,
            ..default()
        };
        for version in [INTRODUCED - 1, INTRODUCED] {
            let a = build_object_at_version(&o, Some(&motion), version);
            let joint = a
                .joints
                .iter()
                .find(|j| j.attachment.as_deref() == Some("contact_laptop_hinge"))
                .unwrap();
            assert_eq!(joint.pivot, hinge);
            let (screen, g) = a
                .parts
                .iter()
                .find(|((s, _), _)| *s == Surface::Screen)
                .unwrap();
            assert!(joint.parts.contains(screen));
            let (size, bottom, z) = if version == INTRODUCED {
                for stock in face.stocks() {
                    assert!(
                        joint.parts.iter().any(|(surface, label)| {
                            crate::scene::procedural_indoor::materials::finish_slot(*surface, label)
                                == Some(stock.slot())
                        }),
                        "{seed} {stock:?}"
                    );
                }
                (
                    face.active_size_m,
                    face.chin_m,
                    p.lid_m * 0.5 - face.screen_recess_m,
                )
            } else {
                (
                    Vec2::new(o.size.x, deck.height) - 2. * p.bezel_m,
                    p.bezel_m,
                    p.lid_m * 0.5 + 0.0012,
                )
            };
            for (v, uv) in g.positions.iter().zip(&g.uvs) {
                let local = inv.transform_point3(Vec3::from_array(*v));
                let expected = Vec2::new(0.5 + local.x / size.x, 1. - (local.y - bottom) / size.y);
                assert!(
                    (local.z - z).abs() < 1e-6 && expected.distance(Vec2::from_array(*uv)) < 2e-5
                );
            }
        }
    }
}
