use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};
use std::collections::BTreeSet;

#[test]
fn library_stacks_fit_both_collections_on_their_decks_and_replay() {
    let mut forms = BTreeSet::new();
    let mut tilts = 0;
    let mut mismatched_faces = 0;
    for seed in 0..64 {
        let mut o = audit::fixture(ObjectKind::LibraryStack, seed);
        o.finish_seed = Some(seed);
        let p = Stack::sample(&o);
        forms.insert((p.open_ends, p.bays, p.levels));
        let mesh = build_object(&o);
        let checked = audit::check(&o, &mesh).unwrap();
        assert!(checked.triangles < 24000);
        assert!(mesh
            .parts
            .keys()
            .all(|(_, label)| matches!(part_label(label), "books" | "bookshelf")));
        assert!(mesh.parts.keys().any(|(s, _)| *s == Surface::PrintedPaper));
        let t = p.board_m;
        let bay_width = (o.size.x - t) / p.bays as f32;
        let rise = (o.size.y - t - p.base_m) / p.levels as f32;
        for i in 0..p.bays {
            for j in 0..p.levels {
                let mut faces = [Vec::new(), Vec::new()];
                for (k, side) in [-1, 1].into_iter().enumerate() {
                    let cell = stock::Cell {
                        left: -o.size.x * 0.5 + t + i as f32 * bay_width,
                        width: bay_width - t,
                        floor: p.base_m + t + j as f32 * rise,
                        height: rise - t,
                        depth: (o.size.z - t) * 0.5,
                        outer: o.size.z * 0.5,
                        side,
                    };
                    let idx = (i * 16 + 2 * j + k) as u64;
                    let mut prev = cell.left;
                    let mut stop_height = None;
                    cell.books(o.seed, idx, p.fill, display::INTRODUCED - 1, |book| {
                        let mut lo = Vec3::splat(f32::INFINITY);
                        let mut hi = Vec3::splat(f32::NEG_INFINITY);
                        for x in [-0.5, 0.5] {
                            for y in [-0.5, 0.5] {
                                for z in [-0.5, 0.5] {
                                    let point =
                                        book.tf.transform_point(book.size * Vec3::new(x, y, z));
                                    lo = lo.min(point);
                                    hi = hi.max(point);
                                }
                            }
                        }
                        assert!(lo.x >= prev - 1e-5 && hi.x <= cell.left + cell.width);
                        let tilt = (book.tf.rotation * Vec3::Y).x;
                        if tilt.abs() > 0.001 {
                            assert!(tilt < 0. && (lo.x - prev - 0.00025).abs() < 1e-5);
                            assert!(
                                stop_height.is_some_and(|y| y >= cell.floor + book.size.y - 1e-5)
                            );
                        }
                        stop_height = (tilt.abs() <= 0.001).then_some(hi.y);
                        prev = hi.x;
                        assert!(
                            (lo.y - cell.floor).abs() < 1e-5 && hi.y < cell.floor + cell.height
                        );
                        assert!(
                            lo.z >= -cell.outer
                                && hi.z <= cell.outer
                                && if side > 0 {
                                    lo.z >= t * 0.5
                                } else {
                                    hi.z <= -t * 0.5
                                }
                        );
                        assert!((book.tf.rotation * Vec3::Z).z * side as f32 > 0.99);
                        tilts += usize::from((book.tf.rotation * Vec3::Y).x.abs() > 0.05);
                        faces[k].push(book.size.to_array().map(f32::to_bits));
                    });
                    let mut replay = Vec::new();
                    cell.books(o.seed, idx, p.fill, display::INTRODUCED - 1, |book| {
                        replay.push(book.size.to_array().map(f32::to_bits))
                    });
                    assert_eq!(faces[k], replay);
                }
                mismatched_faces += usize::from(faces[0] != faces[1]);
            }
        }
    }
    assert!(forms.len() > 12 && tilts > 100 && mismatched_faces > 300);
}
