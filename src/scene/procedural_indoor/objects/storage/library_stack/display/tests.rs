use super::*;
use crate::scene::procedural_indoor::{materials::paper::jacket, objects::audit};

#[test]
fn featured_covers_face_the_aisle_and_rests_bear_the_book_inside_the_reserved_strip() {
    let mut counts = [[0; 2]; 2];
    let mut printed = 0;
    let mut plain_stacks = 0;
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::LibraryStack, seed);
        let stack = Stack::sample(&o);
        assert!(stack.featured_books(&o, INTRODUCED - 1).is_empty());
        let titles = stack.featured_books(&o, INTRODUCED);
        assert_eq!(titles, stack.featured_books(&o, INTRODUCED));
        plain_stacks += usize::from(titles.is_empty());
        for title in titles {
            let original = stack.cell(o.size, title.bay, title.level, title.side);
            let mut remaining = original;
            assert_eq!(
                Selection::for_stack(&o, INTRODUCED).unwrap().fit(
                    o.seed,
                    title.bay,
                    title.level,
                    &mut remaining
                ),
                Some(title)
            );
            let (width, height, thick) =
                (title.cover_width_m, title.cover_height_m, title.thickness_m);
            let aisle = Vec3::Z * f32::from(title.side);
            let tf = title.book_transform();
            assert!((tf.rotation * Vec3::X).dot(aisle) > 0.89);
            assert!((tf.rotation * Vec3::Y).y > 0.91);
            assert!((tf.rotation * Vec3::Z).dot(title.yaw() * Vec3::NEG_X) > 0.99);
            let mut mesh = Assembly::default();
            title.build(&mut mesh, INTRODUCED);
            audit::check(&o, &mesh).unwrap();
            let mut foot = f32::INFINITY;
            let mut lower_contact = f32::INFINITY;
            let mut rear_contact = f32::INFINITY;
            let frame = title.frame();
            for ((surface, label), g) in &mesh.parts {
                for (&point, uv) in g.positions.iter().zip(&g.uvs) {
                    let p = Vec3::from_array(point);
                    assert!(p.x >= original.left && p.x <= original.left + original.width);
                    assert!(
                        p.z * f32::from(title.side) >= original.outer - original.depth
                            && p.z * f32::from(title.side) <= original.outer
                    );
                    assert!(p.y >= original.floor - 1e-5 && p.y < original.floor + original.height);
                    assert!(
                        p.x < remaining.left - 0.012
                            || p.x > remaining.left + remaining.width + 0.012
                    );
                    let local = frame.rotation.inverse() * (p - frame.translation);
                    if label == "bookshelf" {
                        foot = foot.min(p.y - original.floor);
                        if (0.001..thick - 0.001).contains(&local.z) {
                            lower_contact = lower_contact.min(local.y.abs());
                        }
                        if (height * 0.13..height * 0.72).contains(&local.y) {
                            rear_contact = rear_contact.min(local.z.abs());
                        }
                    } else if *surface == Surface::PrintedPaper && (local.z - thick).abs() < 0.00001
                    {
                        // Front panel runs from the spine at the reader's left and
                        // ends at the fore edge on the right. V increases downwards.
                        if uv[0] >= jacket::SPINE[1] {
                            let right = uv[0] > 0.999;
                            assert_eq!(local.x > 0., right);
                            assert_eq!(local.y < height * 0.5, uv[1] > 0.5);
                            printed += 1;
                        }
                    }
                }
                if label == "bookshelf" {
                    for tri in g.indices.as_chunks::<3>().0 {
                        let ps = tri.map(|i| {
                            frame.rotation.inverse()
                                * (Vec3::from_array(g.positions[i as usize]) - frame.translation)
                        });
                        // Sample faces as well as vertices. A stand may contact
                        // the cover or the lower edge, without cutting through it.
                        for local in [
                            ps[0],
                            (ps[0] + ps[1]) * 0.5,
                            (ps[1] + ps[2]) * 0.5,
                            (ps[2] + ps[0]) * 0.5,
                            (ps[0] + ps[1] + ps[2]) / 3.,
                        ] {
                            assert!(
                                !(local.x.abs() < width * 0.5 - 0.0001
                                    && (0.0001..height - 0.0001).contains(&local.y)
                                    && (0.0001..thick - 0.0001).contains(&local.z)),
                                "{title:?} {local:?}"
                            );
                        }
                    }
                }
            }
            assert!(
                foot.abs() < 0.0004 && rear_contact < 0.001,
                "{title:?} {foot} {rear_contact}"
            );
            assert!(lower_contact < 0.001, "{title:?} {lower_contact}");
            counts[usize::from(title.rest == BookRest::GroovedTimber)]
                [usize::from(title.side > 0)] += 1;
        }
    }
    assert!(counts.into_iter().flatten().all(|n| n > 25));
    assert!(printed > 200 && (10..50).contains(&plain_stacks));
}

#[test]
fn spine_runs_pack_both_ways_and_every_lean_has_an_upright_stop() {
    let mut directions = [0; 2];
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::LibraryStack, seed);
        let stack = Stack::sample(&o);
        let displays = Selection::for_stack(&o, INTRODUCED);
        for bay in 0..stack.bays {
            for level in 0..stack.levels {
                for side in [-1, 1] {
                    let mut cell = stack.cell(o.size, bay, level, side);
                    if let Some(displays) = displays {
                        displays.fit(o.seed, bay, level, &mut cell);
                    }
                    let mut previous: Option<(f32, f32, f32, bool)> = None;
                    cell.books(
                        o.seed,
                        cell.index(bay, level),
                        stack.fill,
                        INTRODUCED,
                        |book| {
                            let lateral = book.tf.rotation * Vec3::X * book.size.x;
                            let vertical = book.tf.rotation * Vec3::Y * book.size.y;
                            let half = (lateral.abs() + vertical.abs()) * 0.5;
                            let center = book.tf.translation;
                            let (lo, hi) = (center.x - half.x, center.x + half.x);
                            let tilting = vertical.x.abs() > 0.0001;
                            assert!(lo >= cell.left - 1e-5 && hi <= cell.left + cell.width + 1e-5);
                            assert!(
                                (center.y - half.y - cell.floor).abs() < 1e-5
                                    && center.y + half.y < cell.floor + cell.height
                            );
                            assert!((book.tf.rotation * Vec3::Z).z * f32::from(side) > 0.99);
                            if let Some((left, right, h, upright)) = previous {
                                assert!(hi <= left + 1e-5 || lo >= right - 1e-5);
                                if tilting {
                                    let rightward = vertical.x > 0.;
                                    let gap = if rightward { left - hi } else { lo - right };
                                    assert!(
                                        upright && h >= book.size.y && (gap - 0.00025).abs() < 1e-5
                                    );
                                    directions[usize::from(rightward)] += 1;
                                }
                            } else {
                                assert!(!tilting);
                            }
                            previous = Some((lo, hi, book.size.y, !tilting));
                        },
                    );
                }
            }
        }
    }
    assert!(directions.into_iter().all(|n| n > 100));
}
