use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object_at_version, part_label};
use std::collections::BTreeSet;

fn four_corners(poly: &[Vec2], lo: Vec2, hi: Vec2, margin: f32) -> bool {
    [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
        .iter()
        .all(|&q| {
            (0..poly.len()).all(|i| {
                let edge = poly[(i + 1) % poly.len()] - poly[i];
                edge.perp_dot(q - poly[i]) >= (margin + 0.012) * edge.length()
            })
        })
}

#[test]
fn tabletop_half_planes_match_four_corner_admission_and_four_mounts() {
    let mut admitted = 0;
    let mut rejected = 0;
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::Table, seed);
        let poly = outline(&o);
        let mut rng = stream(seed, 4040);
        for i in 0..poly.len() {
            let edge = poly[(i + 1) % poly.len()] - poly[i];
            let margin = rng.random_range(0.0..0.04);
            let half = Vec2::new(rng.random_range(0.001..0.12), rng.random_range(0.001..0.16));
            let center = match i % 3 {
                0 => (poly[i] + edge * 0.5) + edge.perp().normalize_or_zero() * (margin + 0.012),
                1 => poly[i] * rng.random_range(0.29..0.98),
                _ => Vec2::new(rng.random_range(-0.9..0.9), rng.random_range(-0.7..0.7)),
            };
            for h in [Vec2::ZERO, half, -half] {
                let actual = supports_outline(&poly, center - h, center + h, margin);
                assert_eq!(actual, four_corners(&poly, center - h, center + h, margin));
                if actual {
                    admitted += 1
                } else {
                    rejected += 1
                }
            }
            let span = center.abs();
            let independent = [-1., 1.].into_iter().all(|x| {
                [-1., 1.].into_iter().all(|z| {
                    let q = Vec2::new(x, z) * span;
                    four_corners(&poly, q - half, q + half, margin)
                })
            });
            assert_eq!(
                supports_outline(&poly, -span - half, span + half, margin),
                independent
            );
        }
    }
    assert!(admitted > 1000 && rejected > 1000);
}

#[test]
fn fitted_lounge_shelves_bear_on_their_rails_and_keep_open_space() {
    let mut decks = BTreeSet::new();
    let mut sampled = 0;
    let mut old_frames = 0;
    let mut headings = BTreeSet::new();
    for seed in 0..176 {
        for (kind, size) in [
            (ObjectKind::CoffeeTable, Vec3::new(1.70, 0.33, 0.44)),
            (ObjectKind::CoffeeTable, Vec3::new(0.70, 0.48, 1.18)),
            (ObjectKind::SideTable, Vec3::new(0.42, 0.45, 0.40)),
            (ObjectKind::SideTable, Vec3::new(0.71, 0.64, 0.58)),
        ] {
            let mut o = audit::fixture(kind, seed);
            o.size = size;
            let p = parameters(&o);
            let Some(s) = p.lounge_shelf(&o, INTRODUCED) else {
                old_frames += 1;
                continue;
            };
            sampled += 1;
            assert_eq!(p.lounge_shelf(&o, INTRODUCED), Some(s));
            assert!(p.lounge_shelf(&o, INTRODUCED - 1).is_none());
            decks.insert(s.deck.name());
            headings.insert(s.runs_along_x);
            let under = size.y - p.top_thickness - 0.012;
            let poly = outline(&o);
            let r = Vec2::splat(s.post_half_width_m);
            assert!(four_corners(
                &poly,
                s.center_xz_m - s.post_centers_half_span_m - r,
                s.center_xz_m + s.post_centers_half_span_m + r,
                0.006
            ));
            let q = |l: f32, w: f32, y: f32| {
                let c = s.center_xz_m
                    + if s.runs_along_x {
                        Vec2::new(l, w)
                    } else {
                        Vec2::new(w, l)
                    };
                Vec3::new(c.x, y, c.y)
            };
            let mut only = Assembly::default();
            s.build(&mut only, &o, under);
            assert!(only
                .parts
                .keys()
                .all(|(_, l)| part_label(l) == kind.class_name()));
            let (lo, hi) = only.bounds();
            assert!(lo.y.abs() < 1e-6 && (hi.y - under).abs() < 0.001);
            let a = build_object_at_version(&o, None, INTRODUCED);
            audit::check(&o, &a).unwrap();
            let clear = Clearance::at_version(&o, INTRODUCED);
            for vertex in only.parts.values().flat_map(|g| &g.positions) {
                let at = Vec3::from_array(*vertex);
                assert!(clear.hits_capsule(at, at, 0.00002));
            }
            let (span, cross) = if s.runs_along_x {
                (s.post_centers_half_span_m.x, s.post_centers_half_span_m.y)
            } else {
                (s.post_centers_half_span_m.y, s.post_centers_half_span_m.x)
            };
            let board = (s.width_m - s.gap_m * (s.boards - 1) as f32) / s.boards as f32;
            let first = (board - s.width_m) * 0.5;
            let deck_y = s.top_y_m - s.thickness_m * 0.5;
            assert!(s.length_m * 0.5 > span && s.width_m * 0.5 < cross - s.post_half_width_m);
            for along in [-span, span] {
                for y in [s.top_y_m - s.thickness_m - s.rail_height_m * 0.5, deck_y] {
                    let at = q(along, first, y);
                    assert!(clear.hits_box(at, at, 0.00001), "unsupported deck");
                }
            }
            for at in [
                q(0., 0., (s.top_y_m + s.rim_height_m + under) * 0.5),
                q(0., 0., 0.006),
            ] {
                assert!(!clear.hits_box(at, at, 0.001));
            }
            if s.boards > 1 {
                let gap = q(0., first + (board + s.gap_m) * 0.5, deck_y);
                assert!(!clear.hits_box(gap, gap, s.gap_m * 0.15));
            }
        }
    }
    assert_eq!(decks.len(), 3);
    assert_eq!(headings.len(), 2);
    assert!(sampled > 150 && old_frames > 250);
}

#[test]
fn lower_frames_preserve_tabletop_contact_geometry_and_finish_charts() {
    let caps = |a: Assembly, h: f32| {
        a.parts
            .into_iter()
            .flat_map(|((surface, label), g)| {
                g.positions
                    .into_iter()
                    .zip(g.uvs)
                    .filter(move |(p, _)| p[1] >= h - 0.00001)
                    .map(move |(p, uv)| {
                        (
                            surface,
                            label.clone(),
                            p.map(f32::to_bits),
                            uv.map(f32::to_bits),
                        )
                    })
            })
            .collect::<Vec<_>>()
    };
    let mut shelves = 0;
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::CoffeeTable, seed);
        if parameters(&o).lounge_shelf(&o, INTRODUCED).is_some() {
            let old = build_object_at_version(&o, None, INTRODUCED - 1);
            let current = build_object_at_version(&o, None, INTRODUCED);
            let cap = caps(old, o.size.y);
            assert!(!cap.is_empty());
            assert_eq!(cap, caps(current, o.size.y));
            shelves += 1;
        }
        for kind in [ObjectKind::Table, ObjectKind::Desk] {
            let other = audit::fixture(kind, seed);
            assert!(parameters(&other)
                .lounge_shelf(&other, INTRODUCED)
                .is_none());
        }
    }
    assert!(shelves > 20);
}
