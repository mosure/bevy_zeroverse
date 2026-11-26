use super::*;
use crate::scene::procedural_indoor::{
    materials::{finish_slot, registered_finish},
    objects::{audit, build_object, build_object_at_version, part_label},
};
use std::collections::BTreeSet;

use super::super::test_geometry::{top_surface_at, triangles};

fn top_at(a: &Assembly, q: Vec2) -> Option<f32> {
    top_surface_at(a, q).map(|hit| hit.0)
}

fn horizontal_slice(a: &Assembly, y: f32) -> Vec<[Vec2; 2]> {
    a.parts
        .values()
        .flat_map(triangles)
        .filter_map(|tri| {
            let mut intersections = tri
                .iter()
                .zip(tri.iter().cycle().skip(1))
                .filter(|(p, q)| (p.y <= y) != (q.y <= y))
                .map(|(p, q)| p.xz().lerp(q.xz(), (y - p.y) / (q.y - p.y)));
            Some([intersections.next()?, intersections.next()?])
        })
        .collect()
}

fn distance_to_slice(q: Vec2, slice: &[[Vec2; 2]]) -> f32 {
    slice
        .iter()
        .map(|&[a, b]| {
            let d = b - a;
            let t = ((q - a).dot(d) / d.length_squared().max(1e-18)).clamp(0., 1.);
            q.distance(a + t * d)
        })
        .fold(f32::INFINITY, f32::min)
}

#[test]
fn ceramic_feet_sit_in_the_saucer_well_and_drinks_meet_the_actual_inner_wall() {
    let mut facets = BTreeSet::new();
    let mut drinks = BTreeSet::new();
    let mut saucers = 0;
    for seed in 0..192 {
        for kind in [ObjectKind::Mug, ObjectKind::CoffeeCup] {
            let mut o = audit::fixture(kind, seed);
            if seed % 2 == 0 {
                o.size = if kind == ObjectKind::Mug {
                    Vec3::new(0.101, 0.073, 0.109)
                } else {
                    Vec3::new(0.060, 0.077, 0.060)
                };
            }
            let p = parameters(&o);
            let f = CupForm::sample(&o, &p);
            let section = mesh::Section::new(&f, &p);
            let label = kind.class_name();
            let mut body = Assembly::default();
            section.draw(&mut body, &f, label);
            assert!((body.bounds().0.y - f.base_y_m).abs() < 1e-6);
            if let Some(s) = &f.saucer {
                let mut saucer = Assembly::default();
                s.draw(&mut saucer, label);
                assert!(saucer.bounds().0.y.abs() < 1e-6);
                let mut bearing = 0;
                for pos in body.parts.values().flat_map(|g| &g.positions) {
                    let q = Vec3::from_array(*pos);
                    if q.y <= s.rim_y_m {
                        assert!(q.xz().distance(s.center_m) < s.well_radius_m * (PI / 32.).cos());
                        if (q.y - f.base_y_m).abs() < 1e-7 {
                            assert!((top_at(&saucer, q.xz()).unwrap() - q.y).abs() < 1e-6);
                            bearing += 1;
                        }
                    }
                }
                assert!(bearing >= f.sides);
                let mut handle = Assembly::default();
                section.handle(&mut handle, &f, &p, f.handle_gauge_m.unwrap(), label);
                assert!(handle.bounds().0.y > s.rim_y_m);
                saucers += 1;
            }
            if let Some(brew) = f.brew {
                let mut liquid = Assembly::default();
                section.drink(&mut liquid, &f, &p, brew, label);
                let wet_y = liquid.bounds().1.y;
                let slice = horizontal_slice(&body, wet_y);
                let mut wetted = 0;
                for pos in liquid.parts.values().flat_map(|g| &g.positions) {
                    let q = Vec3::from_array(*pos);
                    if (q.y - wet_y).abs() < 1e-7 {
                        assert!(distance_to_slice(q.xz(), &slice) < 2e-6, "{kind:?}/{seed}");
                        wetted += 1;
                    }
                }
                assert!(wetted >= f.sides);
                facets.insert(f.sides);
                drinks.insert(brew.slot());
            }
            let a = build_object(&o);
            let checked = audit::check(&o, &a).unwrap();
            assert!(checked.max_envelope_overrun_m < 2e-6 && checked.min_y_m.abs() < 1e-6);
            for (surface, tag) in a.parts.keys() {
                assert_eq!(part_label(tag), label);
                if let Some(slot) = finish_slot(*surface, tag) {
                    assert!(registered_finish(*surface, slot));
                }
            }
        }
    }
    assert_eq!(facets, BTreeSet::from([8, 9, 10, 11, 12, 32]));
    assert_eq!(drinks.len(), 4);
    assert!(saucers > 64);
}

#[test]
fn lid_opens_to_the_real_drink_or_floor_and_printed_sleeve_wraps_once() {
    let mut open_to = [0; 2];
    let mut prints = 0;
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::CoffeeCup, seed);
        let p = parameters(&o);
        let f = CupForm::sample(&o, &p);
        let a = build_object(&o);
        let historic = crate::scene::procedural_indoor::materials::cup_board::INTRODUCED - 1;
        let archived = CupForm::sample_at(&o, &p, historic);
        if archived.sleeve.as_ref().is_some_and(|s| s.printed) {
            let archived_mesh = build_object_at_version(&o, None, historic);
            let g = archived_mesh
                .parts
                .iter()
                .find(|((s, _), _)| *s == Surface::Art)
                .unwrap()
                .1;
            let n = f.sides as usize;
            assert_eq!(g.uvs.len(), 2 * (n + 1));
            for (row, band) in g.uvs.chunks_exact(n + 1).enumerate() {
                for (i, &uv) in band.iter().enumerate() {
                    assert_eq!(uv, [i as f32 / n as f32, row as f32]);
                }
            }
            prints += 1;
        }
        let Some(ref lid) = f.lid else { continue };
        let rim = mesh::Section::new(&f, &p).mouth() + 0.4 * f.lip_m;
        let deck = lid.deck_fraction * rim;
        let center = Vec2::Y * (lid.port_offset_fraction * deck);
        let project = |q: Vec2| f.transform().transform_point(Vec3::new(q.x, 0., q.y)).xz();
        let expected = if p.empty {
            f.floor_m
        } else {
            f.height_m * p.fill_fraction
        };
        for delta in [
            Vec2::ZERO,
            Vec2::X * lid.port_half_fraction.x * 0.5,
            Vec2::Y * lid.port_half_fraction.y * 0.5,
        ] {
            let y = top_at(&a, project(center + deck * delta)).unwrap();
            assert!(
                (y - expected).abs() < 1e-6,
                "lid {seed} expected {expected}, hit {y}"
            );
        }
        for delta in [Vec2::ZERO, Vec2::X * 0.4, Vec2::NEG_Y * 0.5] {
            assert!((top_at(&a, project(deck * delta)).unwrap() - o.size.y).abs() < 1e-6);
        }
        open_to[usize::from(p.empty)] += 1;
    }
    assert!(prints > 24 && open_to.into_iter().all(|n| n > 16));
}

#[test]
fn die_cut_stock_owns_the_shell_and_only_trims_the_printed_chart() {
    let mut variants = BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::CoffeeCup, seed);
        o.finish_seed = Some(seed);
        let p = parameters(&o);
        let f = CupForm::sample(&o, &p);
        let a = build_object(&o);
        let archived = build_object_at_version(
            &o,
            None,
            crate::scene::procedural_indoor::materials::cup_board::INTRODUCED - 1,
        );
        // Unsleeved bodies, contents and lids replay the original geometry exactly.
        for ((s, key), g) in &a.parts {
            if f.sleeve.is_none() || *s != Surface::Paper {
                let old = &archived.parts[&(*s, key.clone())];
                assert_eq!(g.positions, old.positions);
                assert_eq!(g.uvs, old.uvs);
                assert_eq!(g.normals, old.normals);
                assert_eq!(g.indices, old.indices);
            }
        }
        let Some(sleeve) = &f.sleeve else {
            assert_eq!(a.parts.len(), archived.parts.len());
            continue;
        };
        assert!(a.parts.keys().all(|(s, _)| *s != Surface::Art));
        let cut = sleeve.cut.as_ref().unwrap();
        variants.insert(format!(
            "{:?}-{:?}-{}",
            cut.trim, cut.stock.press, cut.stock.printed
        ));
        let face = &a.parts[&(Surface::Paper, cut.stock.label())];
        let low = f.height_m * sleeve.low_fraction;
        let high = f.height_m * sleeve.high_fraction;
        let n = face.uvs.len() / 4;
        assert!((32..=34).contains(&n));
        let mut last_u = 1.;
        let (positions, extra_positions) = face.positions.as_chunks::<4>();
        let (uvs, extra_uvs) = face.uvs.as_chunks::<4>();
        assert!(extra_positions.is_empty() && extra_uvs.is_empty());
        assert_eq!(positions.len(), uvs.len());
        for (pos, uv) in positions.iter().zip(uvs) {
            assert_eq!(uv[0][0], last_u);
            assert!(uv[2][0] < last_u && uv[1][1] < uv[0][1]);
            last_u = uv[2][0];
            for (p, uv) in pos.iter().zip(uv) {
                assert!((uv[1] - (high - p[1]) / (high - low)).abs() < 2e-6);
                assert!((-1e-6..=1.000001).contains(&uv[1]));
            }
        }
        assert_eq!(last_u, 0.);
        let (triangles, extra_indices) = face.indices.as_chunks::<3>();
        assert!(extra_indices.is_empty());
        for tri in triangles {
            let [a, b, c] = tri.map(|i| Vec3::from_array(face.positions[i as usize]));
            let normal = Vec3::from_array(face.normals[tri[0] as usize]);
            assert!((b - a).cross(c - a).dot(normal) > 0.);
        }
        // A horizontal ray from inside the vessel must cross the cavity wall
        // and the outside of the jacket exactly once each, including at its lap.
        let section = mesh::Section::new(&f, &p);
        let y = (high + low) * 0.5;
        let slice = horizontal_slice(&a, y);
        let tf = f.transform();
        for turn in [0.071, 0.331, 0.701, 0.984] {
            let dir = Vec2::from_angle(TAU * (turn + f32::from(cut.seam_facet) / 32.));
            let axis = (tf.rotation * Vec3::new(dir.x, 0., dir.y)).xz();
            let mut hits: Vec<_> = slice
                .iter()
                .filter_map(|[a, b]| {
                    let edge = *b - *a;
                    let den = axis.perp_dot(edge);
                    if den.abs() < 1e-11 {
                        return None;
                    }
                    let t = (*a - f.center_m).perp_dot(edge) / den;
                    let along = (*a - f.center_m).perp_dot(axis) / den;
                    (t > 0. && (0.0..1.).contains(&along)).then_some(t)
                })
                .collect();
            hits.sort_by(f32::total_cmp);
            assert_eq!(hits.len(), 2, "seed {seed} {hits:?}");
            let angle = turn * TAU;
            let half = std::f32::consts::PI / f.sides as f32;
            let projection = half.cos() / ((angle.rem_euclid(2. * half)) - half).cos();
            let excess = hits[1] / projection - section.radius_at(y);
            assert!(
                excess > 0.5 * sleeve.thickness_m && excess < 1.002 * sleeve.thickness_m,
                "seed {seed} excess {excess} thickness {}",
                sleeve.thickness_m
            );
        }
    }
    assert_eq!(variants.len(), 18);
}
