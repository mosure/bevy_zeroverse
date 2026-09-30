use super::*;
use crate::scene::procedural_indoor::{layout::IndoorLayout, validation};

#[test]
fn concave_notch_is_empty_and_sweeps_cannot_bridge_it() {
    let p = vec![
        Vec2::new(-4., -4.),
        Vec2::new(-1., -4.),
        Vec2::new(-1., -1.),
        Vec2::new(1., -1.),
        Vec2::new(1., -4.),
        Vec2::new(4., -4.),
        Vec2::new(4., 4.),
        Vec2::new(-4., 4.),
    ];
    polygon::validate(&p).unwrap();
    assert!(!polygon::contains(&p, Vec2::new(0., -2.), 0.));
    assert!(!polygon::segment_inside(
        &p,
        Vec2::new(-2., -2.),
        Vec2::new(2., -2.),
        0.1
    ));
    assert!(polygon::segment_inside(&p, Vec2::ZERO, Vec2::ZERO, 0.1));
    assert!(
        (polygon::triangles(&p)
            .iter()
            .map(|p| polygon::area(p))
            .sum::<f32>()
            - 58.)
            .abs()
            < 1e-4
    );
    let outside = polygon::outside(&p, Vec2::splat(-4.), Vec2::splat(4.));
    assert!((outside.iter().map(|p| polygon::area(p)).sum::<f32>() - 6.).abs() < 1e-4);
}

#[test]
fn architecture_program_replays_and_validates_consecutive_seeds() {
    let mut counts = [0usize; 6];
    for seed in 0..256 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        let e = scene.envelope.as_ref().unwrap();
        e.validate(&scene)
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        validation::validate_layout(&scene).unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        assert_eq!(
            *e,
            serde_json::from_str::<EnvelopeProgram>(&serde_json::to_string(e).unwrap()).unwrap()
        );
        let other =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        assert_eq!(e, other.envelope.as_ref().unwrap());
        counts[0] += usize::from(e.footprint.len() > 4);
        counts[1] += usize::from(e.ceiling_drop.length() > 0.01);
        counts[2] += usize::from(!e.pillars.is_empty());
        counts[3] += usize::from(e.floor_patches.iter().any(|p| p.height > 0.01));
        counts[4] += usize::from(e.minimum_floor() < -0.01);
        counts[5] += usize::from(e.mezzanine.is_some());
    }
    assert!(
        counts.iter().all(|&n| n >= 3),
        "feature coverage {counts:?}"
    );
}

fn ray_triangle(origin: Vec3, ray: Vec3, p: [Vec3; 3]) -> Option<f32> {
    let e = p[1] - p[0];
    let f = p[2] - p[0];
    let h = ray.cross(f);
    let determinant = e.dot(h);
    if determinant.abs() < 1e-8 {
        return None;
    }
    let s = origin - p[0];
    let u = s.dot(h) / determinant;
    let q = s.cross(e);
    let v = ray.dot(q) / determinant;
    let t = f.dot(q) / determinant;
    (u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001 && t > 0.).then_some(t)
}

#[test]
fn built_surfaces_match_floor_support_roof_and_apertures() {
    use crate::scene::procedural_indoor::{architecture, materials::Surface};
    let mut mezzanines = 0;
    for seed in 0..96 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let e = scene.envelope.as_ref().unwrap();
        e.validate(&scene).unwrap();
        let assembly = architecture::architecture(&scene);
        validation::validate_geometry(&scene).unwrap();
        let structural_boxes = e.structural_boxes(scene.room_size);
        for ((_, label), geometry) in &assembly.parts {
            if label == "other_structure" {
                for &v in &geometry.positions {
                    let v = Vec3::from(v);
                    assert!(
                        structural_boxes
                            .iter()
                            .any(|(lo, hi)| v.cmpge(*lo - Vec3::splat(1e-5)).all()
                                && v.cmple(*hi + Vec3::splat(1e-5)).all()),
                        "unbounded structural detail seed {seed}: {v:?}"
                    );
                }
            }
        }
        let raycast = |origin, direction, floor_only: bool| {
            assembly
                .parts
                .iter()
                .filter(|((_, label), _)| {
                    !label.ends_with("#exterior") && (!floor_only || label == "floor")
                })
                .flat_map(|(_, g)| {
                    g.indices.as_chunks::<3>().0.iter().filter_map(move |t| {
                        ray_triangle(
                            origin,
                            direction,
                            [t[0], t[1], t[2]].map(|i| Vec3::from(g.positions[i as usize])),
                        )
                    })
                })
                .min_by(f32::total_cmp)
        };
        for patch in &e.floor_patches {
            let p = (patch.min + patch.max) * 0.5;
            let hit = raycast(Vec3::new(p.x, patch.height + 0.05, p.y), Vec3::NEG_Y, true).unwrap();
            assert!((hit - 0.05).abs() < 1e-4, "floor seed {seed}: {hit}");
        }
        // A roof sample in each triangulated footprint piece must hit its real
        // underside, including the concave bays and oblique edges.
        for t in polygon::triangles(&e.footprint) {
            let p = (t[0] + t[1] + t[2]) / 3.;
            let h = e.ceiling_height(scene.room_size, p);
            let hit = raycast(Vec3::new(p.x, h - 0.03, p.y), Vec3::Y, false).unwrap();
            assert!((hit - 0.03).abs() < 1e-4, "roof seed {seed}: {hit}");
        }
        for ((_, label), g) in &assembly.parts {
            if label != "floor" {
                continue;
            }
            for t in g.indices.as_chunks::<3>().0.iter() {
                let v = [t[0], t[1], t[2]].map(|i| Vec3::from(g.positions[i as usize]));
                if (v[1] - v[0]).cross(v[2] - v[0]).normalize_or_zero().y.abs() < 0.9 {
                    continue;
                }
                let p = (Vec3::from(g.positions[t[0] as usize])
                    + Vec3::from(g.positions[t[1] as usize])
                    + Vec3::from(g.positions[t[2] as usize]))
                    / 3.;
                assert!(
                    p.z >= scene.room_size.z * 0.5 - 1e-4
                        || polygon::contains(&e.footprint, p.xz(), 0.),
                    "floor filled courtyard at seed {seed}: {p:?}"
                );
            }
        }
        // Strip optional shades for this structural opening test. Rays through
        // actual panes must encounter glass, never an uncut wall behind it.
        let mut clear = scene.clone();
        for w in &mut clear.envelope.as_mut().unwrap().walls {
            if let Some(f) = &mut w.facade {
                f.shade = architecture::facade::Shade::None;
            }
        }
        // Isolate exterior construction: a perpendicular interior partition
        // can legitimately occlude part of a pane from a particular view.
        let mut shell = crate::scene::procedural_indoor::objects::Assembly::default();
        let clear_e = clear.envelope.as_ref().unwrap();
        for w in &clear_e.walls {
            construction::wall(
                &mut shell,
                clear_e,
                clear.room_size,
                w.edge,
                w.facade.as_ref().map_or(&[][..], |f| f.openings.as_slice()),
            );
            if let Some(f) = &w.facade {
                architecture::facade::glazing_at(&mut shell, clear_e.wall_transform(w.edge), f);
            }
        }
        for wall in &e.walls {
            let Some(f) = &wall.facade else {
                continue;
            };
            let tf = e.wall_transform(wall.edge);
            for o in &f.openings {
                let x = o.min.x
                    + f.frame_width
                    + (o.max.x - o.min.x - 2. * f.frame_width) / o.columns as f32 * 0.5;
                let y = o.min.y
                    + f.frame_width
                    + (o.max.y - o.min.y - 2. * f.frame_width)
                        * if o.transom > 0. { o.transom * 0.5 } else { 0.5 };
                let origin = tf.transform_point(Vec3::new(x, y, 0.24));
                let direction = tf.rotation * Vec3::NEG_Z;
                let mut glass = false;
                for ((surface, label), g) in &shell.parts {
                    let hit = g
                        .indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .filter_map(|t| {
                            ray_triangle(
                                origin,
                                direction,
                                [t[0], t[1], t[2]].map(|i| Vec3::from(g.positions[i as usize])),
                            )
                        })
                        .any(|t| t < 0.6);
                    if *surface == Surface::Glass {
                        glass |= hit;
                    } else {
                        assert!(
                            !hit,
                            "blocked aperture seed {seed} edge {} {label}/{surface:?} origin={origin:?} opening={o:?} distances={:?}",
                            wall.edge, g.indices.as_chunks::<3>().0.iter().filter_map(|t| ray_triangle(origin, direction, [t[0],t[1],t[2]].map(|i| Vec3::from(g.positions[i as usize])))).filter(|&t|t<0.6).collect::<Vec<_>>()
                        );
                    }
                }
                assert!(glass, "missing glazing seed {seed} edge {}", wall.edge);
            }
        }
        if let Some(m) = &e.mezzanine {
            mezzanines += 1;
            let p = (m.deck.min + m.deck.max) * 0.5;
            let hit =
                raycast(Vec3::new(p.x, m.deck.height + 0.05, p.y), Vec3::NEG_Y, true).unwrap();
            assert!((hit - 0.05).abs() < 1e-4);
            for (lo, hi) in m.stair_boxes() {
                let p = (lo + hi) * 0.5;
                let hit = raycast(Vec3::new(p.x, hi.y + 0.05, p.z), Vec3::NEG_Y, true).unwrap();
                assert!((hit - 0.05).abs() < 1e-4);
            }
        }
    }
    assert!(mezzanines > 0, "mezzanine geometry was not tested");
}
