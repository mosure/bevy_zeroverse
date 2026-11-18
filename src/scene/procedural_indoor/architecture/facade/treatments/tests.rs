use super::*;
use crate::render::semantic::SemanticLabel;
use std::collections::{BTreeMap, BTreeSet};

const KINDS: [Shade; 7] = [
    Shade::Venetian,
    Shade::Roller,
    Shade::Vertical,
    Shade::Roman,
    Shade::Pleated,
    Shade::Curtain,
    Shade::PanelTrack,
];

#[test]
fn split_coverings_reveal_the_window_as_they_retract() {
    // Intersect the actual cloth, not its AABB: folded/stacked covers must leave
    // usable sight lines and never retain a hidden full-width proxy surface.
    for kind in [Shade::Curtain, Shade::Vertical] {
        let mut visible = Vec::new();
        for coverage in [0., 0.55, 1.] {
            let mut f = facade(19, kind, Vec2::new(2.7, 2.5), coverage);
            f.shade_tilt = -0.65;
            f.treatment.as_mut().unwrap().stack = Stack::Pair;
            let mut a = Assembly::default();
            build(&mut a, Transform::IDENTITY, &f, &f.openings[0]);
            let mut hits = 0;
            for i in 0..160 {
                let origin = Vec3::new(-1.30 + 2.60 * (i as f32 + 0.5) / 160., 1.35, 0.25);
                let hit = a
                    .parts
                    .iter()
                    .filter(|((surface, _), _)| *surface == f.treatment.as_ref().unwrap().fabric)
                    .any(|(_, g)| {
                        g.indices.as_chunks::<3>().0.iter().any(|t| {
                            let [a, b, c] = t.map(|i| Vec3::from(g.positions[i as usize]));
                            let ab = b - a;
                            let ac = c - a;
                            let h = Vec3::NEG_Z.cross(ac);
                            let det = ab.dot(h);
                            if det.abs() < 1e-10 {
                                return false;
                            }
                            let s = origin - a;
                            let u = s.dot(h) / det;
                            let q = s.cross(ab);
                            let v = Vec3::NEG_Z.dot(q) / det;
                            let distance = ac.dot(q) / det;
                            u >= 0. && v >= 0. && u + v <= 1. && (0.0..0.25).contains(&distance)
                        })
                    });
                hits += usize::from(hit);
            }
            visible.push(hits);
        }
        assert!(
            visible[0] + 15 < visible[1] && visible[1] + 15 < visible[2],
            "{kind:?}: {visible:?}"
        );
    }
}

fn facade(seed: u64, kind: Shade, size: Vec2, coverage: f32) -> Facade {
    let mut program = ShadeProgram::sample(seed).1;
    let mut rng = stream(seed, 42);
    program.track = (kind == Shade::PanelTrack).then(|| PanelTrack::sample(&mut rng));
    program.textile =
        (kind != Shade::Venetian).then(|| WindowTextile::sample(&mut rng, kind == Shade::Curtain));
    Facade {
        joinery: None,
        side: FacadeSide::Rear,
        openings: vec![WindowOpening {
            glazing: None,
            min: Vec2::new(-size.x * 0.5, 0.05),
            max: Vec2::new(size.x * 0.5, size.y + 0.05),
            columns: 1,
            transoms: Vec::new(),
            transom: 0.,
            operable: true,
        }],
        frame: Surface::Metal,
        frame_width: 0.03,
        frame_depth: 0.08,
        recess: 0.06,
        sill_projection: 0.1,
        treatment: Some(program),
        shade: kind,
        shade_coverage: coverage,
        shade_tilt: if seed.is_multiple_of(2) { -1.2 } else { 1.2 },
    }
}

#[test]
fn seeded_population_spans_coverings_materials_stacks_and_continuous_shapes() {
    let mut kinds = BTreeMap::<String, usize>::new();
    let mut combinations = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    for seed in 0..1024 {
        let (kind, p) = ShadeProgram::sample(seed);
        p.validate().unwrap();
        assert_eq!((kind, p.clone()), ShadeProgram::sample(seed));
        *kinds.entry(format!("{kind:?}")).or_default() += 1;
        combinations.insert(format!(
            "{:?}/{:?}/{}",
            p.stack, p.slat_material, p.finish_slot
        ));
        recipes.insert(serde_json::to_string(&p).unwrap());
        let f = facade(seed, kind, Vec2::splat(2.), 0.5);
        assert_eq!(
            f,
            serde_json::from_str::<Facade>(&serde_json::to_string(&f).unwrap()).unwrap()
        );
    }
    assert_eq!(kinds.len(), 8);
    assert!(kinds.values().all(|&n| n > 40), "{kinds:?}");
    assert!(combinations.len() > 95);
    assert_eq!(recipes.len(), 1024);
}

#[test]
fn coverings_stay_inside_aperture_and_reserved_depth_at_extreme_openings() {
    let mut max_triangles = 0;
    for seed in 0..16 {
        for kind in KINDS {
            for size in [
                Vec2::splat(0.35),
                Vec2::new(5.5, 0.5),
                Vec2::new(0.5, 5.2),
                Vec2::new(4.3, 3.2),
            ] {
                for coverage in [0., 0.5, 1.] {
                    let f = facade(seed, kind, size, coverage);
                    let o = &f.openings[0];
                    let mut a = Assembly::default();
                    build(&mut a, Transform::IDENTITY, &f, o);
                    let (lo, hi) = a.bounds();
                    assert!(lo.x>=o.min.x-1e-5 && hi.x<=o.max.x+1e-5 && lo.y>=o.min.y-1e-5 && hi.y<=o.max.y+1e-5 && lo.z>=0. && hi.z<=0.20,
                        "seed {seed}, {kind:?}, {size:?}, coverage={coverage}, bounds={lo:?}..{hi:?}");
                    let mut tris = 0;
                    for ((surface, label), g) in &a.parts {
                        assert!(variants::supports(*surface));
                        assert_eq!(
                            SemanticLabel::from_label(label.split('#').next().unwrap()),
                            Some(if kind == Shade::Curtain {
                                SemanticLabel::Curtain
                            } else {
                                SemanticLabel::Blinds
                            })
                        );
                        let mut unique = BTreeSet::new();
                        for n in &g.normals {
                            assert!((Vec3::from(*n).length() - 1.).abs() < 1e-4);
                        }
                        for uv in &g.uvs {
                            assert!(Vec2::from(*uv).is_finite());
                        }
                        for t in g.indices.as_chunks::<3>().0 {
                            let p = t.map(|i| Vec3::from(g.positions[i as usize]));
                            let cross = (p[1] - p[0]).cross(p[2] - p[0]);
                            let n: Vec3 =
                                t.iter().map(|&i| Vec3::from(g.normals[i as usize])).sum();
                            assert!(
                                cross.length_squared() > 1e-20 && cross.dot(n) >= -1e-9,
                                "{seed}/{kind:?}, size={size:?}, coverage={coverage}, {label}/{surface:?}: invalid triangle, p={p:?}, n={n:?}, dot={}, area={}", cross.dot(n), cross.length_squared()
                            );
                            let mut vertices =
                                p.map(|v| v.to_array().map(|x| (x * 1e6).round() as i64));
                            vertices.sort();
                            assert!(
                                unique.insert(vertices),
                                "{seed}/{kind:?}: coincident triangles"
                            );
                            tris += 1;
                        }
                    }
                    assert!(a.parts.len() <= 4);
                    max_triangles = max_triangles.max(tris);
                }
            }
        }
    }
    assert!(
        max_triangles < 7500,
        "bounded installation complexity: {max_triangles}"
    );
}

#[test]
fn installed_hardware_blocks_sash_sweep_even_when_retracted_and_legacy_replays() {
    let mut f = facade(5, Shade::Curtain, Vec2::splat(2.), 0.);
    assert!(!f.allows_open_sash());
    f.shade = Shade::None;
    assert!(f.allows_open_sash());
    f.treatment = None;
    f.shade = Shade::Roller;
    let encoded = serde_json::to_string(&f).unwrap();
    assert!(!encoded.contains("treatment"));
    let f: Facade = serde_json::from_str(&encoded).unwrap();
    assert!(f.treatment.is_none() && f.allows_open_sash());
    for field in [
        "slat_pitch",
        "fold_pitch",
        "fold_depth",
        "fullness",
        "hem_sag",
        "split",
        "tieback",
    ] {
        let mut v = serde_json::to_value(ShadeProgram::sample(2).1).unwrap();
        v[field] = serde_json::json!(-10.);
        assert!(serde_json::from_value::<ShadeProgram>(v)
            .unwrap()
            .validate()
            .is_err());
    }
}

#[test]
fn track_meter_matches_frontal_rays_through_the_actual_assembly() {
    const NX: usize = 139;
    const NY: usize = 677;
    for size in [Vec2::new(1.40, 1.75), Vec2::new(3.75, 2.40)] {
        let mut previous = 1.;
        for coverage in [0., 0.3, 0.8, 1.] {
            let f = facade(7, Shade::PanelTrack, size, coverage);
            let track = f.treatment.as_ref().unwrap().track.unwrap();
            let mut a = Assembly::default();
            build(&mut a, Transform::IDENTITY, &f, &f.openings[0]);
            let mut blocked = vec![false; NX * NY];
            // Orthographic rays normal to the aperture, excluding the perimeter
            // brackets. Rasterize actual triangles, including overlaps and ties.
            let lo = Vec2::new(-size.x * 0.5 + 0.03, 0.13);
            let scale = Vec2::new(NX as f32, NY as f32) / (size - Vec2::new(0.06, 0.18));
            for g in a.parts.values() {
                for t in g.indices.as_chunks::<3>().0 {
                    let p =
                        t.map(|i| (Vec3::from(g.positions[i as usize]).truncate() - lo) * scale);
                    let area = (p[1] - p[0]).perp_dot(p[2] - p[0]);
                    if area.abs() < 1e-7 {
                        continue;
                    }
                    let min = p[0].min(p[1]).min(p[2]).floor().as_ivec2().max(IVec2::ZERO);
                    let max = p[0]
                        .max(p[1])
                        .max(p[2])
                        .ceil()
                        .as_ivec2()
                        .min(IVec2::new(NX as i32, NY as i32));
                    for y in min.y..max.y {
                        for x in min.x..max.x {
                            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                            if (0..3).all(|i| {
                                (p[(i + 1) % 3] - p[i]).perp_dot(sample - p[i]) * area >= 0.
                            }) {
                                blocked[y as usize * NX + x as usize] = true;
                            }
                        }
                    }
                }
            }
            let measured = blocked.iter().filter(|&&b| !b).count() as f32 / blocked.len() as f32;
            let estimate = track.clear_fraction(size.x, coverage);
            assert!(
                (estimate - measured).abs() < 0.052,
                "{size:?} {coverage}: {estimate} vs {measured}"
            );
            assert!(
                measured < previous - 0.035,
                "{coverage}: {measured} / {previous}"
            );
            if coverage == 1. {
                assert!((0.06..0.36).contains(&measured));
            }
            if coverage == 0. {
                assert!(measured < 0.90);
            }
            previous = measured;
        }
    }
}

#[test]
fn roller_weave_is_metric_and_sliding_panels_keep_rigid_flow_correspondence() {
    use crate::scene::procedural_indoor::motion::IndoorMotionConfig;
    for kind in [Shade::Roller, Shade::PanelTrack] {
        let mut f = facade(3, kind, Vec2::new(3.0, 2.0), 0.15);
        let mut panels = vec![];
        let mut hem_uv = vec![];
        for coverage in [0.15, 0.88] {
            f.shade_coverage = coverage;
            let mut a = Assembly::default();
            build(&mut a, Transform::IDENTITY, &f, &f.openings[0]);
            if kind == Shade::Roller {
                let g = a
                    .parts
                    .iter()
                    .find(|((_, label), _)| label.ends_with("#roller"))
                    .unwrap()
                    .1;
                // Open roll hardware or a pull cord can share this finish.
                // Isolate the positive-Z cloth face away from the side controls.
                let v: Vec<_> = g
                    .positions
                    .iter()
                    .zip(&g.normals)
                    .zip(&g.uvs)
                    .filter(|((p, n), _)| {
                        p[0].abs() < 0.05 && (p[2] - 0.1126).abs() < 0.007 && n[2] > 0.99
                    })
                    .map(|((p, _), uv)| (p[1], uv[1]))
                    .collect();
                let top = v.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
                let hem = v.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
                assert!(top.0 - hem.0 > 0.1);
                assert!(((top.0 - hem.0) - (hem.1 - top.1)).abs() < 1e-4);
                hem_uv.push(hem.1);
            }
            panels.push(a);
        }
        if kind == Shade::Roller {
            assert!((hem_uv[0] - hem_uv[1]).abs() < 1e-5);
        }
        for (key, start) in &panels[0].parts {
            let end = &panels[1].parts[key];
            assert_eq!(start.indices, end.indices);
            if kind == Shade::PanelTrack {
                for t in start.indices.as_chunks::<3>().0 {
                    let delta = t.map(|i| {
                        Vec3::from(end.positions[i as usize])
                            - Vec3::from(start.positions[i as usize])
                    });
                    assert!(delta[0].y.abs() + delta[0].z.abs() < 1e-5);
                    assert!(delta[1].distance(delta[0]).max(delta[2].distance(delta[0])) < 1e-5);
                }
            }
        }
        let mut moving = Assembly {
            motion_policy: Some(IndoorMotionConfig {
                covering_fraction: 1.,
                ..default()
            }),
            ..default()
        };
        build(&mut moving, Transform::IDENTITY, &f, &f.openings[0]);
        assert_eq!(
            moving.joints.len(),
            1,
            "{kind:?} fell back to a static mesh"
        );
    }
}

#[test]
fn archived_window_program_does_not_acquire_new_stock_during_replay() {
    let mut f = facade(2, Shade::Roller, Vec2::splat(2.), 0.7);
    let mut json = serde_json::to_value(&f).unwrap();
    let p = json["treatment"].as_object_mut().unwrap();
    p.remove("textile");
    p.remove("track");
    f = serde_json::from_value(json).unwrap();
    let mut a = Assembly::default();
    build(&mut a, Transform::IDENTITY, &f, &f.openings[0]);
    assert!(a.parts.keys().all(|(_, label)| label
        == &format!("blinds#finish{}", f.treatment.as_ref().unwrap().finish_slot)));
}

#[test]
fn visible_roller_yarn_tracks_the_hem_at_intermediate_times_with_fixed_uvs() {
    let mut f = facade(5, Shade::Roller, Vec2::new(2.09, 2.81), 0.12);
    let o = f.openings[0].clone();
    let tf = Transform::from_xyz(3.1, 2., -2.3).with_rotation(Quat::from_euler(
        EulerRot::YXZ,
        1.1,
        0.12,
        0.04,
    ));
    let up = tf.rotation * Vec3::Y;
    let feed = mesh::feed_plane(tf, &o);
    let mut endpoints = Vec::new();
    for coverage in [0.12, 0.92] {
        f.shade_coverage = coverage;
        let mut a = Assembly::default();
        mesh::build_with_angle(
            &mut a,
            tf,
            &f,
            &o,
            f.treatment.as_ref().unwrap(),
            None,
            true,
        );
        endpoints.push(
            a.parts
                .into_iter()
                .find(|((_, s), _)| s.ends_with("#roller"))
                .unwrap()
                .1,
        );
    }
    let [start, end] = endpoints.as_slice() else {
        unreachable!()
    };
    assert_eq!(start.indices, end.indices);
    let rail = tf.transform_point(Vec3::Y * (o.max.y - 0.072)).dot(up);
    let available = o.max.y - o.min.y - 0.028 - 0.095;
    let mut emerging = 0;
    for t in [0., 0.23, 0.5, 0.86, 1.] {
        let coverage = 0.12 + (0.92 - 0.12) * t;
        let hem = rail - 0.044 - (available - 0.044) * coverage;
        // The two broad faces use the same yarn coordinates at both endpoints.
        let faces = 2 * 5 * ((available / 0.018).ceil() as usize + 1);
        for i in 0..faces {
            let a = Vec3::from(start.positions[i]);
            let b = Vec3::from(end.positions[i]);
            let actual = feed.interpolate(a, b, t).dot(up);
            assert_eq!(start.uvs[i], end.uvs[i]);
            let expected = hem + available - start.uvs[i][1];
            if expected < rail - 0.035 {
                assert!((actual - expected).abs() < 0.0006, "{actual} vs {expected}");
                if a.dot(up) > rail {
                    emerging += 1;
                }
            } else {
                assert!(actual < rail + 0.014);
            }
        }
    }
    assert!(emerging > 100);
}

#[test]
fn adding_another_roller_does_not_rescale_an_installed_web() {
    let mut f = facade(21, Shade::Roller, Vec2::new(1.4, 2.0), 0.25);
    let mut a = Assembly::default();
    build(&mut a, Transform::IDENTITY, &f, &f.openings[0]);
    let (key, g) = a
        .parts
        .iter()
        .find(|((_, s), _)| s.ends_with("#roller"))
        .unwrap();
    let key = key.clone();
    let first = g.uvs.clone();
    f.shade_coverage = 0.76;
    build(&mut a, Transform::from_xyz(2., 0., 0.), &f, &f.openings[0]);
    assert_eq!(a.parts[&key].uvs[..first.len()], first);
}
