use super::*;
use crate::scene::procedural_indoor::{
    envelope::ray_triangle, geometry::Geometry, materials::glazing::GlazingProgram,
    motion::construction,
};

fn installation(seed: u64, construction: PleatConstruction, draw: f32, size: Vec2) -> Facade {
    let mut f = facade(seed, Shade::Pleated, size, draw);
    let stock = f.treatment.as_mut().unwrap().pleated.as_mut().unwrap();
    stock.construction = construction;
    stock.bond_width_m = if construction == PleatConstruction::Single {
        0.
    } else {
        0.008
    };
    f
}

fn cloth(f: &Facade, tf: Transform) -> Geometry {
    let mut a = Assembly::default();
    mesh::build(
        &mut a,
        tf,
        f,
        &f.openings[0],
        f.treatment.as_ref().unwrap(),
        mesh::Draw {
            coverage: f.shade_coverage,
            fixed_angle: None,
            moving: true,
        },
    );
    a.parts
        .into_iter()
        .find(|((_, label), _)| label.ends_with("#pleat"))
        .unwrap()
        .1
}

#[test]
fn pleat_cuts_are_closed_with_open_cell_ends_and_consistent_winding() {
    for construction in [PleatConstruction::Single, PleatConstruction::Cellular] {
        for draw in [0., 0.37, 1.] {
            let f = installation(12, construction, draw, Vec2::new(2.1, 1.54));
            let p = f.treatment.as_ref().unwrap();
            let stock = p.pleated.unwrap();
            let l = stock.layout(p, &f.openings[0], draw);
            let g = cloth(&f, Transform::IDENTITY);
            let mut edges = BTreeMap::new();
            for ids in g.indices.as_chunks::<3>().0 {
                let tri = ids.map(|i| g.positions[i as usize].map(f32::to_bits));
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let (key, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                    let e = edges.entry(key).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += sign;
                }
            }
            assert!(
                edges.values().all(|x| *x == (2, 0)),
                "{construction:?} {draw}"
            );
            let hits = |origin, dir| {
                g.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter(|t| {
                        ray_triangle(
                            origin,
                            dir,
                            t.map(|i| Vec3::from_array(g.positions[i as usize])),
                        )
                        .is_some()
                    })
                    .count()
            };
            let y = l.top - 4.46 * l.rise;
            let z = pleated::TRACK_Z;
            assert_eq!(
                hits(Vec3::new(-0.123, y, z + 0.18), Vec3::NEG_Z),
                if construction == PleatConstruction::Single {
                    2
                } else {
                    4
                }
            );
            if construction == PleatConstruction::Cellular {
                assert_eq!(hits(Vec3::new(-2., y, z + 0.00013), Vec3::X), 0);
                // The bonded horizontal membrane physically separates cells.
                assert_eq!(
                    hits(
                        Vec3::new(
                            -2.,
                            l.top - 4. * l.rise + stock.cloth_thickness_m * 0.23,
                            z + 0.00013
                        ),
                        Vec3::X
                    ),
                    2
                );
            }
        }
    }
}

#[test]
fn pleat_motion_coincides_with_rebuilt_cloth_at_intermediate_draws() {
    let tf = Transform::from_xyz(3., 0.6, -2.).with_rotation(Quat::from_rotation_y(0.73));
    for seed in 0..8 {
        let base = if seed % 2 == 0 { 0.05 } else { 0.98 };
        let kind = if seed % 4 < 2 {
            PleatConstruction::Single
        } else {
            PleatConstruction::Cellular
        };
        let mut f = installation(seed, kind, base, Vec2::new(2.1, 2.32));
        let mut a = Assembly {
            motion_policy: Some(IndoorMotionConfig {
                covering_fraction: 1.,
                ..default()
            }),
            ..default()
        };
        build(&mut a, tf, &f, &f.openings[0]);
        assert_eq!(a.joints.len(), 1);
        let joint = &a.joints[0];
        let law = joint.cloth_motion.unwrap();
        assert!(joint.attachment.is_some());
        let key = joint
            .parts
            .iter()
            .find(|(_, label)| law.applies(label))
            .unwrap();
        let (first, last) = (&a.parts[key], &joint.morph[key]);
        let end = construction::endpoint(
            installation_seed(f.treatment.as_ref().unwrap(), &f.openings[0]),
            base,
            1.,
        );
        for t in [0., 0.17, 0.5, 0.81, 1.] {
            f.shade_coverage = base.lerp(end, t);
            let stock = f.treatment.as_ref().unwrap().pleated.unwrap();
            let l = stock.layout(
                f.treatment.as_ref().unwrap(),
                &f.openings[0],
                f.shade_coverage,
            );
            let rest = cloth(&f, tf);
            assert_eq!(first.uvs, rest.uvs);
            assert_eq!(first.indices, rest.indices);
            let law = law.at(t);
            for ((a, b), c) in first
                .positions
                .iter()
                .zip(&last.positions)
                .zip(&rest.positions)
            {
                assert!(
                    law.interpolate((*a).into(), (*b).into(), t)
                        .distance((*c).into())
                        < 0.000012
                );
            }
            // One edge on the actual front face from seam to crest. At a
            // grazing angle its length is still the original cut length.
            let a = Vec3::from(rest.positions[4]);
            let b = Vec3::from(rest.positions[5]);
            assert!((a.distance(b) - l.face_m).abs() < 0.000003);
        }
    }
}

#[test]
fn pleat_meter_matches_the_mesh_and_archives_replay_without_the_new_stock() {
    let frit = (0..500)
        .map(|s| GlazingProgram::sample(s, 0, false))
        .find(|p| p.treatment == 6)
        .unwrap();
    for kind in [PleatConstruction::Single, PleatConstruction::Cellular] {
        let size = Vec2::new(2.47, 1.9);
        let mut f = installation(31, kind, 0., size);
        f.openings[0].columns = 3;
        f.openings[0].transoms = vec![0.51];
        f.openings[0].operable = false;
        f.openings[0].glazing = Some(frit.clone());
        let mut previous = f32::INFINITY;
        for draw in [0., 0.19, 0.53, 0.86, 1.] {
            f.shade_coverage = draw;
            let o = &f.openings[0];
            let expected = pleated::visible_area(&f, o, draw).unwrap() / o.area();
            let mut a = Assembly::default();
            glazing_at(&mut a, Transform::IDENTITY, &f);
            let measured = frontal_clear::<509, 271>(&a, o.min, size);
            assert!(
                (measured - expected).abs() < 0.017,
                "{kind:?} {draw}: {measured} vs {expected}"
            );
            assert!(expected < previous);
            previous = expected;
        }
        let mut json = serde_json::to_value(&f).unwrap();
        json["treatment"].as_object_mut().unwrap().remove("pleated");
        let archived: Facade = serde_json::from_value(json).unwrap();
        assert!(pleated::visible_area(&archived, &archived.openings[0], 1.).is_none());
        f.treatment.as_mut().unwrap().pleated = None;
        let mut before = Assembly::default();
        let mut after = Assembly::default();
        build(
            &mut before,
            Transform::IDENTITY,
            &archived,
            &archived.openings[0],
        );
        build(&mut after, Transform::IDENTITY, &f, &f.openings[0]);
        for (k, g) in before.parts {
            assert_eq!(g.positions, after.parts[&k].positions);
            assert_eq!(g.uvs, after.parts[&k].uvs);
        }
    }
}

#[test]
fn cut_stock_includes_fine_and_wide_folds_and_both_rails_cover_the_parked_web() {
    for construction in [PleatConstruction::Single, PleatConstruction::Cellular] {
        let mut depths = Vec::new();
        for seed in 0..128 {
            let f = installation(seed, construction, 0., Vec2::new(1.74, 0.88));
            let o = &f.openings[0];
            let p = f.treatment.as_ref().unwrap();
            let l = p.pleated.unwrap().layout(p, o, 0.);
            depths.push(l.packed_depth);
            let mut a = Assembly::default();
            mesh::build(
                &mut a,
                Transform::IDENTITY,
                &f,
                o,
                p,
                mesh::Draw {
                    coverage: 0.,
                    fixed_angle: None,
                    moving: true,
                },
            );
            let (_, g) = a
                .parts
                .iter()
                .find(|((_, k), _)| k.ends_with("#pleat"))
                .unwrap();
            let (z0, z1) = g
                .positions
                .iter()
                .map(|v| v[2])
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), z| {
                    (lo.min(z), hi.max(z))
                });
            for y in [
                o.max.y - 0.040,
                l.rail_bottom(p.pleated.unwrap()) + p.pleated.unwrap().rail_height_m * 0.5,
            ] {
                // Even the front and rear crests lie inside the track and hem footprints.
                for z in [z0, z1] {
                    let rail_hit = a
                        .parts
                        .iter()
                        .filter(|((s, _), _)| *s == p.slat_material)
                        .any(|(_, g)| {
                            g.indices.as_chunks::<3>().0.iter().any(|ids| {
                                ray_triangle(
                                    Vec3::new(0.064, y, z),
                                    Vec3::Y,
                                    ids.map(|j| g.positions[j as usize].into()),
                                )
                                .is_some_and(|d| d < 0.022)
                            })
                        });
                    assert!(rail_hit, "{construction:?} {seed} {y} {z}");
                }
            }
        }
        depths.sort_by(f32::total_cmp);
        assert!(
            depths[12] < 0.035 && depths[115] > 0.044,
            "{construction:?}: {:?}",
            [depths[12], depths[115]]
        );
    }
}
