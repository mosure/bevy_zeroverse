use super::{
    hinged::{Bank, Face, Hand, Pull},
    *,
};
use crate::scene::procedural_indoor::{envelope::ray_triangle, motion::IndoorMotionConfig};
use std::collections::{BTreeMap, BTreeSet};

fn skin_hit(a: &Assembly, xy: Vec2) -> f32 {
    let from = xy.extend(0.05);
    a.parts
        .values()
        .flat_map(|g| {
            g.indices.as_chunks::<3>().0.iter().filter_map(|t| {
                ray_triangle(
                    from,
                    Vec3::NEG_Z,
                    t.map(|i| Vec3::from_array(g.positions[i as usize])),
                )
            })
        })
        .min_by(f32::total_cmp)
        .map(|t| from.z - t)
        .expect("ray missed cabinet leaf")
}

#[test]
fn panels_close_without_stacked_faces_and_pulls_land_on_solid_stock() {
    let mut styles = BTreeSet::new();
    let mut pulls = BTreeSet::new();
    for seed in 0..160 {
        let o = audit::fixture(ObjectKind::Cabinet, seed);
        let p = parameters(&o);
        let Some(bank) = Bank::new(&o, &p) else {
            continue;
        };
        for d in (0..p.bays.len()).filter_map(|i| bank.leaf(&o, &p, i)) {
            styles.insert(d.face as u8);
            pulls.insert(d.pull as u8);
            let mut body = Assembly::default();
            skin::build(&mut body, &o, &p, &d);
            let mut edges = BTreeMap::new();
            for g in body.parts.values() {
                if matches!(d.face, Face::Fluted | Face::Reeded) {
                    let mut offsets = g
                        .positions
                        .iter()
                        .zip(&g.normals)
                        .zip(&g.uvs)
                        .filter(|((_, n), _)| n[2] > 0.)
                        .map(|((xyz, _), uv)| {
                            Vec2::from_array(*uv) + Vec3::from_array(*xyz).truncate()
                        });
                    let origin = offsets.next().unwrap();
                    assert!(
                        offsets.all(|v| v.abs_diff_eq(origin, 2e-5)),
                        "grain {seed}/{:?}",
                        d.face
                    );
                }
                for t in g.indices.as_chunks::<3>().0 {
                    let xyz = t.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let n = (xyz[1] - xyz[0]).cross(xyz[2] - xyz[0]);
                    let uv = t.map(|i| Vec2::from_array(g.uvs[i as usize]));
                    assert!(
                        (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-11,
                        "UV {seed}/{:?}",
                        d.face
                    );
                    assert!(n.dot(Vec3::from_array(g.normals[t[0] as usize])) > 1e-10);
                    for i in 0..3 {
                        let mut edge =
                            [xyz[i], xyz[(i + 1) % 3]].map(|v| v.to_array().map(f32::to_bits));
                        edge.sort_unstable();
                        *edges.entry(edge).or_insert(0) += 1;
                    }
                }
            }
            assert!(
                edges.values().all(|&n| n == 2),
                "watertight {seed}/{:?}",
                d.face
            );
            if matches!(d.face, Face::Raised | Face::Recessed) {
                let target = if d.face == Face::Raised {
                    -d.relief_m * 0.16
                } else {
                    -d.relief_m
                };
                assert!((skin_hit(&body, Vec2::ZERO) - target).abs() < 1e-5);
            }
            if d.pull == Pull::Touch {
                continue;
            }
            let axis = if d.vertical_pull { Vec2::Y } else { Vec2::X };
            for offset in if d.pull == Pull::Knob {
                [0.; 2]
            } else {
                [-0.5, 0.5]
            } {
                assert!(
                    skin_hit(&body, d.grip_center_m + axis * d.grip_span_m * offset).abs() < 1e-5,
                    "drilling {seed}/{d:?}"
                );
            }
            let mut handle = Assembly::default();
            pull::build(&mut handle, &d);
            let (lo, hi) = handle.bounds();
            assert!(
                lo.z < 0. && hi.z > 0.004 && hi.z <= d.reach_m + 0.0001,
                "{d:?}/{lo:?}/{hi:?}"
            );
        }
    }
    assert_eq!(styles.len(), 5);
    assert_eq!(pulls.len(), 4);
}

#[test]
fn hinge_hands_and_the_entire_swept_leaf_clear_bays_and_shelves() {
    let policy = IndoorMotionConfig {
        cabinet_fraction: 1.,
        ..default()
    };
    let (mut left, mut right, mut paired, mut asymmetric) = (0, 0, 0, 0);
    for seed in 0..384 {
        let mut o = audit::fixture(ObjectKind::Cabinet, seed);
        o.size = [
            Vec3::new(1.67, 0.83, 0.42),
            Vec3::new(0.67, 1.90, 0.31),
            Vec3::new(1.8, 2.1, 0.65),
        ][seed as usize % 3];
        let p = parameters(&o);
        let Some(bank) = Bank::new(&o, &p) else {
            continue;
        };
        let leaves = (0..p.bays.len())
            .filter_map(|i| bank.leaf(&o, &p, i))
            .collect::<Vec<_>>();
        for pair in leaves.windows(2).filter(|a| a[1].bay == a[0].bay + 1) {
            paired += usize::from(pair[0].hand == Hand::Left && pair[1].hand == Hand::Right);
            asymmetric += usize::from((pair[0].rest_radians - pair[1].rest_radians).abs() > 0.012);
        }
        for d in leaves {
            match d.hand {
                Hand::Left => left += 1,
                Hand::Right => right += 1,
            };
            let b = &p.bays[d.bay];
            let mut animated = Assembly {
                motion_policy: Some(policy.clone()),
                ..default()
            };
            d.build(&mut animated, &o, &p);
            let j = &animated.joints[0];
            assert_eq!(j.pivot, d.pivot());
            assert_eq!(j.parts.len(), animated.parts.len());
            assert!(j.angle * d.hand.sign() > 0.);
            let (lo, hi) = animated.bounds();
            assert!(
                lo.z >= p.body_depth_m * 0.5 - 0.045 && hi.z <= o.size.z * 0.5 + 1e-5,
                "rest {seed}/{d:?}"
            );
            let mut still = Assembly::default();
            d.build(&mut still, &o, &p);
            assert!(still.joints.is_empty());
            assert_eq!((lo, hi), still.bounds());
            for n in 0..=16 {
                let tf = j.transform(n as f32 / 16.);
                for g in animated.parts.values() {
                    for &v in &g.positions {
                        let pt = tf.transform_point(Vec3::from_array(v));
                        assert!(
                            pt.x >= b.left - 0.0001
                                && pt.x <= b.right + 0.0001
                                && pt.z >= p.body_depth_m * 0.5 - 0.0451,
                            "sweep {seed} {pt:?} {d:?}"
                        );
                        assert!((pt.y - d.center_m.y).abs() <= d.size_m.y * 0.5 + 1e-5);
                    }
                }
            }
        }
    }
    assert!(
        left > 100 && right > 100 && paired > 35 && asymmetric > 35,
        "{left}/{right}/{paired}/{asymmetric}"
    );
}

#[test]
fn old_cabinet_archives_keep_the_original_front_and_new_materials_preserve_shape() {
    let mut checked = 0;
    for seed in 0..48 {
        let o = audit::fixture(ObjectKind::Cabinet, seed);
        let p = parameters(&o);
        let Some(bank) = Bank::new(&o, &p) else {
            continue;
        };
        if !(0..p.bays.len()).any(|i| bank.leaf(&o, &p, i).is_some()) {
            continue;
        }
        let before = build_object_at_version(&o, None, hinged::INTRODUCED - 1);
        let after = build_object_at_version(&o, None, hinged::INTRODUCED);
        assert!(!before
            .parts
            .keys()
            .any(|(_, l)| l.starts_with("cabinet#uv_local")));
        assert!(after
            .parts
            .keys()
            .any(|(_, l)| l.starts_with("cabinet#uv_local")));
        let mut other = o.clone();
        other.finish_seed = Some(seed + 41);
        // Isolate the finish seed at each version. Shelf contents can acquire
        // new geometry independently of the cabinet front in a later generator.
        for (version, mesh) in [
            (hinged::INTRODUCED, after),
            (
                crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
                build_object(&o),
            ),
        ] {
            assert_eq!(
                audit::descriptor(&o, &mesh),
                audit::descriptor(&other, &build_object_at_version(&other, None, version)),
            );
        }
        checked += 1;
    }
    assert!(checked >= 12);
}

#[test]
fn shallow_bow_pulls_keep_the_inside_bend_open() {
    // This shallow case originally folded the inner triangles of a thick bow.
    let mut o = audit::fixture(ObjectKind::Cabinet, 403);
    o.size = Vec3::new(1.1880105, 1.60352, 0.46270716);
    o.finish_seed = None;
    let p = parameters(&o);
    let bank = Bank::new(&o, &p).unwrap();
    assert!((0..p.bays.len())
        .filter_map(|i| bank.leaf(&o, &p, i))
        .any(|d| d.pull == Pull::Bow));
    audit::check(&o, &build_object(&o)).unwrap();
}
