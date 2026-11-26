use super::*;
use crate::scene::procedural_indoor::{
    motion::IndoorMotionConfig,
    objects::{audit, build_object_at_version},
};

fn chair(seed: u64, variant: u32) -> IndoorObject {
    let mut o = audit::fixture(ObjectKind::Chair, seed);
    o.variant = variant;
    o
}

#[test]
fn mounts_follow_the_chair_construction_and_replay_the_same_order() {
    let mut counts = [0; 4];
    for seed in 0..384 {
        for variant in 0..12 {
            let mut o = chair(seed, variant);
            o.finish_seed = Some(seed / 8);
            o.interaction_target = Some(19);
            let p = parameters(&o);
            assert!(p.back_mount(&o, INTRODUCED - 1).is_none());
            let m = p.back_mount(&o, INTRODUCED);
            assert_eq!(
                m.is_some(),
                matches!(variant, 0..=5 | 9) && p.formed_seat.is_none()
            );
            if let Some(m) = m {
                let replay = p.back_mount(&o, INTRODUCED).unwrap();
                assert_eq!((m.style, m.width_m), (replay.style, replay.width_m));
                assert_eq!(
                    m.style == BackMountStyle::BentPlywood,
                    p.frame_surface == Surface::WoodEdge
                );
                assert!(m.style != BackMountStyle::ForkedSpine || p.base_style < 4);
                counts[m.style as usize] += 1;
            }
        }
    }
    assert!(counts.iter().all(|n| *n > 80), "{counts:?}");
}

fn cap_contains(p: Vec3, tri: [Vec3; 3]) -> bool {
    let [a, b, c] = tri;
    let n = (b - a).cross(c - a).normalize();
    n.dot(p - a).abs() < 1e-5
        && [(a, b), (b, c), (c, a)]
            .into_iter()
            .all(|(v, w)| (w - v).cross(p - v).dot(n) >= -1e-8)
}

#[test]
fn fitted_stock_contacts_the_pan_and_both_sockets_without_piercing_the_cushion() {
    let mut seen = [0; 4];
    for seed in 0..1024 {
        let mut o = chair(seed, (seed % 7) as u32);
        let p = parameters(&o);
        let Some(m) = p.back_mount(&o, INTRODUCED) else {
            continue;
        };
        let bucket = &mut seen[m.style as usize];
        if *bucket == 12 {
            continue;
        }
        *bucket += 1;
        for (reserve, depth_fraction, bottom, socket_y, socket_z) in [
            (0.48, 0.79, 0.423, 0.532, 0.083),
            (0.62, 0.64, 0.386, 0.613, 0.060),
            (0.81, 0.75, 0.398, 0.659, 0.151),
        ] {
            o.size.z = reserve;
            let depth = (reserve * depth_fraction).min(0.55);
            let rear = (depth * 0.465).max(depth * 0.5 - 0.015);
            let mut a = Assembly::default();
            let sockets = [-0.13, 0.13].map(|x| Vec3::new(x, socket_y, socket_z));
            m.build(&mut a, &o, p.frame_surface, depth, bottom, sockets);
            audit::check(&o, &a).unwrap();
            let mut reaches_pan = false;
            let mut reaches_back = [false; 2];
            for ((_, label), g) in &a.parts {
                assert_eq!(label, "chair");
                reaches_pan |= g.positions.iter().copied().map(Vec3::from_array).any(|v| {
                    v.x.abs() < 0.17 && v.z < rear - 0.02 && v.y > bottom && v.y < bottom + 0.003
                });
                for tri in g.indices.as_chunks::<3>().0 {
                    let t = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                    for (hit, socket) in reaches_back.iter_mut().zip(sockets) {
                        *hit |= cap_contains(socket, t);
                    }
                    for (v, w) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        for y in [
                            bottom + 0.023,
                            (bottom + UPRIGHT_SEAT_HEIGHT) * 0.5,
                            UPRIGHT_SEAT_HEIGHT,
                        ] {
                            if (v.y < y) != (w.y < y) {
                                let z = v.lerp(w, (y - v.y) / (w.y - v.y)).z;
                                assert!(
                                    z >= rear - 0.001,
                                    "{:?} penetrates the seat at y={y}: {z} < {rear}",
                                    m.style
                                );
                            }
                        }
                    }
                }
            }
            assert!(
                reaches_pan && reaches_back == [true; 2],
                "{:?}: pan={reaches_pan} back={reaches_back:?}",
                m.style
            );
        }
    }
    assert_eq!(seen, [12; 4]);
}

#[test]
fn backrest_mounts_travel_with_the_swivelling_upper_and_leave_the_base_unchanged() {
    let policy = IndoorMotionConfig {
        chair_fraction: 1.,
        ..default()
    };
    let mut checked = 0;
    for seed in 0..120 {
        let o = chair(seed, [0, 1, 5, 9][seed as usize % 4]);
        if parameters(&o).back_mount(&o, INTRODUCED).is_none() {
            continue;
        }
        let old = build_object_at_version(&o, Some(&policy), INTRODUCED - 1);
        let new = build_object_at_version(&o, Some(&policy), INTRODUCED);
        assert_eq!((old.joints.len(), new.joints.len()), (2, 2));
        for (was, now) in old.joints.iter().zip(&new.joints) {
            assert_eq!(
                (
                    was.pivot,
                    was.axis,
                    was.angle,
                    was.translation,
                    was.interval
                ),
                (
                    now.pivot,
                    now.axis,
                    now.angle,
                    now.translation,
                    now.interval
                )
            );
            assert!(now.parts.iter().all(|key| new.parts.contains_key(key)));
        }
        assert!(new.joints[0].parts.iter().any(|key| old
            .parts
            .get(key)
            .is_none_or(|g| g.positions != new.parts[key].positions)));
        assert_eq!(old.joints[1].parts, new.joints[1].parts);
        for key in &old.joints[1].parts {
            let (x, y) = (&old.parts[key], &new.parts[key]);
            assert!(
                x.positions == y.positions
                    && x.indices == y.indices
                    && x.normals == y.normals
                    && x.uvs == y.uvs
            );
        }
        checked += 1;
    }
    assert!(checked >= 30);
}
