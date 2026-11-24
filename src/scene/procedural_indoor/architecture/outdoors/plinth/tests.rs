use super::{
    mesh::{perimeter, Edge},
    *,
};
use crate::scene::procedural_indoor::architecture::overlap_tests::intersection_area;

#[test]
fn exposed_perimeter_shares_mitres_including_the_inside_corner() {
    let size = Vec2::new(8., 4.);
    for notch in [None, Some(Vec2::new(-0.7, 0.5)), Some(Vec2::new(0.3, 1.92))] {
        let (corners, count) = perimeter(size, notch);
        let corners = &corners[..count];
        let mut tops = Vec::new();
        for i in 0..count {
            let edge = Edge::new(corners, i);
            let next = Edge::new(corners, (i + 1) % count);
            for d in [0., 0.018, 0.053] {
                let section = Vec2::new(d, 0.17);
                assert_eq!(edge.point(edge.length, section), next.point(0., section));
            }
            let inside = Vec2::new(0., 0.17);
            let outside = Vec2::new(0.053, 0.16);
            let [a, b, c, d] = [
                edge.point(0., inside),
                edge.point(edge.length, inside),
                edge.point(edge.length, outside),
                edge.point(0., outside),
            ]
            .map(|v| v.xz());
            tops.extend([[a, b, c], [a, c, d]]);
        }
        for (i, &t) in tops.iter().enumerate() {
            for &u in &tops[i + 1..] {
                assert!(
                    intersection_area(t, u) < 4e-6,
                    "overlapping corner {notch:?}"
                );
            }
        }
    }
}

fn fixture(seed: u64) -> Parcel {
    Parcel {
        center: Vec2::new(9.1, -7.2),
        size: Vec2::new(7.31, 3.8),
        yaw: 1.15,
        kind: 0,
        seed,
    }
}

#[test]
fn each_wings_real_threshold_and_low_window_frames_stay_clear() {
    let site = fixture(37);
    let mut b = Building::sample(&site, 36., FIRST_VERSION);
    b.split = 0.51;
    b.wing_depth = 0.96;
    b.sill = 0.10;
    b.frame_width = 0.064;
    b.window_installation.recess_m = 0.027;
    b.window_installation.rebate_m = 0.012;
    b.bay_pitch = 1.1;
    let notch = Vec2::new(
        site.size.x * (b.split - 0.5),
        site.size.y * (b.wing_depth - 0.5),
    );
    let half = site.size * 0.5;
    let tf = Transform::from_translation(site.center.extend(0.).xzy())
        .with_rotation(Quat::from_rotation_y(site.yaw));
    let mut leafs = Vec::new();
    let doors = [(-half.x, notch.x, half.y), (notch.x, half.x, notch.y)].map(|(left, right, z)| {
        let size = Vec2::new(right - left, z + half.y);
        let wing_site = Parcel {
            center: tf
                .transform_point(Vec3::new((left + right) * 0.5, 0., (z - half.y) * 0.5))
                .xz(),
            size,
            ..site.clone()
        };
        let mut wing = b.clone();
        wing.split = 1.;
        let door = buildings::main_entrance(&wing_site, &wing).unwrap();
        let x = (tf.rotation.inverse() * (door.threshold - site.center).extend(0.).xzy()).x;
        leafs.push((x, z, door.width));
        Some(door)
    });
    for seed in 0..64 {
        let course = Course::sample(seed, b.wall);
        let Some(top) = course.top(&b) else {
            continue;
        };
        let mut a = Assembly::default();
        course.build(&mut a, &site, &b, Some(notch), &doors);
        let g = a.parts.values().next().unwrap();
        let inv = tf.compute_affine().inverse();
        let mut ended = [[false; 2]; 2];
        for &[i, j, k] in g.indices.as_chunks::<3>().0 {
            let tri = [i, j, k].map(|id| inv.transform_point3(g.positions[id as usize].into()));
            assert!((tri[1] - tri[0]).cross(tri[2] - tri[0]).length_squared() > 1e-14);
            for v in tri {
                assert!(
                    v.y <= top + 0.00001 && v.y < FOUNDATION + b.pitch * b.sill - b.frame_width
                );
            }
            let range = |f: fn(Vec3) -> f32| {
                let xyz = tri.map(f);
                (
                    xyz.into_iter().fold(f32::INFINITY, f32::min),
                    xyz.into_iter().fold(f32::NEG_INFINITY, f32::max),
                )
            };
            let (lo, hi) = range(|v| v.x);
            let (front, back) = range(|v| v.z);
            for (n, &(x, z, width)) in leafs.iter().enumerate() {
                if front < z - 1e-5 || back > z + course.fitted_projection(&b, site.size) + 1e-5 {
                    continue;
                }
                let r = width * 0.5 + b.frame_width + JAMB_GAP;
                assert!(
                    hi <= x - r + 2e-5 || lo >= x + r - 2e-5,
                    "course traversed a door {seed}"
                );
                if (hi - lo).abs() < 1e-5 {
                    for (k, limit) in [x - r, x + r].into_iter().enumerate() {
                        ended[n][k] |= (lo - limit).abs() < 2e-5;
                    }
                }
            }
        }
        assert_eq!(ended, [[true; 2]; 2], "both jamb returns exist {seed}");
        if course.stock == Stock::FiredClay {
            let mut fitted = b.clone();
            fitted.wall = Surface::Concrete;
            fitted.frame = Surface::Metal;
            fitted.roof_material = Surface::Metal;
            fitted.plinth = Some(course);
            let mut complete = Assembly::default();
            buildings::build(&mut complete, &site, &fitted, None);
            let built = &complete.parts[a.parts.keys().next().unwrap()];
            assert_eq!(built.indices, g.indices);
            for (v, u) in built.positions.iter().zip(&g.positions) {
                assert!(Vec3::from_array(*v).distance((*u).into()) < 0.00002);
            }
        }
    }
}

#[test]
fn sampled_stock_replays_and_legacy_manifests_keep_the_original_base() {
    use std::collections::BTreeSet;
    let mut stocks = BTreeSet::new();
    let mut heights = BTreeSet::new();
    let mut projections = BTreeSet::new();
    for seed in 0..256 {
        let p = fixture(seed);
        let b = Building::sample(&p, 36., FIRST_VERSION);
        let old = Building::sample(&p, 36., FIRST_VERSION - 1);
        assert!(old.plinth.is_none());
        assert_eq!(
            (old.levels, old.split, old.frame_width, old.sill),
            (b.levels, b.split, b.frame_width, b.sill)
        );
        let course = b.plinth.as_ref().unwrap();
        let mut a = Vec::new();
        let mut replay = Vec::new();
        course.measure(&b, p.size, &mut a);
        Course::sample(seed, b.wall).measure(&b, p.size, &mut replay);
        assert_eq!(a, replay);
        stocks.insert(course.stock as u8);
        if let Some(y) = course.top(&b) {
            heights.insert((y * 1000.) as i32);
            projections.insert((course.fitted_projection(&b, p.size) * 1000.) as i32);
        }
    }
    assert_eq!(stocks.len(), 5);
    assert!(heights.len() > 90 && projections.len() > 50);
}
