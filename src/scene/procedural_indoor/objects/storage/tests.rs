use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint};
use std::collections::BTreeSet;
mod qualification;

#[test]
fn cabinet_compartments_vary_and_contain_meshes_at_independent_metric_extremes() {
    let mut widths = BTreeSet::new();
    let mut gaps = BTreeSet::new();
    let mut mixed = [0; 4];
    let mut styles = BTreeSet::new();
    for seed in 0..96 {
        for size in [
            Vec3::new(0.55, 0.60, 0.28),
            Vec3::new(1.65, 0.72, 0.52),
            Vec3::new(0.66, 1.90, 0.31),
            Vec3::new(1.8, 2.1, 0.65),
        ] {
            let mut o = audit::fixture(ObjectKind::Cabinet, seed);
            o.size = size;
            let p = parameters(&o);
            assert_eq!(
                serde_json::to_value(&p).unwrap(),
                serde_json::to_value(parameters(&o)).unwrap()
            );
            styles.insert(p.front);
            let a = build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{size:?}: {e}"));
            for (i, b) in p.bays.iter().enumerate() {
                assert!(b.left < b.right && b.levels.len() >= 3);
                assert!(b.levels.windows(2).all(|v| v[1] > v[0] + p.panel_m + 0.01));
                if i > 0 {
                    assert!((b.left - p.bays[i - 1].right - p.panel_m).abs() < 1e-6);
                }
                widths.insert(((b.right - b.left) / size.x * 1000.).round() as i32);
                for v in b.levels.windows(2) {
                    gaps.insert(((v[1] - v[0]) * 1000.).round() as i32);
                }
                if p.front == 2 && b.drawers {
                    mixed[i] += 1;
                }
            }
            assert!((p.bays.last().unwrap().right - (size.x * 0.5 - p.panel_m)).abs() < 1e-6);
            // Full declared top footprint is backed by a real horizontal cap.
            for x in [-0.48, 0., 0.48] {
                for z in [-0.48, 0., 0.48] {
                    let origin = Vec3::new(size.x * x, size.y + 0.05, size.z * z);
                    let hit = ray_hit(&a, origin, Vec3::NEG_Y).unwrap();
                    assert!(
                        (hit - 0.05).abs() < 0.003,
                        "unsupported top {seed}/{size:?}"
                    );
                }
            }
        }
    }
    assert_eq!(styles.len(), 6);
    assert!(widths.len() > 150 && gaps.len() > 150);
    assert!(mixed[0] > 0 && mixed[1] > 0);
}

#[test]
fn open_cabinet_contents_fit_and_rest_on_actual_shelves() {
    let mut items = 0;
    let mut positions = BTreeSet::new();
    let (mut loose, mut centered, mut axis) = (0, 0, 0);
    for seed in 0..192 {
        let o = audit::fixture(ObjectKind::Cabinet, seed);
        let p = parameters(&o);
        if p.front != 3 {
            continue;
        }
        let (inside, shelves) = cabinet::content_program(&o, &p);
        for (index, bay) in p.bays.iter().enumerate() {
            for level in 0..program::open_levels(&p, bay) {
                let base = bay.levels[level] + p.panel_m * 0.5;
                let ceiling = bay.levels[level + 1] - p.panel_m * 0.5;
                let rear = -p.body_depth_m * 0.5 + p.panel_m;
                let front = p.body_depth_m * 0.5 - 0.045;
                let mut occupied = Vec::new();
                for mut c in bookcase::contents::placements(&inside, &shelves, index, level) {
                    c.position.z -= 0.0225;
                    let q = Footprint {
                        center: c.position.xz(),
                        half: rotated_half(c.size.xz(), c.yaw),
                        yaw: c.yaw,
                    };
                    assert!(q.inside(Vec2::new(bay.left, rear), Vec2::new(bay.right, front), 0.));
                    assert!(!occupied.iter().any(|&a| q.overlaps(a, 0.)));
                    occupied.push(q);
                    let mut mesh = Assembly::default();
                    bookcase::contents::build(&mut mesh, &inside, &c);
                    let (lo, hi) = mesh.bounds();
                    assert!((lo.y - base).abs() < 0.001 && hi.y <= ceiling);
                    assert!(lo.x >= bay.left && hi.x <= bay.right && lo.z >= rear && hi.z <= front);
                    positions
                        .insert(((c.position.x * 1000.) as i32, (c.position.z * 1000.) as i32));
                    if !c.upright {
                        loose += 1;
                        let uv = (c.position.xz()
                            - Vec2::new((bay.left + bay.right) * 0.5, (rear + front) * 0.5))
                            / Vec2::new(bay.right - bay.left, front - rear);
                        centered += usize::from(uv.abs().max_element() < 0.02);
                        axis += usize::from(
                            (4. * c.yaw).sin().atan2((4. * c.yaw).cos()).abs() * 0.25
                                < 2_f32.to_radians(),
                        );
                    }
                    items += 1;
                }
            }
        }
    }
    assert!(items > 100 && positions.len() > 100);
    assert!(
        loose > 20 && centered * 10 < loose && axis * 5 < loose,
        "loose={loose}, centered={centered}, axis={axis}"
    );
}

#[test]
fn cabinet_finishes_do_not_change_geometry_or_front_motion() {
    let mut finishes = BTreeSet::new();
    for seed in 0..48 {
        let mut o = audit::fixture(ObjectKind::Cabinet, seed);
        let first = audit::descriptor(&o, &build_object(&o));
        o.finish_seed = Some(seed + 11);
        let p = parameters(&o);
        finishes.insert(format!("{:?}/{:?}/{:?}", p.frame, p.shelves, p.fronts));
        assert_eq!(
            first,
            audit::descriptor(&o, &build_object(&o)),
            "seed {seed}"
        );
    }
    assert!(finishes.len() >= 8);
}

fn ray_hit(a: &Assembly, origin: Vec3, direction: Vec3) -> Option<f32> {
    a.parts
        .values()
        .flat_map(|g| {
            g.indices.as_chunks::<3>().0.iter().filter_map(|ids| {
                let p: [Vec3; 3] =
                    std::array::from_fn(|i| Vec3::from_array(g.positions[ids[i] as usize]));
                let e = p[1] - p[0];
                let f = p[2] - p[0];
                let h = direction.cross(f);
                let det = e.dot(h);
                if det.abs() < 1e-9 {
                    return None;
                }
                let s = origin - p[0];
                let u = s.dot(h) / det;
                let q = s.cross(e);
                let v = direction.dot(q) / det;
                let t = f.dot(q) / det;
                (u >= -1e-6 && v >= -1e-6 && u + v <= 1.000001 && t >= 0.).then_some(t)
            })
        })
        .min_by(f32::total_cmp)
}
