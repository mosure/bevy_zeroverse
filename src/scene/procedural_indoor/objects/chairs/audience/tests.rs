use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::BTreeMap;

#[test]
fn retraction_preserves_the_bearing_centre_and_clears_the_back_throughout_its_sweep() {
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 11;
        o.size = Vec3::new(
            0.65,
            0.84 + 0.32 * (seed % 17) as f32 / 16.,
            0.56 + 0.21 * (seed % 9) as f32 / 8.,
        );
        let p = TeachingSeat::sample(&o);
        let local = Vec3::Z * p.hinge.z;
        let nose = Vec3::NEG_Z * p.seat_depth * 0.5;
        let mut last_height = 0.;
        for step in 0..=24 {
            let tf = p.seat_pose(p.fold_angle * step as f32 / 24.);
            assert!(tf.transform_point(local).distance(p.hinge) < 1e-6);
            let front = tf.transform_point(nose);
            assert!(front.y >= last_height);
            last_height = front.y;
            for z in [-0.5, 0.5] {
                for y in [-0.5, 0.5] {
                    let corner = tf.transform_point(Vec3::new(0., y * p.pad, z * p.seat_depth));
                    assert!(corner.z + 0.0078 < p.back.z && corner.y > 0.28 && corner.y < o.size.y);
                }
            }
        }
        assert_eq!(super::super::supports_upright_occupant(&o), !p.folded);
    }
}

#[test]
fn handed_tablets_use_their_actual_side_for_support_geometry_and_clearance() {
    let mut left = 0;
    for seed in 0..160 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 10;
        let p = TeachingSeat::sample(&o);
        let t = p.tablet.unwrap();
        left += usize::from(t.left_handed);
        let bounds = clearance_boxes(&o);
        let contains = |v: Vec3| {
            bounds.iter().any(|(lo, hi)| {
                v.cmpge(*lo - Vec3::splat(1e-6)).all() && v.cmple(*hi + Vec3::splat(1e-6)).all()
            })
        };
        assert!(contains(t.center));
        assert!(!contains(t.center.with_x(-t.center.x)));
        let mut a = Assembly::default();
        t.build(&mut a);
        audit::check(&o, &a).unwrap();
        let (lo, hi) = a.bounds();
        assert!(
            (lo.y - (t.center.y - t.thickness * 0.5)).abs() < 1e-6
                && (hi.y - (t.center.y + t.thickness * 0.5)).abs() < 1e-6
        );
        let mut frame = Assembly::default();
        super::frame::build(&mut frame, &p);
        let installed = frame.parts.values().flat_map(|g| &g.positions).any(|v| {
            let v = Vec3::from_array(*v);
            v.y > lo.y && v.y < t.center.y && v.x > lo.x && v.x < hi.x && v.z > lo.z && v.z < hi.z
        });
        assert!(installed);
    }
    assert!((10..44).contains(&left), "{left}");
}

#[test]
fn mounted_seat_skins_share_watertight_material_boundaries_and_local_uv_frames() {
    let mut o = audit::fixture(ObjectKind::Chair, 77);
    o.variant = 11;
    let p = TeachingSeat::sample(&o);
    let mut a = Assembly::default();
    p.back.build(&mut a, p.face_stock, p.shell_stock);
    assert_eq!(a.parts.len(), 2);
    audit::check(&o, &a).unwrap();
    assert_closed(&a);
    for side in [-1., 1.] {
        let mount = p.back.mount(side);
        assert!(mount.z > p.back.z && mount.z < o.size.z * 0.5);
    }
}

pub(super) fn assert_closed(a: &Assembly) {
    let mut edges = BTreeMap::new();
    for g in a.parts.values() {
        for t in g.indices.as_chunks::<3>().0 {
            let uv = t.map(|i| Vec2::from_array(g.uvs[i as usize]));
            assert!((uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-9);
            let pts = t.map(|i| g.positions[i as usize].map(f32::to_bits));
            for i in 0..3 {
                let (a, b) = (pts[i], pts[(i + 1) % 3]);
                let (key, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                let n = edges.entry(key).or_insert((0, 0));
                n.0 += 1;
                n.1 += sign;
            }
        }
    }
    assert!(edges.values().all(|v| *v == (2, 0)));
}

#[test]
fn tablet_footprints_reserve_the_top_and_cover_all_four_fasteners() {
    let mut classes = std::collections::BTreeSet::new();
    let mut stock_pairs = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 10;
        o.size = Vec3::new(
            0.56 + 0.026 * (seed % 10) as f32,
            0.96,
            0.55 + 0.038 * (seed % 8) as f32,
        );
        o.finish_seed = Some(seed);
        let t = TeachingSeat::sample(&o).tablet.unwrap();
        classes.insert((format!("{:?}", t.outline), t.left_handed, t.yaw == 0.));
        stock_pairs.insert((format!("{:?}", t.face_stock), format!("{:?}", t.edge_stock)));
        let mut a = Assembly::default();
        t.build(&mut a);
        let (lo, hi) = clearance_boxes(&o)[2];
        for point in a.parts.values().flat_map(|g| &g.positions) {
            let p = Vec3::from_array(*point);
            assert!(
                p.cmpge(lo - Vec3::splat(1e-6)).all() && p.cmple(hi + Vec3::splat(1e-6)).all(),
                "{seed}"
            );
        }
        let (plate, size) = t.mounting_plate();
        let bottom = t.center.y - t.thickness * 0.5;
        let underside = a
            .parts
            .values()
            .flat_map(|g| {
                g.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(move |tri| tri.map(|i| Vec3::from_array(g.positions[i as usize])))
            })
            .filter(|points| points.iter().all(|p| (p.y - bottom).abs() < 1e-6))
            .map(|tri| tri.map(|p| p.xz()))
            .collect::<Vec<_>>();
        for sx in [-0.5, 0.5] {
            for sz in [-0.5, 0.5] {
                let contact =
                    plate.transform_point(Vec3::new(sx * size.x, size.y * 0.5, sz * size.z));
                assert!(contact.y > bottom && contact.y < t.center.y);
                assert!(
                    underside
                        .iter()
                        .any(|tri| inside_triangle(*tri, contact.xz())),
                    "unsupported tablet {seed}"
                );
            }
        }
        // Changing the room's finish cannot move a top, its support, or the cutout.
        o.finish_seed = Some(seed + 500);
        let refinished = TeachingSeat::sample(&o).tablet.unwrap();
        assert_eq!(refinished.mounting_plate(), t.mounting_plate());
        let mut b = Assembly::default();
        refinished.build(&mut b);
        let vertices = |a: &Assembly| {
            let mut points: Vec<_> = a
                .parts
                .values()
                .flat_map(|g| g.positions.iter().map(|p| p.map(f32::to_bits)))
                .collect();
            points.sort_unstable();
            points
        };
        assert_eq!(vertices(&a), vertices(&b));
    }
    assert_eq!(classes.len(), 12);
    assert_eq!(stock_pairs.len(), 3);
}

// Orientation-independent point test on the actual triangles used by the renderer.
pub(super) fn inside_triangle([a, b, c]: [Vec2; 3], p: Vec2) -> bool {
    let area = (b - a).perp_dot(c - a);
    if area.abs() < 1e-12 {
        return false;
    }
    let orientation = area.signum();
    [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(u, v)| orientation * (v - u).perp_dot(p - u) >= -1e-9)
}

#[test]
fn tablet_seat_stays_bear_on_rear_plates_without_breaking_through_the_front() {
    for seed in 0..64 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 10;
        let seat = TeachingSeat::sample(&o);
        let mut skin = Assembly::default();
        seat.back
            .build(&mut skin, seat.face_stock, seat.shell_stock);
        let mut stays = Assembly::default();
        frame::build(&mut stays, &seat);
        let front: Vec<_> = skin
            .parts
            .values()
            .flat_map(|g| {
                g.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter(|t| g.normals[t[0] as usize][2] < -0.60)
                    .map(|t| t.map(|i| Vec3::from_array(g.positions[i as usize])))
            })
            .collect();
        let mut samples = 0;
        for g in stays.parts.values() {
            for triangle in g.indices.as_chunks::<3>().0 {
                let tri = triangle.map(|i| Vec3::from_array(g.positions[i as usize]));
                for p in [tri[0], (tri[0] + tri[1] + tri[2]) / 3.] {
                    if p.y < 0.54 || p.z < seat.back.z - 0.02 {
                        continue;
                    }
                    for face in &front {
                        let [a, b, c] = face.map(|p| p.truncate());
                        if !inside_triangle([a, b, c], p.truncate()) {
                            continue;
                        }
                        let area = (b - a).perp_dot(c - a);
                        let v = (p.truncate() - a).perp_dot(c - a) / area;
                        let w = (b - a).perp_dot(p.truncate() - a) / area;
                        let front_z =
                            face[0].z + v * (face[1].z - face[0].z) + w * (face[2].z - face[0].z);
                        assert!(
                            p.z >= front_z - 1e-5,
                            "stay pierces chair {seed} at {p:?} / {front_z}"
                        );
                        samples += 1;
                    }
                }
            }
        }
        assert!(samples > 32, "missed rear mounting plates on chair {seed}");
    }
}

#[test]
fn rehearsal_chairs_fit_the_rear_stays_inside_the_reserved_depth() {
    for seed in 0..96 {
        for (w, d) in [(0.475, 0.465), (0.61, 0.60)] {
            let mut o = audit::fixture(ObjectKind::Chair, seed);
            o.variant = super::super::ENSEMBLE_CHAIR;
            o.size = Vec3::new(w, 0.86, d);
            let seat = TeachingSeat::sample(&o);
            assert!(!seat.floor_mounted && !seat.folded && seat.tablet.is_none());
            let mut a = Assembly::default();
            seat.build(&mut a);
            audit::check(&o, &a).unwrap();
        }
    }
}
