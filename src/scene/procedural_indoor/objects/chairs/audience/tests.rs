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
        t.build(&mut a, p.shell_stock);
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
    for side in [-1., 1.] {
        let mount = p.back.mount(side);
        assert!(mount.z > p.back.z && mount.z < o.size.z * 0.5);
    }
}
