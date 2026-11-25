use super::*;
use crate::scene::procedural_indoor::objects::audit;

#[test]
fn desk_fan_guard_and_bearings_clear_the_whole_swept_disk() {
    let mut styles = [0; 2];
    for seed in 0..160 {
        for size in [[0.18, 0.27, 0.15], [0.28, 0.42, 0.23], [0.18, 0.40, 0.15]] {
            let mut o = audit::fixture(ObjectKind::DeskFan, seed);
            o.size = Vec3::from_array(size);
            let p = parameters(&o);
            let head = FanHead::sample(&o, &p);
            let mut blades = Geometry::default();
            head.rotor.append(&mut blades, seed, |[x, y, z]| [x, -z, y]);
            let mut interval = Vec2::new(f32::INFINITY, f32::NEG_INFINITY);
            for &v in &blades.positions {
                let v = Vec3::from_array(v);
                assert!(v.is_finite() && v.xy().length() <= head.rotor.radius_m + 1e-6);
                interval.x = interval.x.min(v.z);
                interval.y = interval.y.max(v.z);
            }
            assert!(interval.y - interval.x > head.rotor.thickness_m * 2.);
            assert!(
                head.rotor.pitch_radians > head.rotor.tip_pitch_radians
                    && head.rotor.tip_pitch_radians > 0.
            );
            let mut fixed = Geometry::default();
            head.grille(&mut fixed, &p);
            let anchor = head.mount(&mut fixed, &p);
            for tri in fixed.indices.as_chunks::<3>().0 {
                let vertices =
                    [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(fixed.positions[i as usize]));
                let direction = (vertices[0] + vertices[1] + vertices[2])
                    .xy()
                    .normalize_or_zero();
                let outside_disk = vertices
                    .iter()
                    .all(|v| v.xy().dot(direction) > head.rotor.radius_m);
                assert!(
                    outside_disk
                        || vertices.iter().all(|v| v.z > interval.y)
                        || vertices.iter().all(|v| v.z < interval.x),
                    "{seed}/{size:?} {vertices:?} overlaps rotor {interval:?}"
                );
            }
            let base = p
                .head_transform()
                .compute_affine()
                .inverse()
                .transform_point3(Vec3::Y * (p.base.y * 0.78));
            for i in 0..=48 {
                let t = i as f32 / 48.;
                let v = base.lerp(anchor, t);
                let joint = (p.radius * 0.082).min(p.depth * 0.075);
                let stem = (o.size.x * 0.030).lerp(joint * 0.76, t);
                assert!(
                    v.xy().length() - stem > head.rotor.radius_m
                        || v.z + stem < interval.x
                        || v.z - stem > interval.y,
                    "stem {seed}/{size:?}"
                );
            }
            styles[p.yoke as usize] += 1;
        }
    }
    assert!(styles.iter().all(|&n| n > 100));
}

#[test]
fn desk_fan_finish_resampling_preserves_blade_tooling_and_archive_replay() {
    use std::collections::BTreeSet;
    let mut roles = BTreeSet::new();
    let o = audit::fixture(ObjectKind::DeskFan, 37);
    let p = parameters(&o);
    let reference = serde_json::to_value(FanHead::sample(&o, &p).rotor).unwrap();
    for finish in 0..96 {
        let mut item = o.clone();
        item.finish_seed = Some(finish);
        let head = FanHead::sample(&item, &parameters(&item));
        assert_eq!(serde_json::to_value(&head.rotor).unwrap(), reference);
        roles.insert((head.guard, head.housing, head.blades));
    }
    assert_eq!(roles.len(), 12);
    let mut old = Assembly::default();
    let mut replay = Assembly::default();
    legacy::build(&mut old, &o);
    build(&mut replay, &o, FORMED_ROTOR_VERSION - 1);
    for (role, mesh) in old.parts {
        let other = &replay.parts[&role];
        assert_eq!(mesh.positions, other.positions);
        assert_eq!(mesh.normals, other.normals);
        assert_eq!(mesh.uvs, other.uvs);
        assert_eq!(mesh.indices, other.indices);
    }
}
