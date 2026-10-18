use super::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// Frozen scalar predicate, independent of the shared closest-hit helper.
fn reference(t: &TransportTriangle, origin: Vec3, direction: Vec3, max: f32) -> bool {
    let p = direction.cross(t.ac);
    let det = t.ab.dot(p);
    if det.abs() < 1e-9 {
        return false;
    }
    let inverse = det.recip();
    let offset = origin - t.a;
    let u = offset.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = offset.cross(t.ab);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let distance = t.ac.dot(q) * inverse;
    distance > 0.0002 && distance < max
}

#[test]
fn packet_matches_frozen_scalar_at_edges_cutoffs_and_mixed_lanes() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5348_4144_4f57);
    let mut hits = 0;
    let mut misses = 0;
    for case in 0..256 {
        let mut vector = || {
            Vec3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            )
        };
        let triangles = std::array::from_fn(|i| TransportTriangle {
            a: vector() * 10.,
            ab: vector() * [0., 1e-7, 0.1, 1., 5.][(case + i) % 5],
            ac: vector() * 0.5,
            source: i as u32,
        });
        for triangle in &triangles {
            let normal = triangle.ab.cross(triangle.ac).normalize_or(Vec3::Y);
            for direction in [normal, -normal, Vec3::X, Vec3::Y, Vec3::Z] {
                for bary in [0., 0.25, 0.5, 0.500_000_1] {
                    let point = triangle.a + (triangle.ab + triangle.ac) * bary;
                    for distance in [0.000_199_99, 0.0002, 0.000_200_01, 0.25, 10.] {
                        let origin = point - direction * distance;
                        for max in [0., 0.0002, 0.25, 1000.] {
                            let expected = triangles
                                .iter()
                                .any(|t| reference(t, origin, direction, max));
                            assert_eq!(
                                occluded(&triangles, origin, direction, max),
                                expected,
                                "case={case}, origin={origin}, direction={direction}, max={max}"
                            );
                            hits += usize::from(expected);
                            misses += usize::from(!expected);
                        }
                    }
                }
            }
        }
        for value in [0., -0., f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let direction = Vec3::new(value, 0., 1.);
            let expected = triangles
                .iter()
                .any(|t| reference(t, Vec3::ZERO, direction, 1.));
            assert_eq!(occluded(&triangles, Vec3::ZERO, direction, 1.), expected);
        }
    }
    assert!(hits > 1000 && misses > 1000, "exercise both outcomes");
}
