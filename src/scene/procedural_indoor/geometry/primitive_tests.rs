//! Exact geometry oracles for bounded primitive reuse; no graphics device.
use super::*;

fn assert_exact(a: &Geometry, b: &Geometry) {
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&a.positions),
        bytemuck::cast_slice::<_, u8>(&b.positions),
        "positions"
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&a.normals),
        bytemuck::cast_slice::<_, u8>(&b.normals),
        "normals"
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&a.uvs),
        bytemuck::cast_slice::<_, u8>(&b.uvs),
        "UVs"
    );
    assert_eq!(a.indices, b.indices);
}

fn transform(i: usize) -> Transform {
    Transform::from_xyz(i as f32 * 0.13 - 1.5, -0.47, 1.3)
        .with_rotation(Quat::from_euler(EulerRot::YXZ, i as f32 * 0.37, -0.3, 0.23))
        .with_scale(Vec3::new(
            if i.is_multiple_of(2) { 0.31 } else { -0.73 },
            1.7,
            0.63,
        ))
}

#[test]
fn ellipsoid_cache_preserves_mesh_bits_and_tangent_frames() {
    for i in 0..16 {
        let radii = Vec3::new(0.02 + i as f32 * 0.031, 0.4, 0.09);
        let tf = transform(i);
        let mut original = Geometry::default();
        let mut actual = Geometry::default();
        // Nonempty destinations exercise appended vertex/index offsets.
        original.cuboid(Vec3::splat(0.2), 0.01, tf);
        actual.cuboid(Vec3::splat(0.2), 0.01, tf);
        original.mesh(
            Sphere::new(1.0).mesh().uv(20, 12),
            tf.with_scale(tf.scale * radii),
        );
        actual.ellipsoid(radii, tf);
        assert_exact(&original, &actual);
        let original = original.into_mesh();
        let actual = actual.into_mesh();
        for ((id, expected), (actual_id, value)) in original.attributes().zip(actual.attributes()) {
            assert_eq!(id, actual_id);
            assert_eq!(expected.get_bytes(), value.get_bytes());
        }
    }
}

#[test]
fn cached_lathe_angles_preserve_profile_seams_caps_and_signed_bits() {
    let profiles: &[&[(f32, f32)]] = &[
        &[],
        &[(0., 0.)],
        &[(0., -0.4), (0.31, -0.4), (0.31, 0.4), (0., 0.4)],
        &[
            (0., -0.2),
            (0.12, -0.2),
            (0.14, 0.1),
            (0.12, 0.12),
            (0.08, 0.1),
            (0.05, -0.15),
            (0., -0.15),
        ],
        &[(-0., -0.), (0.11, 0.15), (0.11, 0.15), (0., 0.3)],
    ];
    for segments in [0, 1, 2, 3, 7, 20, 48, 96] {
        for (i, profile) in profiles.iter().enumerate() {
            let mut original = Geometry::default();
            let mut actual = Geometry::default();
            original.cuboid(Vec3::ONE, 0., transform(i));
            actual.cuboid(Vec3::ONE, 0., transform(i));
            original_lathe(&mut original, profile, segments, transform(i));
            actual.lathe(profile, segments, transform(i));
            assert_exact(&original, &actual);
        }
    }
}

#[test]
fn primitive_residency_is_fixed_and_nonstandard_angles_are_not_cached() {
    assert!(std::ptr::eq(
        primitives::unit_sphere(),
        primitives::unit_sphere()
    ));
    assert_eq!(primitives::lathe_angles(20).len(), 21);
    assert!(matches!(
        primitives::lathe_angles(20),
        std::borrow::Cow::Borrowed(_)
    ));
    for segments in [3, 7, 19, 21, 97] {
        assert!(matches!(
            primitives::lathe_angles(segments),
            std::borrow::Cow::Owned(_)
        ));
    }
}

#[test]
#[ignore = "bounded CPU primitive microbenchmark; not completed rendering throughput"]
fn benchmark_primitive_reuse() {
    use std::{hint::black_box, time::Instant};
    let profile = [
        (0., -0.2),
        (0.14, -0.2),
        (0.18, 0.12),
        (0.16, 0.15),
        (0.12, 0.12),
        (0.09, -0.15),
        (0., -0.15),
    ];
    for shape in ["ellipsoid", "lathe"] {
        let mut samples = [Vec::new(), Vec::new()];
        for repetition in 0..6 {
            for candidate in if repetition % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let started = Instant::now();
                let mut geometry = Geometry::default();
                for i in 0..2000 {
                    let tf = black_box(transform(i % 16));
                    if shape == "ellipsoid" {
                        let radii = black_box(Vec3::new(0.17, 0.38, 0.21));
                        if candidate {
                            geometry.ellipsoid(radii, tf);
                        } else {
                            geometry.mesh(
                                Sphere::new(1.).mesh().uv(20, 12),
                                tf.with_scale(tf.scale * radii),
                            );
                        }
                    } else if candidate {
                        geometry.lathe(black_box(&profile), 20, tf);
                    } else {
                        original_lathe(&mut geometry, black_box(&profile), 20, tf);
                    }
                    black_box(&geometry);
                    geometry.positions.clear();
                    geometry.normals.clear();
                    geometry.uvs.clear();
                    geometry.indices.clear();
                }
                samples[usize::from(candidate)].push(started.elapsed().as_secs_f64());
            }
        }
        for times in &mut samples {
            times.sort_by(f64::total_cmp);
        }
        let median = |times: &[f64]| (times[2] + times[3]) * 0.5;
        eprintln!(
            "{}",
            serde_json::json!({"shape":shape,"instances_per_repeat":2000,"repeats":6,"reference_median_seconds":median(&samples[0]),"candidate_median_seconds":median(&samples[1]),"speedup":median(&samples[0])/median(&samples[1]),"scope":"CPU primitive construction; final mesh tangent construction excluded"})
        );
    }
}

// Frozen pre-change profile traversal and angular arithmetic.
fn original_lathe(geometry: &mut Geometry, profile: &[(f32, f32)], segments: u32, tf: Transform) {
    // Duplicate rings at profile creases: a pot rim must not average its outer
    // normal with the inward-facing inner wall or invert the lighting normal.
    for pair in profile.windows(2) {
        let offset = geometry.positions.len() as u32;
        let tangent = Vec2::new(pair[1].0 - pair[0].0, pair[1].1 - pair[0].1).normalize();
        for &(r, y) in pair {
            for i in 0..=segments {
                let a = i as f32 / segments as f32 * TAU;
                let normal = Vec3::new(tangent.y * a.cos(), -tangent.x, tangent.y * a.sin());
                geometry.vertex(
                    Vec3::new(r * a.cos(), y, r * a.sin()),
                    normal,
                    Vec2::new(a * pair[0].0.max(pair[1].0).max(0.001), y + r),
                    &tf,
                );
            }
        }
        for i in 0..segments {
            let a = offset + i;
            let b = a + segments + 1;
            if pair[0].0 > 0.00001 {
                geometry.indices.extend([a, b, a + 1]);
            }
            if pair[1].0 > 0.00001 {
                geometry.indices.extend([a + 1, b, b + 1]);
            }
        }
    }
}
