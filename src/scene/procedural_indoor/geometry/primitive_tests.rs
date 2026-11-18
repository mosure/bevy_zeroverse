//! Exact geometry oracles for bounded primitive reuse; no graphics device.
use super::*;

fn assert_spatially_exact(a: &Geometry, b: &Geometry) {
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
    assert_eq!(a.indices, b.indices);
}

fn assert_exact(a: &Geometry, b: &Geometry) {
    assert_spatially_exact(a, b);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&a.uvs),
        bytemuck::cast_slice::<_, u8>(&b.uvs),
        "UVs"
    );
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
fn formed_back_shell_has_a_positive_draw_depth_and_an_outward_closed_rim() {
    let tf = Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.31, 0.7, -0.22))
        .with_translation(Vec3::new(0.14, 0.6, 0.19));
    for height in [0.08, 0.18, 0.53] {
        for depth in [0.012, 0.045] {
            let mut g = Geometry::default();
            g.chair_back_contour(0.32, height, [0.08, 0.17, 0.13, depth, 0.05, 0.08], tf);
            let grid = 9 * 13;
            let draw = tf.rotation * Vec3::Z * depth;
            for i in 0..grid {
                let p = Vec3::from_array(g.positions[i]);
                let rear = Vec3::from_array(g.positions[i + grid]);
                assert!((rear - p - draw).length() < 2e-7);
            }
            for rim in g.uvs[grid * 2..].as_chunks::<4>().0 {
                assert!((rim[2][1] - depth).abs() < 1e-7);
            }
            for tri in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let normal: Vec3 = tri
                    .iter()
                    .map(|&i| Vec3::from_array(g.normals[i as usize]))
                    .sum();
                assert!((b - a).cross(c - a).dot(normal) > 0.);
            }
        }
    }
}

#[test]
fn lathe_mapping_preserves_positions_normals_winding_and_angular_seams() {
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
        &[(-0., -0.), (0.11, 0.15), (0., 0.3)],
    ];
    for segments in [3, 7, 20, 32, 48, 96] {
        for (i, profile) in profiles.iter().enumerate() {
            let mut original = Geometry::default();
            let mut actual = Geometry::default();
            original.cuboid(Vec3::ONE, 0., transform(i));
            actual.cuboid(Vec3::ONE, 0., transform(i));
            original_lathe(&mut original, profile, segments, transform(i));
            actual.lathe(profile, segments, transform(i));
            // The UVs are deliberately reparameterized; spatial attributes and
            // indices must still be bit-identical to the independent traversal.
            assert_spatially_exact(&original, &actual);
        }
    }
}

#[test]
fn lathe_caps_are_metric_and_tapered_stock_has_a_continuous_nonsingular_chart() {
    // The first shoulder rises exactly as far as the radius shrinks. Using
    // height + radius for V collapses its whole PBR chart onto a line.
    let profile = [(0., 0.), (0.4, 0.), (0.2, 0.2), (0.2, 0.55), (0., 0.55)];
    let mut g = Geometry::default();
    let segments = 20;
    g.lathe(&profile, segments, Transform::IDENTITY);
    let ring = segments as usize + 1;
    for (p, uv) in g.positions[..2 * ring].iter().zip(&g.uvs) {
        assert_eq!(*uv, [p[0], p[2]]);
    }
    assert_eq!(&g.uvs[3 * ring..4 * ring], &g.uvs[4 * ring..5 * ring]);
    assert!((g.uvs[3 * ring][1] - g.uvs[2 * ring][1] - 0.2_f32.hypot(0.2)).abs() < 1e-6);
    for tri in g.indices.as_chunks::<3>().0 {
        let uv = |i: usize| Vec2::from_array(g.uvs[tri[i] as usize]);
        assert!((uv(1) - uv(0)).perp_dot(uv(2) - uv(0)).abs() > 1e-6);
    }
    let mut repeated = Geometry::default();
    let mut knots = profile.to_vec();
    knots.insert(2, profile[1]);
    repeated.lathe(&knots, segments, Transform::IDENTITY);
    assert_exact(&g, &repeated);
    for segments in 0..3 {
        let mut empty = Geometry::default();
        empty.lathe(&profile, segments, Transform::IDENTITY);
        assert!(empty.positions.is_empty() && empty.indices.is_empty());
    }
}

#[test]
fn primitive_residency_is_fixed_and_nonstandard_angles_are_not_cached() {
    assert!(std::ptr::eq(
        primitives::unit_sphere(),
        primitives::unit_sphere()
    ));
    for segments in [20, 32] {
        let angles = primitives::lathe_angles(segments);
        assert_eq!(angles.len(), segments as usize + 1);
        assert!(matches!(angles, std::borrow::Cow::Borrowed(_)));
    }
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
