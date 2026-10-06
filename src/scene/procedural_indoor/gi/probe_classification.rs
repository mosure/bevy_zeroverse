//! Full-six-ray oracle for exact probe classification, including touching faces.
use super::*;

fn legacy_inside_solid(scene: &BakeScene, point: Vec3) -> bool {
    let mut backfaces = 0;
    for direction in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
        if let Some((index, _, _)) = scene.hit(point, direction, 1000.0, false) {
            if scene.triangles[index].normal.dot(direction) > 0.0 {
                backfaces += 1;
            }
        }
    }
    backfaces >= 4
}

fn fixture() -> BakeScene {
    let mut scene = BakeScene {
        triangles: Vec::new(),
        nodes: Vec::new(),
        #[cfg(not(target_arch = "wasm32"))]
        transport_tree: default(),
        materials: Vec::new(),
        lights: Vec::new(),
        sun_direction: Vec3::Y,
        sun: Vec3::ZERO,
        sky: Vec3::ONE,
        bounds_min: Vec3::splat(-4.0),
        bounds_max: Vec3::splat(4.0),
        preparation_ms: 0.0,
        world_rotation: Quat::IDENTITY,
    };
    let mut assembly = Assembly::default();
    for x in -2..=2 {
        for z in -2..=2 {
            assembly.box_part(
                Surface::Paint,
                "box",
                Vec3::new(x as f32, (x + z) as f32 * 0.07, z as f32),
                Vec3::new(if x == 0 { 0.0004 } else { 1.0 }, 0.8, 1.0),
                0.0,
            );
        }
    }
    // Exactly touching slabs introduce equidistant, oppositely oriented faces.
    for y in [2.0, 2.5] {
        assembly.box_part(
            Surface::Paint,
            "slab",
            Vec3::new(0.0, y, 0.0),
            Vec3::new(6.0, 0.5, 6.0),
            0.0,
        );
    }
    scene.add_assembly(&assembly, Transform::IDENTITY, &Default::default());
    scene.build_bvh();
    scene
}

fn points(scene: &BakeScene) -> Vec<Vec3> {
    let mut points = Vec::new();
    for x in -8..=8 {
        for y in -2..=6 {
            for z in -8..=8 {
                points.push(Vec3::new(
                    x as f32 * 0.375,
                    y as f32 * 0.5,
                    z as f32 * 0.375,
                ));
            }
        }
    }
    for triangle in scene.triangles.iter().step_by(3) {
        let center = triangle.a + (triangle.ab + triangle.ac) / 3.0;
        for offset in [-0.00021, -0.00019, -0.0, 0.0, 0.00019, 0.00021] {
            points.push(center + triangle.normal * offset);
        }
        points.extend([
            triangle.a,
            triangle.a + triangle.ab,
            triangle.a + triangle.ac,
        ]);
    }
    points
}

#[test]
fn threshold_classification_preserves_all_cardinal_backface_patterns() {
    let directions = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    for pattern in 0_u32..64 {
        let mut scene = fixture();
        scene.triangles.clear();
        for (index, direction) in directions.into_iter().enumerate() {
            let tangent = if direction.x != 0.0 { Vec3::Y } else { Vec3::X };
            let bitangent = direction.cross(tangent);
            scene.triangles.push(Triangle {
                a: direction - tangent - bitangent,
                ab: tangent * 4.0,
                ac: bitangent * 4.0,
                uv: [Vec2::ZERO; 3],
                normal: if pattern & (1 << index) != 0 {
                    direction
                } else {
                    -direction
                },
                material: 0,
            });
        }
        scene.build_bvh();
        let expected = pattern.count_ones() >= 4;
        assert_eq!(
            legacy_inside_solid(&scene, Vec3::ZERO),
            expected,
            "{pattern:06b}"
        );
        assert_eq!(scene.inside_solid(Vec3::ZERO), expected, "{pattern:06b}");
    }
}

#[test]
fn threshold_classification_matches_full_six_rays_on_boundaries_and_ties() {
    let scene = fixture();
    let mut inside = 0;
    let points = points(&scene);
    for (index, point) in points.iter().copied().enumerate() {
        let expected = legacy_inside_solid(&scene, point);
        inside += usize::from(expected);
        assert_eq!(
            scene.inside_solid(point),
            expected,
            "point {index}: {point:?}"
        );
    }
    assert!(inside > 0 && inside < points.len());
    #[cfg(not(target_arch = "wasm32"))]
    {
        scene.prepare_gpu_transport();
        for point in points {
            assert_eq!(
                scene.inside_solid(point),
                legacy_inside_solid(&scene, point)
            );
        }
    }
}

#[test]
#[ignore = "bounded CPU classification microbenchmark; run separately from GPU measurements"]
fn benchmark_threshold_classification() {
    use std::{hint::black_box, time::Instant};
    let scene = fixture();
    let points = points(&scene);
    let mut elapsed = [0.0; 2];
    let mut hits = [0; 2];
    for repetition in 0..8 {
        for candidate in if repetition % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let started = Instant::now();
            let inside = black_box(&points)
                .iter()
                .filter(|&&point| {
                    if candidate {
                        scene.inside_solid(point)
                    } else {
                        legacy_inside_solid(&scene, point)
                    }
                })
                .count();
            elapsed[usize::from(candidate)] += started.elapsed().as_secs_f64();
            hits[usize::from(candidate)] = black_box(inside);
        }
    }
    assert_eq!(hits[0], hits[1]);
    eprintln!("{{\"kernel\":\"threshold_classification\",\"triangles\":{},\"points\":{},\"repetitions\":8,\"legacy_seconds\":{},\"candidate_seconds\":{},\"speedup\":{}}}", scene.triangles.len(), points.len(), elapsed[0] / 8.0, elapsed[1] / 8.0, elapsed[0] / elapsed[1]);
}
