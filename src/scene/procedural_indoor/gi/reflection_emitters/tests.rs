use super::*;
use architecture::fixtures::{parameters, FixtureShape};

fn scene(shape: FixtureShape) -> IndoorManifest {
    let mut scene = IndoorManifest::generate_with_humans(
        6,
        super::super::super::layout::IndoorLayout::Mixed,
        0.65,
        0,
        0.,
    )
    .unwrap();
    scene.program.as_mut().unwrap().fixture_size = Vec2::splat(0.6);
    scene.seed = (0..256)
        .find(|&seed| {
            scene.seed = seed;
            parameters(&scene).shape == shape
        })
        .unwrap();
    scene
}

// Independent f64 triangle oracle, using the actual submitted fixture indices.
fn hit(points: [Vec3; 3], origin: Vec3, direction: Vec3, max: f64) -> Option<f64> {
    let [a, b, c] = points.map(|p| p.as_dvec3());
    let origin = origin.as_dvec3();
    let direction = direction.as_dvec3();
    let (ab, ac) = (b - a, c - a);
    let p = direction.cross(ac);
    let det = ab.dot(p);
    if det <= 1e-9 {
        return None;
    }
    let delta = origin - a;
    let u = delta.dot(p) / det;
    let q = delta.cross(ab);
    let v = direction.dot(q) / det;
    let distance = ac.dot(q) / det;
    (u >= 0. && v >= 0. && u + v <= 1. && distance > 0.0002 && distance < max).then_some(distance)
}

#[test]
fn all_fixture_reflections_match_emissive_meshes_and_photometry() {
    for shape in [
        FixtureShape::Panel,
        FixtureShape::Louver,
        FixtureShape::Downlight,
        FixtureShape::Drum,
        FixtureShape::Ring,
        FixtureShape::Tube,
    ] {
        let scene = scene(shape);
        let emitters = ReflectionEmitters::new(&scene);
        let positions = architecture::fixture_positions(&scene);
        let mut reference = Vec::new();
        for (i, p) in positions.iter().copied().enumerate() {
            let (color, lumens) = architecture::fixture_photometry(&scene, i);
            let expected = linear(Color::srgb(color.x, color.y, color.z))
                * (lumens / (PI * architecture::fixtures::emitting_area_at(&scene, p)));
            for ((surface, _), geometry) in architecture::fixtures::geometry(&scene, i, p).parts {
                if surface != Surface::Light {
                    continue;
                }
                for t in geometry.indices.as_chunks::<3>().0 {
                    reference.push((
                        std::array::from_fn(|j| {
                            Vec3::from_array(geometry.positions[t[j] as usize])
                        }),
                        expected,
                    ));
                }
            }
        }
        assert!(emitters.triangles.len() < positions.len() * 1000);
        let mut hits = 0;
        for y in 0..19 {
            for x in 0..19 {
                let origin = positions[0]
                    + Vec3::new(
                        (x as f32 + 0.173) * 0.04 - 0.36,
                        -0.5,
                        (y as f32 + 0.291) * 0.04 - 0.36,
                    );
                for ray in [Vec3::Y, Vec3::new(0.07, 1., -0.03).normalize()] {
                    let expected = reference
                        .iter()
                        .filter_map(|(t, color)| hit(*t, origin, ray, 1.).map(|d| (d, *color)))
                        .min_by(|a, b| a.0.total_cmp(&b.0));
                    let actual = emitters.radiance(origin, ray, 1.);
                    assert_eq!(
                        actual,
                        expected.map(|v| v.1),
                        "{shape:?} {origin:?} {ray:?}"
                    );
                    hits += usize::from(actual.is_some());
                    assert!(
                        emitters.radiance(origin, ray, 0.1).is_none(),
                        "occluder limit"
                    );
                }
            }
        }
        assert!(hits > 0, "{shape:?} never emitted");
    }
}

#[test]
fn ring_hole_and_tube_gap_do_not_become_rectangular_lights() {
    for shape in [FixtureShape::Ring, FixtureShape::Tube] {
        let scene = scene(shape);
        let p = architecture::fixture_positions(&scene)[0];
        let emitters = ReflectionEmitters::new(&scene);
        assert!(
            emitters.radiance(p - Vec3::Y * 0.5, Vec3::Y, 1.).is_none(),
            "{shape:?} filled its unlit center"
        );
        let s = architecture::fixtures::size(&scene, p);
        let lit = if shape == FixtureShape::Ring {
            Vec3::X * (s.x.min(s.z) * 0.46 * 0.9)
        } else {
            let across = if s.x >= s.z { Vec3::Z } else { Vec3::X };
            across * (s.x.min(s.z) * 0.22)
        };
        let lit = architecture::fixtures::rotation(&scene, p) * lit;
        assert!(emitters
            .radiance(p + lit - Vec3::Y * 0.5, Vec3::Y, 1.)
            .is_some());
        if shape == FixtureShape::Ring {
            assert!(
                emitters
                    .radiance(p + lit + Vec3::Y * 0.5, -Vec3::Y, 1.)
                    .is_none(),
                "ring back faces must match rendered culling"
            );
        }
    }
}
