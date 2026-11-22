use super::*;
use crate::scene::procedural_indoor::photometry::source_color;
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
fn visible_reflection_selects_the_nearest_front_face_and_keeps_opaque_ties() {
    let face = |height, material| Triangle {
        a: Vec3::new(-1., height, -1.),
        ab: Vec3::X * 3.,
        ac: Vec3::Z * 3.,
        normal: Vec3::NEG_Y,
        uv: [Vec2::ZERO; 3],
        material,
    };
    let mut emitters = ReflectionEmitters {
        triangles: vec![face(3., 0), face(2., 1)],
        nodes: Vec::new(),
        emission: vec![Vec3::Z * 80., Vec3::X * 40.],
    };
    emitters.nodes = bvh::build_probe_tree(&mut emitters.triangles);
    let mut scene = super::super::tests::empty();
    scene.materials.push(DiffuseMaterial {
        albedo: Vec3::ZERO,
        emission: Vec3::Y * 2.,
        uv_transform: Affine2::IDENTITY,
        base_map: None,
        emission_map: EmissionMap::Uniform,
    });
    for (height, expected) in [(1., Vec3::Y * 2.), (2., Vec3::Y * 2.), (4., Vec3::X * 40.)] {
        scene.triangles = vec![face(height, 0)];
        drop(scene.build_bvh());
        assert_eq!(
            scene.visible_reflection(Vec3::ZERO, Vec3::Y, 0., &emitters),
            expected
        );
    }
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
            let expected = linear(source_color(&scene, color))
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

#[test]
fn practical_string_reflections_hit_the_real_aperture_once_and_respect_occlusion() {
    let mut room = scene(FixtureShape::Panel);
    let o = (0..128)
        .map(|seed| {
            let mut o = objects::audit::fixture(ObjectKind::StringLights, seed);
            o.variant = 3;
            o.id = 0;
            o.yaw = 0.8;
            o.position = Vec3::new(20., 3., 0.);
            o
        })
        .find(|o| objects::equipment::signals::string_program(o).shape == 4)
        .unwrap();
    for ((surface, label), _) in objects::build_object(&o).parts {
        if surface == Surface::Light {
            assert!(decorative_emitter_slot(&label).is_none());
        }
    }
    let mut micro = o.clone();
    micro.variant = 1;
    assert!(objects::build_object(&micro)
        .parts
        .keys()
        .filter(|(s, _)| *s == Surface::Light)
        .all(|(_, label)| decorative_emitter_slot(label).is_some()));
    room.objects = vec![o];
    let emitters = ReflectionEmitters::new(&room);
    for (_, _, light) in objects::equipment::signals::strings::practical_lights(&room) {
        let front = light.position + 0.10 * light.axis;
        let rgb = emitters.radiance(front, -light.axis, 0.12).unwrap();
        assert!(
            rgb.max_element() > 0.
                && rgb
                    .normalize()
                    .abs_diff_eq(linear(light.color).normalize(), 1e-5)
        );
        assert!(emitters.radiance(front, -light.axis, 0.09).is_none());
        assert!(emitters
            .radiance(light.position - 0.1 * light.axis, light.axis, 0.12)
            .is_none());
    }
}

#[test]
fn static_task_lamp_reflects_from_its_real_diffuser_and_honors_occlusion() {
    let mut room = scene(FixtureShape::Panel);
    let mut lamp = objects::audit::fixture(super::super::super::layout::ObjectKind::DeskLamp, 0);
    lamp.position = Vec3::new(0.3, 0.8, -0.4);
    lamp.yaw = 0.7;
    room.objects.clear();
    let mut seen = [false; 4];
    for seed in 0..64 {
        lamp.seed = seed;
        let power = super::super::super::photometry::lamp(&room, &lamp);
        let head = objects::accessories::lamp_parameters(&lamp).head as usize;
        if seen[head] || power.lumens <= 0. {
            continue;
        }
        seen[head] = true;
        room.objects = vec![lamp.clone()];
        let beam = objects::accessories::lamps::beam(&lamp);
        let eye = beam.position + beam.direction * 0.04;
        let emitters = ReflectionEmitters::new(&room);
        let observed = emitters.radiance(eye, -beam.direction, 0.06).unwrap();
        let area = objects::accessories::lamps::emitting_area(&lamp);
        let material = super::super::super::photometry::lamp_emission(&room, power, area);
        assert_eq!(
            observed,
            Vec3::new(material.red, material.green, material.blue)
        );
        assert!(emitters.radiance(eye, -beam.direction, 0.03).is_none());
        assert!(emitters
            .radiance(beam.position - beam.direction * 0.03, beam.direction, 0.05)
            .is_none());
    }
    assert_eq!(seen, [true; 4]);
}
