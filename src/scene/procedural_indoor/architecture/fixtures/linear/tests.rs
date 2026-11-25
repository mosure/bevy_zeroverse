use super::*;
use crate::scene::procedural_indoor::{envelope::ray_triangle, layout::IndoorLayout};

#[test]
fn stock_fits_its_reservation_with_separate_tubes_or_one_continuous_lens() {
    let mut room =
        IndoorManifest::generate_with_humans(409, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let at = Some(Vec3::new(0.3, 2.9, 0.17));
    let mut forms = [0; 3];
    let mut counts = [0; 4];
    for seed in 0..96 {
        room.seed = seed;
        let mut finishes = None;
        for s in [
            Vec3::new(0.08, 0.03, 0.41),
            Vec3::new(1.6, 0.08, 0.52),
            Vec3::new(0.58, 0.04, 0.58),
            Vec3::new(0.36, 0.15, 1.4),
        ] {
            let f = LinearFixture::fit(&room, FixtureShape::Tube, at, s).unwrap();
            let stock = (f.form, f.housing, f.trim);
            assert_eq!(*finishes.get_or_insert(stock), stock);
            forms[f.form as usize] += 1;
            counts[f.strips - 1] += 1;
            let mut a = Assembly::default();
            f.body(&mut a, "lamp", "trim", Vec3::ZERO);
            let g = a.part(Surface::Light, "lamp");
            f.aperture(g, Vec3::ZERO);
            let triangles: Vec<_> = g
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| t.map(|i| Vec3::from_array(g.positions[i as usize])))
                .collect();
            let area: f32 = triangles
                .iter()
                .map(|[a, b, c]| (b - a).cross(c - a).dot(Vec3::NEG_Y).max(0.) * 0.5)
                .sum();
            assert!((area / f.projected_area_m2() - 1.).abs() < 0.0001);
            let across = Vec3::new(f.along().z, 0., f.along().x);
            let hits = |offset| {
                let origin = across * offset + f.along() * 0.007 + Vec3::NEG_Y;
                triangles
                    .iter()
                    .any(|t| ray_triangle(origin, Vec3::Y, *t).is_some())
            };
            for i in 0..f.strips {
                let center = (i as f32 - (f.strips - 1) as f32 * 0.5) * f.pitch_m;
                for q in [-0.46, 0., 0.46] {
                    assert!(hits(center + q * f.strip_width_m));
                }
                if i + 1 < f.strips {
                    assert!(!hits(center + 0.5 * f.pitch_m));
                }
            }
            for mesh in a.parts.values() {
                for p in &mesh.positions {
                    assert!(
                        Vec3::from_array(*p)
                            .abs()
                            .cmple(s * 0.5 + Vec3::splat(0.00001))
                            .all(),
                        "{seed}: {p:?} outside {s:?}, {f:?}"
                    );
                }
                assert!(mesh
                    .normals
                    .iter()
                    .all(|n| (Vec3::from_array(*n).length() - 1.).abs() < 1e-4));
                assert!(mesh.uvs.iter().flatten().all(|v| v.is_finite()));
                for t in mesh.indices.as_chunks::<3>().0 {
                    let [x, y, z] = t.map(|i| Vec3::from_array(mesh.positions[i as usize]));
                    let face = (y - x).cross(z - x);
                    let normal = t
                        .iter()
                        .map(|&i| Vec3::from_array(mesh.normals[i as usize]))
                        .sum::<Vec3>();
                    assert!(face.dot(normal) > 0., "{seed}: bad winding {f:?}");
                }
            }
            for anchor in f.anchors() {
                assert!(anchor.dot(across).abs() < 1e-6 && anchor.length() < f.body_length_m * 0.5);
                assert!(anchor.length() > 0.2 * f.body_length_m);
            }
        }
    }
    assert!(forms.iter().all(|&n| n >= 10), "{forms:?}");
    assert!(counts.iter().all(|&n| n >= 3), "{counts:?}");
    room.generator_version = 246;
    assert!(LinearFixture::fit(&room, FixtureShape::Tube, at, Vec3::ONE).is_none());
}

#[test]
fn probes_use_exact_visible_apertures_across_circuits_and_legacy_versions() {
    let mut found = [false; 6];
    for seed in 400..432 {
        let mut room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mounts = super::super::super::fixture_positions(&room);
        for version in [246, 247] {
            room.generator_version = version;
            for (i, &p) in mounts.iter().enumerate() {
                let shape = parameters(&room).shape;
                found[shape as usize] = true;
                let a = geometry(&room, i, p);
                let visible = &a.parts[&(Surface::Light, format!("lamp#{i}"))];
                let (area, probe) = reflection_aperture(&room, p);
                assert_eq!(area, emitting_area_at(&room, p));
                assert_eq!(probe.positions, visible.positions);
                assert_eq!(probe.indices, visible.indices);
                assert_eq!(probe.normals, visible.normals);
                assert_eq!(probe.uvs, visible.uvs);
            }
        }
    }
    assert!(found.into_iter().all(|v| v));
}
