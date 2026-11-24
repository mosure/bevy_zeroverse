use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
use std::collections::BTreeSet;

/// First visible part of the case from above, independent of the ring builder.
fn first_hit(assembly: &Assembly, point: Vec2) -> (&str, f32) {
    assembly
        .parts
        .iter()
        .flat_map(|((_, label), mesh)| {
            mesh.indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter_map(move |tri| {
                    let [a, b, c] = tri.map(|i| Vec3::from(mesh.positions[i as usize]));
                    let ab = (b - a).xz();
                    let ac = (c - a).xz();
                    let area = ab.perp_dot(ac);
                    if area >= -1e-12 {
                        return None;
                    }
                    let d = point - a.xz();
                    let u = d.perp_dot(ac) / area;
                    let v = ab.perp_dot(d) / area;
                    (u >= -1e-6 && v >= -1e-6 && u + v <= 1.000001)
                        .then(|| (part_label(label), a.y + u * (b.y - a.y) + v * (c.y - a.y)))
                })
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("case covers its footprint")
}

#[test]
fn refill_and_scored_panel_are_first_surfaces_through_the_real_cutout() {
    let mut states = BTreeSet::new();
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::TissueBox, seed);
        o.size = if seed % 3 == 0 {
            Vec3::new(0.114, 0.169, 0.119)
        } else {
            Vec3::new(0.241, 0.124, 0.127)
        };
        for case in [
            Case::FoldingCarton,
            Case::MoldedCover,
            Case::TimberSleeve,
            Case::MetalCover,
        ] {
            o.variant = case as u32;
            let p = parameters(&o);
            let d = p.dispenser.unwrap();
            assert_eq!(d.case, case);
            let h = o.size.y * p.box_height;
            let mut shell = Assembly::default();
            super::case::build(&mut shell, &o, &p, d);
            let sealed = d.draw == Draw::Sealed;
            for at in [Vec2::ZERO, Vec2::new(-0.44, 0.21), Vec2::new(0.54, -0.22)] {
                let (label, y) = first_hit(
                    &shell,
                    d.mouth_offset_m + at * o.size.xz() * p.opening * 0.5,
                );
                assert_eq!(label, if sealed { "box" } else { "paper" });
                let expected = if sealed {
                    h - d.wall_m * 0.58 * 0.25
                } else {
                    h * d.fill_fraction
                };
                assert!((y - expected).abs() < 0.00001, "{y} vs {expected}");
            }
            let shoulder =
                d.mouth_offset_m + Vec2::new(0.0, o.size.z * (0.25 + 0.25 * p.opening.y));
            let (label, y) = first_hit(&shell, shoulder);
            assert_eq!(label, "box");
            assert!((y - h).abs() < 0.00001);
            if !sealed && seed % 16 == 0 {
                let mut sheet = Assembly::default();
                super::sheet::build(&mut sheet, &o, &p, d);
                let ply = sheet.parts.values().next().unwrap();
                let mut feed_vertices = 0;
                for q in ply.positions.iter().copied().map(Vec3::from) {
                    if (q.y - h).abs() < 1e-6 {
                        assert_eq!(
                            first_hit(&shell, q.xz()).0,
                            "paper",
                            "sheet crosses the cut edge"
                        );
                        feed_vertices += 1;
                    }
                }
                assert!(feed_vertices >= 8);
                assert!(sheet.bounds().0.y < h * d.fill_fraction);
            }
            let assembled = build_object(&o);
            let checked = audit::check(&o, &assembled).unwrap();
            assert!(checked.triangles < 1000 && checked.min_y_m.abs() < 1e-5);
            assert!((assembled.bounds().1.y - o.size.y).abs() < 0.0005);
            let labels: BTreeSet<_> = assembled.parts.keys().map(|(_, s)| part_label(s)).collect();
            assert_eq!(labels.contains("paper"), !sealed);
            assert!(labels.contains("box") && labels.len() <= 2);
            states.insert((case as u32, d.draw as u32));
        }
    }
    assert_eq!(states.len(), 13);
}

#[test]
fn archived_variants_replay_their_old_geometry_and_serialized_fields() {
    for variant in 0..4 {
        let mut o = audit::fixture(ObjectKind::TissueBox, 758 + u64::from(variant));
        o.variant = variant;
        let p = parameters(&o);
        assert_eq!(p, legacy::parameters(&o));
        assert!(serde_json::to_value(&p).unwrap().get("dispenser").is_none());
        let mut expected = Assembly::default();
        legacy::build(&mut expected, &o, &p);
        let mut actual = Assembly::default();
        build(&mut actual, &o);
        assert_eq!(expected.parts.len(), actual.parts.len());
        for (key, mesh) in expected.parts {
            let got = &actual.parts[&key];
            assert_eq!(
                (mesh.positions, mesh.normals, mesh.indices, mesh.uvs),
                (
                    got.positions.clone(),
                    got.normals.clone(),
                    got.indices.clone(),
                    got.uvs.clone()
                )
            );
        }
    }
}
