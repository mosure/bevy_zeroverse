use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
use std::collections::BTreeSet;

#[test]
fn orders_hold_direction_but_cut_different_stock_at_each_object_and_leaf() {
    let mut edges = BTreeSet::new();
    let mut headings = BTreeSet::new();
    let mut matching = BTreeSet::new();
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::Desk, seed);
        o.finish_seed = Some(seed.wrapping_mul(31));
        for surface in [Surface::Ceramic, Surface::Plastic, Surface::Concrete] {
            let cut = SheetCut::for_version(&o, surface, REGISTERED_SHEET_VERSION);
            let mut moved = o.clone();
            moved.position += Vec3::splat(8.15);
            moved.yaw += 1.21;
            assert_eq!(
                cut,
                SheetCut::for_version(&moved, surface, REGISTERED_SHEET_VERSION)
            );
            moved.seed += 983;
            let next = SheetCut::for_version(&moved, surface, REGISTERED_SHEET_VERSION);
            assert_eq!(
                (cut.heading_rad, cut.edge, cut.match_leaves),
                (next.heading_rad, next.edge, next.match_leaves)
            );
            assert_ne!(cut.face_origin_m, next.face_origin_m);
            assert_eq!(
                cut.face_origin_m == cut.back_origin_m,
                surface != Surface::Plastic
            );
            assert_eq!(cut, cut.for_leaf(seed, 0));
            assert_eq!(cut == cut.for_leaf(seed, 1), cut.match_leaves);
            assert_eq!(
                cut.for_leaf(seed, 1) == cut.for_leaf(seed, 2),
                cut.match_leaves
            );
            edges.insert(cut.edge as u8);
            headings.insert(cut.heading_rad.to_degrees().round() as u16);
            matching.insert(cut.match_leaves);
        }
    }
    assert_eq!((edges.len(), matching.len()), (4, 2));
    assert!(headings.len() > 75);
}

#[test]
fn edge_charts_have_area_and_the_wrapped_face_meets_every_bevel_and_corner() {
    let object = audit::fixture(ObjectKind::Desk, 5);
    let outline = [
        Vec2::new(-1., -0.48),
        Vec2::new(0.84, -0.48),
        Vec2::new(1., -0.31),
        Vec2::new(1., 0.48),
        Vec2::new(-1., 0.48),
    ];
    let tf = Transform::from_xyz(0.73, 0.84, -0.28);
    let height = 0.045;
    let profile = Geometry::slab_profile(&outline, height, 0.004);
    for edge in [
        SheetEdge::FormedFoil,
        SheetEdge::ExposedSlab,
        SheetEdge::BondedBand,
        SheetEdge::TimberLip,
    ] {
        let mut cut = SheetCut::for_version(&object, Surface::Ceramic, REGISTERED_SHEET_VERSION);
        cut.edge = edge;
        let rotation = Mat2::from_angle(cut.heading_rad);
        let mut g = Geometry::default();
        cut.edge_uvs(&mut g, &outline, height, profile, &tf, rotation);
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| Vec2::from_array(g.uvs[i as usize]));
            assert!(
                (b - a).perp_dot(c - a).abs() > 1e-6,
                "{edge:?} collapsed UV triangle"
            );
        }
        let mut perimeter = 0.;
        for i in 0..outline.len() {
            let j = (i + 1) % outline.len();
            let uv = |index: usize| Vec2::from_array(g.uvs[index]);
            if matches!(edge, SheetEdge::ExposedSlab | SheetEdge::FormedFoil) {
                let expected =
                    rotation * (outline[i] * profile.1 + tf.translation.xz()) + cut.face_origin_m;
                assert!(expected.distance(uv(i * 12 + 9)) < 1e-6);
                for ring in 0..3 {
                    assert_eq!(uv(i * 12 + ring * 4 + 2), uv(j * 12 + ring * 4 + 1));
                    assert_eq!(uv(i * 12 + ring * 4 + 3), uv(j * 12 + ring * 4));
                }
                // Across the vertical face, advancing through the stock really
                // changes the chart, including for the lowest grazing cameras.
                let n = -(outline[j] - outline[i]).normalize().perp();
                let reveal = rotation.transpose() * (uv(i * 12 + 4) - uv(i * 12 + 5));
                assert!((reveal.dot(n) - (height - 2. * profile.0)).abs() < 1e-5);
            } else {
                let at = if edge == SheetEdge::TimberLip {
                    Vec2::new(height * 0.5, -perimeter)
                } else {
                    Vec2::new(perimeter, height * 0.5)
                };
                assert!(uv(i * 12 + 9).distance(at + cut.band_origin_m) < 1e-6);
            }
            perimeter += outline[i].distance(outline[j]);
        }
        let mesh = g.into_mesh();
        let bevy::mesh::VertexAttributeValues::Float32x4(tangents) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap()
        else {
            unreachable!()
        };
        let bevy::mesh::VertexAttributeValues::Float32x3(normals) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap()
        else {
            unreachable!()
        };
        for (t, n) in tangents.iter().zip(normals) {
            let tangent = Vec4::from_array(*t);
            assert!((tangent.truncate().length() - 1.).abs() < 1e-4);
            assert!(tangent.truncate().dot(Vec3::from_array(*n)).abs() < 1e-4);
            assert_eq!(tangent.w.abs(), 1.);
        }
    }
}

#[test]
fn emitted_face_matches_the_audit_and_legacy_mesh_replays_its_old_registration() {
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        materials::worktop_finish,
        objects::build_manifest_object,
    };
    let mut scene =
        IndoorManifest::generate_with_humans(93405, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
    let mut seen = BTreeSet::new();
    for seed in 100..275 {
        let o = audit::fixture(ObjectKind::Desk, seed);
        let p = parameters(&o);
        let Some(cut) = sheet_cut(&o, &p) else {
            continue;
        };
        let live = build_object(&o);
        scene.generator_version = REGISTERED_SHEET_VERSION - 1;
        let old = build_manifest_object(&scene, &o);
        let label = worktop_finish::table_label(&o, p.top_surface);
        let key = (p.top_surface, label.into_owned());
        let (g, reference) = (&live.parts[&key], &old.parts[&key]);
        assert_eq!(
            live.parts.keys().collect::<Vec<_>>(),
            old.parts.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            (&g.positions, &g.normals, &g.indices),
            (&reference.positions, &reference.normals, &reference.indices)
        );
        let rotation = Mat2::from_angle(cut.heading_rad);
        let mut face_count = 0;
        for (i, n) in g.normals.iter().enumerate() {
            let pos = Vec3::from_array(g.positions[i]);
            if n[1] > 0.99999 && (pos.y - o.size.y).abs() < 1e-5 {
                let expected = rotation * pos.xz() + cut.face_origin_m;
                assert!(Vec2::from_array(g.uvs[i]).distance(expected) < 2e-5);
                if p.top_surface == Surface::Ceramic {
                    assert!(Vec2::from_array(reference.uvs[i]).distance(pos.xz()) < 2e-5);
                }
                face_count += 1;
            }
        }
        assert!(face_count > 3);
        seen.insert(p.top_surface);
    }
    assert_eq!(seen.len(), 3);
}
