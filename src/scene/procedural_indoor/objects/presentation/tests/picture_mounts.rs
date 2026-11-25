use super::*;
use crate::scene::procedural_indoor::{materials::art::frame_parameters, objects::part_label};
use std::collections::BTreeMap;

#[test]
fn assembled_mounts_have_one_closed_surface_across_materials_and_mitres() {
    for seed in 0..96 {
        for size in [
            Vec3::new(0.055, 0.12, 0.008),
            Vec3::new(0.38, 1.45, 0.065),
            Vec3::new(1.80, 0.48, 0.022),
        ] {
            let mut o = audit::fixture(ObjectKind::WallArt, seed);
            o.size = size;
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            let mut edges = BTreeMap::new();
            let mut vertices = BTreeSet::new();
            let mut triangles = 0isize;
            for ((_, label), g) in &a.parts {
                assert_eq!(part_label(label), o.kind.class_name());
                for tri in g.indices.as_chunks::<3>().0 {
                    triangles += 1;
                    let p = tri.map(|v| g.positions[v as usize]);
                    let key = p.map(|p| p.map(|v| if v == 0. { 0 } else { v.to_bits() }));
                    let points = p.map(Vec3::from_array);
                    let n = (points[1] - points[0]).cross(points[2] - points[0]);
                    assert!(tri
                        .iter()
                        .all(|&i| n.dot(Vec3::from_array(g.normals[i as usize])) > 0.));
                    vertices.extend(key);
                    for i in 0..3 {
                        let (a, b) = (key[i], key[(i + 1) % 3]);
                        let (edge, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                        let (count, balance) = edges.entry(edge).or_insert((0, 0));
                        *count += 1;
                        *balance += sign;
                    }
                }
            }
            assert!(
                edges.values().all(|&incidence| incidence == (2, 0)),
                "open or duplicate face on {seed}/{size}"
            );
            assert_eq!(
                vertices.len() as isize - edges.len() as isize + triangles,
                2 * artwork::canvas_set(&o).map_or(1, |set| set.panels.len()) as isize
            );
        }
    }
}

#[test]
fn print_mat_bevel_and_float_channel_have_measurable_depth() {
    let mut mats = 0;
    let mut floats = 0;
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::WallArt, seed);
        o.variant = (seed % 6) as u32;
        let p = frame_parameters(&o);
        let a = build_object(&o);
        let probe = |point: Vec2| {
            let (surface, t) = ray_hit(&a, point.extend(1.), Vec3::NEG_Z).unwrap();
            (surface, 1. - t)
        };
        let half = o.size.xy() * 0.5;
        let mat = p.mat_width * p.mat_scale;
        let center = Vec2::Y * half.y + mat * p.picture_bias;
        let image_half = half - Vec2::splat(p.border_width + p.shadow_gap) - mat;
        let image_z = o.size.z * 0.5 - p.picture_inset;
        let (surface, z) = probe(center);
        assert_eq!(surface, Surface::Art);
        assert!((z - image_z).abs() < 1e-5);
        let edge = center.x + image_half.x;
        let right = half.x - p.border_width;
        if p.mat_thickness_m > 0. {
            mats += 1;
            let (surface, z) = probe(Vec2::new((edge + p.mat_bevel_m + right) * 0.5, center.y));
            assert_eq!(surface, Surface::Paper);
            assert!((z - image_z - p.mat_thickness_m).abs() < 1e-5);
            let (surface, z) = probe(Vec2::new(edge + p.mat_bevel_m * 0.5, center.y));
            assert_eq!(surface, Surface::Paper);
            assert!((z - image_z - p.mat_thickness_m * 0.5).abs() < 1e-5);
        } else if p.float_depth_m > 0. {
            floats += 1;
            let (surface, z) = probe(Vec2::new((edge + right) * 0.5, center.y));
            assert_eq!(surface, Surface::Plastic);
            assert!((z - image_z + p.float_depth_m).abs() < 1e-5);
        }
    }
    assert!(mats > 40 && floats >= 30);
}

#[test]
fn mitred_timber_charts_follow_each_rail_and_wraps_contain_the_image_border() {
    let mut wrapped = 0;
    let mut raw = 0;
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::WallArt, seed);
        o.variant = (seed % 6) as u32;
        let p = frame_parameters(&o);
        let a = build_object(&o);
        if let Some((_, g)) = a.parts.iter().find(|((s, _), _)| *s == Surface::Wood) {
            for (q, uv) in g
                .positions
                .as_chunks::<4>()
                .0
                .iter()
                .zip(g.uvs.as_chunks::<4>().0)
            {
                let p = q.map(Vec3::from_array);
                assert!((p[0].distance(p[1]) - (uv[0][1] - uv[1][1]).abs()).abs() < 1e-5);
                assert_eq!(uv[0][0], uv[1][0]);
                assert!(uv[3][0] > uv[0][0]);
            }
        }
        if matches!(p.style, 0 | 4) {
            let (_, art) = a
                .parts
                .iter()
                .find(|((s, _), _)| *s == Surface::Art)
                .unwrap();
            if p.image_wrap {
                wrapped += 1;
                let uv = art.uvs.as_chunks::<4>().0;
                let front = uv.last().unwrap();
                assert_eq!(uv.len(), 5);
                for (i, edge) in uv[..4].iter().enumerate() {
                    assert_eq!(edge[3], front[i]);
                    assert_eq!(edge[2], front[(i + 1) % 4]);
                    assert!(Vec2::from_array(edge[0]).distance(Vec2::from_array(edge[3])) > 0.);
                    assert!(edge.iter().flatten().all(|x| (0.0..=1.0).contains(x)));
                }
            } else {
                raw += 1;
                assert_eq!(art.indices.len(), 6);
            }
        }
    }
    assert!(wrapped > 5 && raw > 5);
}
