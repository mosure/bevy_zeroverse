use super::super::{program, rugs, Surface};
use super::*;
use bevy::prelude::Vec2;

#[test]
fn installation_tints_stay_in_gamut_across_warm_and_cool_extremes() {
    for seed in 0..8192 {
        for style in 0..3 {
            FloorFinish::sample(seed, style, 4.).validate().unwrap();
        }
    }
    use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
    let scene =
        IndoorManifest::generate_with_humans(929, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
    for r in &scene.program.as_ref().unwrap().materials {
        r.validate().unwrap();
    }
}

#[test]
fn installations_replay_with_metric_cuts_and_palette_variation() {
    let mut layouts = [0; 3];
    let mut carpet_patterns = [0; 3];
    for seed in 0..256 {
        for style in 0..3 {
            let recipes = program::sample_with_floor(seed, style).unwrap();
            let r = &recipes[Surface::Floor as usize];
            r.validate().unwrap();
            let f = r.flooring.as_ref().unwrap();
            assert_eq!(*f, FloorFinish::sample(r.seed, style, r.period_m));
            let cut = f.cut(r, [0.731, 0.127]);
            assert!(cut.uv.into_iter().all(|v| (0.0..1.).contains(&v)));
            assert!(cut.distance_m.is_finite() && cut.distance_m >= 0.);
            assert!(r.map_size(style) <= 512);
            if style == 0 {
                layouts[f.layout as usize] += 1;
            }
            if let Some(c) = &r.carpet {
                carpet_patterns[c.pattern as usize] += 1;
            }
            for uv in [[0.017, 0.851], [0.913, 0.219]] {
                let mut t = program::Texel {
                    color: [0.5; 3],
                    height: 0.,
                    roughness: 0.5,
                    occlusion: 1.,
                };
                f.apply(r, uv, &mut t);
                assert!(t
                    .color
                    .iter()
                    .all(|v| v.is_finite() && (0.0..1.).contains(v)));
            }
        }
    }
    assert!(layouts.iter().all(|n| *n >= 15), "{layouts:?}");
    assert!(carpet_patterns.iter().all(|n| *n >= 45));
}

#[test]
fn carpet_and_rug_fields_close_at_wrap_and_have_real_relief() {
    let mut patterns = [false; 5];
    for seed in 0..32 {
        let rs = program::sample_with_floor(seed, 1).unwrap();
        let rug = rugs::recipe(&rs[Surface::FabricAlt as usize], rugs::slot(seed));
        patterns[rug.carpet.as_ref().unwrap().pattern as usize] = true;
        for r in [&rs[Surface::Floor as usize], &rug] {
            r.validate().unwrap();
            assert_eq!(r.texture_base_color(1), [1.; 3]);
            let mut height = [f32::INFINITY, f32::NEG_INFINITY];
            for i in 0..16 {
                let v = i as f32 / 16.;
                for (a, b) in [([0., v], [1., v]), ([v, 0.], [v, 1.])] {
                    let a = r.texel(a[0], a[1], 1);
                    let b = r.texel(b[0], b[1], 1);
                    for (a, b) in a
                        .color
                        .into_iter()
                        .chain([a.height, a.roughness])
                        .zip(b.color.into_iter().chain([b.height, b.roughness]))
                    {
                        assert!((a - b).abs() < 2e-5);
                    }
                    height[0] = height[0].min(a.height);
                    height[1] = height[1].max(a.height);
                }
            }
            assert!(height[1] - height[0] > 0.00001);
        }
    }
    assert!(patterns.into_iter().all(|p| p));
}

#[test]
fn rugs_resolve_a_distinct_program_without_changing_semantic_or_geometry_bounds() {
    use crate::scene::procedural_indoor::{
        layout::ObjectKind,
        objects::{audit, build_object},
    };
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::Rug, seed);
        let a = build_object(&o);
        let (lo, hi) = a.bounds();
        assert!((hi.y - o.size.y).abs() < 1e-6 && lo.y >= 0.);
        let label = format!(
            "floormat#rug{}",
            rugs::slot(o.finish_seed.unwrap_or(o.seed))
        );
        let field = &a.parts[&(Surface::FabricAlt, label)];
        assert!(field.indices.len() >= 24);
        let mut shared = std::collections::BTreeMap::new();
        let mut boundary_vertices = 0;
        let mut interior_vertices = 0;
        for (position, uv) in field.positions.iter().zip(&field.uvs) {
            assert!(uv.iter().all(|v| (0.0..=1.).contains(v)));
            assert!((position[1] - o.size.y).abs() < 1e-6);
            if let Some(previous) = shared.insert(position.map(f32::to_bits), uv) {
                assert_eq!(previous, uv, "UV seam at a shared rug vertex");
            }
            if position[0] == 0. && position[2] == 0. {
                assert_eq!(uv, &[0.5; 2]);
            } else if uv.iter().any(|v| *v < 1e-6 || *v > 1. - 1e-6) {
                boundary_vertices += 1;
            } else {
                interior_vertices += 1;
            }
        }
        assert!(boundary_vertices >= 16 && interior_vertices >= 32);
        for triangle in field.uvs.as_chunks::<3>().0 {
            let [a, b, c] = triangle.map(Vec2::from_array);
            assert!(
                (b - a).perp_dot(c - a).abs() > 1e-7,
                "collapsed rug UV triangle"
            );
        }
        for (_, label) in a.parts.keys() {
            assert!(label.starts_with("floormat#"));
            assert!(crate::render::semantic::SemanticLabel::from_label(
                label.split('#').next().unwrap()
            )
            .is_some());
        }
    }
}
