use super::*;
use crate::scene::procedural_indoor::materials::program;

#[test]
fn patterned_coverings_are_not_bound_to_ceiling_services_or_fittings() {
    use crate::scene::procedural_indoor::{
        architecture,
        layout::{IndoorLayout, IndoorManifest},
        objects::part_label,
    };
    let mut wall_parts = 0;
    for seed in 0..32 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let a = architecture::architecture(&scene);
        let edges = feature_edges(&scene);
        assert!(edges.len() <= 2);
        assert_eq!(edges, feature_edges(&scene));
        if seed < 8 {
            // Changing the coating only partitions existing triangles by
            // material. Their labels, vertices, normals and UVs stay identical.
            let mut plain = scene.clone();
            plain.program.as_mut().unwrap().materials[Surface::Accent as usize]
                .coating
                .as_mut()
                .unwrap()
                .wall = None;
            let signature = |a: &crate::scene::procedural_indoor::objects::Assembly| {
                let mut triangles = Vec::new();
                for ((_, label), g) in &a.parts {
                    for t in g.indices.as_chunks::<3>().0 {
                        let attrs = t.map(|i| {
                            let i = i as usize;
                            [
                                g.positions[i][0],
                                g.positions[i][1],
                                g.positions[i][2],
                                g.normals[i][0],
                                g.normals[i][1],
                                g.normals[i][2],
                                g.uvs[i][0],
                                g.uvs[i][1],
                            ]
                            .map(f32::to_bits)
                        });
                        triangles.push((label.clone(), attrs));
                    }
                }
                triangles.sort_unstable();
                triangles
            };
            assert_eq!(
                signature(&a),
                signature(&architecture::architecture(&plain)),
                "coating moved geometry, seed {seed}"
            );
        }
        for (surface, label) in a.parts.keys() {
            if *surface == Surface::Accent {
                assert_eq!(
                    part_label(label),
                    "wall",
                    "covering on {label}, seed {seed}"
                );
                wall_parts += 1;
                if let Some(envelope) = &scene.envelope {
                    let g = &a.parts[&(*surface, label.clone())];
                    for t in g.indices.as_chunks::<3>().0 {
                        use bevy::math::{Vec3, Vec3Swizzles};
                        let center = t
                            .iter()
                            .map(|&i| Vec3::from(g.positions[i as usize]))
                            .sum::<Vec3>()
                            / 3.;
                        let normal = Vec3::from(g.normals[t[0] as usize]);
                        assert!(normal.y.abs() < 1e-5, "covering on a reveal/cap");
                        assert!(
                            crate::scene::procedural_indoor::envelope::polygon::contains(
                                &envelope.footprint,
                                (center + normal * 0.002).xz(),
                                0.
                            ),
                            "covering faces outside room"
                        );
                    }
                }
            }
        }
    }
    assert!(wall_parts > 8);
}

#[test]
fn wall_coverings_are_seeded_tileable_and_use_all_component_fields() {
    let mut dominant = [0; 4];
    let mut quiet = 0;
    for seed in 0..256 {
        let recipes = program::sample(seed);
        for surface in [Surface::Paint, Surface::Accent, Surface::Ceiling] {
            let r = &recipes[surface as usize];
            r.validate().unwrap();
            let c = r.coating.as_ref().unwrap();
            let Some(w) = &c.wall else {
                quiet += usize::from(surface == Surface::Paint);
                continue;
            };
            assert_ne!(surface, Surface::Ceiling);
            if surface == Surface::Paint {
                assert_eq!(&w.weights[2..], &[0., 0.]);
            }
            assert_eq!(Some(w.clone()), WallFinish::sample(r.seed, surface));
            let biggest = (0..4)
                .max_by(|&a, &b| w.weights[a].total_cmp(&w.weights[b]))
                .unwrap();
            dominant[biggest] += 1;
            for t in [0., 0.13, 0.57, 0.91] {
                for (a, b) in [([0., t], [1., t]), ([t, 0.], [t, 1.])] {
                    let a = w.evaluate(r, a);
                    let b = w.evaluate(r, b);
                    for (a, b) in a
                        .gain
                        .into_iter()
                        .chain([a.height, a.roughness])
                        .zip(b.gain.into_iter().chain([b.height, b.roughness]))
                    {
                        assert!(a.is_finite() && (a - b).abs() < 2e-6);
                    }
                }
            }
        }
    }
    assert!(dominant.into_iter().all(|n| n > 30), "{dominant:?}");
    assert!(quiet > 80 && quiet < 180);
}

#[test]
fn covering_maps_replay_prepared_scalar_channels_and_detail_interventions() {
    for seed in 0..6 {
        let mut r = program::sample(seed)[Surface::Accent as usize].clone();
        let Some(w) = r.coating.as_mut().unwrap().wall.as_mut() else {
            continue;
        };
        w.weights = [0.; 4];
        w.weights[seed as usize % 4] = 1.;
        let prepared = r.prepare_texels(0).unwrap();
        for y in 0..64 {
            for x in 0..64 {
                let u = x as f32 / 64.;
                let v = y as f32 / 64.;
                program::replay_helpers::same_bits(
                    &r.texel(u, v, 0),
                    &r.texel_prepared(u, v, 0, Some(&prepared)),
                );
            }
        }
        r.detail_scale = Some(0.);
        r.relief_m = 0.;
        assert_eq!(r.texel(0.34, 0.72, 0).height, 0.);
        assert_eq!(
            r.coating
                .as_ref()
                .unwrap()
                .wall
                .as_ref()
                .unwrap()
                .evaluate(&r, [0.34, 0.72])
                .gain,
            [1.; 3]
        );
        let mut json = serde_json::to_value(&r).unwrap();
        json["coating"].as_object_mut().unwrap().remove("wall");
        let legacy: program::MaterialRecipe = serde_json::from_value(json).unwrap();
        assert!(legacy.coating.unwrap().wall.is_none());
    }
}
