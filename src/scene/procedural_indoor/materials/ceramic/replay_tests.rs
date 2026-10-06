//! Copied original fired ceramic program, independent of prepared evaluation.
use super::*;

fn reference(g: &GlazeRecipe, c: &CoatingRecipe, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
    let [u, v] = uv;
    let count = frequency(r.period_m, g.cloud_scale_m);
    let melt = deposit(
        uv,
        [count, count],
        0.12 + g.flow * 0.18,
        r.seed.wrapping_add(401),
    );
    let pool = smooth((melt - 0.25) * 1.5) * g.thickness_variation;
    let grains = frequency(r.period_m, g.body_grain_m);
    let body = periodic_noise(u, v, grains, grains, r.seed.wrapping_add(409)) - 0.5;
    let turns = frequency(r.period_m, g.turning_pitch_m);
    let turn_warp = (periodic_noise(u, v, 7, 3, r.seed.wrapping_add(419)) - 0.5) * 0.30;
    let rings = band(
        ((v * turns as f32 + turn_warp).rem_euclid(1.) - 0.5).abs(),
        0.16,
        turns as f32 / r.map_size(0) as f32,
    ) - 0.32;
    let speck_count = frequency(r.period_m, g.speckle_radius_m * 12.);
    let speck = cellular(uv, [speck_count, speck_count], r.seed.wrapping_add(431));
    let radius = g.speckle_radius_m / r.period_m * speck_count as f32 * (0.6 + speck.dye);
    let inclusion = disk(
        speck.radius,
        radius,
        speck_count as f32 / r.map_size(0) as f32,
    ) * smooth((g.speckle_density - speck.dye) * 18.);
    let cracks = if c.crackle > 0. {
        let n = frequency(r.period_m, g.crack_spacing_m);
        let cell = cellular(uv, [n, n], r.seed.wrapping_add(439));
        band(
            cell.edge * 0.5,
            0.000028 / r.period_m * n as f32,
            n as f32 / r.map_size(0) as f32,
        ) * c.crackle
    } else {
        0.
    };
    // Relative reflectance: the production base multiplier supplies each
    // item's pigment exactly once. Thickness/oxide inclusions share their
    // color, roughness and relief footprints.
    let matrix = tint([0.975; 3], 1. + body * c.pigment_variation - pool * 0.10);
    let reactive = smooth((melt - 0.30) * 1.7) * g.reactive_mix;
    let color = tint(
        mix(
            mix(matrix, g.reactive_color, reactive),
            g.speckle_color,
            inclusion,
        ),
        1. - cracks * 0.30,
    );
    Texel {
        color,
        height: r.relief_m
            * (body * (1. - c.gloss * 0.85) * 0.35
                + rings * g.throwing * 0.65
                + (melt - 0.5) * g.thickness_variation * 0.25
                - inclusion * 0.08
                - cracks * 0.10),
        roughness: (r.roughness + body * (1. - c.gloss) * 0.06 + reactive * 0.13 - pool * 0.07
            + inclusion * 0.12
            + cracks * 0.08)
            .clamp(0.075, 0.97),
        occlusion: (1. - cracks * 0.025).max(0.96),
    }
}

#[test]
fn prepared_glaze_replays_original_texel_bits_and_every_pbr_mip() {
    use super::super::program::replay_helpers::*;
    for seed in [0, 7, 202, 43_084_482, u64::MAX] {
        let mut r =
            super::super::program::sample(seed)[super::super::Surface::Ceramic as usize].clone();
        for crackle in [0., 0.35] {
            r.coating.as_mut().unwrap().crackle = crackle;
            let c = r.coating.as_ref().unwrap();
            let g = c.glaze.as_ref().unwrap();
            check_maps(&r, 0, |u, v| reference(g, c, &r, transformed(&r, u, v)));
        }
    }
}

#[test]
fn prepared_glaze_replays_every_consumed_base_and_structural_map_and_mip() {
    use super::super::super::{
        architecture, humans, layout::IndoorLayout, objects, preparation::SceneGeometry,
    };
    use super::super::program::replay_helpers::*;
    use std::collections::BTreeSet;
    let mut maps = 0;
    for seed in [200, 202, 207, 43_084_482] {
        let scene = super::super::IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Mixed,
            0.65,
            3,
            if seed == 43_084_482 { 0. } else { 0.25 },
        )
        .unwrap();
        let geometry = SceneGeometry {
            architecture: architecture::architecture(&scene),
            objects: scene.objects.iter().map(objects::build_object).collect(),
            humans: scene.humans.iter().map(humans::build_human).collect(),
        };
        let selection = geometry.material_selection(&scene);
        let surface = super::super::Surface::Ceramic;
        let r = &scene.program.as_ref().unwrap().materials[surface as usize];
        let mut structures = BTreeSet::new();
        if selection.needs_maps(&scene, surface) {
            structures.insert(0);
        }
        for &(selected, slot) in &selection.finishes {
            if selected == surface {
                structures.insert(slot % super::super::variants::structure_count(surface));
            }
        }
        for structure in structures {
            let r = r.variant(structure);
            let c = r.coating.as_ref().unwrap();
            let g = c.glaze.as_ref().unwrap();
            check_maps(&r, scene.floor_style, |u, v| {
                reference(g, c, &r, transformed(&r, u, v))
            });
            maps += 1;
        }
    }
    assert_eq!(maps, 9, "qualified consumed ceramic map corpus changed");
}

#[test]
fn authored_glaze_uses_one_selected_cache_and_replays_512_maps() {
    use super::super::program::{replay_helpers::*, PreparedMaterial};
    let recipes = super::super::program::sample(202);
    let mut r = recipes[super::super::Surface::Ceramic as usize].clone();
    r.surface = super::super::Surface::Floor;
    r.mineral = recipes[super::super::Surface::Concrete as usize]
        .mineral
        .clone();
    r.coating.as_mut().unwrap().application =
        Some(super::super::paint::PaintApplication::sample(207));
    r.coating.as_mut().unwrap().crackle = 0.35;
    for style in [0, 2] {
        assert_eq!(r.map_size(style), 512);
        let Some(PreparedMaterial::Glaze(p)) = r.prepare_texels(style) else {
            panic!("authored glaze did not retain precedence");
        };
        assert!(p.bytes() <= super::prepared::FIELD_BUDGET_BYTES);
        let c = r.coating.as_ref().unwrap();
        let g = c.glaze.as_ref().unwrap();
        check_maps(&r, style, |u, v| reference(g, c, &r, transformed(&r, u, v)));
    }
    // The existing leaf-before-coating precedence must allocate no glaze fields.
    r.leaf = recipes[super::super::Surface::Leaf as usize].leaf.clone();
    assert!(r.prepare_texels(0).is_none());
}

#[test]
fn sparse_glaze_budgets_and_maximum_fields_replay_original_float_bits() {
    use super::super::program::replay_helpers::same_bits;
    for seed in [202, 43_084_482] {
        let mut r =
            super::super::program::sample(seed)[super::super::Surface::Ceramic as usize].clone();
        for period in [r.period_m, 10.] {
            r.period_m = period;
            r.coating.as_mut().unwrap().crackle = 0.35;
            let c = r.coating.as_ref().unwrap();
            let g = c.glaze.as_ref().unwrap();
            for budget in [0, 32, 16 * 1024, super::prepared::FIELD_BUDGET_BYTES] {
                let p = PreparedGlaze::with_budget(g, c, &r, budget).unwrap();
                let samples =
                    (0..64).flat_map(|y| (0..64).map(move |x| [x as f32 / 64., y as f32 / 64.]));
                let tiny = f32::from_bits(1);
                for uv in samples.chain([
                    [-1. / 256., 1.],
                    [1., -1. / 256.],
                    [2.3, -0.4],
                    [-tiny, tiny],
                    [-0., 0.],
                    [1. - f32::EPSILON, 1.],
                    [f32::MAX, -f32::MAX],
                    [f32::INFINITY, f32::NEG_INFINITY],
                    [f32::from_bits(0x7fc1_2345), f32::from_bits(0x7fc5_4321)],
                ]) {
                    same_bits(&reference(g, c, &r, uv), &p.evaluate(g, c, &r, uv));
                }
            }
        }
        let c = r.coating.as_ref().unwrap();
        let g = c.glaze.as_ref().unwrap();
        super::super::program::replay_helpers::check_maps(&r, 0, |u, v| {
            reference(
                g,
                c,
                &r,
                super::super::program::replay_helpers::transformed(&r, u, v),
            )
        });
    }
}

#[test]
fn exceptional_authored_glazes_retain_original_scalar_float_and_nan_payloads() {
    use super::super::program::replay_helpers::{same_bits, transformed};
    let r = super::super::program::sample(202)[super::super::Surface::Ceramic as usize].clone();
    let cases = [
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.,
    ];
    for field in 0..19 {
        for value in cases {
            let mut r = r.clone();
            let c = r.coating.as_mut().unwrap();
            let g = c.glaze.as_mut().unwrap();
            match field {
                0 => g.body_grain_m = value,
                1 => g.cloud_scale_m = value,
                2 => g.reactive_mix = value,
                3 => g.reactive_color = [value; 3],
                4 => g.thickness_variation = value,
                5 => g.flow = value,
                6 => g.throwing = value,
                7 => g.turning_pitch_m = value,
                8 => g.speckle_density = value,
                9 => g.speckle_radius_m = value,
                10 => g.speckle_color = [value; 3],
                11 => g.crack_spacing_m = value,
                12 => c.pigment_variation = value,
                13 => c.gloss = value,
                14 => c.crackle = value,
                15 => r.period_m = value,
                16 => r.relief_m = value,
                17 => r.roughness = value,
                18 => {
                    g.reactive_color = [value, f32::from_bits(0x7fc5_4321), value];
                    c.pigment_variation = f32::from_bits(0x7fc9_8765);
                }
                _ => unreachable!(),
            }
            let prepared = r.prepare_texels(0);
            let c = r.coating.as_ref().unwrap();
            let g = c.glaze.as_ref().unwrap();
            assert!(
                prepared.is_none(),
                "exceptional field {field} value {value:?}"
            );
            for uv in [[0., 0.], [0.13, 0.91], [-0.25, 1.125]] {
                // Malformed scales can produce a zero scalar frequency and
                // panic in wrapped hash evaluation. Replay that original
                // outcome as strictly as successful float/NaN channel bits.
                let original =
                    std::panic::catch_unwind(|| reference(g, c, &r, transformed(&r, uv[0], uv[1])));
                let fallback = std::panic::catch_unwind(|| {
                    r.texel_prepared(uv[0], uv[1], 0, prepared.as_ref())
                });
                match (original, fallback) {
                    (Ok(original), Ok(fallback)) => same_bits(&original, &fallback),
                    (Err(original), Err(fallback)) => {
                        assert_eq!(
                            original.as_ref().type_id(),
                            fallback.as_ref().type_id(),
                            "panic type changed: field {field} value {value:?} uv {uv:?}"
                        );
                        let message = |payload: &(dyn std::any::Any + Send)| {
                            payload
                                .downcast_ref::<&str>()
                                .map(|message| (*message).to_owned())
                                .or_else(|| payload.downcast_ref::<String>().cloned())
                        };
                        let original_message = message(original.as_ref());
                        assert!(
                            original_message.is_some(),
                            "original non-text panic cannot be replayed: field {field} value {value:?} uv {uv:?}"
                        );
                        assert_eq!(
                            original_message,
                            message(fallback.as_ref()),
                            "panic message changed: field {field} value {value:?} uv {uv:?}"
                        );
                    }
                    (original, fallback) => panic!(
                        "panic outcome changed: field {field} value {value:?} uv {uv:?}, original_panicked={}, fallback_panicked={}",
                        original.is_err(),
                        fallback.is_err()
                    ),
                }
            }
        }
    }
}

#[test]
#[ignore = "allocation-inclusive counterbalanced CPU diagnostic; run alone"]
fn prepared_glaze_cpu_benchmark() {
    use super::super::program::replay_helpers::*;
    use std::{hint::black_box, time::Instant};
    for size in [256, 512] {
        let mut original = Vec::new();
        let mut prepared = Vec::new();
        for seed in 200..208 {
            let r = &super::super::program::sample(seed)[super::super::Surface::Ceramic as usize];
            let c = r.coating.as_ref().unwrap();
            let g = c.glaze.as_ref().unwrap();
            let run_original = || {
                let started = Instant::now();
                for y in 0..size {
                    for x in 0..size {
                        let uv = transformed(r, x as f32 / size as f32, y as f32 / size as f32);
                        black_box(reference(g, c, black_box(r), uv));
                    }
                }
                started.elapsed().as_secs_f64()
            };
            let run_prepared = || {
                let started = Instant::now();
                let p = PreparedGlaze::new(g, c, black_box(r)).unwrap();
                for y in 0..size {
                    for x in 0..size {
                        let uv = transformed(r, x as f32 / size as f32, y as f32 / size as f32);
                        black_box(p.evaluate(g, c, black_box(r), uv));
                    }
                }
                drop(p);
                started.elapsed().as_secs_f64()
            };
            let (a, b) = if seed % 2 == 0 {
                let a = run_original();
                (a, run_prepared())
            } else {
                let b = run_prepared();
                (run_original(), b)
            };
            original.push(a);
            prepared.push(b);
        }
        report_benchmark(
            "copied original glaze vs prepared; includes field setup/evaluation/drop; excludes maps/mips/workers/GPU",
            size,
            original,
            prepared,
        );
    }
}
