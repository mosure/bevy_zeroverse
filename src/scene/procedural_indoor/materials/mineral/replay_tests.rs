//! Independent pre-preparation mineral texel oracle.
use super::super::program::Texel;
use super::*;

fn reference(m: &MineralRecipe, r: &MaterialRecipe, uv: [f32; 2], floor_style: u32) -> Texel {
    reference_observed(m, r, uv, floor_style).0
}

// The unchanged scalar evaluator also exposes its independently computed
// exposure/gain for coverage diagnostics; it never substitutes an endpoint.
fn reference_observed(
    m: &MineralRecipe,
    r: &MaterialRecipe,
    uv: [f32; 2],
    floor_style: u32,
) -> (Texel, f32, f32) {
    let mut uv = uv;
    let mut seam = 0.;
    let mut footprint = 1. / r.map_size(floor_style) as f32;
    if r.surface == Surface::Floor && floor_style == 2 {
        let [nx, ny] = r.floor_repetitions(2);
        footprint *= nx.max(ny);
        let [u, v] = uv;
        let ix = (u * nx).floor() as u32 % nx as u32;
        let iy = (v * ny).floor() as u32 % ny as u32;
        let x = (u * nx).fract();
        let y = (v * ny).fract();
        let d = (x.min(1. - x) * r.period_m / nx).min(y.min(1. - y) * r.period_m / ny);
        let width = r.joint_width + r.period_m / r.map_size(floor_style) as f32;
        seam = (1. - d / width).clamp(0., 1.) * r.joint_width / width;
        let offset = super::super::hash(ix, iy, r.seed.wrapping_add(157));
        uv = [
            x + offset,
            y + super::super::hash(ix, iy, r.seed.wrapping_add(163)),
        ];
    }

    let [u, v] = uv;
    // Warp the packing as well as individual chip edges. Random sizes and
    // missing chips keep the aggregate from resembling a regular dot grid.
    let packed = [
        u + (periodic_noise(u, v, 11, 13, r.seed.wrapping_add(179)) - 0.5) * 0.9
            / m.aggregate_cells[0] as f32,
        v + (periodic_noise(u, v, 13, 11, r.seed.wrapping_add(181)) - 0.5) * 0.9
            / m.aggregate_cells[1] as f32,
    ];
    let cell = cellular(packed, m.aggregate_cells, r.seed);
    // Crushed minerals have stretched/angular sections and internal grains,
    // rather than identical flat circular dots. The local transform has unit
    // area; shape, size and pigment use distinct cell attributes.
    let axis = 0.25 + cell.shape * 0.70;
    let other = (1. - axis * axis).sqrt();
    let aspect = 0.65 + cell.shape * 0.90;
    let x = (cell.offset[0] * axis + cell.offset[1] * other) * aspect;
    let y = (-cell.offset[0] * other + cell.offset[1] * axis) / aspect;
    let rounded = (x * x + y * y).sqrt();
    let angular = (x.abs() * 0.90 + y.abs() * 0.35).max(y.abs() * 0.90 + x.abs() * 0.35);
    let clast_distance = rounded * (1. - cell.shape * 0.65) + angular * cell.shape * 0.65;
    let edge = periodic_noise(
        u,
        v,
        m.aggregate_cells[0] * 3,
        m.aggregate_cells[1] * 3,
        r.seed.wrapping_add(191),
    );
    let radius = m.aggregate_radius * (0.30 + cell.dye.powi(2) * 1.10) * (0.65 + edge * 0.70);
    let chip_aa =
        (footprint * m.aggregate_cells[0].max(m.aggregate_cells[1]) as f32 * 0.55).max(0.07);
    let chip = smooth((radius + chip_aa - clast_distance) / (2. * chip_aa))
        * smooth((cell.dye - 0.18) * 6.);
    let exposed = chip * m.aggregate_exposure * (1. - m.marble_mix * 0.65);
    let pores = (1. - smooth(cell.radius / 0.13)) * smooth((m.porosity - cell.dye) * 12.);
    let field = if m.marble_mix > 0. {
        deposit(uv, m.vein_cells, m.vein_warp, r.seed.wrapping_add(239))
    } else {
        0.5
    };
    // Integrate thin vein coverage over the atlas footprint. A sub-texel
    // level set must fade continuously instead of becoming isolated dots.
    let aa = footprint * (m.vein_cells[0] + m.vein_cells[1]) as f32 * 0.18;
    let band = |level: f32, width: f32| {
        let filtered = width.max(aa);
        (1. - smooth((field - level).abs() / filtered)) * width / filtered
    };
    let vein = band(0.49, m.vein_width) + 0.35 * band(0.63, m.vein_width * 0.45);
    let vein = (vein * m.vein_strength * m.marble_mix).min(1.);
    let micro = periodic_noise(u, v, 97, 91, r.seed.wrapping_add(271)) - 0.5;
    let binder = periodic_noise(u, v, 9, 11, r.seed.wrapping_add(277)) - 0.5;
    let matrix = tint(r.color, 1. + binder * m.binder_variation + micro * 0.025);
    let chips = tint(
        mix(m.aggregate_color[0], m.aggregate_color[1], cell.dye),
        1. + (edge - 0.5) * 0.28 + micro * 0.20,
    );
    let stone = tint(matrix, 1. + (field - 0.5) * m.marble_mix * 0.20);
    let mut color = mix(mix(stone, chips, exposed), m.vein_color, vein);
    color = tint(color, 1. - pores * 0.65 - seam * 0.35);
    let mut texel = Texel {
        color,
        height: r.relief_m
            * ((chip - 0.5) * m.aggregate_exposure * (1. - m.polish * 0.9)
                + micro * (1. - m.polish) * 0.15
                - pores * 0.9
                + vein * (1. - m.polish) * 0.05)
            - seam * r.joint_width * 0.25,
        roughness: (r.roughness + binder * 0.06 + pores * 0.10 + seam * 0.15
            - exposed * m.polish * 0.08
            - vein * m.polish * 0.04)
            .clamp(0.12, 1.),
        occlusion: (1. - pores * 0.12 - seam * 0.05).clamp(0.8, 1.),
    };
    if let Some(c) = &m.casting {
        super::super::concrete::replay_tests::reference(c, r, uv, m.polish, &mut texel);
    }
    (texel, exposed, 1. + (edge - 0.5) * 0.28 + micro * 0.20)
}

#[test]
fn prepared_mineral_replays_texel_bits_and_every_pbr_mip() {
    use super::super::program::replay_helpers::*;
    for seed in [7, 42_430_575, 43_084_482] {
        for (surface, style) in [
            (Surface::Concrete, 0),
            (Surface::Soil, 0),
            (Surface::Terracotta, 0),
            (Surface::Floor, 2),
        ] {
            let recipes = super::super::program::sample_with_floor(seed, style).unwrap();
            let r = &recipes[surface as usize];
            let m = r.mineral.as_ref().unwrap();
            check_maps(r, style, |u, v| {
                reference(m, r, transformed(r, u, v), style)
            });
        }
    }
}

#[test]
fn carried_colors_replay_original_consumed_mineral_atlases_and_mips() {
    use super::super::super::{
        architecture, humans, layout::IndoorLayout, objects, preparation::SceneGeometry,
    };
    use super::super::program::replay_helpers::*;
    let mut maps = 0;
    for seed in [200, 207, 43_084_482] {
        let scene = super::super::IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Mixed,
            0.65,
            5,
            if seed == 43_084_482 { 0. } else { 0.25 },
        )
        .unwrap();
        let geometry = SceneGeometry {
            architecture: architecture::architecture(&scene),
            objects: scene.objects.iter().map(objects::build_object).collect(),
            humans: scene.humans.iter().map(humans::build_human).collect(),
        };
        let selection = geometry.material_selection(&scene);
        for surface in [
            Surface::Concrete,
            Surface::Floor,
            Surface::Soil,
            Surface::Terracotta,
        ] {
            if !selection.needs_maps(&scene, surface)
                || surface == Surface::Floor && scene.floor_style != 2
            {
                continue;
            }
            let r = &scene.program.as_ref().unwrap().materials[surface as usize];
            let m = r.mineral.as_ref().unwrap();
            check_maps(r, scene.floor_style, |u, v| {
                reference(m, r, transformed(r, u, v), scene.floor_style)
            });
            maps += 1;
        }
    }
    assert!(
        maps >= 3,
        "the consumed-map corpus missed mineral materials"
    );
}

#[test]
fn carried_colors_match_prior_prepared_pipeline_with_malformed_inputs() {
    use super::super::program::{replay_helpers::same_bits, PreparedMineral};
    for seed in [202, 207] {
        let mut r = super::super::program::sample(seed)[Surface::Concrete as usize].clone();
        for color in [
            [-0., 0., 0.5],
            [-0.2, 1.2, f32::MAX],
            [
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(0x7fc0_0123),
            ],
        ] {
            r.color = color;
            r.mineral.as_mut().unwrap().aggregate_color = [color, color];
            r.mineral.as_mut().unwrap().vein_color = color;
            let m = r.mineral.as_ref().unwrap();
            let p = PreparedMineral::new(m, r.color).cache_fields(m, &r);
            for y in 0..16 {
                for x in 0..16 {
                    let uv = [x as f32 / 16., y as f32 / 16.];
                    same_bits(
                        &m.evaluate_color_carry(&r, uv, 0, Some(&p), false),
                        &m.evaluate_color_carry(&r, uv, 0, Some(&p), true),
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "allocation-inclusive counterbalanced CPU diagnostic; run alone"]
fn mineral_color_carry_cpu_benchmark() {
    use super::super::program::{replay_helpers::*, PreparedMineral};
    use std::{hint::black_box, time::Instant};
    for size in [256, 512] {
        let mut prior = Vec::new();
        let mut candidate = Vec::new();
        for seed in 200..208 {
            let r = &super::super::program::sample(seed)[Surface::Concrete as usize];
            let m = r.mineral.as_ref().unwrap();
            let run = |reuse| {
                let started = Instant::now();
                let p = PreparedMineral::new(m, r.color).cache_fields(m, r);
                for y in 0..size {
                    for x in 0..size {
                        let uv = transformed(r, x as f32 / size as f32, y as f32 / size as f32);
                        black_box(m.evaluate_color_carry(r, uv, 0, Some(&p), reuse));
                    }
                }
                drop(p);
                started.elapsed().as_secs_f64()
            };
            let (a, b) = if seed % 2 == 0 {
                let a = run(false);
                (a, run(true))
            } else {
                let b = run(true);
                (run(false), b)
            };
            prior.push(a);
            candidate.push(b);
        }
        report_benchmark(
            "carried decode; same prepared fields, includes field setup/evaluation/drop; excludes maps/mips/GPU",
            size,
            prior,
            candidate,
        );
    }
}

#[test]
fn absent_aggregate_replays_original_float_channels_and_complete_maps() {
    use super::super::program::replay_helpers::*;
    for (seed, surface, style) in [
        (202, Surface::Concrete, 0),
        (207, Surface::Floor, 2),
        (43_084_482, Surface::Soil, 0),
        (43_084_482, Surface::Terracotta, 0),
    ] {
        let mut r = super::super::program::sample_with_floor(seed, style).unwrap()
            [surface as usize]
            .clone();
        r.mineral.as_mut().unwrap().aggregate_exposure = 0.;
        let m = r.mineral.as_ref().unwrap();
        let mut eligible = 0;
        check_maps(&r, style, |u, v| {
            let (texel, exposed, gain) = reference_observed(m, &r, transformed(&r, u, v), style);
            assert!(m.unused_chip_endpoint(exposed, gain));
            eligible += 1;
            texel
        });
        assert_eq!(eligible, r.map_size(style).pow(2));
    }
}

#[test]
fn chip_endpoint_guards_retain_signed_zero_and_malformed_pigment_behavior() {
    use super::super::program::{replay_helpers::*, PreparedMineral};
    let mut r = super::super::program::sample(202)[Surface::Soil as usize].clone();
    r.color = [-0.; 3];
    r.mineral.as_mut().unwrap().aggregate_exposure = 0.;
    for pigment in [
        -0.,
        -0.2,
        1.2,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0123),
        f32::from_bits(0xffc0_0456),
    ] {
        let m = r.mineral.as_mut().unwrap();
        m.aggregate_color = [[pigment; 3]; 2];
        assert!(!m.unused_chip_endpoint(0., 1.));
        let m = r.mineral.as_ref().unwrap();
        // Keep the original scalar field path for malformed authored values;
        // prepared constants still exercise the production endpoint guard.
        let p = PreparedMineral::new(m, r.color);
        for y in 0..8 {
            for x in 0..8 {
                let uv = [x as f32 / 8., y as f32 / 8.];
                same_bits(
                    &reference(m, &r, uv, 0),
                    &m.evaluate_prepared(&r, uv, 0, Some(&p)),
                );
            }
        }
    }
    let m = r.mineral.as_mut().unwrap();
    m.aggregate_color = [[0.; 3]; 2];
    assert!(m.unused_chip_endpoint(0., 1.));
    assert!(m.unused_chip_endpoint(-0., 1.));
    for gain in [-0., -1., f32::NAN, f32::INFINITY] {
        assert!(!m.unused_chip_endpoint(0., gain));
    }
    for exposed in [
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::NAN,
        f32::INFINITY,
    ] {
        assert!(!m.unused_chip_endpoint(exposed, 1.));
    }
    let m = r.mineral.as_ref().unwrap();
    let p = PreparedMineral::new(m, r.color).cache_fields(m, &r);
    for exposure in [0., -0.] {
        let mut authored = r.clone();
        authored.mineral.as_mut().unwrap().aggregate_exposure = exposure;
        let m = authored.mineral.as_ref().unwrap();
        for y in 0..8 {
            for x in 0..8 {
                let uv = [x as f32 / 8., y as f32 / 8.];
                same_bits(
                    &reference(m, &authored, uv, 0),
                    &m.evaluate_prepared(&authored, uv, 0, Some(&p)),
                );
            }
        }
    }
}

#[test]
#[ignore = "full consumed-atlas eligibility diagnostic; run explicitly with --ignored --nocapture"]
fn consumed_mineral_atlases_report_exact_zero_exposure_coverage() {
    use super::super::super::{
        architecture, humans, layout::IndoorLayout, objects, preparation::SceneGeometry,
    };
    use super::super::program::replay_helpers::*;
    let mut texels = 0_u64;
    let mut eligible = 0_u64;
    let mut rooms = Vec::new();
    for seed in [43_084_482, 202, 207] {
        let human_density = if seed == 43_084_482 { 0. } else { 0.25 };
        let scene = super::super::IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Mixed,
            0.65,
            5,
            human_density,
        )
        .unwrap();
        let geometry = SceneGeometry {
            architecture: architecture::architecture(&scene),
            objects: scene.objects.iter().map(objects::build_object).collect(),
            humans: scene.humans.iter().map(humans::build_human).collect(),
        };
        let selection = geometry.material_selection(&scene);
        for surface in [
            Surface::Concrete,
            Surface::Soil,
            Surface::Terracotta,
            Surface::Floor,
        ] {
            if !selection.needs_maps(&scene, surface)
                || surface == Surface::Floor && scene.floor_style != 2
            {
                continue;
            }
            let r = &scene.program.as_ref().unwrap().materials[surface as usize];
            let m = r.mineral.as_ref().unwrap();
            let n = r.map_size(scene.floor_style);
            let mut empty = 0_u64;
            for y in 0..n {
                for x in 0..n {
                    let (_, exposed, gain) = reference_observed(
                        m,
                        r,
                        transformed(r, x as f32 / n as f32, y as f32 / n as f32),
                        scene.floor_style,
                    );
                    empty += u64::from(m.unused_chip_endpoint(exposed, gain));
                }
            }
            let count = u64::from(n) * u64::from(n);
            texels += count;
            eligible += empty;
            rooms.push(serde_json::json!({
                "seed":seed,"surface":surface,"atlas_size":n,
                "texels":count,"unused_chip_texels":empty,
                "unused_chip_fraction":empty as f64/count as f64,
            }));
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "scope":"exact full-atlas independent scalar exposure at actual geometry-selected recipes; no speedup claim",
            "texels":texels,"unused_chip_texels":eligible,
            "unused_chip_fraction":eligible as f64/texels as f64,"atlases":rooms,
        })
    );
}

#[test]
#[ignore = "counterbalanced CPU microbenchmark; run explicitly with --ignored --nocapture"]
fn prepared_mineral_cpu_benchmark() {
    use super::super::program::{replay_helpers::*, PreparedMineral};
    for size in [256, 512] {
        let mut old = Vec::new();
        let mut new = Vec::new();
        for seed in 200..208 {
            let recipes = super::super::program::sample(seed);
            let r = &recipes[Surface::Concrete as usize];
            let m = r.mineral.as_ref().unwrap();
            let p = PreparedMineral::new(m, r.color).cache_fields(m, r);
            let (a, b) = bench_texels(
                size,
                |uv| reference(m, r, uv, 0),
                |uv| m.evaluate_prepared(r, uv, 0, Some(&p)),
                seed % 2 == 1,
            );
            old.push(a);
            new.push(b);
        }
        report_benchmark("mineral bounded spatial fields", size, old, new);
    }
}

#[test]
fn mineral_spatial_fields_share_one_map_budget_and_fallback_exactly() {
    use super::super::program::PreparedMineral;
    for seed in [202, 207, 43_084_482] {
        let recipes = super::super::program::sample(seed);
        let mut r = recipes[Surface::Concrete as usize].clone();
        let m = r.mineral.as_mut().unwrap();
        m.aggregate_cells = [96, 96];
        m.marble_mix = 1.;
        m.vein_cells = [16, 16];
        let c = m.casting.as_mut().unwrap();
        c.bughole_radius_m = 0.000005;
        c.board_width_m = 0.04;
        c.trowel_scale_m = 0.01;
        let m = r.mineral.as_ref().unwrap();
        for budget in [0, 32, prepared::FIELD_BUDGET_BYTES] {
            let mut p = PreparedMineral::new(m, r.color);
            let fields = MineralFields::with_budget(m, &r, |dye| p.aggregate(dye), budget);
            assert!(fields.allocated_bytes <= budget);
            assert_eq!(fields.bytes(), fields.allocated_bytes);
            p.set_test_fields(fields);
            for y in 0..64 {
                for x in 0..64 {
                    let uv = [x as f32 / 64., y as f32 / 64.];
                    let expected = reference(m, &r, uv, 0);
                    let actual = m.evaluate_prepared(&r, uv, 0, Some(&p));
                    let bits = |t: Texel| {
                        [
                            t.color[0],
                            t.color[1],
                            t.color[2],
                            t.height,
                            t.roughness,
                            t.occlusion,
                        ]
                        .map(f32::to_bits)
                    };
                    assert_eq!(
                        bits(expected),
                        bits(actual),
                        "seed {seed} budget {budget} uv {uv:?}"
                    );
                }
            }
        }
    }
}
