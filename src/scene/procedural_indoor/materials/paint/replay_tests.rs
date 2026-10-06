//! Independent pre-preparation paint oracle, retained for exact floating-point replay.
use super::*;

fn reference(a: &PaintApplication, c: &CoatingRecipe, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
    let [u, v] = uv;
    let spray_n = frequency(r.period_m, a.spray_spacing_m);
    let spray = cellular(uv, [spray_n, spray_n], r.seed.wrapping_add(503));
    let aa = spray_n as f32 / r.map_size(0) as f32;
    let splat = disk(spray.radius, 0.20 + spray.dye * 0.28, aa);
    // Rounded droplets become broad, flattened islands when knocked down.
    let plateau = smooth(splat * 1.4).min(1. - c.knockdown * 0.55);
    let roller_n = frequency(r.period_m, a.roller_spacing_m);
    let roller = periodic_noise(
        u,
        v,
        roller_n,
        (roller_n as f32 / a.roller_stretch).round().max(1.) as u32,
        r.seed.wrapping_add(509),
    ) - 0.5;
    let fine_n = frequency(r.period_m, a.orange_peel_m);
    let film = periodic_noise(u, v, fine_n, fine_n, r.seed.wrapping_add(521)) - 0.5;
    let brush_n = frequency(r.period_m, a.brush_spacing_m);
    let brush = periodic_noise(u, v, brush_n, 5, r.seed.wrapping_add(523)) - 0.5;
    // Unresolved brush strokes contribute scattering, not aliasing relief.
    let resolved_brush = (r.map_size(0) as f32 / brush_n as f32 * 0.30).min(1.);
    let trowel_n = frequency(r.period_m, a.trowel_scale_m);
    let passes = deposit(uv, [trowel_n, trowel_n], 0.12, r.seed.wrapping_add(541));
    let lap = smooth((passes - 0.35) * 1.5);
    let repair = smooth((passes - 0.65) * 6.) * a.repair_mix;
    let holes = disk(spray.radius, 0.08, aa) * smooth((c.pinholes - spray.dye) * 10.);
    let texture = (plateau - 0.28) * c.texture_mix * (1. - repair);
    Texel {
        color: tint(
            r.color,
            1. + (passes - 0.5) * c.pigment_variation - holes * 0.035,
        ),
        height: r.relief_m * (texture * 0.80 + (lap - 0.5) * c.trowel * 0.22 - holes * 0.40)
            + 0.000025 * (roller * c.roller + film * a.orange_peel)
            + 0.000016 * brush * a.brush * resolved_brush,
        roughness: (r.roughness
            + film * 0.025
            + roller * c.roller * 0.035
            + (lap - 0.5) * a.lap_variation
            + holes * 0.04
            - repair * c.gloss * 0.10)
            .clamp(0.16, 0.99),
        occlusion: (1. - holes * 0.025).max(0.97),
    }
}

fn same_bits(expected: Texel, actual: Texel, seed: u64, uv: [f32; 2]) {
    assert_eq!(
        [
            expected.color[0].to_bits(),
            expected.color[1].to_bits(),
            expected.color[2].to_bits(),
            expected.height.to_bits(),
            expected.roughness.to_bits(),
            expected.occlusion.to_bits(),
        ],
        [
            actual.color[0].to_bits(),
            actual.color[1].to_bits(),
            actual.color[2].to_bits(),
            actual.height.to_bits(),
            actual.roughness.to_bits(),
            actual.occlusion.to_bits(),
        ],
        "paint texel changed: seed {seed} uv {uv:?}"
    );
}

fn transformed(r: &MaterialRecipe, u: f32, v: f32) -> [f32; 2] {
    let (u, v) = r.layers.as_ref().map_or((u, v), |l| l.rotate(u, v));
    [
        (u + r.phase[0]).rem_euclid(1.),
        (v + r.phase[1]).rem_euclid(1.),
    ]
}

#[test]
fn prepared_paint_replays_every_texel_channel_bit() {
    use super::super::Surface;
    for (seed, size) in [
        (0, 256),
        (7, 256),
        (202, 256),
        (1_013_005, 256),
        (42_430_575, 256),
        (43_084_584, 256),
        (43_084_482, 256),
        (u64::MAX, 256),
        (99, 512),
    ] {
        let recipes = super::super::program::sample(seed);
        for surface in [Surface::Paint, Surface::Accent, Surface::Ceiling] {
            let r = &recipes[surface as usize];
            let c = r.coating.as_ref().unwrap();
            let a = c.application.as_ref().unwrap();
            let prepared = r.prepare_texels(0).unwrap();
            for y in 0..size {
                for x in 0..size {
                    let (u, v) = (x as f32 / size as f32, y as f32 / size as f32);
                    let uv = transformed(r, u, v);
                    same_bits(
                        reference(a, c, r, uv),
                        r.texel_prepared(u, v, 0, Some(&prepared)),
                        seed,
                        uv,
                    );
                }
            }
            for (u, v) in [(-1. / 256., 1.), (1., -1. / 256.), (2.3, -0.4)] {
                let uv = transformed(r, u, v);
                same_bits(
                    reference(a, c, r, uv),
                    r.texel_prepared(u, v, 0, Some(&prepared)),
                    seed,
                    uv,
                );
            }
        }
    }
}

#[test]
fn authored_glaze_retains_precedence_over_paint_application() {
    let recipes = super::super::program::sample(7);
    let mut recipe = recipes[super::super::Surface::Ceramic as usize].clone();
    let coating = recipe.coating.as_mut().unwrap();
    assert!(coating.glaze.is_some());
    coating.application = Some(PaintApplication::sample(8));
    assert!(matches!(
        recipe.prepare_texels(0),
        Some(super::super::program::PreparedMaterial::Glaze(_))
    ));
}

#[test]
fn maximum_frequency_paint_cache_replays_original_complete_maps() {
    for seed in [202, 43_084_482] {
        let mut r =
            super::super::program::sample(seed)[super::super::Surface::Paint as usize].clone();
        r.period_m = 10.;
        let coating = r.coating.as_ref().unwrap();
        let application = coating.application.as_ref().unwrap();
        assert_eq!(frequency(r.period_m, application.spray_spacing_m), 192);
        super::super::program::replay_helpers::check_maps(&r, 0, |u, v| {
            reference(application, coating, &r, transformed(&r, u, v))
        });
    }
}

#[test]
#[ignore = "counterbalanced CPU microbenchmark; run explicitly with --ignored --nocapture"]
fn prepared_paint_cpu_benchmark() {
    use std::{hint::black_box, time::Instant};
    for size in [256, 512] {
        let mut reference_seconds = Vec::new();
        let mut hoist_only_seconds = Vec::new();
        let mut prepared_seconds = Vec::new();
        let mut cell_grids = Vec::new();
        for seed in 200..212 {
            let recipes = super::super::program::sample(seed);
            let r = &recipes[super::super::Surface::Paint as usize];
            let c = r.coating.as_ref().unwrap();
            let a = c.application.as_ref().unwrap();
            let old = || {
                let start = Instant::now();
                for y in 0..size {
                    for x in 0..size {
                        let uv = [x as f32 / size as f32, y as f32 / size as f32];
                        black_box(reference(a, c, black_box(r), uv));
                    }
                }
                start.elapsed().as_secs_f64()
            };
            let cached = || {
                let start = Instant::now();
                // Include construction of the bounded grid in per-map cost.
                let super::super::program::PreparedMaterial::Paint(prepared) =
                    r.prepare_texels(0).unwrap()
                else {
                    unreachable!("paint recipe")
                };
                for y in 0..size {
                    for x in 0..size {
                        let uv = [x as f32 / size as f32, y as f32 / size as f32];
                        black_box(prepared.evaluate(a, c, black_box(r), uv));
                    }
                }
                start.elapsed().as_secs_f64()
            };
            let hoisted = || {
                let start = Instant::now();
                let prepared = a.prepare(r);
                for y in 0..size {
                    for x in 0..size {
                        let uv = [x as f32 / size as f32, y as f32 / size as f32];
                        black_box(prepared.evaluate(a, c, black_box(r), uv));
                    }
                }
                start.elapsed().as_secs_f64()
            };
            let (before, hoist_only, after) = match seed % 3 {
                0 => (old(), hoisted(), cached()),
                1 => {
                    let h = hoisted();
                    let after = cached();
                    (old(), h, after)
                }
                _ => {
                    let after = cached();
                    let before = old();
                    (before, hoisted(), after)
                }
            };
            reference_seconds.push(before);
            hoist_only_seconds.push(hoist_only);
            prepared_seconds.push(after);
            let cells = frequency(r.period_m, a.spray_spacing_m);
            cell_grids.push(serde_json::json!({
                "seed": seed,
                "count": cells,
                "requested_halo_bytes": (cells + 2) * (cells + 2) * 4 * 4,
            }));
        }
        reference_seconds.sort_by(f64::total_cmp);
        hoist_only_seconds.sort_by(f64::total_cmp);
        prepared_seconds.sort_by(f64::total_cmp);
        let median = |values: &[f64]| (values[5] + values[6]) * 0.5;
        let before = median(&reference_seconds);
        let after = median(&prepared_seconds);
        eprintln!(
            "{}",
            serde_json::json!({
                "atlas_size": size,
                "rooms": 12,
                "reference_median_seconds": before,
                "hoist_only_median_seconds": median(&hoist_only_seconds),
                "prepared_median_seconds": after,
                "kernel_speedup": before / after,
                "reference_seconds": reference_seconds,
                "hoist_only_seconds": hoist_only_seconds,
                "prepared_seconds": prepared_seconds,
                "cell_grids": cell_grids,
                "scope": "CPU paint texels plus per-map preparation/grid construction; three-way counterbalanced seeds; no mip/filter/GPU or end-to-end throughput claim"
            })
        );
    }
}
