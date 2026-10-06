//! Independent original weave loop and complete-map replay, including normal/ORM mips.
use super::super::{program::replay_helpers::*, Surface};
use super::*;

fn reference(t: &TextileRecipe, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
    let [u, v] = uv;
    if t.knit {
        return knit::texel(t, r, u, v);
    }
    let n = t.yarns.map(|n| n as f32);
    let x = u * n[0];
    let y = v * n[1];
    let ix = x.floor() as u32 % t.yarns[0];
    let iy = y.floor() as u32 % t.yarns[1];
    let noise = |nx, ny, salt| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(salt));
    let slub = [noise(t.yarns[0], 3, 11), noise(3, t.yarns[1], 17)];
    let jitter = [hash(ix, 0, r.seed) - 0.5, hash(iy, 1, r.seed) - 0.5];
    let profile = |p: f32, w: f32| {
        let q = (p / (w * 0.5)).abs();
        // Rounded yarn crowns taper smoothly into gaps; no full-tile wave.
        (1. - q * q).max(0.).sqrt() * (1. - smooth((q - 0.82) / 0.18))
    };
    let p = [
        x.fract() - 0.5 - jitter[0] * 0.05,
        y.fract() - 0.5 - jitter[1] * 0.05,
    ];
    let crown = [0, 1].map(|i| profile(p[i], t.width[i] * (1. + t.slub * (slub[i] - 0.5))));
    let row = iy / t.bundle;
    let row = if t.herringbone {
        let turn = row % (2 * t.repeat);
        turn.min(2 * t.repeat - 1 - turn)
    } else {
        row
    };
    let warp_over = ((ix / t.bundle + row * t.advance) % t.repeat < t.float_length) as u32 as f32;
    // Crimp depresses the lower yarn at a crossing, not along the entire UV.
    let h = [
        crown[0] * (1. - t.crimp * crown[1] * (1. - warp_over)),
        crown[1] * (1. - t.crimp * crown[0] * warp_over),
    ];
    let top = if h[0] + h[1] > 0.0001 {
        h[0] / (h[0] + h[1])
    } else {
        0.5
    };
    let coverage = crown[0].max(crown[1]);
    let dye = [hash(ix, 13, r.seed), hash(iy, 19, r.seed)];
    let bands = if let Some(l) = &r.layers {
        [0, 1].map(|axis| {
            let count = l.bands[axis];
            let index = [ix, iy][axis];
            if count > 0 && (index * count / t.yarns[axis]).is_multiple_of(3) {
                1. - l.stripe_strength * 0.25
            } else {
                1.
            }
        })
    } else {
        [1.; 2]
    };
    let twist = t.twist * (noise(3, 3, 83) - 0.5);
    let filament = [ridge(x * 2. + twist), ridge(y * 2. - twist)];
    let fibres = top * filament[0] + (1. - top) * filament[1];
    let nap = noise(79, 83, 73) - 0.5;
    let yarn = [0, 1, 2].map(|c| {
        let a = t.yarn_tint[0][c] * (0.94 + t.dye_variation * (dye[0] - 0.5)) * bands[0];
        let b = t.yarn_tint[1][c] * (0.94 + t.dye_variation * (dye[1] - 0.5)) * bands[1];
        (0.76 + (a * top + b * (1. - top) - 0.76) * coverage + nap * t.fuzz * 0.035).clamp(0.25, 1.)
    });
    let height =
        r.relief_m * (h[0].max(h[1]) - 0.5 + fibres * 0.045 * (1. - t.fuzz) + nap * t.fuzz * 0.09);
    let pile = noise(37, 41, 101);
    Texel {
        color: yarn.map(|c| (c * (1. - t.pile * 0.08 * (pile - 0.5))).clamp(0., 1.)),
        height: height * (1. - t.pile * 0.45) + t.pile * r.relief_m * (pile - 0.5),
        roughness: (r.roughness + t.fuzz * nap * 0.10 + (1. - coverage) * 0.10
            - t.lustre * (fibres - 0.5) * 0.06)
            .clamp(0.30, 1.),
        occlusion: 0.93 + 0.07 * coverage,
    }
}

fn resolved(r: &MaterialRecipe) -> TextileRecipe {
    let mut t = r
        .textile
        .clone()
        .unwrap_or_else(|| TextileRecipe::sample(r.seed));
    if r.surface == Surface::Floor && t.pile == 0. {
        t.pile = 0.85;
    }
    t
}

#[test]
fn prepared_weave_replays_original_texel_channels_and_complete_pbr_mips() {
    for seed in [202, 207, 43_084_482] {
        for (surface, style) in [
            (Surface::Fabric, 0),
            (Surface::FabricAlt, 0),
            (Surface::Floor, 1),
        ] {
            let recipes = super::super::program::sample_with_floor(seed, style).unwrap();
            let r = &recipes[surface as usize];
            let t = resolved(r);
            check_maps(r, style, |u, v| reference(&t, r, transformed(r, u, v)));
        }
    }
}

#[test]
fn prepared_weave_budget_fallbacks_and_authored_yarn_extremes_preserve_bits() {
    for seed in [0, 202, 43_084_482, u64::MAX] {
        let recipes = super::super::program::sample(seed);
        for authored in [false, true] {
            let mut r = recipes[Surface::Fabric as usize].clone();
            if authored {
                let t = r.textile.as_mut().unwrap();
                t.yarns = [48, 48];
                t.repeat = 2;
                t.bundle = 3;
                t.float_length = 1;
                t.advance = 1;
                t.herringbone = true;
                t.width = [0.5, 0.99];
                t.slub = 1.;
                t.fuzz = 1.;
                t.twist = -2.;
                t.dye_variation = 1.;
                t.pile = 1.;
                t.yarn_tint = [[0.5; 3], [1.; 3]];
                let l = r.layers.as_mut().unwrap();
                l.bands = [17, 0];
                l.stripe_strength = 1.;
            }
            let t = resolved(&r);
            for budget in [0, 32, 1024, prepared::TABLE_BUDGET_BYTES] {
                let p = PreparedTextile::with_budget(&r, budget);
                assert_eq!(p.allocated_bytes(), p.bytes());
                assert!(p.bytes() <= budget);
                let samples =
                    (0..32).flat_map(|y| (0..32).map(move |x| [x as f32 / 32., y as f32 / 32.]));
                for uv in samples.chain([
                    [-1. / 256., 1.],
                    [1., -1. / 256.],
                    [2.3, -0.4],
                    [-f32::EPSILON, f32::EPSILON],
                ]) {
                    same_bits(&reference(&t, &r, uv), &p.evaluate(&r, uv));
                }
            }
        }
    }
}

#[test]
fn prepared_weave_keeps_fallback_recipe_carpet_pile_and_knit_dispatch() {
    for surface in [Surface::Fabric, Surface::Floor] {
        let mut r =
            super::super::program::sample_with_floor(207, 1).unwrap()[surface as usize].clone();
        r.textile = None;
        let t = resolved(&r);
        check_maps(&r, 1, |u, v| reference(&t, &r, transformed(&r, u, v)));
        r.textile = Some(t);
        r.textile.as_mut().unwrap().knit = true;
        assert!(r.prepare_texels(1).is_none());
        for uv in [[0., 0.], [0.3, 0.7], [1., -1. / 256.]] {
            let t = resolved(&r);
            let p = transformed(&r, uv[0], uv[1]);
            same_bits(&reference(&t, &r, p), &r.texel(uv[0], uv[1], 1));
        }
    }
}

#[test]
#[ignore = "allocation-inclusive weave CPU kernel; run separately with --ignored --nocapture"]
fn prepared_weave_allocation_inclusive_cpu_benchmark() {
    use std::{hint::black_box, time::Instant};
    for size in [256, 512] {
        let mut old = Vec::new();
        let mut new = Vec::new();
        for seed in 200..208 {
            let r = super::super::program::sample(seed)[Surface::Fabric as usize].clone();
            let scalar = || {
                let started = Instant::now();
                let t = resolved(&r);
                for y in 0..size {
                    for x in 0..size {
                        black_box(reference(
                            &t,
                            &r,
                            [x as f32 / size as f32, y as f32 / size as f32],
                        ));
                    }
                }
                started.elapsed().as_secs_f64()
            };
            let cached = || {
                let started = Instant::now();
                let p = PreparedTextile::new(&r);
                for y in 0..size {
                    for x in 0..size {
                        black_box(p.evaluate(&r, [x as f32 / size as f32, y as f32 / size as f32]));
                    }
                }
                drop(p);
                started.elapsed().as_secs_f64()
            };
            let (a, b) = if seed % 2 == 0 {
                (scalar(), cached())
            } else {
                let b = cached();
                (scalar(), b)
            };
            old.push(a);
            new.push(b);
        }
        report_benchmark(
            "woven textile including map context allocations/setup/free; excludes mips/GPU",
            size,
            old,
            new,
        );
    }
}
