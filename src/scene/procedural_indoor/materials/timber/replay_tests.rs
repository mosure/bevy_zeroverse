//! Independent pre-preparation timber texel oracle.
use super::super::{field::*, program::Texel};
use super::*;

fn reference(w: &WoodFinish, r: &MaterialRecipe, [u, v]: [f32; 2]) -> Texel {
    let g = grain(r, u, v);
    let absorbed = (w.stain_strength * (0.92 + (0.65 - g) * 0.22)).clamp(0., 1.);
    let base = mix(
        mix(r.color, w.stain_color, absorbed),
        [0.88, 0.87, 0.82],
        w.bleach,
    );
    let gain = (1. + (g - 0.58) * r.contrast * w.ring_contrast * 2.).clamp(0.35, 1.20);
    Texel {
        color: tint(base, gain),
        height: (g - 0.5) * r.relief_m * (1. - w.pore_fill * 0.85),
        roughness: (r.roughness + (0.55 - g) * 0.18 * (1. - w.clearcoat * 0.65)).clamp(0.12, 0.95),
        occlusion: 1. - (0.5 - g).max(0.) * 0.04 * (1. - w.pore_fill),
    }
}

#[test]
fn prepared_wood_replays_texel_bits_and_every_pbr_mip() {
    use super::super::program::replay_helpers::*;
    for seed in [207, 43_084_584, 43_084_482] {
        let recipes =
            super::super::program::replay_helpers::legacy_sample_with_floor(seed, 0).unwrap();
        for surface in [Surface::Wood, Surface::WoodEdge, Surface::Floor] {
            let r = &recipes[surface as usize];
            let w = r.wood.as_ref().unwrap();
            check_maps(r, 0, |u, v| {
                let uv = transformed(r, u, v);
                let mut t = reference(w, r, uv);
                if surface == Surface::Floor {
                    floor_joints(r, uv, &mut t);
                }
                t
            });
        }
    }
}

#[test]
fn timber_finish_treatments_replay_and_span_dark_stain_pale_wash_and_clearcoat() {
    let mut stain = 0;
    let mut wash = 0;
    let mut coat = 0;
    for seed in 0..512 {
        let w = WoodFinish::sample(seed);
        assert_eq!(w, WoodFinish::sample(seed));
        w.validate().unwrap();
        stain += usize::from(w.stain_strength > 0.60);
        wash += usize::from(w.bleach > 0.30);
        coat += usize::from(w.clearcoat > 0.65);
        assert!(w.bleach <= 0.65 * (1. - w.stain_strength * 0.85));
        assert!(w.pore_fill + 1e-6 >= w.clearcoat * 0.48);
    }
    assert!(
        stain > 80 && wash > 20 && coat > 35,
        "finish diversity collapsed: {stain} {wash} {coat}"
    );
}

#[test]
#[ignore = "counterbalanced CPU microbenchmark; run explicitly with --ignored --nocapture"]
fn prepared_wood_cpu_benchmark() {
    use super::super::program::{replay_helpers::*, PreparedWood};
    for size in [256, 512] {
        let mut old = Vec::new();
        let mut new = Vec::new();
        for seed in 200..208 {
            let recipes = super::super::program::sample(seed);
            let r = &recipes[Surface::Wood as usize];
            let w = r.wood.as_ref().unwrap();
            let p = PreparedWood::new(w, r.color);
            let (a, b) = bench_texels(
                size,
                |uv| reference(w, r, uv),
                |uv| w.evaluate_prepared(r, uv, Some(&p)),
                seed % 2 == 1,
            );
            old.push(a);
            new.push(b);
        }
        report_benchmark("wood constant colors", size, old, new);
    }
}
