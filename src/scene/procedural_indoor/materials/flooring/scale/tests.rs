use super::super::super::{program, Surface};
use super::*;
use std::collections::BTreeMap;

fn floor(seed: u64, style: u32) -> MaterialRecipe {
    program::sample_with_floor(seed, style)
        .unwrap()
        .remove(Surface::Floor as usize)
}

#[test]
fn physical_formats_cover_fine_and_large_products_in_each_installation() {
    for style in 0..3 {
        let mut support = BTreeMap::<usize, (usize, f32, f32)>::new();
        let mut regular = 0;
        let mut variable = 0;
        for seed in 0..1024 {
            let r = floor(seed, style);
            r.validate().unwrap();
            let f = r.flooring.as_ref().unwrap();
            let size = r.period_m / f.cells[0] as f32;
            let (_, lo, hi) = support
                .entry(f.layout as usize)
                .or_insert((0, f32::MAX, 0.));
            *lo = lo.min(size);
            *hi = hi.max(size);
            support.get_mut(&(f.layout as usize)).unwrap().0 += 1;
            assert_eq!(r.map_size(style), 512);
            assert!((0.55..=6.4).contains(&r.period_m));
            if matches!(f.layout, FloorLayout::Straight | FloorLayout::BrokenBond) {
                let jitter = f.stock.as_ref().unwrap().cut_jitter;
                regular += usize::from(jitter == [0.; 2]);
                variable += usize::from(jitter != [0.; 2]);
                // Opposite displacements give the worst possible cut size.
                // Floor pieces must not become thin slivers under any phase.
                let minimum = if style == 0 { [0.04, 0.12] } else { [0.04; 2] };
                for axis in 0..2 {
                    let shortest = r.period_m / f.cells[axis] as f32 * (1. - 2. * jitter[axis]);
                    assert!(
                        shortest + 1e-6 >= minimum[axis],
                        "{seed}/{style}/{axis}: {shortest}"
                    );
                }
            }
        }
        for (layout, (n, lo, hi)) in &support {
            assert!(*n > 25, "{style}/{layout} has insufficient support: {n}");
            assert!(
                hi / lo > 2.5,
                "{style}/{layout} scale collapsed: {lo}..{hi}"
            );
        }
        let (_, lo, hi) = support[&(FloorLayout::Straight as usize)];
        assert!(lo < [0.09, 0.30, 0.12][style as usize]);
        assert!(hi > [0.30, 0.85, 1.20][style as usize]);
        assert!(regular > 20);
        if style != 1 {
            assert!(variable > 80);
        }
    }
}

#[test]
fn broadloom_has_no_hidden_tile_color_height_or_roughness_grid() {
    let mut broadloom = 0;
    let mut soft_tiles = 0;
    let mut defined_tiles = 0;
    let mut motifs = [f32::MAX, 0.0_f32];
    for seed in 0..512 {
        let r = floor(seed, 1);
        let f = r.flooring.as_ref().unwrap();
        soft_tiles += usize::from(f.tile_definition > 0. && f.tile_definition < 0.3);
        defined_tiles += usize::from(f.tile_definition > 0.7);
        let carpet = r.carpet.as_ref().unwrap();
        for count in carpet.repeats {
            let m = r.period_m / count as f32;
            motifs[0] = motifs[0].min(m);
            motifs[1] = motifs[1].max(m);
        }
        if f.tile_definition != 0. {
            continue;
        }
        broadloom += 1;
        assert_eq!(f.piece_roughness, 0.);
        let mut without_tiles = r.clone();
        without_tiles.flooring = None;
        for i in 0..128 {
            let uv = [
                (i as f32 * 0.7548777).fract(),
                (i as f32 * 0.5698403).fract(),
            ];
            let a = r.texel(uv[0], uv[1], 1);
            let b = without_tiles.texel(uv[0], uv[1], 1);
            assert_eq!(a.color, b.color);
            assert_eq!(a.height, b.height);
            assert_eq!(a.roughness, b.roughness);
            assert_eq!(a.occlusion, b.occlusion);
        }
    }
    assert!((160..260).contains(&broadloom), "{broadloom}");
    assert!(soft_tiles > 70 && defined_tiles > 30);
    assert!(motifs[0] < 0.15 && motifs[1] > 2.5, "{motifs:?}");
}

#[test]
fn floor_scale_sampling_preserves_other_roles_and_resolved_replay() {
    for seed in [0, 1, 202, 66248449, u64::MAX] {
        let base = program::sample(seed);
        for style in 0..3 {
            let sampled = program::sample_with_floor(seed, style).unwrap();
            for (a, b) in sampled.iter().zip(&base) {
                if a.surface != Surface::Floor {
                    assert_eq!(a, b);
                }
            }
            let r = &sampled[Surface::Floor as usize];
            let copy: MaterialRecipe =
                serde_json::from_slice(&serde_json::to_vec(r).unwrap()).unwrap();
            assert_eq!(*r, copy);
            assert_eq!(
                r.evaluate(0.173, 0.687, style),
                copy.evaluate(0.173, 0.687, style)
            );
        }
    }
}
