//! Shared test scaffolding; callers supply independent pre-preparation texel oracles.
use super::*;
use bevy::prelude::*;

pub(in super::super) fn transformed(r: &MaterialRecipe, u: f32, v: f32) -> [f32; 2] {
    let (u, v) = r.layers.as_ref().map_or((u, v), |l| l.rotate(u, v));
    [
        (u + r.phase[0]).rem_euclid(1.),
        (v + r.phase[1]).rem_euclid(1.),
    ]
}

pub(in super::super) fn same_bits(a: &Texel, b: &Texel) {
    assert_eq!(
        [
            a.color[0],
            a.color[1],
            a.color[2],
            a.height,
            a.roughness,
            a.occlusion
        ]
        .map(f32::to_bits),
        [
            b.color[0],
            b.color[1],
            b.color[2],
            b.height,
            b.roughness,
            b.occlusion
        ]
        .map(f32::to_bits),
    );
}

pub(in super::super) fn floor_joints(r: &MaterialRecipe, uv: [f32; 2], t: &mut Texel) {
    r.floor_joints(uv, t);
}

pub(in super::super) fn check_maps(
    r: &MaterialRecipe,
    style: u32,
    mut reference: impl FnMut(f32, f32) -> Texel,
) {
    let n = r.map_size(style) as usize;
    let prepared = r.prepare_texels(style).unwrap();
    let mut heights = vec![0.; n * n];
    let mut colors = Vec::with_capacity(n * n * 4);
    let mut data = Vec::with_capacity(n * n * 4);
    for y in 0..n {
        for x in 0..n {
            let u = x as f32 / n as f32;
            let v = y as f32 / n as f32;
            let t = reference(u, v);
            same_bits(&t, &r.texel_prepared(u, v, style, Some(&prepared)));
            heights[y * n + x] = t.height;
            let c = t.color.map(|c| (c.clamp(0., 1.) * 255.) as u8);
            colors.extend([c[0], c[1], c[2], 255]);
            data.extend([
                (t.occlusion * 255.).round() as u8,
                (t.roughness.clamp(0.05, 1.) * 255.) as u8,
                0,
                255,
            ]);
        }
    }
    let slope = Vec2::splat(n as f32 * 0.5) / r.period_uv();
    let mut normals = Vec::with_capacity(n * n * 4);
    for y in 0..n {
        for x in 0..n {
            let dx = heights[y * n + (x + 1) % n] - heights[y * n + (x + n - 1) % n];
            let dy = heights[((y + 1) % n) * n + x] - heights[((y + n - 1) % n) * n + x];
            normals.extend(super::super::filter::encode(
                Vec3::new(-dx * slope.x, -dy * slope.y, 1.).normalize(),
            ));
        }
    }
    let expected = super::super::mapped_images((colors, normals, data), n as u32, Some(r));
    let actual = r.maps(style);
    for (a, b) in expected.iter().zip(&actual) {
        assert_eq!(a.texture_descriptor, b.texture_descriptor);
        assert_eq!(
            a.data, b.data,
            "map or mip changed: {:?} seed {}",
            r.surface, r.seed
        );
        assert_eq!(format!("{:?}", a.sampler), format!("{:?}", b.sampler));
    }
}

pub(in super::super) fn bench_texels(
    size: u32,
    mut before: impl FnMut([f32; 2]) -> Texel,
    mut after: impl FnMut([f32; 2]) -> Texel,
    reverse: bool,
) -> (f64, f64) {
    use std::{hint::black_box, time::Instant};
    let run = |f: &mut dyn FnMut([f32; 2]) -> Texel| {
        let started = Instant::now();
        for y in 0..size {
            for x in 0..size {
                black_box(f([x as f32 / size as f32, y as f32 / size as f32]));
            }
        }
        started.elapsed().as_secs_f64()
    };
    if reverse {
        let b = run(&mut after);
        (run(&mut before), b)
    } else {
        (run(&mut before), run(&mut after))
    }
}

pub(in super::super) fn report_benchmark(label: &str, size: u32, mut a: Vec<f64>, mut b: Vec<f64>) {
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let median = |v: &[f64]| (v[v.len() / 2 - 1] + v[v.len() / 2]) * 0.5;
    eprintln!(
        "{}",
        serde_json::json!({
            "program": label, "atlas_size": size, "seeds": a.len(),
            "reference_median_seconds": median(&a), "prepared_median_seconds": median(&b),
            "kernel_speedup": median(&a) / median(&b), "reference_seconds": a, "prepared_seconds": b,
            "scope": "counterbalanced CPU kernel diagnostic; timing boundary is documented by the calling fixture; no GPU/end-to-end capture claim"
        })
    );
}
