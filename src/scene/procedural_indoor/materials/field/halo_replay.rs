//! Frozen pre-halo hash-cache evaluator. Do not derive the reference from the
//! candidate's stored positions, primary-only search or wrapped index helper.
use super::*;

struct OriginalPrepared {
    count: [u32; 2],
    attributes: Vec<[f32; 4]>,
}

impl OriginalPrepared {
    fn new(count: [u32; 2], seed: u64) -> Self {
        let mut attributes = Vec::with_capacity((count[0] * count[1]) as usize);
        for y in 0..count[1] {
            for x in 0..count[0] {
                attributes.push([
                    hash(x, y, seed),
                    hash(x, y, seed.wrapping_add(31)),
                    hash(x, y, seed.wrapping_add(71)),
                    hash(x, y, seed.wrapping_add(113)),
                ]);
            }
        }
        Self { count, attributes }
    }

    fn sample_indexed(&self, [u, v]: [f32; 2]) -> (Cell, usize) {
        let x = u.rem_euclid(1.) * self.count[0] as f32;
        let y = v.rem_euclid(1.) * self.count[1] as f32;
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let mut nearest = [f32::INFINITY; 2];
        let mut dye = 0.;
        let mut offset = [0.; 2];
        let mut shape = 0.;
        let mut selected = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (ix + dx, iy + dy);
                let hx = cx.rem_euclid(self.count[0] as i32) as u32;
                let hy = cy.rem_euclid(self.count[1] as i32) as u32;
                let index = (hy * self.count[0] + hx) as usize;
                let attributes = self.attributes[index];
                let px = cx as f32 + 0.05 + 0.90 * attributes[0];
                let py = cy as f32 + 0.05 + 0.90 * attributes[1];
                let d = (px - x).powi(2) + (py - y).powi(2);
                if d < nearest[0] {
                    nearest = [d, nearest[0]];
                    dye = attributes[2];
                    offset = [px - x, py - y];
                    shape = attributes[3];
                    selected = index;
                } else if d < nearest[1] {
                    nearest[1] = d;
                }
            }
        }
        (
            Cell {
                radius: nearest[0].sqrt(),
                edge: nearest[1].sqrt() - nearest[0].sqrt(),
                dye,
                offset,
                shape,
            },
            selected,
        )
    }
}

fn bits(cell: &Cell) -> [u32; 6] {
    [
        cell.radius,
        cell.edge,
        cell.dye,
        cell.offset[0],
        cell.offset[1],
        cell.shape,
    ]
    .map(f32::to_bits)
}

fn compare(original: &OriginalPrepared, halo: &PreparedCellular, uv: [f32; 2]) {
    let (expected, index) = original.sample_indexed(uv);
    assert_eq!(bits(&halo.sample(uv)), bits(&expected), "uv {uv:?}");
    let (actual, actual_index) = halo.sample_primary_indexed(uv);
    assert_eq!(actual_index, index, "winner changed at {uv:?}");
    for actual in [actual, halo.sample_primary(uv)] {
        assert_eq!(
            [
                actual.radius,
                actual.dye,
                actual.offset[0],
                actual.offset[1],
                actual.shape,
            ]
            .map(f32::to_bits),
            [
                expected.radius,
                expected.dye,
                expected.offset[0],
                expected.offset[1],
                expected.shape,
            ]
            .map(f32::to_bits),
            "primary feature changed at {uv:?}"
        );
    }
}

#[test]
fn cellular_halo_replays_original_full_atlases_and_wrapped_winners() {
    for count in [[48, 64], [192, 192]] {
        for seed in [202, 43_084_482] {
            let original = OriginalPrepared::new(count, seed);
            let halo = PreparedCellular::new(count, seed);
            let dyes = halo.dyes().collect::<Vec<_>>();
            assert_eq!(dyes.len(), original.attributes.len());
            assert_eq!(dyes.capacity(), original.attributes.len());
            assert_eq!(
                dyes.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                original
                    .attributes
                    .iter()
                    .map(|a| a[2].to_bits())
                    .collect::<Vec<_>>()
            );
            for y in 0..256 {
                for x in 0..256 {
                    compare(&original, &halo, [x as f32 / 256., y as f32 / 256.]);
                }
            }
        }
    }
}

#[test]
fn cellular_halo_preserves_rounding_fallback_signed_zero_and_nonfinite_inputs() {
    for count in [[1, 1], [3, 5], [191, 192]] {
        for seed in [0, 207, u64::MAX] {
            let original = OriginalPrepared::new(count, seed);
            let halo = PreparedCellular::new(count, seed);
            for uv in [
                [-f32::MIN_POSITIVE, f32::MIN_POSITIVE],
                [f32::MIN_POSITIVE, -f32::MIN_POSITIVE],
                [-f32::from_bits(1), 0.],
                [0., -f32::from_bits(1)],
                [-0., 0.],
                [1., 1.],
                [-f32::EPSILON, 1. - f32::EPSILON],
                [-1. / 256., 1.],
                [2.3, -0.4],
                [f32::NAN, 0.],
                [0., f32::INFINITY],
                [f32::NEG_INFINITY, 1.],
            ] {
                compare(&original, &halo, uv);
                assert_eq!(
                    bits(&cellular(uv, count, seed)),
                    bits(&original.sample_indexed(uv).0)
                );
            }
        }
    }
}

#[test]
fn cellular_halo_debits_actual_capacity_and_retains_small_budget_fallback() {
    for count in [[1, 1], [48, 64], [192, 192]] {
        let bytes = ((count[0] + 2) * (count[1] + 2)) as usize * 16;
        for budget in [0, 16, bytes - 1, bytes, bytes + 64] {
            let mut remaining = budget;
            let prepared = PreparedCellular::with_budget(count, 202, &mut remaining);
            if let Some(prepared) = prepared {
                assert_eq!(prepared.bytes() + remaining, budget);
                assert!(prepared.bytes() >= bytes);
            } else {
                assert_eq!(remaining, budget);
            }
        }
    }
}

#[test]
#[ignore = "allocation-inclusive counterbalanced cellular kernel timing; root coordinates"]
fn cellular_halo_cpu_benchmark() {
    use std::{hint::black_box, time::Instant};
    let seeds = [202, 207, 43_084_482, 43_084_584, 42430575, 7, 1013005, 0];
    let mut original_seconds = Vec::new();
    let mut halo_seconds = Vec::new();
    for (round, seed) in seeds.into_iter().enumerate() {
        let original = || {
            let start = Instant::now();
            let context = OriginalPrepared::new([48, 64], seed);
            for y in 0..256 {
                for x in 0..256 {
                    let (cell, index) =
                        context.sample_indexed(black_box([x as f32 / 256., y as f32 / 256.]));
                    black_box((cell.radius, cell.dye, cell.offset, cell.shape, index));
                }
            }
            start.elapsed().as_secs_f64()
        };
        let halo = || {
            let start = Instant::now();
            let context = PreparedCellular::new([48, 64], seed);
            for y in 0..256 {
                for x in 0..256 {
                    let (cell, index) = context
                        .sample_primary_indexed(black_box([x as f32 / 256., y as f32 / 256.]));
                    black_box((cell.radius, cell.dye, cell.offset, cell.shape, index));
                }
            }
            start.elapsed().as_secs_f64()
        };
        let (a, b) = if round % 2 == 0 {
            (original(), halo())
        } else {
            let b = halo();
            (original(), b)
        };
        original_seconds.push(a);
        halo_seconds.push(b);
    }
    println!(
        "{}",
        serde_json::json!({
            "scope": "allocation-inclusive cellular evaluator only; original prepared hash cache versus absolute halo/primary search; not atlas/capture throughput",
            "seeds": seeds,
            "original_seconds": original_seconds,
            "halo_seconds": halo_seconds,
            "halo_capacity_bytes": PreparedCellular::new([48,64],202).bytes(),
        })
    );
}
