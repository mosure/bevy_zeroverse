//! Immutable per-map paint constants. The texel program and precision are unchanged.
use super::*;

// Cells and noise share the original combined cap. The absolute-position halo
// can borrow the few extra bytes it needs from noise grids rather than making
// the largest spray field fall back to per-texel hash evaluation.
const MAX_CACHE_BYTES_PER_MAP: usize = 192 * 192 * std::mem::size_of::<[f32; 4]>() + 256 * 1024;

pub(in super::super) struct PreparedPaint {
    spray_n: u32,
    roller_n: u32,
    roller_rows: u32,
    fine_n: u32,
    brush_n: u32,
    trowel_n: u32,
    aa: f32,
    resolved_brush: f32,
    linear_color: [f32; 3],
    spray_cells: Option<PreparedCellular>,
    noise: Option<[PreparedPeriodic; 8]>,
}

impl PreparedPaint {
    pub(super) fn new(a: &PaintApplication, r: &MaterialRecipe) -> Self {
        let spray_n = frequency(r.period_m, a.spray_spacing_m);
        let roller_n = frequency(r.period_m, a.roller_spacing_m);
        let brush_n = frequency(r.period_m, a.brush_spacing_m);
        Self {
            spray_n,
            roller_n,
            roller_rows: (roller_n as f32 / a.roller_stretch).round().max(1.) as u32,
            fine_n: frequency(r.period_m, a.orange_peel_m),
            brush_n,
            trowel_n: frequency(r.period_m, a.trowel_scale_m),
            aa: spray_n as f32 / r.map_size(0) as f32,
            resolved_brush: (r.map_size(0) as f32 / brush_n as f32 * 0.30).min(1.),
            linear_color: r.color.map(super::super::srgb_to_linear),
            spray_cells: None,
            noise: None,
        }
    }
    pub(in super::super) fn cache_cells(mut self, seed: u64) -> Self {
        let mut remaining = MAX_CACHE_BYTES_PER_MAP;
        self.spray_cells = PreparedCellular::with_budget(
            [self.spray_n, self.spray_n],
            seed.wrapping_add(503),
            &mut remaining,
        );
        let mut make = |count, seed| PreparedPeriodic::new(count, seed, &mut remaining);
        let deposit_seed = seed.wrapping_add(541);
        self.noise = Some([
            make([self.roller_n, self.roller_rows], seed.wrapping_add(509)),
            make([self.fine_n, self.fine_n], seed.wrapping_add(521)),
            make([self.brush_n, 5], seed.wrapping_add(523)),
            make([3, 5], deposit_seed),
            make([5, 3], deposit_seed.wrapping_add(17)),
            make(
                [self.trowel_n, self.trowel_n],
                deposit_seed.wrapping_add(47),
            ),
            make(
                [self.trowel_n * 2, self.trowel_n * 2],
                deposit_seed.wrapping_add(61),
            ),
            make(
                [self.trowel_n * 4, self.trowel_n * 4],
                deposit_seed.wrapping_add(83),
            ),
        ]);
        self
    }

    pub(in super::super) fn evaluate(
        &self,
        a: &PaintApplication,
        c: &CoatingRecipe,
        r: &MaterialRecipe,
        uv: [f32; 2],
    ) -> Texel {
        let [u, v] = uv;
        let spray = self.spray_cells.as_ref().map_or_else(
            || cellular(uv, [self.spray_n, self.spray_n], r.seed.wrapping_add(503)).into(),
            |cells| cells.sample_primary(uv),
        );
        let splat = disk(spray.radius, 0.20 + spray.dye * 0.28, self.aa);
        let plateau = smooth(splat * 1.4).min(1. - c.knockdown * 0.55);
        let (roller, film, brush, passes) = if let Some(noise) = &self.noise {
            let roller = noise[0].sample(u, v) - 0.5;
            let film = noise[1].sample(u, v) - 0.5;
            let brush = noise[2].sample(u, v) - 0.5;
            let x = u + (noise[3].sample(u, v) - 0.5) * 0.12;
            let y = v + (noise[4].sample(u, v) - 0.5) * 0.12;
            let passes = 0.64 * noise[5].sample(x, y)
                + 0.25 * noise[6].sample(x, y)
                + 0.11 * noise[7].sample(x, y);
            (roller, film, brush, passes)
        } else {
            let roller = periodic_noise(
                u,
                v,
                self.roller_n,
                self.roller_rows,
                r.seed.wrapping_add(509),
            ) - 0.5;
            let film =
                periodic_noise(u, v, self.fine_n, self.fine_n, r.seed.wrapping_add(521)) - 0.5;
            let brush = periodic_noise(u, v, self.brush_n, 5, r.seed.wrapping_add(523)) - 0.5;
            let passes = deposit(
                uv,
                [self.trowel_n, self.trowel_n],
                0.12,
                r.seed.wrapping_add(541),
            );
            (roller, film, brush, passes)
        };
        let lap = smooth((passes - 0.35) * 1.5);
        let repair = smooth((passes - 0.65) * 6.) * a.repair_mix;
        let holes = disk(spray.radius, 0.08, self.aa) * smooth((c.pinholes - spray.dye) * 10.);
        let texture = (plateau - 0.28) * c.texture_mix * (1. - repair);
        let gain = 1. + (passes - 0.5) * c.pigment_variation - holes * 0.035;
        Texel {
            color: self
                .linear_color
                .map(|color| super::super::linear_to_srgb((color * gain).clamp(0., 1.))),
            height: r.relief_m * (texture * 0.80 + (lap - 0.5) * c.trowel * 0.22 - holes * 0.40)
                + 0.000025 * (roller * c.roller + film * a.orange_peel)
                + 0.000016 * brush * a.brush * self.resolved_brush,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::materials::{program, Surface};

    #[test]
    fn scalar_paint_preparation_has_no_grids_and_map_noise_budget_is_bounded() {
        for seed in [0, 202, 207, 43_084_584, u64::MAX] {
            let recipes = program::sample(seed);
            for surface in [Surface::Paint, Surface::Accent, Surface::Ceiling] {
                let mut r = recipes[surface as usize].clone();
                for period in [r.period_m, 10.] {
                    r.period_m = period;
                    let a = r.coating.as_ref().unwrap().application.as_ref().unwrap();
                    let scalar = a.prepare(&r);
                    assert!(scalar.spray_cells.is_none());
                    assert!(scalar.noise.is_none());
                    let prepared = scalar.cache_cells(r.seed);
                    let cell_bytes = prepared
                        .spray_cells
                        .as_ref()
                        .map_or(0, PreparedCellular::bytes);
                    let noise_bytes: usize = prepared
                        .noise
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(PreparedPeriodic::bytes)
                        .sum();
                    assert!(cell_bytes + noise_bytes <= MAX_CACHE_BYTES_PER_MAP);
                    if period == 10. {
                        assert_eq!(prepared.spray_n, 192);
                        assert!(prepared.spray_cells.is_some());
                        assert!(cell_bytes >= 194 * 194 * std::mem::size_of::<[f32; 4]>());
                    }
                }
            }
        }
    }
}
