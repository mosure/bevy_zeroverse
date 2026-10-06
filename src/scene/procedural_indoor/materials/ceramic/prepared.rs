//! Map-owned fired ceramic fields and pigment endpoints. The caller prepares
//! only the selected coating branch, so one atlas owns one shared cache cap.
use super::*;

// Match the existing mineral field cap; all cellular halos and noise grids
// debit the same actual allocated-capacity budget, including sparse fallbacks.
pub(super) const FIELD_BUDGET_BYTES: usize = 1024 * 1024;

pub(in super::super) struct PreparedGlaze {
    turns: u32,
    turn_footprint: f32,
    speck_count: u32,
    speck_radius: f32,
    speck_footprint: f32,
    crack_count: u32,
    crack_width: f32,
    crack_footprint: f32,
    matrix_color: [f32; 3],
    reactive_color: [f32; 3],
    speckle_color: [f32; 3],
    speckles: Option<PreparedCellular>,
    cracks: Option<PreparedCellular>,
    noise: [PreparedPeriodic; 2],
    melt: PreparedDeposit,
    #[cfg(test)]
    allocated_bytes: usize,
}

impl PreparedGlaze {
    pub(in super::super) fn new(
        g: &GlazeRecipe,
        c: &CoatingRecipe,
        r: &MaterialRecipe,
    ) -> Option<Self> {
        Self::with_budget(g, c, r, FIELD_BUDGET_BYTES)
    }

    pub(super) fn with_budget(
        g: &GlazeRecipe,
        c: &CoatingRecipe,
        r: &MaterialRecipe,
        budget: usize,
    ) -> Option<Self> {
        // The original scalar expression owns exceptional authored inputs,
        // including NaN operand/payload propagation through tint and mix.
        // Do not validate unrelated coating fields or reject a glaze merely
        // because an unused authored application is present.
        if g.validate().is_err()
            || !r.period_m.is_finite()
            || r.period_m <= 0.
            || !r.relief_m.is_finite()
            || r.relief_m < 0.
            || !r.roughness.is_finite()
            || !(0.0..=1.).contains(&r.roughness)
            || [c.pigment_variation, c.gloss, c.crackle]
                .into_iter()
                .any(|v| !v.is_finite() || !(0.0..=1.).contains(&v))
        {
            return None;
        }
        let size = r.map_size(0) as f32;
        let turns = frequency(r.period_m, g.turning_pitch_m);
        let speck_count = frequency(r.period_m, g.speckle_radius_m * 12.);
        let speck_radius = g.speckle_radius_m / r.period_m * speck_count as f32;
        let crack_count = frequency(r.period_m, g.crack_spacing_m);
        let crack_width = 0.000028 / r.period_m * crack_count as f32;
        if !speck_radius.is_finite() || !crack_width.is_finite() {
            return None;
        }
        let mut remaining = budget;
        // Speckles need only their primary neighbor; cracks consume both
        // distances. Allocate no crack halo when the selected recipe is clear.
        let speckles = PreparedCellular::with_budget(
            [speck_count, speck_count],
            r.seed.wrapping_add(431),
            &mut remaining,
        );
        let cracks = (c.crackle > 0.)
            .then(|| {
                PreparedCellular::with_budget(
                    [crack_count, crack_count],
                    r.seed.wrapping_add(439),
                    &mut remaining,
                )
            })
            .flatten();
        let grains = frequency(r.period_m, g.body_grain_m);
        let noise = [
            PreparedPeriodic::new([grains, grains], r.seed.wrapping_add(409), &mut remaining),
            PreparedPeriodic::new([7, 3], r.seed.wrapping_add(419), &mut remaining),
        ];
        let count = frequency(r.period_m, g.cloud_scale_m);
        let melt = PreparedDeposit::new(
            [count, count],
            0.12 + g.flow * 0.18,
            r.seed.wrapping_add(401),
            &mut remaining,
        );
        Some(Self {
            turns,
            turn_footprint: turns as f32 / size,
            speck_count,
            speck_radius,
            speck_footprint: speck_count as f32 / size,
            crack_count,
            crack_width,
            crack_footprint: crack_count as f32 / size,
            matrix_color: [0.975; 3].map(super::super::srgb_to_linear),
            reactive_color: g.reactive_color.map(super::super::srgb_to_linear),
            speckle_color: g.speckle_color.map(super::super::srgb_to_linear),
            speckles,
            cracks,
            noise,
            melt,
            #[cfg(test)]
            allocated_bytes: budget - remaining,
        })
    }

    pub(in super::super) fn evaluate(
        &self,
        g: &GlazeRecipe,
        c: &CoatingRecipe,
        r: &MaterialRecipe,
        uv: [f32; 2],
    ) -> Texel {
        let [u, v] = uv;
        if !u.is_finite() || !v.is_finite() {
            return g.evaluate(c, r, uv);
        }
        let melt = self.melt.sample(uv);
        let pool = smooth((melt - 0.25) * 1.5) * g.thickness_variation;
        let body = self.noise[0].sample(u, v) - 0.5;
        let turn_warp = (self.noise[1].sample(u, v) - 0.5) * 0.30;
        let rings = band(
            ((v * self.turns as f32 + turn_warp).rem_euclid(1.) - 0.5).abs(),
            0.16,
            self.turn_footprint,
        ) - 0.32;
        let speck = self.speckles.as_ref().map_or_else(
            || {
                cellular(
                    uv,
                    [self.speck_count, self.speck_count],
                    r.seed.wrapping_add(431),
                )
                .into()
            },
            |cells| cells.sample_primary(uv),
        );
        let radius = self.speck_radius * (0.6 + speck.dye);
        let inclusion = disk(speck.radius, radius, self.speck_footprint)
            * smooth((g.speckle_density - speck.dye) * 18.);
        let cracks = if c.crackle > 0. {
            let cell = self.cracks.as_ref().map_or_else(
                || {
                    cellular(
                        uv,
                        [self.crack_count, self.crack_count],
                        r.seed.wrapping_add(439),
                    )
                },
                |cells| cells.sample_edges(uv),
            );
            band(cell.edge * 0.5, self.crack_width, self.crack_footprint) * c.crackle
        } else {
            0.
        };
        let matrix_gain = 1. + body * c.pigment_variation - pool * 0.10;
        let matrix = self
            .matrix_color
            .map(|c| super::super::linear_to_srgb((c * matrix_gain).clamp(0., 1.)));
        let reactive = smooth((melt - 0.30) * 1.7) * g.reactive_mix;
        // Retain both intermediate encodes and their subsequent decodes. Only
        // fixed endpoints are hoisted; f32 sRGB round trips are not identities.
        let melted = mixed(matrix, self.reactive_color, reactive);
        let speckled = mixed(melted, self.speckle_color, inclusion);
        let color = tint(speckled, 1. - cracks * 0.30);
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

    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.speckles.as_ref().map_or(0, PreparedCellular::bytes)
            + self.cracks.as_ref().map_or(0, PreparedCellular::bytes)
            + self
                .noise
                .iter()
                .map(PreparedPeriodic::bytes)
                .sum::<usize>()
            + self.melt.bytes()
    }
}

fn mixed(color: [f32; 3], endpoint: [f32; 3], amount: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| {
        super::super::linear_to_srgb(
            super::super::srgb_to_linear(color[i]) * (1. - amount) + endpoint[i] * amount,
        )
    })
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn glaze_halos_and_noise_debit_one_actual_capacity_budget() {
        for seed in [202, 43_084_482] {
            let mut r = super::super::super::program::sample(seed)
                [super::super::super::Surface::Ceramic as usize]
                .clone();
            for period in [r.period_m, 10.] {
                r.period_m = period;
                for crackle in [0., 0.3] {
                    r.coating.as_mut().unwrap().crackle = crackle;
                    let c = r.coating.as_ref().unwrap();
                    let g = c.glaze.as_ref().unwrap();
                    for budget in [0, 32, 16 * 1024, FIELD_BUDGET_BYTES] {
                        let p = PreparedGlaze::with_budget(g, c, &r, budget).unwrap();
                        assert_eq!(p.bytes(), p.allocated_bytes);
                        assert!(p.bytes() <= budget);
                        assert!(crackle > 0. || p.cracks.is_none());
                        if period == 10. && budget == FIELD_BUDGET_BYTES {
                            assert_eq!(p.speck_count, 192);
                            assert!(p.speckles.is_some());
                            // Two maximum halos exceed the one-map budget.
                            assert!(p.cracks.is_none());
                        }
                    }
                }
            }
        }
    }
}
