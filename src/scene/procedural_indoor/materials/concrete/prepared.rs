//! Immutable casting fields sharing the mineral atlas's single allocation budget.
use super::*;

pub(in super::super) struct PreparedConcrete {
    pub(super) boards: u32,
    pub(super) seam_width: f32,
    pub(super) board_footprint: f32,
    pub(super) void_n: u32,
    pub(super) void_radius: f32,
    pub(super) void_footprint: f32,
    pub(super) noise: [PreparedPeriodic; 4],
    pub(super) passes: PreparedDeposit,
    pub(super) cure: PreparedDeposit,
    pub(super) voids: Option<PreparedCellular>,
    board_tones: Option<Vec<f32>>,
}

impl PreparedConcrete {
    pub(in super::super) fn new(
        c: &ConcreteFinish,
        r: &MaterialRecipe,
        remaining: &mut usize,
    ) -> Self {
        let boards = frequency(r.period_m, c.board_width_m);
        let pass_n = frequency(r.period_m, c.trowel_scale_m);
        let void_n = frequency(r.period_m, c.bughole_radius_m * 14.);
        let voids =
            PreparedCellular::with_budget([void_n, void_n], r.seed.wrapping_add(643), remaining);
        let bytes = boards as usize * std::mem::size_of::<f32>();
        let board_tones = (bytes <= *remaining).then(|| {
            *remaining -= bytes;
            (0..boards)
                .map(|board| {
                    (super::super::hash(0, board, r.seed.wrapping_add(615)) - 0.5)
                        * c.formwork
                        * 0.12
                })
                .collect()
        });
        Self {
            boards,
            seam_width: c.seam_width_m / r.period_m * boards as f32,
            board_footprint: boards as f32 / r.map_size(0) as f32,
            void_n,
            void_radius: c.bughole_radius_m / r.period_m * void_n as f32,
            void_footprint: void_n as f32 / r.map_size(0) as f32,
            noise: [
                PreparedPeriodic::new([3, 5], r.seed.wrapping_add(607), remaining),
                PreparedPeriodic::new([4, boards * 19], r.seed.wrapping_add(613), remaining),
                PreparedPeriodic::new([21, 3], r.seed.wrapping_add(631), remaining),
                PreparedPeriodic::new([173, 167], r.seed.wrapping_add(641), remaining),
            ],
            passes: PreparedDeposit::new(
                [pass_n, pass_n],
                0.16,
                r.seed.wrapping_add(617),
                remaining,
            ),
            cure: PreparedDeposit::new([3, 5], 0.17, r.seed.wrapping_add(619), remaining),
            voids,
            board_tones,
        }
    }

    #[cfg(test)]
    pub(in super::super) fn bytes(&self) -> usize {
        self.voids.as_ref().map_or(0, PreparedCellular::bytes)
            + self
                .board_tones
                .as_ref()
                .map_or(0, |v| v.len() * std::mem::size_of::<f32>())
            + self
                .noise
                .iter()
                .map(PreparedPeriodic::bytes)
                .sum::<usize>()
            + self.passes.bytes()
            + self.cure.bytes()
    }

    pub(super) fn board_tone(&self, c: &ConcreteFinish, r: &MaterialRecipe, board: u32) -> f32 {
        self.board_tones.as_ref().map_or_else(
            || (super::super::hash(0, board, r.seed.wrapping_add(615)) - 0.5) * c.formwork * 0.12,
            |tones| tones[board as usize],
        )
    }
}
