//! Map-owned spatial fields: one shared, explicit allocation cap including pigment tables.
use super::*;

pub(in super::super) const FIELD_BUDGET_BYTES: usize = 1024 * 1024;

pub(in super::super) struct MineralFields {
    pub(super) aggregate: Option<PreparedCellular>,
    pub(super) aggregate_colors: Option<Vec<[f32; 3]>>,
    pub(super) noise: [PreparedPeriodic; 5],
    pub(super) veins: Option<PreparedDeposit>,
    pub(super) casting: Option<super::super::concrete::PreparedConcrete>,
    #[cfg(test)]
    pub(in super::super) allocated_bytes: usize,
}
impl MineralFields {
    pub(in super::super) fn new(
        m: &MineralRecipe,
        r: &MaterialRecipe,
        pigment: impl Fn(f32) -> [f32; 3],
    ) -> Self {
        Self::with_budget(m, r, pigment, FIELD_BUDGET_BYTES)
    }
    #[cfg(test)]
    pub(in super::super) fn bytes(&self) -> usize {
        self.aggregate.as_ref().map_or(0, PreparedCellular::bytes)
            + self
                .aggregate_colors
                .as_ref()
                .map_or(0, |v| v.capacity() * std::mem::size_of::<[f32; 3]>())
            + self
                .noise
                .iter()
                .map(PreparedPeriodic::bytes)
                .sum::<usize>()
            + self.veins.as_ref().map_or(0, PreparedDeposit::bytes)
            + self
                .casting
                .as_ref()
                .map_or(0, super::super::concrete::PreparedConcrete::bytes)
    }
    pub(in super::super) fn with_budget(
        m: &MineralRecipe,
        r: &MaterialRecipe,
        pigment: impl Fn(f32) -> [f32; 3],
        budget: usize,
    ) -> Self {
        let mut remaining = budget;
        let aggregate = PreparedCellular::with_budget(m.aggregate_cells, r.seed, &mut remaining);
        let aggregate_colors = aggregate.as_ref().and_then(|cells| {
            let bytes = m.aggregate_cells[0] as usize
                * m.aggregate_cells[1] as usize
                * std::mem::size_of::<[f32; 3]>();
            (bytes <= remaining).then(|| {
                remaining -= bytes;
                cells.dyes().map(&pigment).collect()
            })
        });
        // Cache the expensive casting-cell attributes before smaller noise grids.
        // Every allocation debits this same budget; misses retain scalar evaluation.
        let casting = m
            .casting
            .as_ref()
            .map(|c| super::super::concrete::PreparedConcrete::new(c, r, &mut remaining));
        let noise = [
            PreparedPeriodic::new([11, 13], r.seed.wrapping_add(179), &mut remaining),
            PreparedPeriodic::new([13, 11], r.seed.wrapping_add(181), &mut remaining),
            PreparedPeriodic::new(
                [m.aggregate_cells[0] * 3, m.aggregate_cells[1] * 3],
                r.seed.wrapping_add(191),
                &mut remaining,
            ),
            PreparedPeriodic::new([97, 91], r.seed.wrapping_add(271), &mut remaining),
            PreparedPeriodic::new([9, 11], r.seed.wrapping_add(277), &mut remaining),
        ];
        let veins = (m.marble_mix > 0.).then(|| {
            PreparedDeposit::new(
                m.vein_cells,
                m.vein_warp,
                r.seed.wrapping_add(239),
                &mut remaining,
            )
        });
        Self {
            aggregate,
            aggregate_colors,
            noise,
            veins,
            casting,
            #[cfg(test)]
            allocated_bytes: budget - remaining,
        }
    }
}

#[cfg(test)]
mod halo_budget_tests {
    use super::*;

    #[test]
    fn maximum_cell_halo_and_pigments_fit_the_shared_actual_capacity_budget() {
        let mut recipe = super::super::super::program::sample(202)
            [super::super::super::Surface::Concrete as usize]
            .clone();
        recipe.mineral.as_mut().unwrap().aggregate_cells = [192, 192];
        let mineral = recipe.mineral.as_ref().unwrap();
        for budget in [0, 32, FIELD_BUDGET_BYTES] {
            let fields = MineralFields::with_budget(mineral, &recipe, |dye| [dye; 3], budget);
            assert_eq!(fields.bytes(), fields.allocated_bytes);
            assert!(fields.bytes() <= budget);
            if budget == FIELD_BUDGET_BYTES {
                assert!(fields.aggregate.is_some());
                let colors = fields.aggregate_colors.as_ref().unwrap();
                assert_eq!(colors.len(), 192 * 192);
                assert_eq!(colors.capacity(), 192 * 192);
            }
        }
    }
}
