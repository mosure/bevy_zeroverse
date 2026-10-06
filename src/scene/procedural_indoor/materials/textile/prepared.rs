//! Exact, map-owned weave fields. Every heap table shares one fixed budget.
use super::super::{field::PreparedPeriodic, Surface};
use super::*;

pub(in super::super) const TABLE_BUDGET_BYTES: usize = 64 * 1024;

struct Yarn {
    jitter: f32,
    pigment: [f32; 3],
}

pub(in super::super) struct PreparedTextile {
    recipe: TextileRecipe,
    yarns: Option<[Vec<Yarn>; 2]>,
    noise: [PreparedPeriodic; 5],
    #[cfg(test)]
    allocated_bytes: usize,
}
impl PreparedTextile {
    pub(in super::super) fn new(r: &MaterialRecipe) -> Self {
        Self::with_budget(r, TABLE_BUDGET_BYTES)
    }
    pub(super) fn with_budget(r: &MaterialRecipe, budget: usize) -> Self {
        // The scalar dispatch resolves an absent textile and carpet pile this
        // same way. Resolve it once; no sampled recipe or distribution changes.
        let mut recipe = r
            .textile
            .clone()
            .unwrap_or_else(|| TextileRecipe::sample(r.seed));
        if r.surface == Surface::Floor && recipe.pile == 0. {
            recipe.pile = 0.85;
        }
        let mut remaining = budget;
        let bytes = recipe
            .yarns
            .into_iter()
            .try_fold(0usize, |total, n| total.checked_add(n as usize))
            .and_then(|count| count.checked_mul(std::mem::size_of::<Yarn>()));
        let yarns = bytes.filter(|bytes| *bytes <= remaining).map(|bytes| {
            remaining -= bytes;
            [0, 1].map(|axis| {
                (0..recipe.yarns[axis])
                    .map(|index| {
                        let dye = hash(index, [13, 19][axis], r.seed);
                        let band = band(&recipe, r, axis, index);
                        Yarn {
                            jitter: hash(index, axis as u32, r.seed) - 0.5,
                            pigment: recipe.yarn_tint[axis].map(|color| {
                                color * (0.94 + recipe.dye_variation * (dye - 0.5)) * band
                            }),
                        }
                    })
                    .collect()
            })
        });
        let noise = [
            PreparedPeriodic::new(
                [recipe.yarns[0], 3],
                r.seed.wrapping_add(11),
                &mut remaining,
            ),
            PreparedPeriodic::new(
                [3, recipe.yarns[1]],
                r.seed.wrapping_add(17),
                &mut remaining,
            ),
            PreparedPeriodic::new([3, 3], r.seed.wrapping_add(83), &mut remaining),
            PreparedPeriodic::new([79, 83], r.seed.wrapping_add(73), &mut remaining),
            PreparedPeriodic::new([37, 41], r.seed.wrapping_add(101), &mut remaining),
        ];
        Self {
            recipe,
            yarns,
            noise,
            #[cfg(test)]
            allocated_bytes: budget - remaining,
        }
    }
    pub(in super::super) fn evaluate(&self, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
        self.recipe.evaluate_prepared(r, uv, Some(self))
    }
    pub(super) fn noise(&self, index: usize, [u, v]: [f32; 2]) -> f32 {
        self.noise[index].sample(u, v)
    }
    pub(super) fn jitter(&self, r: &MaterialRecipe, axis: usize, index: u32) -> f32 {
        self.yarns.as_ref().map_or_else(
            || hash(index, axis as u32, r.seed) - 0.5,
            |yarns| yarns[axis][index as usize].jitter,
        )
    }
    pub(super) fn pigment(&self, r: &MaterialRecipe, axis: usize, index: u32) -> [f32; 3] {
        self.yarns.as_ref().map_or_else(
            || {
                let dye = hash(index, [13, 19][axis], r.seed);
                let band = band(&self.recipe, r, axis, index);
                self.recipe.yarn_tint[axis]
                    .map(|color| color * (0.94 + self.recipe.dye_variation * (dye - 0.5)) * band)
            },
            |yarns| yarns[axis][index as usize].pigment,
        )
    }
    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.yarns.as_ref().map_or(0, |yarns| {
            yarns
                .iter()
                .map(|v| v.len() * std::mem::size_of::<Yarn>())
                .sum()
        }) + self
            .noise
            .iter()
            .map(PreparedPeriodic::bytes)
            .sum::<usize>()
    }
    #[cfg(test)]
    pub(super) fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
}

fn band(t: &TextileRecipe, r: &MaterialRecipe, axis: usize, index: u32) -> f32 {
    if let Some(l) = &r.layers {
        let count = l.bands[axis];
        if count > 0 && (index * count / t.yarns[axis]).is_multiple_of(3) {
            return 1. - l.stripe_strength * 0.25;
        }
    }
    1.
}
