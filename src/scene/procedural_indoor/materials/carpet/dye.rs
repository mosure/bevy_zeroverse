//! Space-dyed, heathered and patterned commercial carpet. All fields tile and
//! remain smooth at atlas edges. Geometry/annotation targets are unaffected.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarpetDye {
    /// Heather, striae, offset weave, clouds, stipple and yarn blocks.
    pub weights: [f32; 6],
    /// Shared low-frequency dye/warp domain; independent of tuft size.
    pub dye_cells: [u32; 2],
    /// Nap, slub and fleck frequency pairs, in the atlas domain.
    pub grain_cells: [u32; 6],
    pub warp: f32,
    pub brokenness: f32,
    /// Integer lattice shear preserves exact wrapping without axis lock-in.
    pub shear: i32,
}
impl CarpetDye {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x4341525045544459);
        let anchor = rng.random_range(0..6);
        let spread = rng.random_range(0.02..0.45);
        let mut weights = std::array::from_fn(|_| rng.random_range(0.0..spread));
        weights[anchor] += rng.random_range(0.55..1.);
        let sum: f32 = weights.iter().sum();
        Self {
            weights: weights.map(|w| w / sum),
            dye_cells: [rng.random_range(2..=12), rng.random_range(2..=12)],
            grain_cells: [
                rng.random_range(96..=220),
                rng.random_range(96..=220),
                rng.random_range(3..=23),
                rng.random_range(32..=96),
                rng.random_range(18..=84),
                rng.random_range(18..=84),
            ],
            warp: rng.random_range(0.0..0.8),
            brokenness: rng.random_range(0.0..1.),
            shear: rng.random_range(-1..=1),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self
            .weights
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.).contains(x))
            || (self.weights.iter().sum::<f32>() - 1.).abs() > 1e-5
            || self.dye_cells.iter().any(|x| !(1..=16).contains(x))
            || self.grain_cells.iter().any(|x| !(1..=240).contains(x))
            || !(0.0..=1.).contains(&self.warp)
            || !(0.0..=1.).contains(&self.brokenness)
            || !(-1..=1).contains(&self.shear)
        {
            return Err("invalid carpet dye mixture, frequencies or shear".into());
        }
        Ok(())
    }

    pub(super) fn mask(&self, c: &CarpetDesign, r: &MaterialRecipe, [u, v]: [f32; 2]) -> f32 {
        let smooth = |x: f32| {
            let x = x.clamp(0., 1.);
            x * x * (3. - 2. * x)
        };
        let a = periodic_noise(
            u,
            v,
            self.dye_cells[0],
            self.dye_cells[1],
            r.seed.wrapping_add(743),
        );
        let b = periodic_noise(
            u,
            v,
            self.dye_cells[1],
            self.dye_cells[0],
            r.seed.wrapping_add(751),
        );
        let nx = c.repeats[0] as f32;
        let ny = c.repeats[1] as f32;
        let row = (v * ny).floor() as u32 % c.repeats[1];
        let shift = hash(row, 0, r.seed.wrapping_add(757)) * self.brokenness;
        let x = ((u + self.shear as f32 * v) * nx + (a - 0.5) * self.warp).rem_euclid(1.);
        let y = (v * ny).rem_euclid(1.);
        // Filtering also covers the shear and warp footprint at the highest
        // supported pattern frequencies. No hard step-shaped normal ridges.
        let aa = (nx * (1. + self.shear.abs() as f32)
            + ny
            + self.dye_cells[0].max(self.dye_cells[1]) as f32 * self.warp)
            / r.map_size(1) as f32;
        let stripe = smooth((c.line_width - (x - 0.5).abs()) / aa + 0.5);
        let weave = smooth(
            (c.line_width * 0.45 - ((x + shift).fract() - 0.5).abs().min((y - 0.5).abs())) / aa
                + 0.5,
        );
        let block = hash(
            (u * nx).floor() as u32 % c.repeats[0],
            row,
            r.seed.wrapping_add(761),
        );
        let fields = [
            a,
            stripe,
            weave,
            smooth(a * 0.65 + b * 0.35),
            smooth((a - (0.38 + c.line_width * 0.5)) * 6.),
            block,
        ];
        fields
            .into_iter()
            .zip(self.weights)
            .map(|(a, w)| a * w)
            .sum::<f32>()
            .clamp(0., 1.)
    }
}
