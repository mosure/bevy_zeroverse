//! Cement paste, exposed aggregate, mineral deposits and polished stone.
use super::{
    field::*,
    periodic_noise,
    program::{MaterialRecipe, Texel},
    Surface,
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MineralRecipe {
    pub aggregate_cells: [u32; 2],
    pub aggregate_exposure: f32,
    pub aggregate_radius: f32,
    pub aggregate_color: [[f32; 3]; 2],
    pub porosity: f32,
    pub marble_mix: f32,
    pub vein_cells: [u32; 2],
    pub vein_width: f32,
    pub vein_warp: f32,
    pub vein_strength: f32,
    pub vein_color: [f32; 3],
    pub polish: f32,
    pub binder_variation: f32,
}
impl MineralRecipe {
    pub fn sample(seed: u64, surface: Surface) -> Self {
        let mut rng = stream(seed, 0x4d494e4552414c);
        let earthy = matches!(surface, Surface::Terracotta | Surface::Soil);
        let chip = [0, 1].map(|_| {
            let n = rng.random_range(0.18_f32..0.85);
            let warm = rng.random_range(-0.10_f32..0.16);
            [
                n,
                (n * (1. - warm)).min(0.95),
                (n * (1. - warm * 1.6)).min(0.95),
            ]
        });
        let n = rng.random_range(0.08..0.90);
        let warm = rng.random_range(-0.10..0.18);
        Self {
            aggregate_cells: [rng.random_range(24..=64), rng.random_range(24..=64)],
            aggregate_exposure: rng.random_range(0.0_f32..1.0).powi(2),
            aggregate_radius: rng.random_range(0.18..0.46),
            aggregate_color: chip,
            porosity: rng.random_range(0.0..if earthy { 0.32 } else { 0.16 }),
            marble_mix: if surface == Surface::Floor {
                rng.random_range(0.0_f32..1.0).sqrt()
            } else if surface == Surface::Concrete {
                rng.random_range(0.0_f32..1.0).powi(3)
            } else {
                0.
            },
            vein_cells: [rng.random_range(2..=7), rng.random_range(2..=9)],
            vein_width: rng.random_range(0.006..0.060),
            vein_warp: rng.random_range(0.02..0.35),
            vein_strength: rng.random_range(0.12..0.90),
            vein_color: [
                n,
                (n * (1. - warm)).min(0.95),
                (n * (1. - warm * 1.8)).min(0.95),
            ],
            polish: if earthy {
                0.
            } else {
                rng.random_range(0.0_f32..1.0).powi(2)
            },
            binder_variation: rng.random_range(0.02..0.15),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.aggregate_cells.iter().any(|c| !(12..=96).contains(c))
            || self.vein_cells.iter().any(|c| !(1..=16).contains(c))
            || [
                self.aggregate_exposure,
                self.aggregate_radius,
                self.porosity,
                self.marble_mix,
                self.vein_width,
                self.vein_warp,
                self.vein_strength,
                self.polish,
                self.binder_variation,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || self.vein_width <= 0.
            || self
                .aggregate_color
                .iter()
                .flatten()
                .chain(self.vein_color.iter())
                .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
        {
            return Err("invalid mineral substrate program".into());
        }
        Ok(())
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, uv: [f32; 2], floor_style: u32) -> Texel {
        let mut uv = uv;
        let mut seam = 0.;
        let mut footprint = 1. / r.map_size(floor_style) as f32;
        if r.surface == Surface::Floor && floor_style == 2 {
            let [nx, ny] = r.floor_repetitions(2);
            footprint *= nx.max(ny);
            let [u, v] = uv;
            let ix = (u * nx).floor() as u32 % nx as u32;
            let iy = (v * ny).floor() as u32 % ny as u32;
            let x = (u * nx).fract();
            let y = (v * ny).fract();
            let d = (x.min(1. - x) * r.period_m / nx).min(y.min(1. - y) * r.period_m / ny);
            let width = r.joint_width + r.period_m / r.map_size(floor_style) as f32;
            seam = (1. - d / width).clamp(0., 1.) * r.joint_width / width;
            let offset = super::hash(ix, iy, r.seed.wrapping_add(157));
            uv = [
                x + offset,
                y + super::hash(ix, iy, r.seed.wrapping_add(163)),
            ];
        }
        let [u, v] = uv;
        // Warp the packing as well as individual chip edges. Random sizes and
        // missing chips keep the aggregate from resembling a regular dot grid.
        let packed = [
            u + (periodic_noise(u, v, 11, 13, r.seed.wrapping_add(179)) - 0.5) * 0.9
                / self.aggregate_cells[0] as f32,
            v + (periodic_noise(u, v, 13, 11, r.seed.wrapping_add(181)) - 0.5) * 0.9
                / self.aggregate_cells[1] as f32,
        ];
        let cell = cellular(packed, self.aggregate_cells, r.seed);
        let edge = periodic_noise(
            u,
            v,
            self.aggregate_cells[0] * 3,
            self.aggregate_cells[1] * 3,
            r.seed.wrapping_add(191),
        );
        let radius =
            self.aggregate_radius * (0.30 + cell.dye.powi(2) * 1.10) * (0.65 + edge * 0.70);
        let chip_aa =
            (footprint * self.aggregate_cells[0].max(self.aggregate_cells[1]) as f32 * 0.55)
                .max(0.07);
        let chip = smooth((radius + chip_aa - cell.radius) / (2. * chip_aa))
            * smooth((cell.dye - 0.18) * 6.);
        let exposed = chip * self.aggregate_exposure * (1. - self.marble_mix * 0.65);
        let pores = (1. - smooth(cell.radius / 0.13)) * smooth((self.porosity - cell.dye) * 12.);
        let field = deposit(
            uv,
            self.vein_cells,
            self.vein_warp,
            r.seed.wrapping_add(239),
        );
        // Integrate thin vein coverage over the atlas footprint. A sub-texel
        // level set must fade continuously instead of becoming isolated dots.
        let aa = footprint * (self.vein_cells[0] + self.vein_cells[1]) as f32 * 0.18;
        let band = |level: f32, width: f32| {
            let filtered = width.max(aa);
            (1. - smooth((field - level).abs() / filtered)) * width / filtered
        };
        let vein = band(0.49, self.vein_width) + 0.35 * band(0.63, self.vein_width * 0.45);
        let vein = (vein * self.vein_strength * self.marble_mix).min(1.);
        let micro = periodic_noise(u, v, 97, 91, r.seed.wrapping_add(271)) - 0.5;
        let binder = periodic_noise(u, v, 9, 11, r.seed.wrapping_add(277)) - 0.5;
        let matrix = tint(r.color, 1. + binder * self.binder_variation + micro * 0.025);
        let chips = mix(self.aggregate_color[0], self.aggregate_color[1], cell.dye);
        let stone = tint(matrix, 1. + (field - 0.5) * self.marble_mix * 0.20);
        let mut color = mix(mix(stone, chips, exposed), self.vein_color, vein);
        color = tint(color, 1. - pores * 0.65 - seam * 0.35);
        Texel {
            color,
            height: r.relief_m
                * ((chip - 0.5) * self.aggregate_exposure * (1. - self.polish * 0.9)
                    + micro * (1. - self.polish) * 0.15
                    - pores * 0.9
                    + vein * (1. - self.polish) * 0.05)
                - seam * r.joint_width * 0.25,
            roughness: (r.roughness + binder * 0.06 + pores * 0.10 + seam * 0.15
                - exposed * self.polish * 0.08
                - vein * self.polish * 0.04)
                .clamp(0.12, 1.),
            occlusion: (1. - pores * 0.12 - seam * 0.05).clamp(0.8, 1.),
        }
    }
}
