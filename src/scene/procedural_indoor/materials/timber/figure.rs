//! Longitudinal cuts in furniture stock. Growth rings, vessel lumens and rays
//! contribute differently to pigment and to the relief of the finished board.
use super::{periodic_noise, smooth, wrapped, MaterialRecipe, WoodFinish};
use crate::scene::procedural_indoor::{
    layout::stream,
    materials::program::{PreparedWood, Texel},
};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FurnitureGrain {
    /// Nominal growth intervals for a board at the reference grain frequency of 40.
    pub rings: u32,
    pub vessel_cells: [u32; 2],
    /// Signed deflection around the pith, from straight grain to cathedral figure.
    pub crown: f32,
    pub crown_width: f32,
    pub wander: f32,
    pub latewood_width: f32,
    pub center: [f32; 2],
    pub knot_center: [f32; 2],
    pub knot: f32,
    pub vessel_strength: f32,
    /// Fraction concentrating visible vessels in the earlywood of each ring.
    pub ring_porosity: f32,
    pub ray_strength: f32,
}

impl FurnitureGrain {
    pub(super) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x4752_4149_4e43_5554);
        let cut: f32 = rng.random();
        let pore: f32 = rng.random();
        Self {
            rings: rng.random_range(10..=48),
            vessel_cells: [rng.random_range(85..=126), rng.random_range(17..=45)],
            crown: cut
                * cut
                * rng.random_range(0.22..0.42)
                * if rng.random_bool(0.5) { -1. } else { 1. },
            crown_width: rng.random_range(0.19..0.44),
            wander: rng.random_range(0.022..0.114),
            latewood_width: rng.random_range(0.12..0.34),
            center: rng.random(),
            knot_center: rng.random(),
            knot: if rng.random_bool(0.32) {
                rng.random_range(0.18..1.)
            } else {
                0.
            },
            vessel_strength: pore * pore,
            ring_porosity: rng.random::<f32>() * pore,
            ray_strength: rng.random::<f32>().powi(2) * (1. - cut),
        }
    }

    /// Whole intervals across the baked tile. The upper limit leaves about
    /// five base texels per interval in the 256-pixel furniture atlas.
    pub fn rings_across(&self, r: &MaterialRecipe) -> u32 {
        (self.rings as f32 * (0.5 + r.grain_frequency / 80.))
            .round()
            .clamp(4., 52.) as u32
    }

    /// Nominal projected spacing on the cut face before the local grain bends.
    pub fn ring_pitch_m(&self, r: &MaterialRecipe) -> f32 {
        let period = r.period_uv();
        let across = if r.layers.as_ref().is_some_and(|l| l.quarter_turn % 2 == 1) {
            period.y
        } else {
            period.x
        };
        across / self.rings_across(r) as f32
    }

    pub(super) fn valid(&self) -> bool {
        (4..=64).contains(&self.rings)
            && self.vessel_cells.iter().all(|v| (8..=128).contains(v))
            && (-0.45..=0.45).contains(&self.crown)
            && (0.12..=0.49).contains(&self.crown_width)
            && (0.0..=0.16).contains(&self.wander)
            && (0.07..=0.36).contains(&self.latewood_width)
            && [
                self.knot,
                self.vessel_strength,
                self.ring_porosity,
                self.ray_strength,
            ]
            .iter()
            .chain(&self.center)
            .chain(&self.knot_center)
            .all(|x| (0.0..=1.).contains(x))
    }

    pub(super) fn evaluate(
        &self,
        r: &MaterialRecipe,
        finish: &WoodFinish,
        uv: [f32; 2],
        stock: &PreparedWood,
    ) -> Texel {
        let [u, v] = uv;
        let noise = |x, y, n, m, salt| periodic_noise(x, y, n, m, r.seed.wrapping_add(salt));
        let ellipse = |center: [f32; 2], width, length| {
            let dx: f32 = wrapped(u - center[0]) / width;
            let dy: f32 = wrapped(v - center[1]) / length;
            (1. - dx * dx - dy * dy).max(0.).powi(2)
        };
        let drift = noise(u, v, 4, 2, 311) - 0.5;
        let knot = self.knot * ellipse(self.knot_center, 0.071, 0.098);
        let track = u
            + self.crown * ellipse(self.center, self.crown_width, 0.48)
            + stock.wander * drift
            + 0.075 * knot;
        let growth = noise(track, v, 7, 3, 317) - 0.5;
        let age = track * stock.rings as f32 + growth * 0.72;
        let year = (age.floor() as i32).rem_euclid(stock.rings as i32) as u32;
        let phase = age.rem_euclid(1.);
        // Each year has its own latewood peak within the growth interval and a
        // separate latewood density. Both shoulders meet the earlywood smoothly.
        let growth_rate = super::hash(year, 11, r.seed.wrapping_add(353));
        let density = super::hash(year, 23, r.seed.wrapping_add(359));
        let peak = 0.50 + growth_rate * 0.23;
        let width = (self.latewood_width * (0.60 + density * 0.55)).min(0.99 - peak);
        let late = smooth(1. - (phase - peak).abs() / width) * (0.45 + density * 0.77);
        let early = smooth(1. - wrapped(phase - 0.16).abs() / 0.38);
        let fibres = noise(track, v, 89, 9, 331) - 0.5;
        let vessel =
            smooth((noise(track, v, self.vessel_cells[0], self.vessel_cells[1], 337) - 0.56) * 4.3)
                * self.vessel_strength
                * (1. - self.ring_porosity * (1. - early));
        // Radial flecks are most visible on quarter-sawn stock and cross the axial fibres.
        let ray = smooth((noise(track, v, 13, 81, 347) - 0.71) * 5.) * self.ray_strength;
        let detail = r.detail_scale.unwrap_or(1.);
        let absorbed =
            (finish.stain_strength * (1. + (0.28 * vessel - 0.07 * late) * detail)).clamp(0., 1.);
        // Interlocked fibre bundles change colour gradually between growth
        // bands. A finished face rarely shows a dark line at every boundary.
        let bundle = noise(track, v, 23, 4, 367) - 0.5;
        let figure = (0.17 - late) * (0.28 + 0.38 * self.ring_porosity + 0.35 * growth)
            + growth * 0.52
            + bundle * 0.64
            + fibres * 0.40
            - 0.30 * vessel
            + 0.22 * ray
            - 0.73 * knot;
        let gain = (1. + figure * r.contrast * finish.ring_contrast * 1.65).clamp(0.48, 1.45);
        let open = 1. - finish.pore_fill;
        Texel {
            color: stock.figure_color(absorbed, gain),
            height: r.relief_m
                * ((0.09 * late + 0.12 * fibres) * open - 0.86 * vessel * open - 0.07 * knot),
            roughness: (r.roughness
                + detail
                    * (1. - finish.clearcoat * 0.77)
                    * (0.19 * vessel * open + 0.06 * fibres - 0.032 * late - 0.075 * ray))
                .clamp(0.12, 0.95),
            occlusion: 1. - open * vessel * 0.065 * detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::materials::{program, Surface};

    #[test]
    fn furniture_stock_is_shared_replayable_and_covers_sawn_and_pore_continua() {
        let mut counts = [0; 6];
        for seed in 0..384 {
            let palette = program::sample(seed);
            let board = &palette[Surface::Wood as usize];
            let edge = &palette[Surface::WoodEdge as usize];
            assert!(palette[Surface::Floor as usize]
                .wood
                .as_ref()
                .unwrap()
                .figure
                .is_none());
            for slot in 0..3 {
                let r = board.variant(slot);
                let w = r.wood.as_ref().unwrap();
                let f = w.figure.unwrap();
                assert_eq!(edge.variant(slot).wood.unwrap().figure, Some(f));
                assert_eq!(f, FurnitureGrain::sample(w.seed));
                r.validate().unwrap();
                if slot == 0 {
                    for (n, yes) in counts.iter_mut().zip([
                        f.crown.abs() < 0.025,
                        f.crown.abs() > 0.18,
                        f.vessel_strength < 0.07,
                        f.vessel_strength > 0.6,
                        f.knot == 0.,
                        f.ray_strength > 0.32,
                    ]) {
                        *n += u32::from(yes);
                    }
                }
            }
        }
        assert!(counts.into_iter().all(|n| n >= 24), "{counts:?}");
    }

    #[test]
    fn surfaces_tile_and_finish_controls_relief_separately_from_pigment() {
        let mut r = program::sample(202).remove(Surface::Wood as usize);
        let mut w = r.wood.take().unwrap();
        let f = w.figure.as_mut().unwrap();
        f.knot = 0.;
        f.vessel_strength = 1.;
        f.ring_porosity = 0.83;
        let mut filled = w.clone();
        w.pore_fill = 0.;
        filled.pore_fill = 1.;
        let mut tiled = r.clone();
        tiled.wood = Some(w.clone());
        let mut recessed = 0usize;
        for y in 0..32 {
            for x in 0..32 {
                let uv = [(x as f32 + 0.5) / 32., (y as f32 + 0.5) / 32.];
                let t = w.evaluate(&r, uv);
                let sealed = filled.evaluate(&r, uv);
                assert_eq!(t.color, sealed.color);
                assert_eq!(sealed.height, 0.);
                assert_eq!(sealed.occlusion, 1.);
                if t.occlusion < 0.99 {
                    assert!(t.height < 0.);
                    recessed += 1;
                }
                let base = tiled.texel(uv[0], uv[1], 0);
                for shift in [
                    [uv[0] + 1., uv[1]],
                    [uv[0], uv[1] - 1.],
                    [uv[0] - 17., uv[1] + 32.],
                ] {
                    let repeated = tiled.texel(shift[0], shift[1], 0);
                    program::replay_helpers::same_bits(&base, &repeated);
                }
            }
        }
        assert!(recessed > 35);
        r.detail_scale = Some(0.);
        r.contrast = 0.;
        r.relief_m = 0.;
        let t = w.evaluate(&r, [0., 0.]);
        for uv in [[0.32, 0.58], [0.81, 0.21], [0.67, 0.27]] {
            let s = w.evaluate(&r, uv);
            assert_eq!(
                (t.color, t.height, t.roughness, t.occlusion),
                (s.color, s.height, s.roughness, s.occlusion)
            );
        }
    }
}
