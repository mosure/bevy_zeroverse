//! Placement in the whole-rug UV chart. The first coordinate follows the row.
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WovenLayout {
    pub count: u8,
    pub first_center: [f32; 2],
    pub pitch: f32,
    pub inverse_radii: [f32; 2],
    /// Fill radius and half-width of an optional outline, in medallion radii.
    pub fill: f32,
    pub outline: f32,
    /// Framing lines measured inward from the edge of the outer border.
    pub guard_offsets: [f32; 2],
    pub guard_half_widths: [f32; 2],
    /// Relative pile relief on a high/low woven design. Zero gives flat dye.
    pub pile_contrast: f32,
}

impl WovenLayout {
    pub(super) fn sample(seed: u64, border: f32) -> Self {
        let mut rng = stream(seed, 0x574f56454e525547);
        let guards = if border == 0. || rng.random_bool(0.24) {
            0
        } else if rng.random_bool(0.62) {
            1
        } else {
            2
        };
        let mut offsets = [0.; 2];
        let mut widths = [0.; 2];
        let mut margin = border;
        for i in 0..guards {
            widths[i] = rng.random_range(0.003..0.012);
            offsets[i] = margin - border + rng.random_range(0.005..0.027) + widths[i];
            margin = border + offsets[i] + widths[i];
        }

        // Reserve the complete framing before sizing or shifting a motif.
        let field = 1. - 2. * (margin + 0.016);
        let count = if rng.random_bool(0.60) {
            1
        } else {
            rng.random_range(2..=4)
        };
        let pitch = field / f32::from(count) * rng.random_range(0.78..1.);
        let radii = [
            pitch * rng.random_range(0.22..0.46),
            field * rng.random_range(0.13..0.44),
        ];
        let half_row = f32::from(count - 1) * pitch * 0.5;
        let mut center = [0.5; 2];
        if rng.random_bool(0.61) {
            for (value, slack) in center
                .iter_mut()
                .zip([field * 0.5 - half_row - radii[0], field * 0.5 - radii[1]])
            {
                *value += slack * rng.random_range(-0.88..0.88);
            }
        }
        let outline = if rng.random_bool(0.78) {
            rng.random_range(0.032..0.105)
        } else {
            0.
        };
        Self {
            count,
            first_center: [center[0] - half_row, center[1]],
            pitch,
            inverse_radii: radii.map(f32::recip),
            fill: rng.random_range(0.62..(0.96 - 2. * outline)),
            outline,
            guard_offsets: offsets,
            guard_half_widths: widths,
            pile_contrast: if rng.random_bool(0.38) {
                rng.random_range(0.08..0.30)
            } else {
                0.
            },
        }
    }

    pub(super) fn valid(&self, border: f32) -> bool {
        if self.guard_half_widths[0] == 0. && self.guard_half_widths[1] > 0. {
            return false;
        }
        let mut margin = border;
        for (&offset, &width) in self.guard_offsets.iter().zip(&self.guard_half_widths) {
            if !(0.0..=0.16).contains(&offset) || !(0.0..=0.04).contains(&width) {
                return false;
            }
            if width > 0. {
                if border + offset - width < margin {
                    return false;
                }
                margin = border + offset + width;
            }
        }
        if !(1..=4).contains(&self.count)
            || !(0.01..=1.).contains(&self.pitch)
            || self.inverse_radii.iter().any(|v| !(2.0..=128.).contains(v))
            || !(0.3..=1.).contains(&self.fill)
            || !(0.0..=0.2).contains(&self.outline)
            || self.fill + 2. * self.outline > 1.
            || !(0.0..=0.4).contains(&self.pile_contrast)
        {
            return false;
        }
        let [x, y] = self.first_center;
        let [rx, ry] = self.inverse_radii.map(f32::recip);
        x - rx >= margin
            && x + f32::from(self.count - 1) * self.pitch + rx <= 1. - margin
            && y - ry >= margin
            && y + ry <= 1. - margin
            && (self.count == 1 || 2. * rx <= self.pitch)
    }

    /// Coverage of a finite band. Subtracting the two filtered edges preserves
    /// a subpixel line's area and makes a zero-width line exactly invisible.
    fn band(x: f32, center: f32, half_width: f32, resolution: f32) -> f32 {
        super::smooth((x - center + half_width) * resolution + 0.5)
            - super::smooth((x - center - half_width) * resolution + 0.5)
    }

    pub(super) fn center_masks(&self, uv: [f32; 2], roundness: f32, pixels: f32) -> [f32; 2] {
        let [along, cross] = uv;
        let offset = along - self.first_center[0];
        // Only the nearest medallion can contribute.
        let i = if self.count == 1 {
            0.
        } else {
            (offset / self.pitch + 0.5)
                .floor()
                .clamp(0., f32::from(self.count - 1))
        };
        let a = (offset - i * self.pitch).abs() * self.inverse_radii[0];
        let b = (cross - self.first_center[1]).abs() * self.inverse_radii[1];
        let radius = (a + b) * (1. - roundness) + (a * a + b * b).sqrt() * roundness;
        let resolution = pixels / self.inverse_radii[0].max(self.inverse_radii[1]);
        [
            super::smooth((self.fill - radius) * resolution + 0.5),
            Self::band(radius, 1. - self.outline, self.outline, resolution),
        ]
    }

    pub(super) fn guards(&self, edge_from_border: f32, pixels: f32) -> f32 {
        if self.guard_half_widths[0] == 0. {
            return 0.;
        }
        let first = Self::band(
            edge_from_border,
            self.guard_offsets[0],
            self.guard_half_widths[0],
            pixels,
        );
        if self.guard_half_widths[1] == 0. {
            return first;
        }
        first
            + Self::band(
                edge_from_border,
                self.guard_offsets[1],
                self.guard_half_widths[1],
                pixels,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::materials::rug_design::{RugDesign, WOVEN_LAYOUT_VERSION};

    #[test]
    fn rows_fit_the_frame_and_the_inactive_patterns_stay_absent() {
        let mut counts = [0; 5];
        let mut guards = [0; 3];
        let mut shifted = 0;
        for seed in 0..2048 {
            let design = RugDesign::sample(seed);
            let w = design.weave.unwrap();
            assert!(w.valid(design.border_width));
            counts[if design.center > 0. {
                usize::from(w.count)
            } else {
                0
            }] += 1;
            guards[w
                .guard_half_widths
                .iter()
                .filter(|&&width| width > 0.)
                .count()] += 1;
            if design.center == 0. {
                continue;
            }
            shifted += usize::from((w.first_center[1] - 0.5).abs() > 0.035);
            for i in 0..w.count {
                let x = w.first_center[0] + f32::from(i) * w.pitch;
                assert_eq!(
                    w.center_masks([x, w.first_center[1]], design.center_roundness, 256.),
                    [1., 0.]
                );
                if i + 1 < w.count {
                    assert_eq!(
                        w.center_masks(
                            [x + w.pitch * 0.5, w.first_center[1]],
                            design.center_roundness,
                            512.
                        ),
                        [0., 0.]
                    );
                }
            }
            assert_eq!(
                w.center_masks([0., w.first_center[1]], design.center_roundness, 256.),
                [0., 0.]
            );
        }
        assert!(counts.iter().all(|&n| n > 85), "{counts:?}");
        assert!(guards.iter().all(|&n| n > 250), "{guards:?}");
        assert!(shifted > 180 && shifted < 650, "{shifted}");
    }

    #[test]
    fn filtered_guard_keeps_subpixel_coverage_and_zero_width_is_invisible() {
        let pixels = 128.;
        let guard = 0.003;
        for phase in [0., 0.17, 0.49, 0.78] {
            let coverage: f32 = (0..16384)
                .map(|x| WovenLayout::band((x as f32 + phase) / 16384., 0.32, guard, pixels))
                .sum::<f32>()
                / 16384.;
            assert!((coverage - 2. * guard).abs() < 0.00001);
            assert_eq!(WovenLayout::band(phase, 0.32, 0., pixels), 0.);
        }
    }

    #[test]
    fn archived_dye_replays_without_sampling_a_weave_or_changing_its_palette_stream() {
        for seed in 0..32 {
            let old = RugDesign::sample_at_version(seed, WOVEN_LAYOUT_VERSION - 1);
            let mut current = RugDesign::sample(seed);
            assert!(current.weave.take().is_some());
            assert_eq!(current, old);
            let encoded = serde_json::to_value(&old).unwrap();
            assert!(encoded.get("weave").is_none());
            assert_eq!(serde_json::from_value::<RugDesign>(encoded).unwrap(), old);
        }
        let mut malformed = RugDesign::sample(17);
        malformed.weave.as_mut().unwrap().first_center[0] = -0.1;
        assert!(malformed.validate().is_err());
    }
}
