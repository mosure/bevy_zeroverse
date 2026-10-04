//! A soft, body-derived envelope for cloth. All distances are rest-body metres.
//! Cross-sections bridge small anatomical depressions without imposing a box on
//! the chest. Smooth interpolation and shoulder/hem fades avoid offset ledges.
use bevy::prelude::*;

const SECTIONS: usize = 12;

pub(crate) struct TorsoFit {
    bottom: f32,
    top: f32,
    // half width, front, back
    sections: [Vec3; SECTIONS],
    roundness: f32,
    drape: f32,
}

impl TorsoFit {
    pub fn new(vertices: &[&Vec3], waist: f32, shoulder: f32) -> Self {
        let bottom = waist - 0.06;
        let top = shoulder + 0.05;
        let step = (top - bottom) / (SECTIONS - 1) as f32;
        let mut sections = std::array::from_fn(|i| {
            let y = bottom + i as f32 * step;
            let mut section = Vec3::new(0.04, f32::INFINITY, f32::NEG_INFINITY);
            for &&p in vertices {
                if (p.y - y).abs() < step * 1.8 {
                    section.x = section.x.max(p.x.abs());
                    section.y = section.y.min(p.z);
                    section.z = section.z.max(p.z);
                }
            }
            section
        });
        // End sections can be above/below the torso rig's ownership boundary.
        for i in 0..SECTIONS {
            if !sections[i].is_finite() {
                sections[i] = (0..SECTIONS)
                    .filter(|&j| sections[j].is_finite())
                    .min_by_key(|&j| i.abs_diff(j))
                    .map(|j| sections[j])
                    .unwrap_or(Vec3::new(0.15, -0.08, 0.08));
            }
        }
        // Below the bust cloth hangs from its support instead of shrinking
        // into each rib/abdomen contour. A little taper retains a tailored hem.
        for i in (0..SECTIONS - 1).rev() {
            sections[i].y = sections[i].y.min(sections[i + 1].y + step * 0.035);
            sections[i].z = sections[i].z.max(sections[i + 1].z - step * 0.04);
        }
        for _ in 0..3 {
            let previous = sections;
            for i in 1..SECTIONS - 1 {
                sections[i] = previous[i - 1] * 0.25 + previous[i] * 0.5 + previous[i + 1] * 0.25;
            }
        }
        Self {
            bottom,
            top,
            sections,
            roundness: 2.05,
            drape: 0.55,
        }
    }

    pub fn with_program(mut self, program: &super::GarmentProgram) -> Self {
        self.roundness = program.section_roundness;
        self.drape = program.drape;
        self
    }

    pub fn delta(&self, p: Vec3, waist: f32, shoulder: f32, neck: f32) -> Vec3 {
        let t = ((p.y - self.bottom) / (self.top - self.bottom)).clamp(0.0, 1.0)
            * (SECTIONS - 1) as f32;
        let i = (t.floor() as usize).min(SECTIONS - 2);
        // Catmull-Rom keeps both value and slope continuous between sections.
        let [a, b, c, d] =
            [i.saturating_sub(1), i, i + 1, (i + 2).min(SECTIONS - 1)].map(|j| self.sections[j]);
        let f = t - i as f32;
        let section = 0.5
            * (2.0 * b
                + (-a + c) * f
                + (2.0 * a - 5.0 * b + 4.0 * c - d) * f * f
                + (-a + 3.0 * b - 3.0 * c + d) * f * f * f);
        let center = (section.y + section.z) * 0.5;
        let radii = Vec2::new(section.x, (section.z - section.y) * 0.5).max(Vec2::splat(0.025));
        let q = Vec2::new(p.x, p.z - center);
        let normalized = (q / (radii * 1.015)).abs();
        let radius = (normalized.x.powf(self.roundness) + normalized.y.powf(self.roundness))
            .powf(1.0 / self.roundness);
        let correction = if (0.1..1.).contains(&radius) {
            q * (radius.recip() - 1.)
        } else {
            Vec2::ZERO
        };
        // Saturate gradually instead of a hard 5 cm clamp. At the hem and
        // shoulder the fitted envelope fades into the original body's shell.
        let length = correction.length();
        let mut correction = correction * (0.09 / (0.09 + length));
        // A radial expansion alone retains the cleavage groove and makes a
        // shirt look painted onto the body. Bridge its supported front panel
        // in depth while keeping the side and shoulder fades smooth.
        let x = (p.x.abs() / radii.x).min(1.0);
        let front = section.y + radii.y * smooth(0.45, 0.98, x).powf(1.3);
        let bridge = -(p.z - front).max(0.0);
        let front_mask = 1. - smooth(center - 0.025, center + 0.025, p.z);
        correction.y = correction
            .y
            .min(bridge * (0.075 / (0.075 + bridge.abs())) * front_mask);
        let mask = smooth(waist, waist + 0.08, p.y)
            * (1.0 - smooth(shoulder - 0.045, neck - 0.005, p.y))
            * (1.0 - smooth(radii.x * 0.84, radii.x * 1.16, p.x.abs()));
        Vec3::new(correction.x, 0.0, correction.y) * mask * (0.70 + self.drape * 0.30)
    }
}

pub(super) fn smooth(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo).max(1e-5)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitted_cloth_has_no_chest_or_hem_ledge() {
        let points: Vec<_> = (0..30)
            .flat_map(|y| {
                (0..32).map(move |a| {
                    let angle = a as f32 * std::f32::consts::TAU / 32.0;
                    Vec3::new(
                        angle.cos() * 0.22,
                        0.9 + y as f32 * 0.02,
                        angle.sin() * 0.12,
                    )
                })
            })
            .collect();
        let fit = TorsoFit::new(&points.iter().collect::<Vec<_>>(), 1.0, 1.4);
        let mut last = Vec3::ZERO;
        let mut max_delta: f32 = 0.0;
        for i in 0..800 {
            let p = Vec3::new(0.05, 0.8 + i as f32 * 0.001, -0.085);
            let d = fit.delta(p, 1.0, 1.4, 1.5);
            assert!(d.is_finite() && d.length() < 0.09);
            assert!((d - last).length() < 0.0015, "cloth discontinuity at {p:?}");
            max_delta = max_delta.max(d.length());
            last = d;
        }
        assert!(
            max_delta > 0.009,
            "fit must still bridge anatomical grooves"
        );
    }
}
