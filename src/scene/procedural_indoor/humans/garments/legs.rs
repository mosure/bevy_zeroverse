//! A rest-body trouser envelope. Calf taper is a sewn cut, not muscle topology.
use super::{fit::smooth, GarmentProgram};
use bevy::prelude::*;

const RINGS: usize = 12;
pub(in crate::scene::procedural_indoor::humans) struct LegFit {
    ankle: Vec3,
    up: Vec3,
    right: Vec3,
    back: Vec3,
    length: f32,
    radii: [Vec2; RINGS],
    straightness: f32,
    hem_width: f32,
}
impl LegFit {
    pub fn new(vertices: &[Vec3], hip: Vec3, ankle: Vec3, program: &GarmentProgram) -> Self {
        let length = hip.distance(ankle).max(0.1);
        let up = (hip - ankle) / length;
        let right = (Vec3::X - up * up.x).normalize_or(Vec3::X);
        let back = right.cross(up);
        let radii = std::array::from_fn(|i| {
            let height = i as f32 / (RINGS - 1) as f32;
            let mut radii = Vec2::splat(0.026);
            for &p in vertices {
                let d = p - ankle;
                if (d.dot(up) / length - height).abs() < 0.11 {
                    radii = radii.max(Vec2::new(d.dot(right).abs(), d.dot(back).abs()));
                }
            }
            radii
        });
        Self {
            ankle,
            up,
            right,
            back,
            length,
            radii,
            straightness: program.leg_straightness,
            hem_width: program.hem_width,
        }
    }
    pub fn delta(&self, p: Vec3, cuff: (Vec3, Vec3), waist: f32) -> Vec3 {
        let d = p - self.ankle;
        let height = (d.dot(self.up) / self.length).clamp(0.0, 1.0);
        let ring = height * (RINGS - 1) as f32;
        let i = (ring.floor() as usize).min(RINGS - 2);
        let actual = self.radii[i].lerp(self.radii[i + 1], smooth(0., 1., ring - i as f32));
        let calf = self.radii[2..7].iter().copied().fold(Vec2::ZERO, Vec2::max);
        let target = actual
            .max(calf * self.hem_width)
            .lerp(actual, smooth(0.38, 0.70, height))
            + Vec2::splat(0.003);
        let q = Vec2::new(d.dot(self.right), d.dot(self.back));
        let radius = (q / target.max(Vec2::splat(0.025))).length();
        if !(0.1..1.0).contains(&radius) {
            return Vec3::ZERO;
        }
        let correction = q * (radius.recip() - 1.);
        let length = correction.length();
        let correction = correction * (0.035 / (0.035 + length));
        let coverage = -(p - cuff.0).dot(cuff.1);
        let mask = smooth(0., 0.018, coverage) * (1. - smooth(waist - 0.09, waist, p.y));
        (self.right * correction.x + self.back * correction.y) * mask * self.straightness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn straight_trousers_bridge_the_calf_and_keep_bare_ankles_unchanged() {
        let vertices: Vec<_> = (0..25)
            .flat_map(|i| {
                (0..32).map(move |j| {
                    let y = i as f32 * 0.032;
                    let radius = 0.032 + 0.055 * (-(y - 0.32).powi(2) / 0.025).exp();
                    let a = j as f32 * std::f32::consts::TAU / 32.;
                    Vec3::new(radius * a.cos(), y, radius * a.sin())
                })
            })
            .collect();
        let fit = LegFit::new(
            &vertices,
            Vec3::Y * 0.8,
            Vec3::ZERO,
            &GarmentProgram {
                leg_straightness: 0.95,
                hem_width: 1.25,
                ..Default::default()
            },
        );
        let cuff = (Vec3::Y * 0.06, Vec3::NEG_Y);
        assert!(fit.delta(Vec3::new(0.04, 0.15, 0.), cuff, 0.85).x > 0.01);
        assert_eq!(
            fit.delta(Vec3::new(0.035, 0.03, 0.), cuff, 0.85),
            Vec3::ZERO
        );
        let mut previous = Vec3::ZERO;
        for i in 0..900 {
            let d = fit.delta(Vec3::new(0.035, i as f32 * 0.001, 0.), cuff, 0.85);
            assert!(d.is_finite() && d.length() < 0.035);
            assert!((d - previous).length() < 0.002);
            previous = d;
        }
    }
}
