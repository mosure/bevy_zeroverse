//! Bounded local crease fields around sewn hems and bending joints. This is a
//! rest garment approximation, not a cloth simulation; playback skins it once.
use super::super::{stream, IndoorHuman};
use bevy::prelude::*;
use rand::Rng;

struct Crease {
    centre: Vec3,
    along: Vec3,
    across: Vec3,
    length: f32,
    width: f32,
    amplitude: f32,
}
pub(in crate::scene::procedural_indoor::humans) struct Drape(Vec<Crease>);
impl Drape {
    pub fn new(
        h: &IndoorHuman,
        waist: f32,
        chest: f32,
        depth: [f32; 2],
        joints: [Vec3; 4],
    ) -> Self {
        let mut rng = stream(h.seed, 0x435245415345);
        let amplitude = h.appearance.as_ref().map_or(0.0015, |a| a.fold_amplitude);
        let scale = h
            .appearance
            .as_ref()
            .map_or(1.0, |a| (45.0 / a.fold_frequency).clamp(0.65, 1.6));
        let mut folds = Vec::with_capacity(20);
        for i in 0..20 {
            let (centre, along, across) = if i < 8 {
                let front = i % 2;
                let centre = Vec3::new(
                    rng.random_range(-0.14..0.14),
                    waist + (chest - waist) * rng.random_range(0.02..0.55),
                    depth[front],
                );
                let angle: f32 = rng.random_range(-0.8..0.8);
                (
                    centre,
                    Vec3::new(angle.sin(), angle.cos(), 0.0),
                    Vec3::new(angle.cos(), -angle.sin(), 0.0),
                )
            } else {
                let joint = joints[(i - 8) / 3];
                let angle: f32 = rng.random_range(0.0..std::f32::consts::TAU);
                let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
                (
                    joint
                        + radial * rng.random_range(0.035..0.065)
                        + Vec3::Y * rng.random_range(-0.05..0.05),
                    radial.cross(Vec3::Y),
                    Vec3::Y,
                )
            };
            folds.push(Crease {
                centre,
                along,
                across,
                length: rng.random_range(0.035..0.11) * scale.sqrt(),
                width: rng.random_range(0.006..0.016) * scale,
                amplitude: amplitude * rng.random_range(0.35..1.1),
            });
        }
        Self(folds)
    }
    pub fn offset(&self, p: Vec3) -> f32 {
        let mut sum = 0.0;
        for fold in &self.0 {
            let d = p - fold.centre;
            if d.length_squared() > 0.035 {
                continue;
            }
            let a = d.dot(fold.along) / fold.length;
            let b = d.dot(fold.across) / fold.width;
            let n = d.dot(fold.along.cross(fold.across)) / 0.035;
            let radius = a * a + b * b + n * n;
            if radius < 12.0 {
                sum += fold.amplitude * (-radius).exp() * (1.0 - 2.0 * b * b);
            }
        }
        sum.clamp(-0.005, 0.006)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_creases_are_bounded_and_do_not_deform_uncovered_hands() {
        let h = super::super::super::sample_person(
            6,
            0,
            Vec3::ZERO,
            0.0,
            super::super::super::HumanPoseKind::StandingRelaxed,
            None,
            false,
        );
        let fold = Drape::new(&h, 0.9, 1.3, [-0.12, 0.12], [Vec3::new(0.5, 1.1, 0.0); 4]);
        for i in 0..500 {
            let p = Vec3::new(i as f32 * 0.001 - 0.25, 0.98, -0.12);
            assert!((-0.005..=0.006).contains(&fold.offset(p)));
            assert!((fold.offset(p + Vec3::X * 0.0001) - fold.offset(p)).abs() < 0.00015);
        }
        assert_eq!(fold.offset(Vec3::new(0.8, 0.5, 0.0)), 0.0);
    }
}
