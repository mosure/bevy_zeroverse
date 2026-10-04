//! Seeded adult Anny shape controls, resolved without loading the body model.
//! Stature is metric; the remaining axes are interpolation anchors, not BMI,
//! physical ages or demographic labels.
use super::{stream, IndoorHuman};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BodyProgram {
    /// Anny male-to-female shape interpolation, independent of pigmentation.
    pub gender: f64,
    /// Young-to-old adult anchors. Anny's young anchor is 2/3, not 1/2.
    pub age: f64,
    pub muscle: f64,
    pub proportions: f64,
}

impl BodyProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x424f445953484150);
        Self {
            gender: rng.random_range(0.0..1.0),
            age: rng.random_range(0.67..1.0),
            muscle: rng.random_range(0.12..0.88),
            proportions: rng.random_range(0.02..0.90),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, value, lo, hi) in [
            ("gender", self.gender, 0., 1.),
            ("adult age", self.age, 0.67, 1.),
            ("muscle", self.muscle, 0., 1.),
            ("proportions", self.proportions, 0., 1.),
        ] {
            if !value.is_finite() || !(lo..=hi).contains(&value) {
                return Err(format!("invalid Anny {name} anchor"));
            }
        }
        Ok(())
    }
}

/// Exact controls shared by static meshes, motion rest bodies and diagnostics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Phenotype {
    pub gender: f64,
    pub age: f64,
    pub muscle: f64,
    pub weight: f64,
    pub height: f64,
    pub proportions: f64,
}

impl Phenotype {
    pub fn value(self, name: &str) -> f64 {
        match name {
            "gender" => self.gender,
            "age" => self.age,
            "muscle" => self.muscle,
            "weight" => self.weight,
            "height" => self.height,
            "proportions" => self.proportions,
            _ => 0.5,
        }
    }
}

pub fn phenotype(h: &IndoorHuman) -> Phenotype {
    // Keep old serialized people replayable when no explicit program exists.
    // Consume the same stream even when the legacy gender override is present.
    let mut rng = stream(h.seed, 43);
    let gender = rng.random_range(0.0..1.0);
    let age = rng.random_range(0.60..0.95);
    let muscle = rng.random_range(0.25..0.70);
    let proportions = rng.random_range(0.30..0.70);
    let a = h.appearance.as_ref();
    let p = a.and_then(|a| a.body_program.as_ref());
    let build = ((h.build as f64 - 0.82) / 0.40).clamp(0., 1.);
    Phenotype {
        gender: a
            .and_then(|a| a.body_gender)
            .unwrap_or_else(|| p.map_or(gender, |p| p.gender)),
        age: p.map_or(age, |p| p.age),
        muscle: p.map_or(muscle, |p| p.muscle),
        weight: if p.is_some() {
            build.mul_add(0.68, 0.16)
        } else {
            build.mul_add(0.42, 0.30)
        },
        // Anny's full height anchors span roughly 1.2–2.4 m in the reference,
        // and alter proportions. They are not the endpoints of our adult
        // 1.50–1.95 m prior. Keep proportion changes near the adult centre and
        // let the uniform mesh scale establish the requested metric stature.
        height: {
            let t = ((h.stature as f64 - 1.50) / 0.45).clamp(0., 1.);
            if p.is_some() {
                t.mul_add(0.30, 0.35)
            } else {
                t
            }
        },
        proportions: p.map_or(proportions, |p| p.proportions),
    }
}

/// Dimensions of the unclothed, uniformly scaled Anny rest body. Torso extents
/// exclude arms/hair/clothing; posed mesh bounds are reported separately.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct BodyMeasurements {
    pub rest_stature_metres: f32,
    pub rest_shoulder_bone_span_metres: f32,
    pub rest_torso_width_metres: f32,
    pub rest_torso_depth_metres: f32,
    /// Static retargeting displacement between the torso-attached Anny shoulder
    /// joints and the requested arm roots; large values pull the chest apart.
    #[serde(default)]
    pub shoulder_attachment_offset_metres: f32,
}

/// Shoulder-joint span, rather than the outer clothed shoulder width. Keeping
/// this proportional to stature avoids pulling a short Anny torso apart.
pub fn shoulder_span(stature: f32, build: f32, span_at_175_cm: f32) -> f32 {
    span_at_175_cm * (stature / 1.75) * build.powf(0.15)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{humans, layout::*};
    use bevy::prelude::*;

    #[test]
    fn adult_shape_program_is_continuous_replayable_and_rejects_bad_anchors() {
        let mut unique = std::collections::HashSet::new();
        for seed in 0..1024 {
            let p = BodyProgram::sample(seed);
            assert_eq!(p, BodyProgram::sample(seed));
            p.validate().unwrap();
            assert!(p.age > 0.6666666865348816);
            unique.insert(p.muscle.to_bits());
        }
        assert_eq!(unique.len(), 1024);
        for value in [f64::NAN, f64::INFINITY, -0.1, 1.1, 0.6] {
            let mut p = BodyProgram::sample(0);
            p.age = value;
            assert!(p.validate().is_err());
        }
    }

    #[test]
    fn stature_and_shape_reach_anny_and_dressed_extremes_fit_placement_bounds() {
        let scene =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 1.).unwrap();
        let mut h = scene.humans[0].clone();
        let mut widths = Vec::new();
        let mut depths = Vec::new();
        for case in 0..64 {
            h.stature = if case & 1 == 0 { 1.5 } else { 1.95 };
            h.build = if case & 2 == 0 { 0.82 } else { 1.22 };
            h.shoulder_width = shoulder_span(h.stature, h.build, 0.37);
            let a = h.appearance.as_mut().unwrap();
            a.body_program = Some(BodyProgram {
                gender: if case & 4 == 0 { 0.05 } else { 0.95 },
                age: if case & 8 == 0 { 0.67 } else { 1. },
                muscle: if case & 16 == 0 { 0.12 } else { 0.88 },
                proportions: if case & 32 == 0 { 0.02 } else { 0.90 },
            });
            h.outfit = [
                humans::HumanOutfit::Shirt,
                humans::HumanOutfit::Knitwear,
                humans::HumanOutfit::Blazer,
                humans::HumanOutfit::Tee,
                humans::HumanOutfit::Polo,
                humans::HumanOutfit::Cardigan,
            ][case % 6];
            h.hairstyle = (case % humans::hair::HairStyle::ALL.len()) as u8;
            h.pose = if case % 3 == 0 {
                humans::HumanPoseKind::SeatedWorking
            } else {
                humans::HumanPoseKind::StandingRelaxed
            };
            let pose = humans::poses::PoseProgram::sample(h.seed, h.pose);
            h.joints = pose.solve(h.stature, h.build, h.shoulder_width, h.pose.seated());
            h.pose_program = Some(pose);
            humans::update_bounds(&mut h);
            let mesh = humans::build_human(&h);
            let (lo, hi) = mesh.bounds();
            assert!(
                lo.cmpge(h.bounds_min - Vec3::splat(0.001)).all()
                    && hi.cmple(h.bounds_max + Vec3::splat(0.001)).all(),
                "case {case}: actual {lo:?}..{hi:?}, reserved {:?}..{:?}",
                h.bounds_min,
                h.bounds_max
            );
            assert!(lo.y.abs() < 0.003, "case {case}: shoes float {lo:?}");
            let m = mesh.body_measurements;
            assert!(
                m.shoulder_attachment_offset_metres < 0.045,
                "case {case}: detached shoulder anchors {m:?}"
            );
            assert!((m.rest_stature_metres - h.stature).abs() < 1e-5);
            widths.push(m.rest_torso_width_metres / h.stature);
            depths.push(m.rest_torso_depth_metres / h.stature);
        }
        let spread = |xs: Vec<f32>| {
            xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - xs.iter().copied().fold(f32::INFINITY, f32::min)
        };
        assert!(
            spread(widths) > 0.02,
            "shape did not change Anny torso width"
        );
        assert!(
            spread(depths) > 0.02,
            "shape did not change Anny torso depth"
        );
    }

    #[test]
    fn legacy_body_stream_and_gender_override_remain_replayable() {
        let scene =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 1.).unwrap();
        let mut h = scene.humans[0].clone();
        h.appearance.as_mut().unwrap().body_program = None;
        let mut rng = stream(h.seed, 43);
        let expected = [
            rng.random_range(0.0..1.0),
            rng.random_range(0.60..0.95),
            rng.random_range(0.25..0.70),
            ((h.build as f64 - 0.82) / 0.40).mul_add(0.42, 0.30),
            ((h.stature as f64 - 1.50) / 0.45).clamp(0., 1.),
            rng.random_range(0.30..0.70),
        ];
        let labels = ["gender", "age", "muscle", "weight", "height", "proportions"];
        assert_eq!(labels.map(|label| phenotype(&h).value(label)), expected);
        h.appearance.as_mut().unwrap().body_gender = Some(0.95);
        assert_eq!(phenotype(&h).gender, 0.95);
        assert_eq!(phenotype(&h).muscle, expected[2]);
        h.appearance.as_mut().unwrap().body_program = Some(BodyProgram::sample(h.seed));
        assert_eq!(phenotype(&h).gender, 0.95);
    }
}
