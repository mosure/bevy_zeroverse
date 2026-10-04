//! Continuous cut, fall and grooming controls, independent of phenotype/palette.
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// IDs 0–7 retain the legacy selector mapping. Each selector supplies a topology;
/// length, layers, parting, sweep and curl are sampled within it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum HairStyle {
    Buzz,
    SidePart,
    Bob,
    Swept,
    TightCurls,
    Bald,
    Bun,
    Ponytail,
    StraightLong,
    WavyLong,
    CurlyLong,
    LayeredLong,
    AsymmetricBob,
    Braid,
    TwinBraids,
    HighPonytail,
    Locs,
    Afro,
    Pixie,
}
impl HairStyle {
    pub const ALL: [Self; 19] = [
        Self::Buzz,
        Self::SidePart,
        Self::Bob,
        Self::Swept,
        Self::TightCurls,
        Self::Bald,
        Self::Bun,
        Self::Ponytail,
        Self::StraightLong,
        Self::WavyLong,
        Self::CurlyLong,
        Self::LayeredLong,
        Self::AsymmetricBob,
        Self::Braid,
        Self::TwinBraids,
        Self::HighPonytail,
        Self::Locs,
        Self::Afro,
        Self::Pixie,
    ];
    pub fn from_id(id: u8) -> Option<Self> {
        Self::ALL.get(id as usize).copied()
    }
    pub fn loose(self) -> bool {
        matches!(
            self,
            Self::Bob
                | Self::StraightLong
                | Self::WavyLong
                | Self::CurlyLong
                | Self::LayeredLong
                | Self::AsymmetricBob
                | Self::Locs
        )
    }
    pub fn falls(self) -> bool {
        self.loose()
            || matches!(
                self,
                Self::Braid | Self::TwinBraids | Self::Ponytail | Self::HighPonytail
            )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HairProgram {
    /// Free hair below the jaw/nape reference, metres at nominal adult stature.
    pub drop_m: f32,
    pub layers: f32,
    pub spread: f32,
    pub sweep: f32,
    pub wave_length_m: f32,
    pub curl_radius_m: f32,
    pub clump_width_m: f32,
    pub bangs: f32,
    pub tie_height: f32,
    pub flyaways: f32,
}
impl Default for HairProgram {
    fn default() -> Self {
        Self {
            drop_m: 0.32,
            layers: 0.35,
            spread: 1.,
            sweep: 0.1,
            wave_length_m: 0.19,
            curl_radius_m: 0.012,
            clump_width_m: 0.013,
            bangs: 0.,
            tie_height: 0.45,
            flyaways: 0.15,
        }
    }
}
impl HairProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x4841495243555453);
        Self {
            drop_m: rng.random_range(0.16..0.56),
            layers: rng.random_range(0.10..0.85),
            spread: rng.random_range(0.85..1.28),
            sweep: rng.random_range(-0.65..0.65),
            wave_length_m: rng.random_range(0.11..0.29),
            curl_radius_m: rng.random_range(0.006..0.027),
            clump_width_m: rng.random_range(0.009..0.022),
            bangs: if rng.random_bool(0.28) {
                rng.random_range(0.35..0.95)
            } else {
                0.
            },
            tie_height: rng.random_range(0.18..0.78),
            flyaways: rng.random_range(0.05..0.35),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        for (v, lo, hi) in [
            (self.drop_m, 0.08, 0.65),
            (self.layers, 0., 1.),
            (self.spread, 0.7, 1.4),
            (self.sweep, -1., 1.),
            (self.wave_length_m, 0.08, 0.40),
            (self.curl_radius_m, 0.002, 0.04),
            (self.clump_width_m, 0.006, 0.03),
            (self.bangs, 0., 1.),
            (self.tie_height, 0., 1.),
            (self.flyaways, 0., 0.5),
        ] {
            if !v.is_finite() || !(lo..=hi).contains(&v) {
                return Err("invalid continuous hair cut or grooming parameter".into());
            }
        }
        Ok(())
    }
}

/// Supported lower hair follows the torso while scalp/temple roots follow the
/// head. Used identically by static construction and GPU/annotation skinning.
pub(crate) fn torso_weight(style: u8, below_head_m: f32, stature: f32) -> f32 {
    if !HairStyle::from_id(style).is_some_and(HairStyle::falls) {
        return 0.;
    }
    let scale = stature / 1.75;
    let t = ((below_head_m / scale - 0.045) / 0.24).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuts_are_seeded_continuous_and_legacy_ids_stay_stable() {
        let mut lengths = std::collections::HashSet::new();
        for seed in 0..512 {
            let p = HairProgram::sample(seed);
            assert_eq!(p, HairProgram::sample(seed));
            p.validate().unwrap();
            lengths.insert(p.drop_m.to_bits());
        }
        assert!(lengths.len() > 500);
        assert_eq!(HairStyle::from_id(5), Some(HairStyle::Bald));
        assert_eq!(HairStyle::from_id(7), Some(HairStyle::Ponytail));
        assert_eq!(HairStyle::from_id(19), None);
        assert_eq!(torso_weight(8, 0., 1.75), 0.);
        assert_eq!(torso_weight(8, 0.4, 1.75), 1.);
        let invalid = HairProgram {
            drop_m: f32::NAN,
            ..HairProgram::default()
        };
        assert!(invalid.validate().is_err());
    }
}
