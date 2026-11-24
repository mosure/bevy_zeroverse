//! Vision openings in the wall between the room and its adjoining circulation.
//! Intervals run away from the entry jamb on each side, so a sidelight stays
//! attached to its own door even when the entrance changes position.
use super::*;
use crate::scene::procedural_indoor::materials::glazing::GlazingProgram;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SharedOpenings {
    /// Fractions of the available wall from each jamb towards its corner.
    pub jamb_intervals: [Vec2; 2],
    /// Vertical interval within the clear height above the sill and below the header.
    pub height_interval: Vec2,
}

/// Fixed vision panels along the entry wall. Heights and privacy film follow
/// one installation across the doorway. End bays can be cut to fit each run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionPanelStock {
    pub glass: GlazingProgram,
    /// For each side of the entry, widths at the jamb and corner in nominal bays.
    pub end_modules: [Vec2; 2],
}

impl VisionPanelStock {
    fn sample_glass(seed: u64) -> GlazingProgram {
        let mut r = stream(seed, u64::from_le_bytes(*b"visnfilm"));
        let mut glass = GlazingProgram::sample(r.random(), 0, false);
        // Every opening in this installation has the same sill and head. A
        // height-only finish can therefore share one material at any bay width.
        glass.treatment = match r.random_range(0..20) {
            0..=7 => 0,   // clear, including tinted substrates
            8..=9 => 1,   // full frost
            10..=14 => 2, // eye-level privacy band
            15..=16 => 3, // graded privacy
            17..=18 => 5, // etched stripes
            _ => 6,       // opaque ceramic markings
        };
        glass
    }

    /// A material-only intervention can change substrate tint and frost
    /// strength. The cut glass, opaque markings and mullion layout stay fitted.
    pub(crate) fn apply_material_seed(&mut self, seed: u64) {
        let sampled = Self::sample_glass(seed);
        self.glass.installation = sampled.installation;
        self.glass.frost_roughness = sampled.frost_roughness;
    }

    pub(super) fn sample(seed: u64) -> Self {
        let glass = Self::sample_glass(seed);
        let mut r = stream(seed, u64::from_le_bytes(*b"visnbays"));
        let end_modules = std::array::from_fn(|_| {
            if r.random_bool(0.30) {
                return Vec2::ONE;
            }
            let end = r.random_range(0.60..1.40);
            let other = if r.random_bool(0.55) {
                1.
            } else {
                r.random_range(0.60..1.40)
            };
            if r.random() {
                Vec2::new(end, other)
            } else {
                Vec2::new(other, end)
            }
        });
        Self { glass, end_modules }
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        self.glass.validate()?;
        if self.glass.opening != 0.
            || self.glass.treatment == 4
            || self
                .end_modules
                .iter()
                .flat_map(|m| m.to_array())
                .any(|v| !(0.60..=1.40).contains(&v))
        {
            return Err("invalid fixed vision panel stock".into());
        }
        Ok(())
    }

    /// Fraction along an ordered run. The first side runs from corner to jamb.
    pub(super) fn boundary(&self, side: usize, i: usize, count: usize) -> f32 {
        if i == 0 {
            return 0.;
        }
        if i == count {
            return 1.;
        }
        let ends = self.end_modules[side];
        let first = if side == 0 { ends.y } else { ends.x };
        (first + (i - 1) as f32) / (ends.x + ends.y + (count - 2) as f32)
    }
}

impl SharedOpenings {
    pub(super) fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x5649_5349_4f4e_5741);
        let design = r.random::<f32>();
        let height_interval = if design < 0.20 {
            Vec2::new(r.random_range(0.42..0.74), r.random_range(0.9..1.0))
        } else {
            Vec2::new(r.random_range(0.0..0.14), r.random_range(0.78..1.0))
        };
        let mut jamb_intervals = [Vec2::ZERO; 2];
        let primary = usize::from(r.random::<bool>());
        for (i, interval) in jamb_intervals.iter_mut().enumerate() {
            let width = match design {
                // Private walls can have only the entry itself; other installations
                // range from one vision panel to nearly continuous office glazing.
                d if d > 0.93 || (d > 0.64 && i != primary) => continue,
                d if d > 0.64 => r.random_range(0.20..0.72),
                d if d > 0.30 => r.random_range(0.67..1.0),
                _ => r.random_range(0.35..0.96),
            };
            let offset = if design > 0.64 {
                (1. - width) * r.random_range(0.0..0.25)
            } else {
                (1. - width) * r.random::<f32>()
            };
            *interval = Vec2::new(offset, offset + width);
        }
        Self {
            jamb_intervals,
            height_interval,
        }
    }

    pub(super) fn valid(self) -> bool {
        let bounded = |v: Vec2| v.is_finite() && v.x >= 0. && v.y <= 1. && v.x <= v.y;
        self.jamb_intervals.into_iter().all(bounded)
            && bounded(self.height_interval)
            && self.height_interval.y > self.height_interval.x
    }
}
