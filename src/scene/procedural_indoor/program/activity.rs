//! Activity names bias a continuous mixture; they are not fixed room templates.
use super::super::layout::{stream, IndoorLayout};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityMix {
    /// Workstations, shared meetings, soft seating, learning/project tables.
    pub weights: [f32; 4],
    pub group_scale: f32,
    pub storage_bias: f32,
}
impl ActivityMix {
    pub fn sample(seed: u64, zone: usize, profile: IndoorLayout) -> Self {
        use IndoorLayout::*;
        let prior = match profile {
            Conference => [0.10, 0.75, 0.10, 0.05],
            OpenOffice => [0.78, 0.10, 0.07, 0.05],
            Lounge => [0.04, 0.06, 0.85, 0.05],
            Training => [0.10, 0.10, 0.05, 0.75],
            Coworking => [0.45, 0.25, 0.25, 0.05],
            Breakroom => [0.05, 0.30, 0.60, 0.05],
            Reception => [0.18, 0.07, 0.70, 0.05],
            Library => [0.25, 0.05, 0.30, 0.40],
            Workshop => [0.20, 0.35, 0.05, 0.40],
            Studio => [0.40, 0.25, 0.20, 0.15],
            Mixed => [0.25; 4],
        };
        let mut rng = stream(seed, 3010 + zone as u64);
        let blend = rng.random_range(0.10..0.52);
        let mut weights = prior.map(|p| p * (1.0 - blend) + blend * rng.random_range(0.01..1.0));
        let total: f32 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= total);
        Self {
            weights,
            group_scale: rng.random_range(0.70..1.35),
            storage_bias: if profile == Library {
                rng.random_range(0.60..0.95)
            } else {
                rng.random_range(0.08..0.50)
            },
        }
    }
    pub fn choose(&self, rng: &mut impl Rng) -> IndoorLayout {
        let mut draw = rng.random_range(0.0..1.0);
        for (weight, activity) in self.weights.iter().zip([
            IndoorLayout::OpenOffice,
            IndoorLayout::Conference,
            IndoorLayout::Lounge,
            IndoorLayout::Training,
        ]) {
            draw -= weight;
            if draw <= 0.0 {
                return activity;
            }
        }
        IndoorLayout::Training
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || (self.weights.iter().sum::<f32>() - 1.0).abs() > 1e-4
            || !(0.5..=1.5).contains(&self.group_scale)
            || !(0.0..=1.0).contains(&self.storage_bias)
        {
            return Err("invalid activity mixture".into());
        }
        Ok(())
    }
}
