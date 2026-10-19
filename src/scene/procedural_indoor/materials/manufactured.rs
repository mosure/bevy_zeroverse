//! Manufacturing finish independent of pigment: brushed/polished plating,
//! powder-coated or lightly peened metal, and fine/coarse molded polymers.
use super::{periodic_noise, program::MaterialRecipe, Surface};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManufacturedFinish {
    pub cells: [u32; 2],
    pub structure: f32,
    pub peening: f32,
}
impl ManufacturedFinish {
    pub fn sample(seed: u64, surface: Surface) -> Option<Self> {
        if !matches!(
            surface,
            Surface::Metal | Surface::Chrome | Surface::Plastic | Surface::Rubber
        ) {
            return None;
        }
        let mut rng = stream(seed, 0x4d414e55464143);
        let directional = rng.random_bool(if surface == Surface::Chrome {
            0.78
        } else {
            0.18
        });
        Some(Self {
            cells: [
                rng.random_range(20..=112),
                if directional {
                    rng.random_range(2..=12)
                } else {
                    rng.random_range(20..=112)
                },
            ],
            structure: rng.random_range(0.30..0.95),
            peening: if surface == Surface::Chrome {
                0.
            } else {
                rng.random_range(0.0_f32..1.).powi(3) * 0.65
            },
        })
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.cells.iter().any(|n| !(2..=128).contains(n))
            || [self.structure, self.peening]
                .into_iter()
                .any(|v| !v.is_finite() || !(0.0..=1.).contains(&v))
        {
            return Err("invalid manufactured finish".into());
        }
        Ok(())
    }
    pub(super) fn value(&self, r: &MaterialRecipe, uv: [f32; 2], micro: f32, meso: f32) -> f32 {
        let structure = periodic_noise(
            uv[0],
            uv[1],
            self.cells[0],
            self.cells[1],
            r.seed.wrapping_add(751),
        );
        let fine = micro * (1. - self.structure) + structure * self.structure;
        fine * (1. - self.peening) + super::field::smooth((meso - 0.25) * 2.) * self.peening
    }
}
