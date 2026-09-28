//! Tileable finish mixtures; intensity and scale vary independently of colour.
use super::{program::MaterialRecipe, Surface};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinishLayers {
    pub quarter_turn: u8,
    pub bands: [u32; 2],
    pub stripe_strength: f32,
    pub fleck_strength: f32,
    pub vein_strength: f32,
}
impl FinishLayers {
    pub fn validate(&self) -> Result<(), String> {
        if self.quarter_turn > 3
            || !(1..=100).contains(&self.bands[0])
            || self.bands[1] > 100
            || [
                self.stripe_strength,
                self.fleck_strength,
                self.vein_strength,
            ]
            .into_iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        {
            return Err("invalid procedural finish layers".into());
        }
        Ok(())
    }
    pub fn sample(surface: Surface, rng: &mut impl Rng) -> Self {
        let textile = matches!(
            surface,
            Surface::Fabric | Surface::FabricAlt | Surface::Floor
        );
        let mineral = matches!(
            surface,
            Surface::Concrete | Surface::Ceramic | Surface::Floor
        );
        let wood = matches!(surface, Surface::Wood | Surface::WoodEdge | Surface::Bark);
        Self {
            // Object UVs align timber with its long axis and bark with the trunk.
            // Preserve that direction; floors and textiles may rotate freely.
            quarter_turn: if wood {
                rng.random_range(0..2) * 2
            } else {
                rng.random_range(0..4)
            },
            bands: [rng.random_range(1..13), rng.random_range(0..5)],
            stripe_strength: if textile {
                rng.random_range(0.0_f32..1.0).powi(2)
                    * if surface == Surface::Floor {
                        0.85
                    } else {
                        0.35
                    }
            } else {
                0.0
            },
            fleck_strength: if mineral {
                rng.random_range(0.0..0.9)
            } else {
                0.0
            },
            vein_strength: if mineral || wood {
                rng.random_range(0.0..0.8)
            } else {
                0.0
            },
        }
    }
    pub fn rotate(&self, u: f32, v: f32) -> (f32, f32) {
        match self.quarter_turn {
            1 => (v, 1.0 - u),
            2 => (1.0 - u, 1.0 - v),
            3 => (1.0 - v, u),
            _ => (u, v),
        }
    }
    pub fn apply(
        &self,
        recipe: &MaterialRecipe,
        uv: [f32; 2],
        noise: [f32; 3],
        finish: &mut [f32; 3],
        floor_style: u32,
    ) {
        use std::f32::consts::TAU;
        let [u, v] = uv;
        let [_, meso, micro] = noise;
        let band = (TAU * (u * self.bands[0] as f32 + v * self.bands[1] as f32)).sin();
        let textile = matches!(recipe.surface, Surface::Fabric | Surface::FabricAlt)
            || recipe.surface == Surface::Floor && floor_style == 1;
        let mineral = matches!(recipe.surface, Surface::Concrete | Surface::Ceramic)
            || recipe.surface == Surface::Floor && floor_style == 2;
        // Timber gets its directional grain from the base recipe, not mineral
        // veins or textile stripes spanning every plank in the room.
        let stripes = if textile {
            (0.5 + 0.5 * band).powi(4) * self.stripe_strength
        } else {
            0.0
        };
        let flecks = if mineral {
            ((micro - 0.57) * 5.0).clamp(0.0, 1.0) * self.fleck_strength
        } else {
            0.0
        };
        // Stone tiles are cut from different parts of a slab. A room-wide
        // sinusoid made unrelated tiles look like one striped sheet.
        let (vein_u, vein_v, phase) = if recipe.surface == Surface::Floor && floor_style == 2 {
            let [nx, ny] = recipe.floor_repetitions(floor_style);
            let x = (u * nx).floor() as u32;
            let y = (v * ny).floor() as u32;
            let phase = super::hash(x, y, recipe.seed.wrapping_add(97));
            let uv = ((u * nx).fract(), (v * ny).fract());
            let uv = if phase > 0.5 { (uv.1, uv.0) } else { uv };
            (uv.0, uv.1, phase * TAU)
        } else {
            (u, v, 0.0)
        };
        // Narrow level sets of a warped multi-octave mineral field produce
        // branching deposits instead of equally spaced sinusoidal marble bands.
        let mineral_field = super::periodic_noise(vein_u + phase, vein_v, 4, 5, recipe.seed)
            + 0.32
                * super::periodic_noise(
                    vein_u,
                    vein_v + phase,
                    11,
                    13,
                    recipe.seed.wrapping_add(33),
                );
        let veins = ((mineral_field - 0.62).abs() * 14.0).clamp(0.0, 1.0);
        let vein = if mineral {
            (1.0 - veins).powi(6) * self.vein_strength
        } else {
            0.0
        };
        finish[0] *= 1.0 - stripes * 0.40 - vein * 0.28 + flecks * 0.18;
        finish[1] += recipe.relief_m * (stripes * 0.35 - vein * 0.25 + flecks * (meso - 0.5));
        finish[2] += stripes * 0.06 + vein * 0.09 - flecks * 0.10;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mixed_finishes_remain_tileable_and_finite() {
        for seed in 0..24 {
            for recipe in super::super::program::sample(seed) {
                for t in [0.0, 0.123, 0.5, 0.937, 1.0] {
                    for (a, b) in [
                        (recipe.evaluate(0.0, t, 2), recipe.evaluate(1.0, t, 2)),
                        (recipe.evaluate(t, 0.0, 0), recipe.evaluate(t, 1.0, 0)),
                    ] {
                        assert!(
                            (a.0 - b.0).abs() < 0.0001
                                && (a.1 - b.1).abs() < 0.0001
                                && (a.2 - b.2).abs() < 0.0001,
                            "nonperiodic {:?}: {a:?} {b:?}",
                            recipe.surface
                        );
                        assert!(a.0.is_finite() && a.1.is_finite() && (0.12..=1.0).contains(&a.2));
                    }
                }
            }
        }
    }
}
