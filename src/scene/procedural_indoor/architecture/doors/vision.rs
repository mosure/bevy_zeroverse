//! Machined openings specified in metres before fitting to the available stiles.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VisionShape {
    Porthole {
        diameter_m: f32,
    },
    Slot {
        width_m: f32,
        height_m: f32,
    },
    Rounded {
        width_m: f32,
        height_m: f32,
        radius_m: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VisionLight {
    pub shape: VisionShape,
    /// Height above the bottom of the leaf.
    pub center_height_m: f32,
    /// Fraction of the available lateral travel, positive toward the latch.
    pub latch_bias: f32,
    /// Exposed metal glazing bead, otherwise finished to match the leaf.
    pub metal_bead: bool,
}

impl VisionLight {
    pub(super) fn sample(seed: u64, panel: DoorPanel) -> Option<Self> {
        if panel != DoorPanel::VisionLite {
            return None;
        }
        let mut rng = stream(seed, u64::from_le_bytes(*b"DoorLite"));
        let shape = match rng.random_range(0..100) {
            0..45 => return None,
            45..60 => VisionShape::Porthole {
                diameter_m: rng.random_range(0.22..0.52),
            },
            60..80 => VisionShape::Slot {
                width_m: rng.random_range(0.14..0.32),
                height_m: rng.random_range(0.49..1.13),
            },
            _ => VisionShape::Rounded {
                width_m: rng.random_range(0.27..0.75),
                height_m: rng.random_range(0.38..1.05),
                radius_m: rng.random_range(0.045..0.16),
            },
        };
        Some(Self {
            shape,
            center_height_m: rng.random_range(1.26..1.64),
            latch_bias: rng.random_range(-0.38..0.91),
            metal_bead: rng.random_bool(0.49),
        })
    }

    pub(super) fn valid(self) -> bool {
        let dimension = |x: f32| (0.08..=1.8).contains(&x);
        (0.4..=2.2).contains(&self.center_height_m)
            && (-1.0..=1.0).contains(&self.latch_bias)
            && match self.shape {
                VisionShape::Porthole { diameter_m } => dimension(diameter_m),
                VisionShape::Slot { width_m, height_m } => {
                    dimension(width_m) && dimension(height_m) && height_m > width_m
                }
                VisionShape::Rounded {
                    width_m,
                    height_m,
                    radius_m,
                } => dimension(width_m) && dimension(height_m) && (0.01..=0.5).contains(&radius_m),
            }
    }

    pub(super) fn fit(self, field: Rect, bottom: f32, side: f32) -> Rect {
        let desired = match self.shape {
            VisionShape::Porthole { diameter_m } => Vec2::splat(diameter_m),
            VisionShape::Slot { width_m, height_m }
            | VisionShape::Rounded {
                width_m, height_m, ..
            } => Vec2::new(width_m, height_m),
        };
        // Scale uniformly on a narrow paired/telescoping leaf so a circle stays circular.
        let size = desired * (field.size() / desired).min_element().min(1.);
        let travel = (field.size() - size).max(Vec2::ZERO) * 0.5;
        let offset = Vec2::new(
            -side * self.latch_bias * travel.x,
            (bottom + self.center_height_m - field.center().y).clamp(-travel.y, travel.y),
        );
        Rect::from_center_size(field.center() + offset, size)
    }

    pub(super) fn radius(self, bounds: Rect) -> f32 {
        let half = 0.5 * bounds.size().min_element();
        match self.shape {
            VisionShape::Porthole { .. } | VisionShape::Slot { .. } => half,
            VisionShape::Rounded { radius_m, .. } => radius_m.min(half * 0.90),
        }
    }
}
