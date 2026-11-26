//! Forearm pads fitted to the upper chair frame. Dimensions are in metres.
use super::*;

mod mesh;
#[cfg(test)]
mod tests;

const INTRODUCED: u32 = 252;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmPadStock {
    SoftTouch,
    MouldedShell,
    Wrapped,
    Timber,
}

impl ArmPadStock {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::SoftTouch => "soft_touch",
            Self::MouldedShell => "moulded_shell",
            Self::Wrapped => "wrapped",
            Self::Timber => "timber",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ArmPad {
    pub stock: ArmPadStock,
    pub face: Surface,
    pub table_order: bool,
    pub gauge_m: f32,
    /// Fraction of the pad gauge between the upper shoulder and the centre of the crown.
    pub crown_fraction: f32,
    /// Fractional narrowing relative to the middle. Negative taper flares a paddle outwards.
    pub front_taper: f32,
    pub rear_taper: f32,
    pub nose_length_fraction: f32,
    pub heel_length_fraction: f32,
    pub tip_width_fraction: f32,
    /// Signed lateral sweep towards the nose. The two pads mirror each other.
    pub sweep_m: f32,
    /// Fore-aft installation offset, positive toward the backrest.
    pub offset_z_m: f32,
}

impl ChairProgram {
    pub fn arm_pad(&self, o: &IndoorObject, version: u32) -> Option<ArmPad> {
        if version < INTRODUCED
            || !self.armrests
            || o.kind != ObjectKind::Chair
            || !matches!(o.variant, 0..=5 | 9)
        {
            return None;
        }
        let mut item = stream(o.seed, u64::from_le_bytes(*b"ArmPdFit"));
        let order = o
            .finish_seed
            .zip(o.interaction_target)
            .filter(|_| item.random_bool(0.80));
        let offset_z_m = item.random_range(-0.009..0.009);
        let mut lot = order.map(|(finish, table)| {
            stream(
                finish,
                u64::from_le_bytes(*b"ArmPdLot")
                    ^ (table as u64).rotate_left(17)
                    ^ u64::from(o.variant),
            )
        });
        let stock_rng = lot.as_mut().unwrap_or(&mut item);
        use ArmPadStock::*;
        let choice = stock_rng.random::<f32>();
        let (timber, wrapped, soft) = match o.variant {
            3 => (0.76, 0.14, 0.02),
            2 => (0.17, 0.23, 0.22),
            9 => (0.03, 0.67, 0.23),
            4 => (0.02, 0.10, 0.12),
            _ => (0., 0.25, 0.45),
        };
        let stock = if choice < timber {
            Timber
        } else if choice < timber + wrapped {
            Wrapped
        } else if choice < timber + wrapped + soft {
            SoftTouch
        } else {
            MouldedShell
        };
        let (face, gauge, crown, tip) = match stock {
            Timber => (Surface::Wood, 0.013..0.029, 0.025..0.13, 0.38..0.79),
            MouldedShell => (Surface::Plastic, 0.012..0.026, 0.05..0.16, 0.20..0.70),
            SoftTouch => (Surface::Rubber, 0.018..0.037, 0.10..0.23, 0.16..0.56),
            Wrapped => {
                let face = match self.seat_surface {
                    surface @ (Surface::Leather | Surface::Fabric | Surface::FabricAlt) => surface,
                    _ => Surface::Leather,
                };
                (face, 0.024..0.047, 0.12..0.28, 0.25..0.68)
            }
        };
        Some(ArmPad {
            stock,
            face,
            table_order: order.is_some(),
            gauge_m: stock_rng.random_range(gauge),
            crown_fraction: stock_rng.random_range(crown),
            front_taper: stock_rng.random_range(-0.20..0.32),
            rear_taper: stock_rng.random_range(-0.12..0.34),
            nose_length_fraction: stock_rng.random_range(0.07..0.25),
            heel_length_fraction: stock_rng.random_range(0.045..0.20),
            tip_width_fraction: stock_rng.random_range(tip),
            sweep_m: stock_rng.random_range(-0.105..0.105) * self.arm_pad_width_m,
            offset_z_m,
        })
    }
}

impl ArmPad {
    pub(super) fn underside(self, p: &ChairProgram) -> f32 {
        p.arm_height + 0.010 - self.gauge_m
    }

    fn lower_surface(self) -> Surface {
        match self.stock {
            ArmPadStock::Wrapped | ArmPadStock::SoftTouch => Surface::Plastic,
            ArmPadStock::Timber => Surface::WoodEdge,
            ArmPadStock::MouldedShell => self.face,
        }
    }
}
