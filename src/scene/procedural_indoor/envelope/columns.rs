//! Architectural column finishes share a specification across a room. All
//! dimensions fit the shaft and base already reserved by layout and cameras.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorLayout},
    materials::Surface,
};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnStock {
    CastConcrete,
    PaintedPlaster,
    VeneerWrap,
    MetalJacket,
}

impl ColumnStock {
    pub(crate) fn surface(self) -> Surface {
        match self {
            Self::CastConcrete => Surface::Concrete,
            Self::PaintedPlaster => Surface::Paint,
            Self::VeneerWrap => Surface::Wood,
            Self::MetalJacket => Surface::Metal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColumnConstruction {
    pub stock: ColumnStock,
    pub base_surface: Surface,
    pub base_projection_m: f32,
    /// Full-height wraps and monolithic substrates have no horizontal joint.
    pub sleeve_pitch_m: Option<f32>,
    pub joint_width_m: f32,
    pub joint_depth_m: f32,
}

impl ColumnConstruction {
    pub(super) fn sample(seed: u64, layout: IndoorLayout) -> Self {
        use ColumnStock::*;
        use IndoorLayout::*;
        let mut r = stream(seed, u64::from_le_bytes(*b"ColStock"));
        let general = match r.random_range(0..12) {
            0..=3 => CastConcrete,
            4..=7 => PaintedPlaster,
            8..=9 => VeneerWrap,
            _ => MetalJacket,
        };
        let stock = if r.random_bool(0.48) {
            match layout {
                Workshop | Studio => CastConcrete,
                Kitchen | Kitchenette | ElevatorLanding => MetalJacket,
                Reception | Lounge | Library | Conference => VeneerWrap,
                _ => general,
            }
        } else {
            general
        };
        let base_surface = if r.random_bool(0.54) {
            stock.surface()
        } else if matches!(stock, PaintedPlaster | VeneerWrap) && r.random_bool(0.42) {
            Surface::WoodEdge
        } else {
            Surface::Metal
        };
        let sleeve_pitch_m = (matches!(stock, VeneerWrap | MetalJacket) && r.random_bool(0.78))
            .then(|| r.random_range(0.88..2.65));
        Self {
            stock,
            base_surface,
            base_projection_m: r.random_range(0.002..0.024),
            sleeve_pitch_m,
            joint_width_m: r.random_range(0.004..0.013),
            joint_depth_m: r.random_range(0.0015..0.005),
        }
    }

    pub(super) fn valid(self) -> bool {
        matches!(
            self.base_surface,
            Surface::Concrete | Surface::Paint | Surface::Metal | Surface::Wood | Surface::WoodEdge
        ) && (0.001..=0.025).contains(&self.base_projection_m)
            && self.sleeve_pitch_m.is_none_or(|p| {
                matches!(
                    self.stock,
                    ColumnStock::VeneerWrap | ColumnStock::MetalJacket
                ) && (0.8..=2.8).contains(&p)
            })
            && (0.003..=0.015).contains(&self.joint_width_m)
            && (0.001..=0.006).contains(&self.joint_depth_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_program_admits_every_stock_and_replays_one_specification_per_room() {
        for layout in IndoorLayout::PROFILES {
            let mut counts = [0; 4];
            for seed in 0..256 {
                let finish = ColumnConstruction::sample(seed, layout);
                assert!(finish.valid());
                assert_eq!(finish, ColumnConstruction::sample(seed, layout));
                counts[finish.stock as usize] += 1;
            }
            assert!(counts.iter().all(|&n| n > 5), "{layout:?}: {counts:?}");
        }
        let profile = ColumnProfile::sample(10, 0);
        let mut json = serde_json::to_value(&profile).unwrap();
        json.as_object_mut().unwrap().remove("construction");
        assert!(serde_json::from_value::<ColumnProfile>(json)
            .unwrap()
            .construction
            .is_none());
    }
}
