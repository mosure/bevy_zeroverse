//! A heating circuit may share a manufactured series while each unit fits its
//! own bay. The purchasing stream never advances the spatial placement RNG.
use super::*;
use crate::scene::procedural_indoor::materials::{coated_stock, variants};

#[derive(Clone, Copy)]
#[repr(u32)]
enum Model {
    ElectricConvector = 1,
    HydronicConvector,
    Pressed,
    Tubular,
}

impl Model {
    fn sample(r: &mut impl Rng) -> Self {
        match r.random_range(0..100) {
            0..23 => Self::ElectricConvector,
            23..43 => Self::HydronicConvector,
            43..79 => Self::Pressed,
            _ => Self::Tubular,
        }
    }
    fn decode(variant: u32) -> Option<Self> {
        match (variant >> 8) & 15 {
            1 => Some(Self::ElectricConvector),
            2 => Some(Self::HydronicConvector),
            3 => Some(Self::Pressed),
            4 => Some(Self::Tubular),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct HeatingSeries {
    model: Model,
    finish_lot: usize,
}

impl HeatingSeries {
    pub(crate) fn for_room(seed: u64) -> Option<Self> {
        let mut r = stream(seed, u64::from_le_bytes(*b"HeatLoop"));
        r.random_bool(0.68).then(|| Self {
            model: Model::sample(&mut r),
            finish_lot: r.random_range(0..variants::COUNT),
        })
    }

    pub(crate) fn install(self, o: &mut IndoorObject) {
        // The original low byte remains available for existing object variants.
        o.variant =
            (o.variant & 0xff) | ((self.model as u32) << 8) | ((self.finish_lot as u32) << 12);
    }

    fn for_object(o: &IndoorObject) -> Self {
        let shared_lot = (o.variant >> 12) as usize;
        let shared = Model::decode(o.variant).filter(|_| shared_lot < variants::COUNT);
        let finish_lot = match (o.finish_seed, shared) {
            (None, Some(_)) => shared_lot,
            (seed, _) => stream(seed.unwrap_or(o.seed), u64::from_le_bytes(*b"HeatCoat"))
                .random_range(0..variants::COUNT),
        };
        Self {
            model: shared.unwrap_or_else(|| {
                Model::sample(&mut stream(o.seed, u64::from_le_bytes(*b"HeatUnit")))
            }),
            finish_lot,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(tag = "construction", rename_all = "snake_case")]
pub enum Emitter {
    Convector { hydronic: bool },
    Pressed(PressedPanel),
    Tubular(TubeBank),
}

impl Emitter {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Convector { hydronic: false } => "electric_convector",
            Self::Convector { hydronic: true } => "hydronic_convector",
            Self::Pressed(p) if p.bank_spacing_m.is_some() => "double_pressed_panel",
            Self::Pressed(_) => "single_pressed_panel",
            Self::Tubular(_) => "horizontal_tubes",
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Installation {
    pub emitter: Emitter,
    pub finish_lot: usize,
}

impl Installation {
    pub fn for_object(o: &IndoorObject) -> Self {
        let series = HeatingSeries::for_object(o);
        let emitter = match series.model {
            Model::Pressed => Emitter::Pressed(PressedPanel::fit(o)),
            Model::Tubular => Emitter::Tubular(TubeBank::fit(o)),
            Model::ElectricConvector => Emitter::Convector { hydronic: false },
            Model::HydronicConvector => Emitter::Convector { hydronic: true },
        };
        Self {
            emitter,
            finish_lot: series.finish_lot,
        }
    }

    pub(super) fn casing_label(&self) -> String {
        let slot = coated_stock::slot(Surface::Metal, self.finish_lot).expect("metal coating slot");
        format!("appliance#finish{slot}")
    }
}
