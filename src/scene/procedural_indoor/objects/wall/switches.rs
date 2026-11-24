//! Installed switch plates. Product selection belongs to the room, while the
//! position of a lever, paddle or dimmer belongs to an individual circuit.
use super::*;
use crate::scene::procedural_indoor::materials::{coated_stock, variants};
mod mesh;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 213;
// Preserve the electrical convention in variant % 3, as for the outlet banks.
const FIRST: u32 = 900;

pub(super) fn primary(o: &IndoorObject) -> Option<u8> {
    (o.kind == ObjectKind::LightSwitch && (FIRST..FIRST + 18).contains(&o.variant))
        .then(|| ((o.variant - FIRST) / 3) as u8)
}

pub(crate) fn install(o: &mut IndoorObject, room_seed: u64, standard: u32) {
    let mut choice = stream(o.seed, u64::from_le_bytes(*b"SwRefits"));
    let family_seed = if choice.random_bool(0.78) {
        room_seed
    } else {
        o.seed
    };
    let primary = match stream(family_seed, u64::from_le_bytes(*b"SwFamily")).random_range(0..100) {
        0..=32 => 0,
        33..=52 => 1,
        53..=64 => 2,
        65..=80 => 3,
        81..=91 => 4,
        _ => 5,
    };
    o.variant = FIRST + 3 * primary + standard % 3;
    if choice.random_bool(0.77) {
        o.finish_seed = Some(stream(room_seed, u64::from_le_bytes(*b"SwFinish")).random());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub enum Actuator {
    Paddle { angle_rad: f32 },
    Toggle { angle_rad: f32 },
    Rotary { angle_rad: f32 },
    Split { angles_rad: [f32; 2] },
    Slider { level: f32 },
    Buttons { latched: Option<usize> },
}

impl Actuator {
    pub fn name(self) -> &'static str {
        match self {
            Self::Paddle { .. } => "paddle",
            Self::Toggle { .. } => "toggle",
            Self::Rotary { .. } => "rotary_dimmer",
            Self::Split { .. } => "split_paddle",
            Self::Slider { .. } => "slide_dimmer",
            Self::Buttons { .. } => "push_buttons",
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SwitchBank {
    pub modules: Vec<Actuator>,
    pub case_size: Vec3,
    pub module_size: Vec2,
    pub rear_z_m: f32,
    pub face_z_m: f32,
    pub pocket_z_m: f32,
    pub gap_m: f32,
    pub corner_m: f32,
    pub edge_m: f32,
    pub finish: Surface,
    pub screws: bool,
    pub finish_lot: usize,
}

impl SwitchBank {
    pub(super) fn fitted(o: &IndoorObject, p: &FixtureProgram, primary: u8) -> Self {
        let product = o.finish_seed.unwrap_or(o.seed);
        let mut stock = stream(product, u64::from_le_bytes(*b"SwCover "));
        let size = o.size * Vec3::new(p.case_fraction.x, p.case_fraction.y, 1.);
        let rear = -0.35 * size.z;
        let thick = stock.random_range(0.0032_f32..0.0068).min(size.z * 0.37);
        let face = rear + thick;
        let module = Vec2::new(size.x / p.gangs as f32, size.y) * p.control_fraction;
        let gap = stock
            .random_range(0.00065_f32..0.00145)
            .min(module.min_element() * 0.05)
            .min(thick * 0.24);
        let mut bank = Self {
            modules: Vec::with_capacity(p.gangs),
            case_size: size,
            module_size: module,
            rear_z_m: rear,
            face_z_m: face,
            pocket_z_m: rear + thick * 0.15,
            gap_m: gap,
            corner_m: (stock.random_range(0.0014_f32..0.009)).min(size.x.min(size.y) * 0.09),
            edge_m: (stock.random_range(0.00055_f32..0.0019)).min(thick * 0.34),
            finish: if stock.random_bool(0.24) {
                Surface::Metal
            } else {
                Surface::Plastic
            },
            screws: stock.random_bool(0.49),
            finish_lot: stock.random_range(0..variants::COUNT),
        };
        let max_tilt = ((bank.front_limit() - bank.rocker_low()) / bank.rocker_height()).atan();
        for i in 0..p.gangs {
            let mut r = stream(o.seed, u64::from_le_bytes(*b"SwGang  ") + i as u64);
            let kind = if i > 0 && r.random_bool(0.28) {
                match primary {
                    0 | 3 => [0, 2, 3, 4][r.random_range(0..4)],
                    1 | 2 | 4 => [0, 1, 2, 4][r.random_range(0..4)],
                    _ => 5,
                }
            } else {
                primary
            };
            let signed = |mag: f32, up: bool| if up { mag } else { -mag };
            let mut tilt = || signed(r.random_range(0.083_f32..0.215).min(max_tilt), r.random());
            bank.modules.push(match kind {
                0 => Actuator::Paddle { angle_rad: tilt() },
                1 => Actuator::Toggle {
                    angle_rad: signed(r.random_range(0.38..0.70), r.random()),
                },
                2 => Actuator::Rotary {
                    angle_rad: r.random_range(-2.36..2.36),
                },
                3 => Actuator::Split {
                    angles_rad: [tilt(), tilt()],
                },
                4 => Actuator::Slider {
                    level: r.random_range(0.0..=1.),
                },
                _ => Actuator::Buttons {
                    latched: r.random_bool(0.56).then(|| r.random_range(0..2)),
                },
            });
        }
        bank
    }

    fn center(&self, i: usize) -> Vec2 {
        let n = self.modules.len() as f32;
        Vec2::new(
            (i as f32 + 0.5 - n * 0.5) * self.case_size.x / n,
            self.case_size.y * 0.5,
        )
    }
    fn front_limit(&self) -> f32 {
        self.case_size.z * 0.48
    }
    fn rocker_low(&self) -> f32 {
        self.face_z_m - (self.face_z_m - self.pocket_z_m) * 0.42
    }
    fn rocker_height(&self) -> f32 {
        self.module_size.y - 2. * self.gap_m
    }
    fn label(&self, surface: Surface, class: &str) -> String {
        let slot =
            coated_stock::slot(surface, self.finish_lot).expect("metal or molded polymer stock");
        format!("{class}#finish{slot}")
    }
    pub(super) fn build(&self, a: &mut Assembly, label: &str) {
        mesh::build(self, a, label);
    }
}
