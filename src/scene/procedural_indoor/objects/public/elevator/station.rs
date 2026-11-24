//! A landing owns its service directions. Door travel, individual car position
//! and contact hardware remain car-specific, independently seeded decisions.
use super::*;
use crate::scene::procedural_indoor::materials::appliance_panel::Film;

mod mesh;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) struct Levels {
    pub floor: u8,
    pub highest: u8,
}

impl Levels {
    // Preserve the historical bare core variant (20). The tag distinguishes the
    // two five-bit indices from other categories' unrelated variation codes.
    const TAG: u32 = 1 << 16;
    pub(crate) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"LiftFlrs"));
        let highest = if rng.random_bool(0.72) {
            rng.random_range(2..=9)
        } else {
            rng.random_range(10..=29)
        };
        let floor = match rng.random_range(0..100) {
            0..=28 => 0,
            29..=41 => highest,
            _ => rng.random_range(1..highest),
        };
        Self { floor, highest }
    }
    pub(crate) fn core_variant(self) -> u32 {
        Self::TAG | 20 | (u32::from(self.floor) << 5) | (u32::from(self.highest) << 10)
    }
    pub(super) fn decode(variant: u32) -> Option<Self> {
        let levels = Self {
            floor: ((variant >> 5) & 31) as u8,
            highest: ((variant >> 10) & 31) as u8,
        };
        (variant == levels.core_variant()
            && (1..=29).contains(&levels.highest)
            && levels.floor <= levels.highest)
            .then_some(levels)
    }
    pub(super) fn for_car(o: &IndoorObject) -> Self {
        Self::decode(o.variant).unwrap_or_else(|| Self::sample(o.seed))
    }
    pub(super) fn display(self, seed: u64, opening: f32) -> u8 {
        if opening > 0. {
            return self.floor;
        }
        let mut rng = stream(seed, u64::from_le_bytes(*b"LiftPosn"));
        if rng.random_bool(0.53) {
            self.floor
        } else {
            rng.random_range(0..=self.highest)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Direction {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReadoutMount {
    Header,
    Jamb,
    Integrated,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Station {
    pub levels: Levels,
    pub displayed_floor: u8,
    pub call_directions: [Option<Direction>; 2],
    pub registered: Option<Direction>,
    pub horizontal: bool,
    pub mount: ReadoutMount,
    pub plate_center_m: Vec2,
    pub plate_size_m: Vec2,
    pub button_center_m: Vec2,
    pub button_diameter_m: f32,
    pub button_pitch_m: f32,
    pub button_exponent: f32,
    pub readout_center_m: Vec2,
    pub readout_size_m: Vec2,
    pub face_z_m: f32,
    pub plate_depth_m: f32,
    pub key_film: Film,
    pub display_film: Film,
    pub plate_finish: Surface,
    pub brushed_stock: BrushedHardware,
    pub amber_confirmation: bool,
}

impl Station {
    pub(crate) fn for_object(o: &IndoorObject) -> Self {
        Self::sample(o, car_size(o), &sample(o))
    }
    pub(super) fn sample(o: &IndoorObject, car: Vec3, p: &ElevatorProgram) -> Self {
        use Direction::{Down, Up};
        let levels = Levels::for_car(o);
        let calls = match levels.floor {
            0 => [Some(Up), None],
            n if n == levels.highest => [Some(Down), None],
            _ => [Some(Up), Some(Down)],
        };
        let mut rng = stream(o.seed, u64::from_le_bytes(*b"LiftCall"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"LiftFace"),
        );
        let core = is_core(o);
        let reveal = if core { o.size.x * 0.016 } else { 0. };
        let half = car.x * 0.5;
        let left = p.portal_left * car.x - reveal;
        let right = p.portal_right * car.x + reveal;
        let jambs = [(-half, left), (right, half)];
        let mut index = usize::from(p.control_side > 0.);
        if jambs[index].1 - jambs[index].0 < 0.145
            && jambs[1 - index].1 - jambs[1 - index].0 > jambs[index].1 - jambs[index].0
        {
            index = 1 - index;
        }
        let (lo, hi) = jambs[index];
        let clear = hi - lo;
        let width = rng.random_range(0.086..0.153_f32).min(clear * 0.78);
        let x = rng.random_range(lo + width * 0.55..hi - width * 0.55);
        let diameter = rng.random_range(0.032..0.048_f32).min(width * 0.52);
        let pitch = diameter * rng.random_range(1.34..1.68);
        let two = calls[1].is_some();
        let horizontal = two && width > pitch + diameter * 1.38 && rng.random_bool(0.24);
        let mount = match rng.random_range(0..100) {
            0..=49 => ReadoutMount::Header,
            50..=78 => ReadoutMount::Jamb,
            _ => ReadoutMount::Integrated,
        };
        let integrated = mount == ReadoutMount::Integrated;
        let button_height = diameter + if two && !horizontal { pitch } else { 0. };
        let inset = rng.random_range(0.015..0.034);
        let button_y = rng.random_range(0.93..1.18);
        let button_center = Vec2::new(x, button_y);
        let jamb_readout_width = 1.8 * (x - lo).min(hi - x);
        let display_width = match mount {
            ReadoutMount::Header => rng.random_range(0.136..0.252),
            ReadoutMount::Jamb => rng.random_range(0.109..0.192_f32).min(jamb_readout_width),
            ReadoutMount::Integrated => width * rng.random_range(0.66..0.81),
        };
        let display_size = Vec2::new(
            display_width,
            (display_width * rng.random_range(0.37..0.57)).min(0.099),
        );
        let extra = if integrated {
            display_size.y + inset
        } else {
            0.
        };
        let portal_top = p.portal_top * car.y + if core { 0.016 } else { 0. };
        let readout_center = match mount {
            ReadoutMount::Header => Vec2::new(
                (left + right) * 0.5,
                (portal_top + rng.random_range(0.10..0.21))
                    .min(o.size.y - display_size.y * 0.5 - 0.015),
            ),
            ReadoutMount::Jamb => Vec2::new(x, rng.random_range(1.54..1.88)),
            ReadoutMount::Integrated => {
                Vec2::new(x, button_y + (button_height + display_size.y) * 0.5 + inset)
            }
        };
        let face = o.size.z * if core { 0.48 } else { 0.49 };
        let registered = if p.opening == 0. && rng.random_bool(0.57) {
            if two && rng.random_bool(0.5) {
                calls[1]
            } else {
                calls[0]
            }
        } else {
            None
        };
        Self {
            levels,
            displayed_floor: levels.display(o.seed, p.opening),
            call_directions: calls,
            registered,
            horizontal,
            mount,
            plate_center_m: button_center + Vec2::Y * (extra * 0.5),
            plate_size_m: Vec2::new(width, button_height + inset * 2. + extra),
            button_center_m: button_center,
            button_diameter_m: diameter,
            button_pitch_m: pitch,
            button_exponent: if rng.random_bool(0.66) {
                2.
            } else {
                rng.random_range(3.7..7.3)
            },
            readout_center_m: readout_center,
            readout_size_m: display_size,
            face_z_m: face,
            plate_depth_m: rng
                .random_range(0.0042..0.0085_f32)
                .min((o.size.z * 0.50 - face) * 0.48),
            key_film: if finish.random_bool(0.58) {
                Film::DarkKey
            } else {
                Film::LightKey
            },
            display_film: if finish.random_bool(0.81) {
                Film::Led
            } else {
                Film::ReflectiveLcd
            },
            plate_finish: [Surface::Metal, Surface::Chrome, Surface::Plastic]
                [finish.random_range(0..3)],
            brushed_stock: BrushedHardware::for_landing(o),
            amber_confirmation: finish.random_bool(0.58),
        }
    }
    pub(super) fn build(&self, a: &mut Assembly) {
        mesh::build(a, self);
    }
}
