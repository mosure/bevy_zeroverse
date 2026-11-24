//! Operator controls fitted to a fixed chassis or a bottom-hinged dishwasher door.
use super::*;
pub use crate::scene::procedural_indoor::materials::appliance_panel::{Film, Glyph};
use rand_chacha::ChaCha8Rng;

pub(super) mod legacy;
mod mesh;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 227;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mount {
    Chassis,
    DropDoor,
}

pub(super) fn door_console(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::Dishwasher
        && stream(o.seed, u64::from_le_bytes(*b"WashDoor")).random_bool(0.78)
}

/// The original front reserve has 68 mm beyond the chassis even before
/// door-travel expansion. Spend at most 45 mm on additional leaf length;
/// the remaining service plinth sets the lower edge of the rack opening.
pub(super) fn dishwasher_hinge_y(o: &IndoorObject) -> f32 {
    (o.size.y - 0.010 - program::dishwasher_door_height(o) - 0.045).max(0.075 + 0.024)
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Fascia {
    /// Local centre of the exposed face, in metres.
    pub face: Vec3,
    pub size: Vec2,
    pub mount: Mount,
    pub orientation: Quat,
    thickness: f32,
}

impl Fascia {
    pub(super) fn fitted(
        o: &IndoorObject,
        p: &KitchenProgram,
        product: &program::ApplianceProgram,
        front: f32,
    ) -> Self {
        let w = o.size.x - 0.018;
        let h = o.size.y;
        if o.kind == ObjectKind::Dishwasher {
            let low = h * (1. - product.fascia_fraction);
            let high = h - 0.01;
            let mut face = Vec3::new(0., (low + high) * 0.5, front + 0.025);
            let mounted = door_console(o);
            let orientation = if mounted {
                let hinge = Vec3::new(0., dishwasher_hinge_y(o), front + 0.014);
                let turn = Quat::from_rotation_x(p.opening_radians);
                face = hinge + turn * (face - hinge);
                turn
            } else {
                Quat::IDENTITY
            };
            return Self {
                face,
                size: Vec2::new(w * if mounted { 0.96 } else { 0.97 }, high - low),
                orientation,
                mount: if mounted {
                    Mount::DropDoor
                } else {
                    Mount::Chassis
                },
                thickness: 0.022,
            };
        }
        let fraction = product.control_fraction;
        let (center, size) = if p.controls == 2 {
            (Vec2::new(0., h * 0.17), Vec2::new(w * 0.96, h * 0.21))
        } else {
            let sign = if p.controls == 1 { -1. } else { 1. };
            (
                Vec2::new(sign * w * (1. - fraction) * 0.5, h * 0.51),
                Vec2::new(w * fraction - 0.006, h * 0.86),
            )
        };
        Self {
            face: center.extend(front + 0.0285),
            size,
            thickness: 0.025,
            orientation: Quat::IDENTITY,
            mount: Mount::Chassis,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Mechanical,
    DialAndKeys,
    Keypad,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Dial {
    pub center: Vec2,
    pub radius_m: f32,
    pub depth_m: f32,
    pub angle_rad: f32,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Key {
    pub center: Vec2,
    pub size: Vec2,
    pub glyph: Glyph,
}

impl Key {
    fn layout(
        family: Family,
        wash: bool,
        face_size: Vec2,
        offset: Vec2,
        axis: Vec2,
        r: &mut ChaCha8Rng,
    ) -> Vec<Self> {
        let portrait = axis.y != 0.;
        let mut presets = [Glyph::Blank; 6];
        let (symbols, cols, t, span): (&[Glyph], usize, f32, Vec2) = match family {
            Family::Mechanical if wash => (&[Glyph::Start], 1, 0.22, Vec2::new(0.16, 0.55)),
            Family::Mechanical => return Vec::new(),
            Family::DialAndKeys => (
                &[Glyph::Stop, Glyph::Start],
                2,
                0.,
                if portrait {
                    Vec2::new(0.76, 0.14)
                } else {
                    Vec2::new(0.24, 0.50)
                },
            ),
            Family::Keypad if !wash && portrait => (
                &[
                    Glyph::Number(1),
                    Glyph::Number(2),
                    Glyph::Number(3),
                    Glyph::Number(4),
                    Glyph::Number(5),
                    Glyph::Number(6),
                    Glyph::Number(7),
                    Glyph::Number(8),
                    Glyph::Number(9),
                    Glyph::Stop,
                    Glyph::Number(0),
                    Glyph::Start,
                ],
                3,
                0.09,
                Vec2::new(0.84, 0.58),
            ),
            Family::Keypad => {
                // Horizontal microwave panels use cooking shortcuts. The first
                // numbered dishwasher buttons select wash programs.
                let count = r.random_range(4..=6);
                for (i, glyph) in presets[..count - 2].iter_mut().enumerate() {
                    *glyph = Glyph::Number((i + 1) as u8);
                }
                presets[count - 2] = if wash { Glyph::Plus } else { Glyph::Stop };
                presets[count - 1] = Glyph::Start;
                (&presets[..count], count, 0.145, Vec2::new(0.57, 0.56))
            }
        };
        let rows = symbols.len().div_ceil(cols);
        let cell = face_size * span / Vec2::new(cols as f32, rows as f32);
        let height = (cell.y * r.random_range(0.70..0.89)).min(0.022);
        let width = (cell.x * r.random_range(0.73..0.91)).min(height * 1.62);
        let origin = offset + axis * t;
        symbols
            .iter()
            .enumerate()
            .map(|(i, &glyph)| {
                let grid = Vec2::new(
                    (i % cols) as f32 - (cols - 1) as f32 * 0.5,
                    (rows - 1) as f32 * 0.5 - (i / cols) as f32,
                );
                Self {
                    center: origin + grid * cell,
                    size: Vec2::new(width, height),
                    glyph,
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Readout {
    pub center: Vec2,
    pub size: Vec2,
    pub film: Film,
    /// A set or remaining duration. On an open door the timer is idle.
    pub timer_seconds: u32,
    pub blank: bool,
    pub hours_minutes: bool,
}

impl Readout {
    fn sample(
        o: &IndoorObject,
        opening: f32,
        face_size: Vec2,
        offset: Vec2,
        axis: Vec2,
        lcd: bool,
        r: &mut ChaCha8Rng,
    ) -> Self {
        let wash = o.kind == ObjectKind::Dishwasher;
        let portrait = axis.y != 0.;
        let aspect = r.random_range(2.8..3.9);
        let width = (face_size.x * if portrait { 0.74 } else { 0.24 })
            .min(face_size.y * 0.57 * aspect)
            .min(r.random_range(0.054..0.098));
        let mut state = stream(o.seed, u64::from_le_bytes(*b"CookTime"));
        let blank = state.random_bool(if wash { 0.39 } else { 0.21 });
        let idle = blank || opening > 0.02 || state.random_bool(0.33);
        Readout {
            center: offset - axis * r.random_range(0.26..0.32),
            size: Vec2::new(width, width / aspect),
            film: match (lcd, blank) {
                (true, false) => Film::ReflectiveLcd,
                (true, true) => Film::BlankLcd,
                (false, false) => Film::Led,
                (false, true) => Film::DarkWindow,
            },
            blank,
            hours_minutes: wash,
            timer_seconds: if idle {
                0
            } else if wash {
                state.random_range(12..=219) * 60
            } else {
                state.random_range(5..=899)
            },
        }
    }

    fn glyphs(self) -> [Glyph; 5] {
        if self.blank {
            return [Glyph::Blank; 5];
        }
        let units = if self.hours_minutes {
            self.timer_seconds / 60
        } else {
            self.timer_seconds
        };
        let major = units / 60;
        let minor = units % 60;
        [
            if major >= 10 {
                Glyph::Number((major / 10) as u8)
            } else {
                Glyph::Blank
            },
            Glyph::Number((major % 10) as u8),
            Glyph::Colon,
            Glyph::Number((minor / 10) as u8),
            Glyph::Number((minor % 10) as u8),
        ]
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Controls {
    pub fascia: Fascia,
    pub family: Family,
    pub dials: Vec<Dial>,
    pub keys: Vec<Key>,
    pub key_film: Film,
    pub dial_surface: Surface,
    pub readout: Option<Readout>,
}

impl Controls {
    pub(super) fn sample(o: &IndoorObject, p: &KitchenProgram, fascia: Fascia) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"CookCtrl"));
        let wash = o.kind == ObjectKind::Dishwasher;
        let family = match r.random_range(0..10) {
            0..=2 => Family::Mechanical,
            3..=5 => Family::DialAndKeys,
            _ => Family::Keypad,
        };
        let mut stock = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"KeyFilm!"),
        );
        let key_film = if stock.random_bool(0.69) {
            Film::DarkKey
        } else {
            Film::LightKey
        };
        let dial_surface =
            [Surface::Metal, Surface::Plastic, Surface::Rubber][stock.random_range(0..3)];
        let portrait = fascia.size.y > fascia.size.x;
        let sign = if !portrait && r.random_bool(0.48) {
            -1.
        } else {
            1.
        };
        // Across a horizontal strip, or down a side column. Positions are
        // relative to the fascia, never to the width of the appliance door.
        let axis = if portrait {
            Vec2::NEG_Y * fascia.size.y
        } else {
            Vec2::X * sign * fascia.size.x
        };
        let minor = fascia.size.min_element();
        let offset = axis * r.random_range(-0.020..0.020)
            + axis.normalize().perp() * (minor * r.random_range(-0.033..0.033));
        let mut dials = Vec::new();
        let mut dial = |t: f32, radius: f32, r: &mut ChaCha8Rng| {
            let radius_m = radius.min(minor * 0.32);
            dials.push(Dial {
                center: offset + axis * t,
                radius_m,
                depth_m: (radius_m * r.random_range(0.42..0.76)).min(0.014),
                angle_rad: r.random_range(-2.55..2.55),
            });
        };
        match family {
            Family::Mechanical if !wash => {
                let span = r.random_range(0.165..0.26);
                for t in [-span, span] {
                    let radius = r.random_range(0.013..0.024);
                    dial(t, radius, &mut r);
                }
            }
            Family::Mechanical | Family::DialAndKeys => {
                let radius = r.random_range(0.014..0.025);
                dial(
                    if wash && family == Family::Mechanical {
                        -0.26
                    } else {
                        0.27
                    },
                    radius,
                    &mut r,
                );
            }
            Family::Keypad => {}
        }
        let readout = (family != Family::Mechanical).then(|| {
            Readout::sample(
                o,
                p.opening_radians,
                fascia.size,
                offset,
                axis,
                stock.random_bool(0.52),
                &mut r,
            )
        });
        let keys = Key::layout(family, wash, fascia.size, offset, axis, &mut r);
        Self {
            fascia,
            family,
            keys,
            dials,
            key_film,
            dial_surface,
            readout,
        }
    }
}

/// Used by full material installation and catalog diagnostics. Selected scene
/// preparation reads the already-built mesh's finish slots directly.
pub fn for_object(o: &IndoorObject) -> Option<Controls> {
    if !matches!(o.kind, ObjectKind::Microwave | ObjectKind::Dishwasher) {
        return None;
    }
    let p = parameters(o);
    let (d, z) = body(o, &p);
    Some(Controls::sample(
        o,
        &p,
        Fascia::fitted(o, &p, &program::parameters(o), z + d * 0.5),
    ))
}
