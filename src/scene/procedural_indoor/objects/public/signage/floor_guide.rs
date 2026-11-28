//! Landing identification carried by the admitted stand, including archived rooms.
use super::*;

pub(crate) const INTRODUCED: u32 = 322;
const OBJECT_TAG: u32 = 1 << 20;
pub(crate) const FIRST_SLOT: usize = 1 << 14;
pub(crate) const END_SLOT: usize = 2 * FIRST_SLOT;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Connection {
    Lift,
    Stairs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct FloorGuide {
    pub connection: Connection,
    pub level: u8,
    pub highest: u8,
}

impl FloorGuide {
    pub(crate) fn lift(level: u8, highest: u8) -> Self {
        Self {
            connection: Connection::Lift,
            level,
            highest,
        }
    }

    pub(crate) fn stair(elevation: f32, ground: f32, deck_height: f32) -> Option<Self> {
        let upper = (elevation - deck_height).abs() < 0.01;
        (upper || (elevation - ground).abs() < 0.01).then_some(Self {
            connection: Connection::Stairs,
            level: u8::from(upper),
            highest: 1,
        })
    }

    fn code(self) -> u32 {
        u32::from(self.level)
            | (u32::from(self.highest) << 5)
            | (u32::from(self.connection == Connection::Stairs) << 10)
    }

    fn decode(code: u32) -> Option<Self> {
        let level = (code & 31) as u8;
        let highest = ((code >> 5) & 31) as u8;
        let stair = code & 1024 != 0;
        (code < 2048 && (1..=29).contains(&highest) && level <= highest && (!stair || highest == 1))
            .then_some(Self {
                connection: if stair {
                    Connection::Stairs
                } else {
                    Connection::Lift
                },
                level,
                highest,
            })
    }

    pub(crate) fn identify(self, o: &mut IndoorObject) {
        // Low byte retains the original furniture variation for archived consumers.
        o.variant = OBJECT_TAG | (self.code() << 8) | (o.variant & 255);
    }

    pub fn for_object(o: &IndoorObject, version: u32) -> Option<Self> {
        if version < INTRODUCED
            || o.kind != ObjectKind::InformationStand
            || o.variant & !0x7ffff != OBJECT_TAG
        {
            return None;
        }
        Self::decode((o.variant >> 8) & 2047)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Edition {
    pub guide: FloorGuide,
    pub landscape: bool,
    pub stock: u8,
}

impl Edition {
    pub fn aspect(self) -> f32 {
        if self.landscape {
            std::f32::consts::SQRT_2
        } else {
            std::f32::consts::FRAC_1_SQRT_2
        }
    }
    pub fn slot(self) -> usize {
        FIRST_SLOT
            + self.guide.code() as usize
            + (usize::from(self.landscape) << 11)
            + (usize::from(self.stock) << 12)
    }
    pub fn from_slot(slot: usize) -> Option<Self> {
        if !(FIRST_SLOT..END_SLOT).contains(&slot) {
            return None;
        }
        let bits = slot - FIRST_SLOT;
        Some(Self {
            guide: FloorGuide::decode((bits & 2047) as u32)?,
            landscape: bits & 2048 != 0,
            stock: (bits >> 12) as u8,
        })
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Sheet {
    pub edition: Edition,
    pub center_m: Vec2,
    pub size_m: Vec2,
}

impl Sheet {
    pub fn for_stand(o: &IndoorObject, version: u32, p: &SignageProgram) -> Option<Self> {
        let guide = FloorGuide::for_object(o, version)?;
        let panel = o.size.truncate() * Vec2::new(p.width_fraction, p.height_fraction);
        let b = panel.min_element() * p.border_fraction;
        let lo = -0.5 * panel + panel * Vec2::new(p.margins[0], p.margins[2]) + b;
        let hi = 0.5 * panel - panel * Vec2::new(p.margins[1], p.margins[3]) - b;
        let available = hi - lo;
        let edition = Edition {
            guide,
            landscape: available.x > available.y,
            stock: stream(
                o.finish_seed.unwrap_or(o.seed),
                u64::from_le_bytes(*b"WayStock"),
            )
            .random_range(0..4),
        };
        let height = available.y.min(available.x / edition.aspect());
        Some(Self {
            edition,
            center_m: (hi + lo) * 0.5,
            size_m: height * Vec2::new(edition.aspect(), 1.),
        })
    }

    /// The frame terminates at the printed insert. No broad backing face sits behind the ink.
    pub(super) fn panel(self, board: &mut Assembly, finish: Surface, label: &str, size: Vec3) {
        let half = size.truncate() * 0.5;
        let paper = self.size_m * 0.5;
        let front = size.z * 0.5;
        let bevel = (size.z * 0.14).min(self.size_m.min_element() * 0.012);
        let ring = |center: Vec2, half: Vec2, z: f32| {
            [
                Vec2::new(-1., -1.),
                Vec2::new(1., -1.),
                Vec2::ONE,
                Vec2::new(-1., 1.),
            ]
            .map(|corner| (center + half * corner).extend(z))
        };
        let back = ring(Vec2::ZERO, half, -front);
        let edge = ring(Vec2::ZERO, half, front - bevel);
        let rim = ring(Vec2::ZERO, half - bevel, front);
        let lip = ring(self.center_m, paper + bevel * 0.6, front);
        let insert = ring(self.center_m, paper, front - bevel);
        let g = board.part(finish, label);
        g.metric_quad(back, Vec3::NEG_Z);
        for (i, outside) in [Vec3::NEG_Y, Vec3::X, Vec3::Y, Vec3::NEG_X]
            .into_iter()
            .enumerate()
        {
            let j = (i + 1) % 4;
            for (a, b, normal) in [
                (back, edge, outside),
                (edge, rim, Vec3::Z),
                (rim, lip, Vec3::Z),
                (lip, insert, Vec3::Z),
            ] {
                g.metric_quad([a[i], a[j], b[j], b[i]], normal);
            }
        }
        let label = format!("{label}#finish{}#uv_local", self.edition.slot());
        board.part(Surface::PrintedPaper, &label).vertex_quad(
            insert,
            [Vec3::Z; 4],
            [[0., 1.], [1., 1.], [1., 0.], [0., 0.]],
        );
    }
}

#[cfg(test)]
mod tests;
