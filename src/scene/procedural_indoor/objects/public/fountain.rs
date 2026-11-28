//! Wall-hung drinking fountains. Local +Z faces the user. The installation
//! height, pressed bowl, water outlets and catalogue bounds use the same stock.
use super::*;
mod basin;
mod mesh;
#[cfg(test)]
mod tests;
pub(super) use mesh::build;

pub(crate) const VERSION: u32 = 324;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Form {
    Bracket,
    Cooler,
    BottleFiller,
}

struct Stock {
    form: Form,
    size: Vec3,
    deck: f32,
}

impl Stock {
    fn sample(seed: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"FtnStock"));
        let form = match r.random_range(0..100) {
            0..30 => Form::Bracket,
            30..62 => Form::Cooler,
            _ => Form::BottleFiller,
        };
        let (width, depth, deck) = if form == Form::Bracket {
            (
                r.random_range(0.325..0.464),
                r.random_range(0.35..0.493),
                r.random_range(0.215..0.292),
            )
        } else {
            (
                r.random_range(0.419..0.542),
                r.random_range(0.421..0.526),
                r.random_range(0.295..0.385),
            )
        };
        let rise = if form == Form::BottleFiller {
            r.random_range(0.404..0.560)
        } else {
            r.random_range(0.093..0.146)
        };
        Self {
            form,
            size: Vec3::new(width, deck + rise, depth),
            deck,
        }
    }
}

pub(super) fn dimensions(seed: u64) -> Vec3 {
    Stock::sample(seed).size
}

/// Bottom of the installed unit above its approach floor. This is also used in
/// broad-phase clearance tests, so resolving it requires no RNG construction.
pub fn mount_height_m(seed: u64) -> f32 {
    let fraction = (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 48) as u16;
    0.505 + 0.155 * f32::from(fraction) / f32::from(u16::MAX)
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Fountain {
    pub form: Form,
    pub bowl: Surface,
    pub cabinet: Surface,
    pub roundness: f32,
    pub deck_m: f32,
    pub basin_depth_m: f32,
    pub throat_radius_m: f32,
    pub left_bubbler: bool,
    pub push_width_fraction: f32,
    pub guard_length_m: f32,
}

impl Fountain {
    pub fn for_object(o: &IndoorObject) -> Self {
        let stock = Stock::sample(o.seed);
        let mut r = stream(o.seed, u64::from_le_bytes(*b"FtnPress"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"FtnFinsh"),
        );
        Self {
            form: stock.form,
            deck_m: stock.deck * (o.size.y / stock.size.y),
            bowl: if stock.form == Form::Bracket && finish.random_bool(0.43) {
                Surface::Ceramic
            } else {
                Surface::Metal
            },
            cabinet: [
                Surface::Metal,
                Surface::Metal,
                Surface::Plastic,
                Surface::Ink,
            ][finish.random_range(0..4)],
            roundness: if stock.form == Form::Bracket {
                r.random_range(2.0..4.2)
            } else {
                r.random_range(3.0..6.8)
            },
            basin_depth_m: o.size.z * r.random_range(0.135..0.205),
            throat_radius_m: r.random_range(0.014..0.021),
            left_bubbler: r.random(),
            push_width_fraction: r.random_range(0.38..0.68),
            guard_length_m: r.random_range(0.054..0.073),
        }
    }

    fn bowl_center(size: Vec3) -> Vec3 {
        Vec3::Z * (0.012 * size.z)
    }

    fn face(self, size: Vec3) -> Transform {
        let side = if self.left_bubbler { -1. } else { 1. };
        let base = Vec3::new(side * 0.439 * size.x, self.deck_m - 0.003, 0.035 * size.z);
        Transform::from_translation(base).with_rotation(Quat::from_rotation_z(side * 0.69))
    }
}
