//! Freestanding bottled-water dispensers. Local +Z is the cup approach.
//! Catalog dimensions, installed geometry and reported water volume share one stock.
use super::*;
mod bottle;
mod mesh;
#[cfg(test)]
mod tests;
pub use bottle::Bottle;
pub(super) use mesh::build;
pub(crate) const VERSION: u32 = 302;
const CLASS: &str = "appliance";
const STEEL: &str = "appliance#finish111#uv_local";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Loading {
    TopBottle,
    TopVacant,
    BottomCabinet,
}

struct Stock {
    loading: Loading,
    size: Vec3,
    body_height: f32,
    bottle: Option<Bottle>,
}

impl Stock {
    fn sample(seed: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"ColdVend"));
        let loading = match r.random_range(0..100) {
            0..57 => Loading::TopBottle,
            57..64 => Loading::TopVacant,
            _ => Loading::BottomCabinet,
        };
        let width: f32 = r.random_range(0.293..0.397);
        let depth = r.random_range(0.306..0.414);
        let body_height = r.random_range(0.938..1.116);
        let bottle =
            (loading == Loading::TopBottle).then(|| Bottle::sample(seed, width.min(depth)));
        let height = body_height + bottle.as_ref().map_or(0., |b| b.height_m - b.insertion_m);
        Self {
            loading,
            size: Vec3::new(width, height, depth),
            body_height,
            bottle,
        }
    }
}

pub(super) fn dimensions(seed: u64) -> Vec3 {
    Stock::sample(seed).size
}

pub(crate) fn approach_m(seed: u64) -> f32 {
    0.57 + 0.22 * f32::from((seed >> 15) as u16) / f32::from(u16::MAX)
}

#[derive(Debug, Clone, Serialize)]
pub struct Cooler {
    pub loading: Loading,
    pub body: Surface,
    pub buttons: bool,
    pub taps: usize,
    pub tray_bars: usize,
    pub roundedness: f32,
    pub body_height_m: f32,
    pub tray_height_m: f32,
    pub cup_clearance_m: f32,
    pub outlet_z_m: f32,
    pub bottle: Option<Bottle>,
}
impl Cooler {
    pub fn for_object(o: &IndoorObject) -> Self {
        let stock = Stock::sample(o.seed);
        let scale_y = o.size.y / stock.size.y;
        let mut r = stream(o.seed, u64::from_le_bytes(*b"CoolTaps"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"CoolTrim"),
        );
        let body_height_m = stock.body_height * scale_y;
        let buttons = r.random_bool(0.55);
        let taps = if buttons {
            r.random_range(1..=3)
        } else {
            r.random_range(2..=3)
        };
        Self {
            loading: stock.loading,
            body: [Surface::Metal, Surface::Plastic, Surface::Ink][finish.random_range(0..3)],
            buttons,
            taps,
            tray_bars: r.random_range(7..=12),
            roundedness: r.random_range(3.0..7.8),
            body_height_m,
            tray_height_m: body_height_m * r.random_range(0.54..0.63),
            cup_clearance_m: body_height_m * r.random_range(0.192..0.246),
            outlet_z_m: o.size.z * r.random_range(0.24..0.325),
            bottle: stock.bottle.map(|b| b.fit(scale_y, o.size.x.min(o.size.z))),
        }
    }
}
