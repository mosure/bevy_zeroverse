//! A fresh-market table carries removable crates on one or two bearing surfaces.
//! Stock, the catalog and room audits use the same metric locations.
use super::*;
use crate::scene::procedural_indoor::objects::kitchen::produce::{market_lot, Fruit};
use serde::Serialize;

mod bins;
mod frame;
mod stock;
#[cfg(test)]
mod tests;
pub(crate) const VERSION: u32 = 325;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ProduceBase {
    OpenFrame,
    EndPanels,
    Plinth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CrateStyle {
    TimberSlats,
    VentedTote,
    FormedTray,
}

impl CrateStyle {
    fn sample(r: &mut impl Rng) -> Self {
        match r.random_range(0..10) {
            0..4 => Self::TimberSlats,
            4..8 => Self::VentedTote,
            _ => Self::FormedTray,
        }
    }

    fn wall_m(self) -> f32 {
        match self {
            Self::TimberSlats => 0.014,
            Self::VentedTote => 0.008,
            Self::FormedTray => 0.009,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ProduceBin {
    /// Fixture-local coordinates of the centre of the crate's bearing face.
    pub base_m: Vec3,
    pub size_xz_m: Vec2,
    pub rim_height_m: f32,
    pub style: CrateStyle,
    pub lot: usize,
    pub stock: Vec<ProducePiece>,
}

impl ProduceBin {
    fn floor_m(&self) -> f32 {
        self.base_m.y + self.style.wall_m()
    }
}

#[derive(Debug, Serialize)]
pub struct ProducePiece {
    /// Position relative to the bin centre, sitting on its interior floor.
    pub position_xz_m: Vec2,
    pub yaw: f32,
    pub shape: usize,
}

#[derive(Debug, Serialize)]
pub struct ProduceStand {
    pub base: ProduceBase,
    pub front_deck_m: f32,
    /// Z coordinate of the boundary between rows. Local +Z faces the customer aisle.
    pub row_split_m: Option<f32>,
    /// Zero also admits a two-row display on one level.
    pub rear_rise_m: f32,
    pub bins: Vec<ProduceBin>,
    pub resin_slot: usize,
    #[serde(skip)]
    lots: Vec<[Fruit; 3]>,
}

pub fn produce_stand(o: &IndoorObject, version: u32) -> Option<ProduceStand> {
    if version < VERSION || o.kind != ObjectKind::GroceryDisplay || o.size.y >= 1.15 {
        return None;
    }
    let Installation::Produce { crates, fill, .. } = parameters(o) else {
        return None;
    };
    Some(ProduceStand::sample(o, crates, fill))
}

pub(crate) fn resin_slot(o: &IndoorObject) -> usize {
    let mut r = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"CrateDye"),
    );
    crate::scene::procedural_indoor::materials::basket_resin::sample_slot(&mut r)
}

impl ProduceStand {
    pub(in super::super) fn sample(o: &IndoorObject, crates: usize, fill: f32) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"FruitBay"));
        let base = match r.random_range(0..10) {
            0..4 => ProduceBase::OpenFrame,
            4..7 => ProduceBase::EndPanels,
            _ => ProduceBase::Plinth,
        };
        let rows = if r.random_bool(0.64) { 2 } else { 1 };
        let split = (rows == 2).then(|| r.random_range(-0.12..0.15) * o.size.z);
        let rise = if rows == 2 && r.random_bool(0.73) {
            r.random_range(0.085..0.193)
        } else {
            0.
        };
        let front = o.size.y - r.random_range(0.154..0.192) - rise;
        let edge = r.random_range(0.017..0.038);
        let gap = r.random_range(0.012..0.031);
        let primary_style = CrateStyle::sample(&mut r);
        let lot_count = r.random_range(1..=(crates * rows).min(5));
        let lots = (0..lot_count)
            .map(|lot| market_lot(o.seed, o.finish_seed.unwrap_or(o.seed), lot))
            .collect::<Vec<_>>();
        let mut bins = Vec::with_capacity(crates * rows);
        for row in 0..rows {
            let mut near = -0.5 * o.size.z + edge;
            let mut far = 0.5 * o.size.z - edge;
            let mut y = front;
            if let Some(split) = split {
                if row == 0 {
                    near = split + gap * 0.5;
                } else {
                    far = split - gap * 0.5;
                    y += rise;
                }
            }
            let requested = if row == 1 {
                r.random_range(crates.saturating_sub(1).max(1)..=(crates + 1).min(4))
            } else {
                crates
            };
            // A wide sales deck has multiple handling bays even if the rear row is sparse.
            let usable_width = o.size.x - 2. * edge;
            let minimum = ((usable_width + gap) / (0.98 + gap)).ceil().clamp(1., 4.) as usize;
            let count = requested.max(minimum);
            let weights: [f32; 4] = std::array::from_fn(|_| r.random_range(0.72..1.30));
            let unit =
                (usable_width - (count - 1) as f32 * gap) / weights[..count].iter().sum::<f32>();
            let mut x = -0.5 * o.size.x + edge;
            let mut lot = r.random_range(0..lot_count);
            for &weight in &weights[..count] {
                if r.random_bool(0.63) {
                    lot = r.random_range(0..lot_count);
                }
                let width = unit * weight;
                let style = if r.random_bool(0.79) {
                    primary_style
                } else {
                    CrateStyle::sample(&mut r)
                };
                let mut bin = ProduceBin {
                    base_m: Vec3::new(x + width * 0.5, y, (near + far) * 0.5),
                    size_xz_m: Vec2::new(width, far - near),
                    rim_height_m: r.random_range(0.065..0.127),
                    style,
                    lot,
                    stock: Vec::new(),
                };
                let mut stock =
                    stream(o.seed, u64::from_le_bytes(*b"FruitPos") ^ bins.len() as u64);
                bin.fit_to_bay(&mut stock);
                bin.load(&lots[lot], fill, o.size.y, &mut stock);
                bins.push(bin);
                x += width + gap;
            }
        }
        Self {
            base,
            front_deck_m: front,
            row_split_m: split,
            rear_rise_m: rise,
            bins,
            resin_slot: resin_slot(o),
            lots,
        }
    }

    pub(in super::super) fn build(&self, a: &mut Assembly, o: &IndoorObject, finish: Surface) {
        let mut cuts = TimberCuts::new(o.seed, u64::from_le_bytes(*b"FruitCut"));
        self.frame(a, o, finish, &mut cuts);
        let resin = format!("{FIXTURE}#finish{}", self.resin_slot);
        for bin in &self.bins {
            bin.crate_mesh(a, &mut cuts, &resin);
        }
        self.build_stock(a);
    }
}
