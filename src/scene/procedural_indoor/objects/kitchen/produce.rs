//! Countertop produce stays at grocery scale. A fruit is admitted only when its
//! silhouette clears the vessel and its neighbours while its foot rests on the base.
use super::*;
use crate::scene::procedural_indoor::materials::produce_skin::Skin;
mod fruit;
mod legacy;
#[cfg(test)]
mod tests;
mod vessel;
pub(crate) use fruit::{Fruit, Variety};
pub(crate) use vessel::Vessel;

pub(crate) const INTRODUCED: u32 = 230;

#[derive(Debug, serde::Serialize)]
pub(crate) struct Arrangement {
    pub vessel: Vessel,
    pub fruit: Vec<Fruit>,
}

impl Arrangement {
    pub(crate) fn sample(o: &IndoorObject) -> Self {
        let vessel = Vessel::sample(o);
        let mut choice = stream(o.seed, u64::from_le_bytes(*b"FruitBag"));
        let mut location = stream(o.seed, u64::from_le_bytes(*b"FruitPos"));
        let mut grade = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"FruitLot"),
        );
        let main = Variety::draw(&mut choice);
        let second = Variety::draw(&mut choice);
        let homogeneity = choice.random_range(0.12..0.98);
        let requested = match choice.random_range(0..100) {
            0..=2 => 0,
            3..=18 => 1,
            _ => choice.random_range(2..=8),
        };
        let mut fruit: Vec<Fruit> = Vec::with_capacity(requested);
        // Avoid shrinking later pieces to squeeze in an impossible order.
        for _ in 0..requested * 2 {
            if fruit.len() == requested {
                break;
            }
            let kind = if choice.random_bool(homogeneity) {
                main
            } else if choice.random_bool(0.52) {
                second
            } else {
                Variety::draw(&mut choice)
            };
            let mut piece = Fruit::sample(kind, &mut choice, &mut grade);
            if vessel.floor_m + piece.total_height() > o.size.y {
                continue;
            }
            let radius = piece.clearance_radius();
            let limit = piece.center_limit(&vessel);
            if limit <= 0. {
                continue;
            }
            for _ in 0..18 {
                let turn = location.random_range(0.0..TAU);
                let d = Vec2::from_angle(turn);
                let position = if !fruit.is_empty() && location.random_bool(0.53) {
                    let n = &fruit[location.random_range(0..fruit.len())];
                    n.center_m
                        + d * (n.clearance_radius() + radius + location.random_range(0.0008..0.012))
                } else {
                    d * vessel.rim_axes_m * (limit * location.random::<f32>().sqrt())
                };
                if (position / vessel.rim_axes_m).length_squared() <= limit * limit
                    && fruit.iter().all(|other| {
                        other.center_m.distance_squared(position)
                            >= (other.clearance_radius() + radius + 0.0006).powi(2)
                    })
                {
                    piece.center_m = position;
                    fruit.push(piece);
                    break;
                }
            }
        }
        Self { vessel, fruit }
    }
}

/// Several real pieces from the same cultivar, shared by facings in one retail
/// crate. Each shape retains its own lean, lobing, filtered peel and stem.
pub(in crate::scene::procedural_indoor::objects) fn crate_lot(
    r: &mut impl Rng,
) -> [(f32, Assembly); 3] {
    let kind = Variety::draw(r);
    let mut finish = stream(r.random(), u64::from_le_bytes(*b"ShopPeel"));
    lot_pieces(kind, r, &mut finish).map(|piece| {
        let mut mesh = Assembly::default();
        piece.build(&mut mesh, 0.);
        (piece.clearance_radius() + 0.001, mesh)
    })
}

pub(in crate::scene::procedural_indoor::objects) fn market_lot(
    seed: u64,
    finish_seed: u64,
    lot: usize,
) -> [Fruit; 3] {
    let mut shape = stream(seed, u64::from_le_bytes(*b"BinFruit") ^ lot as u64);
    let mut finish = stream(finish_seed, u64::from_le_bytes(*b"BinGrade") ^ lot as u64);
    lot_pieces(Variety::draw(&mut shape), &mut shape, &mut finish)
}

fn lot_pieces(kind: Variety, r: &mut impl Rng, finish: &mut impl Rng) -> [Fruit; 3] {
    let mut cultivar = None;
    std::array::from_fn(|_| {
        let mut piece = Fruit::sample(kind, r, finish);
        piece.skin = *cultivar.get_or_insert(piece.skin);
        piece
    })
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if version < INTRODUCED {
        return legacy::build(a, o);
    }
    let arrangement = Arrangement::sample(o);
    arrangement.vessel.build(a, o);
    for piece in &arrangement.fruit {
        piece.build(a, arrangement.vessel.floor_m);
    }
}
