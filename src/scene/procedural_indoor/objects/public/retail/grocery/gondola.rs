//! Adjustable, double-sided shelves for ambient groceries.
use super::*;

mod stock;
#[cfg(test)]
mod tests;

pub(in super::super) const VERSION: u32 = 309;

#[derive(Clone, Copy)]
struct Deck {
    top: f32,
    projection: f32,
    clearance: f32,
}

struct Gondola {
    seed: u64,
    width: f32,
    spine: f32,
    plinth: f32,
    thickness: f32,
    lip: f32,
    pitch: f32,
    facings: usize,
    stock: f32,
    decks: Vec<Deck>,
    products: [(Sku, Vec3); 4],
}

impl Gondola {
    fn sample(o: &IndoorObject, levels: usize, facings: usize, stock: f32) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"ShopDeck"));
        let plinth = r.random_range(0.073..0.14);
        let thickness = r.random_range(0.016..0.029);
        let lip = r.random_range(0.010..0.027);
        let spine = r.random_range(0.027..0.052);
        let taper = r.random_range(0.0..0.28);
        let pitch = (o.size.x - 0.08) / facings as f32;
        let mut top = plinth + thickness;
        let weights: Vec<f32> = (0..levels).map(|_| r.random_range(0.79..1.24)).collect();
        let unit = (o.size.y - top) / weights.iter().sum::<f32>();
        let mut decks = Vec::with_capacity(levels);
        for (i, weight) in weights.into_iter().enumerate() {
            let step = weight * unit;
            let fraction = i as f32 / (levels - 1) as f32;
            decks.push(Deck {
                top,
                projection: 0.5 * (o.size.z - spine) * (1. - taper * fraction),
                clearance: if i + 1 == levels {
                    step
                } else {
                    step - thickness - lip
                },
            });
            top += step;
        }
        let clearance = decks
            .iter()
            .map(|deck| deck.clearance)
            .fold(f32::INFINITY, f32::min);
        let bags = r.random_range(0.16..0.80);
        let products = assortment(o).map(|mut sku| {
            let mut shape = stream(o.seed, u64::from_le_bytes(*b"ShopWrap") ^ sku.lot as u64);
            if sku.pack == Pack::Carton && shape.random_bool(bags) {
                sku.pack = Pack::PaperBag;
            }
            let size = sku.pack.size(pitch, clearance, &mut shape);
            (sku, size)
        });
        Self {
            seed: o.seed,
            width: o.size.x,
            spine,
            plinth,
            thickness,
            lip,
            pitch,
            facings,
            stock,
            decks,
            products,
        }
    }

    fn frame(&self, a: &mut Assembly, size: Vec3, frame: Surface) {
        let Self {
            width,
            spine,
            plinth,
            thickness,
            lip,
            ..
        } = *self;
        board(
            a,
            frame,
            FIXTURE,
            Vec3::Y * (plinth * 0.5),
            Vec3::new(width - 0.08, plinth, size.z - 0.09),
        );
        board(
            a,
            frame,
            FIXTURE,
            Vec3::Y * ((size.y + plinth) * 0.5),
            Vec3::new(width - 0.05, size.y - plinth, spine),
        );
        for deck in &self.decks {
            for sign in [-1., 1.] {
                board(
                    a,
                    frame,
                    FIXTURE,
                    Vec3::new(
                        0.,
                        deck.top - thickness * 0.5,
                        sign * (spine + deck.projection) * 0.5,
                    ),
                    Vec3::new(width, thickness, deck.projection),
                );
                // The shelf edge turns down below its bearing surface. Recess
                // the return slightly so its face never coincides with the edge.
                let outer = spine * 0.5 + deck.projection;
                board(
                    a,
                    frame,
                    FIXTURE,
                    Vec3::new(0., deck.top - thickness - lip * 0.5, sign * (outer - 0.007)),
                    Vec3::new(width - 0.012, lip, 0.010),
                );
            }
        }
    }
}

pub(in super::super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    levels: usize,
    facings: usize,
    stock: f32,
    frame: Surface,
) {
    let rack = Gondola::sample(o, levels, facings, stock);
    rack.frame(a, o.size, frame);
    let mut lots: [Vec<(Vec3, f32, f32)>; 4] = Default::default();
    rack.place(|p| {
        let (sin, cos) = p.yaw.sin_cos();
        lots[p.sku].push((p.base, sin, cos));
    });
    build_lots(a, rack.products, lots);
}

/// Stock on one shelf face, in increasing fixture-local X. Occupied slots
/// identify a product lot, including when its foremost copy is set back.
#[derive(Debug, serde::Serialize, PartialEq)]
pub struct GondolaRow {
    pub level: usize,
    pub positive_z: bool,
    pub slots: Vec<Option<usize>>,
    pub side_by_side_copies: usize,
    pub rear_copies: usize,
}

/// Metric deck heights, usable depths on each side, and replayable merchandise
/// positions by shelf row. Product lot indices are local to this fixture.
#[derive(Debug, serde::Serialize, PartialEq)]
pub struct GondolaStock {
    pub deck_heights_m: Vec<f32>,
    pub deck_depths_m: Vec<f32>,
    pub package_forms: [&'static str; 4],
    pub rows: Vec<GondolaRow>,
}

/// Replay the same placements used by the mesh without constructing any geometry.
pub fn gondola_stock(o: &IndoorObject, generator_version: u32) -> Option<GondolaStock> {
    if generator_version < VERSION
        || o.kind != ObjectKind::GroceryDisplay
        || o.size.y < 1.15
        || is_refrigerated(o)
    {
        return None;
    }
    let Installation::Gondola {
        levels,
        facings,
        stock,
        ..
    } = parameters(o)
    else {
        return None;
    };
    let rack = Gondola::sample(o, levels, facings, stock);
    let mut rows: Vec<_> = (0..levels * 2)
        .map(|i| GondolaRow {
            level: i / 2,
            positive_z: i % 2 == 1,
            slots: vec![None; facings],
            side_by_side_copies: 0,
            rear_copies: 0,
        })
        .collect();
    rack.place(|p| {
        let row = &mut rows[p.row];
        row.slots[p.slot] = Some(p.sku);
        match p.facing {
            stock::Facing::Front => (),
            stock::Facing::Beside => row.side_by_side_copies += 1,
            stock::Facing::Behind => row.rear_copies += 1,
        }
    });
    Some(GondolaStock {
        deck_heights_m: rack.decks.iter().map(|d| d.top).collect(),
        deck_depths_m: rack.decks.iter().map(|d| d.projection).collect(),
        package_forms: rack.products.map(|(s, _)| match s.pack {
            Pack::Carton => "carton",
            Pack::PaperBag => "paper bag",
            Pack::Can => "can",
            Pack::Bottle => "bottle",
            Pack::Gable => "gable carton",
            Pack::Tub => "tub",
        }),
        rows,
    })
}
