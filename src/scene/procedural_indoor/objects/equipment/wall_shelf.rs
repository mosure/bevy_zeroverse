//! Wall-mounted timber, laminate and folded-steel shelves with fitted brackets.
use super::*;
mod deck;
mod legacy;
mod mesh;
pub use deck::{EndCut, ShelfDeck, ShelfEnd, ShelfStock};
pub(crate) const DECK_VERSION: u32 = 244;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ShelfProgram {
    /// Triangulated rods, angle straps, curved braces, or hidden supports.
    pub style: u8,
    pub thickness_m: f32,
    pub bracket_x_m: Vec<f32>,
    pub strap_width_m: f32,
    pub strap_thickness_m: f32,
    pub reach_fraction: f32,
    pub brace_radius_m: f32,
    pub brace_bend: f32,
    pub board: Surface,
    pub hardware: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deck: Option<ShelfDeck>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edging: Option<Surface>,
}

pub fn parameters(o: &IndoorObject) -> ShelfProgram {
    parameters_at(o, bevy_zeroverse_capture::GENERATOR_VERSION)
}

pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> ShelfProgram {
    let mut r = stream(o.seed, 0x57414c4c5348454c);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x5348454c46494e49);
    let style = r.random_range(0..4);
    let thickness = r
        .random_range(if style == 3 {
            0.030_f32..0.065
        } else {
            0.014..0.038
        })
        .min(o.size.y * 0.30);
    let count = if style == 3 {
        0
    } else {
        (o.size.x / r.random_range(0.44..0.75)).ceil().clamp(2., 4.) as usize
    };
    let span = o.size.x * r.random_range(0.46..0.82);
    let offset = (o.size.x - span) * r.random_range(-0.12..0.12);
    let regular = r.random_bool(0.35);
    let mut bracket_x = Vec::with_capacity(count);
    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        let jitter = if regular || i == 0 || i + 1 == count {
            0.
        } else {
            r.random_range(-0.14..0.14) / count as f32
        };
        bracket_x.push(offset + span * (t + jitter - 0.5));
    }
    let strap_thickness = r.random_range(0.006_f32..0.014).min(o.size.y * 0.09);
    let mut p = ShelfProgram {
        style,
        thickness_m: thickness,
        bracket_x_m: bracket_x,
        strap_width_m: r.random_range(0.014_f32..0.036).min(o.size.x * 0.045),
        strap_thickness_m: strap_thickness,
        reach_fraction: r.random_range(0.62..0.94),
        brace_radius_m: r
            .random_range(0.0035_f32..0.0070)
            .min(strap_thickness * 0.48),
        brace_bend: r.random_range(0.18..0.72),
        board: [Surface::Wood, Surface::Plastic, Surface::Metal][f.random_range(0..3)],
        hardware: [Surface::Metal, Surface::Chrome, Surface::WoodEdge][f.random_range(0..3)],
        deck: None,
        edging: None,
    };
    if version >= DECK_VERSION {
        let deck = ShelfDeck::sample(o);
        p.thickness_m = deck.depth_m;
        p.hardware = if f.random_bool(0.23) {
            Surface::Chrome
        } else {
            Surface::Metal
        };
        p.board = match deck.stock {
            ShelfStock::RoutedTimber => Surface::Wood,
            ShelfStock::BandedPanel => {
                if f.random_bool(0.47) {
                    Surface::Wood
                } else {
                    Surface::Plastic
                }
            }
            ShelfStock::FoldedSteel => Surface::Metal,
        };
        if deck.stock == ShelfStock::BandedPanel && f.random_bool(0.60) {
            p.edging = Some(Surface::WoodEdge);
        }
        p.deck = Some(deck);
    }
    p
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    if let Some(deck) = &p.deck {
        mesh::build(a, o, &p, deck);
    } else {
        legacy::build(a, o, &p);
    }
}
