use super::{
    super::{finishes::LayerFinishes, HumanOutfit},
    GarmentCut,
};
use bevy::prelude::*;

pub(super) fn band_center(cut: &GarmentCut, outfit: HumanOutfit, y: f32, side: f32) -> f32 {
    if outfit.open_front() {
        side * (cut.opening_half_width(y) + cut.program.placket_width * 0.60)
    } else {
        0.
    }
}

pub(super) fn buttons<'a>(
    cut: &'a GarmentCut,
    outfit: HumanOutfit,
    f: &'a LayerFinishes,
) -> impl Iterator<Item = (Vec2, f32)> + 'a {
    let (min, count_range, low, high, radius) = match outfit {
        HumanOutfit::Polo => (2, 3, cut.neck - 0.125, cut.neck - 0.05, 0.87),
        HumanOutfit::Blazer => (
            1,
            3,
            cut.waist + 0.033,
            cut.waist + (cut.chest - cut.waist) * 0.52,
            1.45,
        ),
        HumanOutfit::Cardigan => (3, 4, cut.waist + 0.025, cut.neck - 0.072, 1.13),
        HumanOutfit::Shirt => (5, 4, cut.waist + 0.022, cut.neck - 0.052, 0.90),
        _ => (0, 1, 0., 0., 0.),
    };
    let rows = min
        + (f.fastening_density * count_range as f32)
            .floor()
            .min((count_range - 1) as f32) as usize;
    (0..rows).map(move |i| {
        let t = if rows == 1 {
            0.55
        } else {
            i as f32 / (rows - 1) as f32
        };
        let y = low + (high - low) * t;
        let radius = f.button_radius * radius;
        let x = if outfit.open_front() {
            f.button_side
                * (cut.opening_half_width(y)
                    + (radius + 0.001).max(cut.program.placket_width * 0.60))
        } else {
            0.
        };
        (Vec2::new(x, y), radius)
    })
}
