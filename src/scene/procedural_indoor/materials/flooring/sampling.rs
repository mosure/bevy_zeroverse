//! Independent product, pigment, cut-stock and finish domains. Continuous
//! variation stays inside buildable floor products, not arbitrary RGB noise.
use super::super::{carpet::dye::CarpetDye, Surface};
use super::*;

pub(in super::super) fn expand(r: &mut MaterialRecipe, style: u32) {
    debug_assert_eq!(r.surface, Surface::Floor);
    let mut product = stream(r.seed, 0x464c50524f445543);
    let mut pigment = stream(r.seed, 0x464c5049474d454e);
    let mut finish = stream(r.seed, 0x464c46494e495348);
    let seamless = r
        .flooring
        .as_ref()
        .is_some_and(|f| f.layout == FloorLayout::Seamless);
    r.period_m = super::scale::period(r.seed, seamless);
    let hue = pigment.random_range(if style == 0 { 18.0..48.0 } else { 0.0..360.0 });
    let saturation = if style == 0 {
        pigment.random_range(0.18..0.48)
    } else if pigment.random_bool(0.65) {
        // Quiet gray/warm-white floors are common. Keep a broad colored tail
        // without turning every office into a saturated carpet/tile showroom.
        pigment.random_range(0.005..0.07)
    } else {
        pigment.random_range(0.08..0.30)
    };
    let lightness = pigment.random_range(if style == 1 { 0.12..0.58 } else { 0.17..0.78 });
    let c = bevy::prelude::Color::hsl(hue, saturation, lightness).to_srgba();
    r.color = [c.red, c.green, c.blue];
    match style {
        0 => {
            let w = r.wood.as_mut().unwrap();
            w.clearcoat = finish.random_range(0.0_f32..1.).powi(2) * 0.9;
            w.coat_roughness = finish.random_range(0.09..0.44);
            w.stain_strength = pigment.random_range(0.0_f32..1.).powi(2) * 0.85;
            w.bleach = pigment.random_range(0.0_f32..1.).powi(3) * 0.65 * (1. - w.stain_strength);
            w.pore_fill = (finish.random_range(0.05..0.65) + w.clearcoat * 0.3).min(0.95);
            r.roughness = finish.random_range(0.30..0.82);
            r.relief_m = finish.random_range(0.00003..0.00020) * (1. - w.pore_fill * 0.7);
            r.grain_frequency = product.random_range(17.0..78.0);
        }
        1 => {
            r.roughness = finish.random_range(0.82..0.98);
            r.relief_m = finish.random_range(0.00015..0.00065);
            let c = r.carpet.as_mut().unwrap();
            c.dye = Some(CarpetDye::sample(r.seed));
            c.contrast = pigment.random_range(0.08..0.72);
            c.line_width = product.random_range(0.035..0.34);
            c.palette[0] = r.color;
            let pale = pigment.random_range(0.08..0.55);
            c.palette[1] = r.color.map(|v| (v + (0.9 - v) * pale).min(0.94));
            let dark = pigment.random_range(0.28..0.85);
            c.palette[2] = r.color.map(|v| v * dark);
            c.pile = finish.random_range(0.10..0.95);
        }
        _ => {
            let m = r.mineral.as_mut().unwrap();
            m.polish = finish.random_range(0.0..1.);
            r.roughness = 0.86 - m.polish * 0.66;
            r.relief_m = finish.random_range(0.000025..0.00016) * (1. - m.polish * 0.7);
            // Chip sizing and vein geometry vary independently of polish and
            // installation. Keep quiet porcelain and strongly figured stone.
            m.aggregate_radius = product.random_range(0.12..0.45);
            m.vein_width = product.random_range(0.003..0.028);
            m.vein_warp = product.random_range(0.03..0.42);
        }
    }
    r.relief_m *= r.detail_scale.unwrap_or(1.);
    let mut f = FloorFinish::sample(r.seed, style, r.period_m);
    let tone: f32 = pigment.random_range(if style == 0 { 0.035..0.18 } else { 0.12..0.82 });
    f.stock = Some(stock::FloorStock {
        cut_jitter: [0.; 2],
        mottle_cells: [product.random_range(1..=11), product.random_range(1..=11)],
        joint_color: [tone, tone * 0.97, tone * 0.94],
        joint_roughness: finish.random_range(0.65..0.97),
    });
    r.joint_width = product.random_range(if style == 0 {
        0.0005..0.0025
    } else {
        0.001..0.006
    });
    r.flooring = Some(f);
    super::scale::resolve(r, style);
}
