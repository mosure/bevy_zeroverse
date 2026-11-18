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
    match style {
        0 => {
            let hue = pigment.random_range(18.0..48.0);
            let sat = pigment.random_range(0.18..0.48);
            let value = pigment.random_range(0.17..0.78);
            r.color = super::pigment::hsl(hue, sat, value);
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
            let c = r.carpet.as_mut().unwrap();
            c.dye = Some(CarpetDye::sample(r.seed));
            c.contrast = pigment.random_range(0.08..0.72);
            c.line_width = product.random_range(0.035..0.34);
            c.palette = super::pigment::yarns(&mut pigment);
            r.color = c.palette[0];
            c.pile = finish.random_range(0.10..0.95);
            // Unresolved fibres broaden the highlight. The normal map captures
            // the relief of yarn bundles that survive the atlas footprint.
            r.roughness = 0.80 + 0.12 * c.pile + finish.random_range(0.0..0.065);
            r.relief_m = 0.00010 + c.pile * finish.random_range(0.00024..0.00061);
        }
        _ => {
            let m = r.mineral.as_mut().unwrap();
            r.color = super::pigment::mineral(&mut pigment, m.marble_mix);
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
    let joint_color = if style == 0 {
        let tone = pigment.random_range(0.035..0.18);
        [tone, tone * 0.97, tone * 0.94]
    } else {
        super::pigment::joint(&mut pigment, r.color, style == 1)
    };
    f.stock = Some(stock::FloorStock {
        cut_jitter: [0.; 2],
        mottle_cells: [product.random_range(1..=11), product.random_range(1..=11)],
        joint_color,
        joint_roughness: if style == 1 {
            finish.random_range(0.84..0.98)
        } else {
            finish.random_range(0.65..0.97)
        },
    });
    r.joint_width = product.random_range(if style == 0 {
        0.0005..0.0025
    } else {
        0.001..0.006
    });
    r.flooring = Some(f);
    super::scale::resolve(r, style);
}
